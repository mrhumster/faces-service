import logging
import os

import numpy as np

from .clustering import face_is_large_enough

logger = logging.getLogger("faces-service")


class FaceModel:
    """InsightFace buffalo_l (det+recognition) loaded lazily with the model root baked
    into the image. CPUExecutionProvider only."""

    def __init__(self) -> None:
        self._app = None
        self._load()

    def _load(self) -> None:
        try:
            # Import the submodule directly: insightface/app/__init__.py pulls
            # app/mask_renderer.py -> albumentations, which we do not bundle.
            # Face comes from the same module, so no new import path is opened.
            from insightface.app.face_analysis import Face, FaceAnalysis
        except ImportError as e:  # pragma: no cover
            raise RuntimeError("insightface not installed") from e

        self._face_cls = Face
        root = os.path.abspath(os.path.expanduser(os.getenv("MODELS_ROOT", "/models")))
        # insightface's ensure_available('models', name, root) resolves the pack
        # to $(MODELS_ROOT)/models/<name> — check the exact dir it will load.
        if not os.path.exists(os.path.join(root, "models", "buffalo_l")):
            raise RuntimeError(f"buffalo_l model not found under MODELS_ROOT={root}")
        app = FaceAnalysis(
            name="buffalo_l",
            root=root,
            allowed_modules=["detection", "recognition"],
        )
        app.prepare(ctx_id=-1, det_size=(640, 640))  # -1 = CPU
        self._app = app
        logger.info("insightface buffalo_l loaded from %s", root)

    def _detect(
        self,
        img: np.ndarray,
        det_size: tuple[int, int] | None,
        min_face_ratio: float = 0.0,
    ):
        """Run detection (+ recognition for every hit) like FaceAnalysis.get,
        but allow a smaller detection input and a size floor for the interactive
        path.

        SCRFD scales the long side to `input_size` and pads to a square, then
        divides the boxes by that scale — so bboxes come back in the pixels of
        `img` either way, and the embeddings are produced by exactly the same
        kps-aligned crops. The only difference is how much compute the detector
        burns, which is quadratic in the size. The ONNX sessions are shared, so
        this costs no extra memory.

        `min_face_ratio` skips recognition for faces below that share of the
        frame height — recognition dominates the request cost, and a face that
        small cannot be matched anyway.
        """
        if det_size is None and min_face_ratio <= 0:
            return self._app.get(img)
        app = self._app
        bboxes, kpss = app.det_model.detect(img, input_size=det_size, metric="default")
        if bboxes.shape[0] == 0:
            return []
        out = []
        for i in range(bboxes.shape[0]):
            bbox = bboxes[i, 0:4]
            if not face_is_large_enough(bbox, img.shape, min_face_ratio):
                continue
            face = self._face_cls(
                bbox=bbox,
                kps=None if kpss is None else kpss[i],
                det_score=bboxes[i, 4],
            )
            for taskname, model in app.models.items():
                if taskname == "detection":
                    continue
                model.get(img, face)
            out.append(face)
        return out

    def embed(
        self,
        img: np.ndarray,
        detect_threshold: float,
        det_size: int | None = None,
        min_face_ratio: float = 0.0,
    ) -> list[dict]:
        """Return [{embedding: np.ndarray (512,), confidence: float, bbox: [x1,y1,x2,y2] | None}].

        `det_size` shrinks the detector's input for the interactive frame assist,
        where a whole paused frame has to be answered within a few hundred ms.
        `min_face_ratio` drops faces too small to recognise. Both are left unset
        for the batch pipeline, which wants full recall.
        """
        if self._app is None:
            self._load()
        self._app.det_thresh = detect_threshold
        size = None if not det_size or det_size <= 0 else (det_size, det_size)
        faces = self._detect(img, size, min_face_ratio)
        out = []
        for f in faces:
            if f.embedding is None or f.det_score is None:
                continue
            emb = np.asarray(f.embedding, dtype=np.float32).flatten()
            norm = np.linalg.norm(emb)
            if norm == 0 or norm > 100:
                continue
            emb = emb / norm
            det = {"embedding": emb, "confidence": float(f.det_score)}
            bbox = getattr(f, "bbox", None)
            if bbox is not None and len(bbox) == 4:
                det["bbox"] = [float(v) for v in bbox]
            else:
                det["bbox"] = None
            out.append(det)
        return out