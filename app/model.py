import logging
import os

import numpy as np

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
            from insightface.app.face_analysis import FaceAnalysis
        except ImportError as e:  # pragma: no cover
            raise RuntimeError("insightface not installed") from e

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

    def embed(self, img: np.ndarray, detect_threshold: float) -> list[dict]:
        """Return [{embedding: np.ndarray (512,), confidence: float, bbox: [x1,y1,x2,y2] | None}]."""
        if self._app is None:
            self._load()
        self._app.det_thresh = detect_threshold
        faces = self._app.get(img)
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