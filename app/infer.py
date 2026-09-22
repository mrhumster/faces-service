import logging

import cv2
import numpy as np

from . import config, db
from .clustering import match_cluster, merge_centroid
from .model import FaceModel
from .minio_client import FrameStore

logger = logging.getLogger("faces-service")

SAMPLE_INTERVAL_SECONDS = 3.0


def _thresholds() -> tuple[float, float]:
    return (config.Config.match_threshold, config.Config.unknown_threshold)


class InferEngine:
    def __init__(self, model: FaceModel, store: FrameStore) -> None:
        self.model = model
        self.store = store

    def run(self, stream_id: str, owner_id: str, frames_prefix: str, count: int) -> dict:
        """Reads `count` frames from MinIO under frames_prefix, runs face detection/
        embedding per frame and clusters into this owner's clusters."""
        clusters = db.get_clusters_for_owner(owner_id)
        cluster_map: dict[str, dict] = {c["id"]: c for c in clusters}
        drift: dict[str, list[float]] = {}  # cluster_id -> updated centroid

        items: list[tuple[str, list[float], float, float]] = []
        faces_detected = 0
        frame_skipped = 0
        created_ids: list[str] = []

        for idx in range(count):
            buf = self.store.get_frame_io(frames_prefix, idx)
            if buf is None:
                frame_skipped += 1
                continue
            raw, _ = buf
            img = decode_img(raw)
            if img is None:
                frame_skipped += 1
                continue

            t_seconds = (idx + 0.5) * SAMPLE_INTERVAL_SECONDS
            match_threshold, unknown_threshold = _thresholds()
            embeddings = self.model.embed(img, config.Config.detect_threshold)
            for det in embeddings:
                faces_detected += 1
                emb = det["embedding"]
                cluster_id, sim = match_cluster(
                    emb, clusters, match_threshold, unknown_threshold
                )
                if cluster_id is None or cluster_id not in cluster_map:
                    centroid = emb.astype(np.float32)
                    new_id = db.upsert_cluster(owner_id, None, False, centroid.tolist())
                    logger.info("new anonymous cluster %s (sim=%.3f)", new_id, sim)
                    cluster_id = new_id
                    nc = {
                        "id": new_id,
                        "name": None,
                        "is_named": False,
                        "centroid": centroid.tolist(),
                        "sample_count": 0,
                    }
                    clusters.append(nc)
                    cluster_map[new_id] = nc
                    drift[new_id] = centroid.tolist()
                    created_ids.append(new_id)
                else:
                    c = cluster_map[cluster_id]
                    centroid = np.asarray(c["centroid"], dtype=np.float32)
                    merged = merge_centroid(centroid, c["sample_count"], emb)
                    c["centroid"] = merged.tolist()
                    c["sample_count"] += 1
                    drift[cluster_id] = merged.tolist()
                items.append(
                    (cluster_id, emb.astype(np.float32).tolist(), t_seconds, det["confidence"])
                )

        written = db.append_stream_occurrences(owner_id, stream_id, items)
        for cluster_id, centroid in drift.items():
            db.update_cluster_centroid(cluster_id, owner_id, centroid)

        return {
            "faces_detected": faces_detected,
            "frames_processed": count - frame_skipped,
            "frames_skipped": frame_skipped,
            "occurrences_written": written,
            "clusters_created": created_ids,
        }


def decode_img(raw: bytes) -> np.ndarray | None:
    arr = np.frombuffer(raw, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None or img.shape[0] == 0 or img.shape[1] == 0:
        return None
    return img