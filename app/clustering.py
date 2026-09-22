import logging

import numpy as np

logger = logging.getLogger("faces-service")


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    if norm == 0:
        return 0.0
    return float(np.dot(a, b) / norm)


def match_cluster(
    embedding: np.ndarray,
    clusters: list[dict],
    match_threshold: float,
    unknown_threshold: float,
) -> tuple[str | None, float]:
    """Return (cluster_id, similarity) of the best match, or (None, best_sim).

    Named clusters require >= match_threshold (lenient); anonymous clusters require
    >= unknown_threshold (stricter) — the idea being that a named identity is an
    intentional, consistent person, while anonymous blobs should not swallow new faces.
    """
    best_id: str | None = None
    best_sim = match_threshold
    for c in clusters:
        centroid = np.asarray(c["centroid"], dtype=np.float32)
        sim = cosine_similarity(embedding, centroid)
        threshold = match_threshold if c["is_named"] else unknown_threshold
        if sim >= threshold and (best_id is None or sim > best_sim):
            best_id = c["id"]
            best_sim = sim
    return best_id, best_sim


def merge_centroid(existing: np.ndarray, count: int, new: np.ndarray) -> np.ndarray:
    """Running mean: (old*count + new) / (count+1)."""
    if count <= 0:
        return new.astype(np.float32)
    return ((existing * count) + new) / (count + 1)