import logging
import time

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


def build_suggestion_index(clusters: list[dict]) -> tuple[np.ndarray, list[dict]]:
    """Stack cluster centroids into one L2-normalized matrix so that matching a
    face becomes a single matrix-vector product.

    Rebuilding this per paused frame is what made the interactive assist slow:
    with a few thousand people the centroid list is megabytes and a per-cluster
    Python loop dominated the request. The matrix is built once and reused, so
    this is only paid on the (rare) invalidation.
    """
    vectors: list[np.ndarray] = []
    meta: list[dict] = []
    for c in clusters:
        centroid = c.get("centroid")
        if not centroid:
            continue
        vectors.append(np.asarray(centroid, dtype=np.float32))
        meta.append(c)
    if not vectors:
        return np.zeros((0, 0), dtype=np.float32), []
    matrix = np.stack(vectors).astype(np.float32)
    norms = np.linalg.norm(matrix, axis=1)
    norms[norms == 0] = 1.0
    return matrix / norms[:, None], meta


def best_match(
    matrix: np.ndarray,
    meta: list[dict],
    embedding: np.ndarray,
    floor: float,
) -> tuple[dict | None, float]:
    """Nearest cluster to a face the user just paused on, ignoring the batch
    match/unknown thresholds: the assist is meant to *offer* a candidate and let
    a human decide, so the only gate is `floor` (how similar is too far to even
    mention).

    Returns (cluster | None, similarity). None means "no candidate worth showing"
    — the caller then offers to create a new person. Whether a candidate is
    strong enough to attach silently is the caller's call (it needs the config
    threshold).
    """
    if matrix.size == 0:
        return None, 0.0
    query = np.asarray(embedding, dtype=np.float32)
    norm = float(np.linalg.norm(query))
    if norm == 0:
        return None, 0.0
    sims = matrix @ (query / norm)
    index = int(np.argmax(sims))
    similarity = float(sims[index])
    if similarity < floor:
        return None, similarity
    return meta[index], similarity


class SuggestionIndexCache:
    """Per-owner cache of the suggestion index.

    Entries are dropped by `bump(owner_id)` from every write path, so a fresh
    read is the normal case and the TTL is only a backstop against a write path
    that forgets to bump. The TTL is deliberately generous: a cold build costs
    about two seconds for a large library, so a short one would rebuild constantly.
    """

    def __init__(self, ttl_seconds: float = 300.0) -> None:
        self._ttl = ttl_seconds
        self._entries: dict[str, tuple[int, float, tuple[np.ndarray, list[dict]]]] = {}
        self._epoch: dict[str, int] = {}
        self._clock = time.monotonic

    def epoch(self, owner_id: str) -> int:
        return self._epoch.get(owner_id, 0)

    def bump(self, owner_id: str) -> None:
        """Drop the owner's index; the next read rebuilds it."""
        self._epoch[owner_id] = self.epoch(owner_id) + 1
        self._entries.pop(owner_id, None)

    def get(self, owner_id: str, builder) -> tuple[np.ndarray, list[dict]]:
        epoch = self.epoch(owner_id)
        entry = self._entries.get(owner_id)
        if entry is not None:
            stored_epoch, stored_at, value = entry
            if stored_epoch == epoch and (self._clock() - stored_at) < self._ttl:
                return value
        value = builder()
        self._entries[owner_id] = (epoch, self._clock(), value)
        return value

    def clear(self) -> None:
        self._entries.clear()


# Shared by the reader routes and every write path in db.py.
SUGGESTION_INDEX = SuggestionIndexCache()


def iou(a: list[float], b: list[float]) -> float:
    """Intersection-over-union of two [x1, y1, x2, y2] boxes."""
    ax1, ay1, ax2, ay2 = (float(v) for v in a)
    bx1, by1, bx2, by2 = (float(v) for v in b)
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0


SIMILARITY_THRESHOLD = 0.5


def build_similarity_groups(clusters: list[dict], threshold: float = SIMILARITY_THRESHOLD):
    """Group near-duplicate clusters by pairwise centroid cosine similarity
    (union-find connected components). Mirrors the browser-side implementation
    so the API never ships overlapping duplicates to the UI.

    Returns a list of groups:
        {"rep": dict, "members": [{"cluster": dict, "sim": float}], "maxSim": float}
    rep is the named cluster with the largest sample_count (or just the largest
    sample_count when none are named); members are sorted by similarity to rep
    descending; groups are ordered by maxSim descending. Clusters without a
    centroid are ignored. Clusters not part of any edge stay out of the result.
    """
    with_vec = [c for c in clusters if c.get("centroid")]
    if len(with_vec) < 2:
        return []

    centroids = np.stack(
        [np.asarray(c["centroid"], dtype=np.float32) for c in with_vec]
    )
    norms = np.linalg.norm(centroids, axis=1)
    norms[norms == 0] = 1.0
    centroids = centroids / norms[:, None]
    sims = np.clip(centroids @ centroids.T, -1.0, 1.0)

    idx = {c["id"]: i for i, c in enumerate(with_vec)}

    parent = {c["id"]: c["id"] for c in with_vec}

    def find(x: str) -> str:
        root = x
        while parent[root] != root:
            root = parent[root]
        while parent[x] != root:
            nxt = parent[x]
            parent[x] = root
            x = nxt
        return root

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for i in range(len(with_vec)):
        for j in range(i + 1, len(with_vec)):
            if float(sims[i, j]) >= threshold:
                union(with_vec[i]["id"], with_vec[j]["id"])

    by_root: dict[str, list[dict]] = {}
    for c in with_vec:
        by_root.setdefault(find(c["id"]), []).append(c)

    def member_sim(rep_id: str, c: dict) -> float:
        return float(sims[idx[rep_id], idx[c["id"]]])

    groups = []
    for members_list in by_root.values():
        if len(members_list) < 2:
            continue
        rep = max(
            members_list,
            key=lambda c: (int(c["is_named"]), c["sample_count"]),
        )
        members = [
            {"cluster": c, "sim": member_sim(rep["id"], c)}
            for c in members_list
            if c["id"] != rep["id"]
        ]
        members.sort(key=lambda m: m["sim"], reverse=True)
        max_sim = max((m["sim"] for m in members), default=0.0)
        groups.append({"rep": rep, "members": members, "maxSim": max_sim})

    groups.sort(key=lambda g: g["maxSim"], reverse=True)
    return groups