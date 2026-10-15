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


def face_is_large_enough(
    bbox: list[float],
    frame_shape: tuple[int, int, int] | tuple[int, int],
    min_ratio: float,
) -> bool:
    """Whether a detected face is worth recognising.

    Recognition is by far the most expensive part of the paused-frame assist
    (measured ~589ms per face), and a face that small cannot produce an
    embedding good enough to match anyone — so the interactive path drops it
    before paying for it, and the UI never shows a box that has no name behind
    it. `min_ratio` is relative to the frame height; 0 keeps everything, which
    is what the batch pipeline relies on for recall.
    """
    if min_ratio <= 0:
        return True
    height = float(bbox[3]) - float(bbox[1])
    if height <= 0:
        return False
    return height >= min_ratio * float(frame_shape[0])


SIMILARITY_THRESHOLD = 0.5

# Similarity that renders as "100%" in the UI (the client rounds to whole
# percents), so a group whose *every* member clears this is an exact duplicate
# pair that needs no human decision.
EXACT_DUPLICATE_THRESHOLD = 0.995


def build_similarity_groups(clusters: list[dict], threshold: float = SIMILARITY_THRESHOLD):
    """Group clusters into cliques of mutually similar centroids.

    A greedy clique grows from each unassigned cluster (highest degree /
    named / most samples first): a candidate joins only if it is >= threshold
    to *every* member already in the clique. Connected-component grouping used
    to chain A~B~C into a group where C could be far from the displayed
    representative (even negative) — this keeps every pair, and therefore
    every member relative to the rep, at or above the threshold.

    A clique holds at most one *named* cluster: a group is a merge suggestion,
    and two named people must never be suggested for merging (the second named
    candidate is left unassigned so it can seed its own clique).

    Returns a list of groups:
        {"rep": dict, "members": [{"cluster": dict, "sim": float}], "maxSim": float}
    rep is the named clique member with the largest sample_count (or just the
    largest sample_count when none are named); members are sorted by similarity
    to rep descending; groups are ordered by maxSim descending. Clusters without
    a centroid, or with no similar peer, stay out of the result.
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
    similar = sims >= threshold
    np.fill_diagonal(similar, False)

    degrees = similar.sum(axis=1)
    order = sorted(
        range(len(with_vec)),
        key=lambda i: (int(degrees[i]), int(with_vec[i]["is_named"]), with_vec[i]["sample_count"]),
        reverse=True,
    )

    assigned = np.zeros(len(with_vec), dtype=bool)
    groups = []
    for i in order:
        if assigned[i]:
            continue
        neighbors = [int(j) for j in np.where(similar[i])[0] if not assigned[j]]
        if not neighbors:
            continue
        neighbors.sort(key=lambda j: float(sims[i, j]), reverse=True)
        clique = [i]
        clique_has_named = bool(with_vec[i]["is_named"])
        for candidate in neighbors:
            # Never suggest merging two named people: a second named candidate
            # is skipped (left unassigned) so it can seed its own clique.
            if with_vec[candidate]["is_named"] and clique_has_named:
                continue
            if all(sims[candidate, member] >= threshold for member in clique):
                clique.append(candidate)
                clique_has_named = clique_has_named or bool(
                    with_vec[candidate]["is_named"]
                )
        if len(clique) < 2:
            continue
        for member in clique:
            assigned[member] = True
        rep = max(
            clique,
            key=lambda k: (int(with_vec[k]["is_named"]), with_vec[k]["sample_count"]),
        )
        members = [
            {"cluster": with_vec[k], "sim": float(sims[rep, k])}
            for k in clique
            if k != rep
        ]
        members.sort(key=lambda m: m["sim"], reverse=True)
        max_sim = max((m["sim"] for m in members), default=0.0)
        groups.append({"rep": with_vec[rep], "members": members, "maxSim": max_sim})

    groups.sort(key=lambda g: g["maxSim"], reverse=True)
    return groups


def select_exact_duplicate_groups(
    groups: list[dict], threshold: float = EXACT_DUPLICATE_THRESHOLD
) -> list[dict]:
    """Groups whose every member is >= threshold: identity is not in question,
    so merging them needs no human decision. The default threshold is what the
    UI renders as "100%". A group with even one weaker member is left for the
    user (it is exactly the case where a decision is required)."""
    return [
        g
        for g in groups
        if g["members"] and all(m["sim"] >= threshold for m in g["members"])
    ]