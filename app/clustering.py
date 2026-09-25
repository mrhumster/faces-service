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