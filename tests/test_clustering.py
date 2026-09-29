"""Faces-service unit tests (no infra required; cv2/insightface not needed)."""
import sys
import os

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import clustering  # noqa: E402


def test_cosine_identical():
    a = np.array([1.0, 0, 0], dtype=np.float32)
    assert clustering.cosine_similarity(a, a) > 0.999


def test_cosine_opposite():
    a = np.array([1.0, 0, 0], dtype=np.float32)
    b = np.array([-1.0, 0, 0], dtype=np.float32)
    assert clustering.cosine_similarity(a, b) < -0.999


def test_match_cluster_prefers_named():
    emb = np.array([1.0, 0, 0], dtype=np.float32)
    clusters = [
        {"id": "n", "centroid": [1, 0, 0], "is_named": True},
        {"id": "u", "centroid": [0, 1, 0], "is_named": False},
    ]
    cid, sim = clustering.match_cluster(emb, clusters, 0.4, 0.5)
    assert cid == "n"
    assert sim >= 0.99


def test_match_cluster_anonymous_stricter():
    # anonymous cluster requires >= unknown_threshold (0.5)
    emb = np.array([0.55, 0.6, 0.0], dtype=np.float32)
    emb = emb / np.linalg.norm(emb)
    clusters = [
        {"id": "u", "centroid": [1, 0, 0], "is_named": False},
    ]
    cid, sim = clustering.match_cluster(emb, clusters, 0.4, 0.5)
    assert cid == "u"


def test_match_cluster_no_match():
    emb = np.array([1.0, 0, 0], dtype=np.float32)
    clusters = [
        {"id": "u", "centroid": [0, 1, 0], "is_named": False},
        {"id": "n", "centroid": [0, -1, 0], "is_named": True},
    ]
    cid, sim = clustering.match_cluster(emb, clusters, 0.4, 0.5)
    assert cid is None


def test_merge_centroid():
    out = clustering.merge_centroid(np.array([2.0, 0, 0]), 1, np.array([4.0, 0, 0]))
    assert abs(out[0] - 3.0) < 1e-6
    out0 = clustering.merge_centroid(np.zeros(3), 0, np.array([4.0, 0, 0]))
    assert abs(out0[0] - 4.0) < 1e-6


def _cluster(cid, vec, is_named=False, sample_count=1, name=None):
    return {
        "id": cid,
        "centroid": [float(v) for v in vec],
        "is_named": is_named,
        "sample_count": sample_count,
        "name": name,
    }


def test_build_similarity_groups_links_near_duplicates():
    v = [1.0, 0.0, 0.0]
    clusters = [
        _cluster("a", v, sample_count=5),
        _cluster("b", v, sample_count=2),
        _cluster("c", [0.0, 1.0, 0.0]),
    ]
    groups = clustering.build_similarity_groups(clusters, 0.5)
    assert len(groups) == 1
    g = groups[0]
    assert g["rep"]["id"] == "a"
    assert {m["cluster"]["id"] for m in g["members"]} == {"b"}
    assert g["maxSim"] > 0.99


def test_build_similarity_groups_named_rep_wins():
    v = [1.0, 0.0, 0.0]
    clusters = [
        _cluster("unnamed-big", v, sample_count=9),
        _cluster("named", v, is_named=True, sample_count=1, name="Alex"),
    ]
    groups = clustering.build_similarity_groups(clusters, 0.5)
    assert len(groups) == 1
    assert groups[0]["rep"]["id"] == "named"


def test_build_similarity_groups_members_sorted_by_sim():
    clusters = [
        _cluster("rep", [1.0, 0.0, 0.0], is_named=True, name="A"),
        _cluster("near", [0.99, 0.01, 0.0]),
        _cluster("far", [0.5, 0.866, 0.0]),
    ]
    groups = clustering.build_similarity_groups(clusters, 0.3)
    assert len(groups) == 1
    ids = [m["cluster"]["id"] for m in groups[0]["members"]]
    assert ids == ["near", "far"]


def test_build_similarity_groups_groups_ordered_by_maxsim():
    clusters = [
        _cluster("c", [1.0, 0.0, 0.0], sample_count=4),  # maxSim 1.0 (identical)
        _cluster("c2", [1.0, 0.0, 0.0], sample_count=4),
        _cluster("d", [0.0, 1.0, 0.0], sample_count=3),
        _cluster("d2", [0.0, 0.999, 0.0], sample_count=3),  # maxSim < 1.0
    ]
    groups = clustering.build_similarity_groups(clusters, 0.9)
    assert [g["rep"]["id"] for g in groups] == ["c", "d"]
    assert groups[0]["maxSim"] >= groups[1]["maxSim"]


def test_build_similarity_groups_excludes_singles_and_below_threshold():
    clusters = [
        _cluster("n1", [1.0, 0.0, 0.0]),
        _cluster("n2", [0.999, 0.0, 0.0]),
        _cluster("solo", [0.0, 1.0, 0.0]),
        _cluster("other", [0.0, 0.0, 1.0]),
    ]  # n1/n2 group, then two unrelated
    groups = clustering.build_similarity_groups(clusters, 0.9)
    assert len(groups) == 1
    all_ids = {g["rep"]["id"] for g in groups}
    all_ids |= {m["cluster"]["id"] for g in groups for m in g["members"]}
    assert all_ids == {"n1", "n2"}


def test_build_similarity_groups_empty_and_no_centroid():
    assert clustering.build_similarity_groups([], 0.5) == []
    clusters = [_cluster("x", [])]
    assert clustering.build_similarity_groups(clusters, 0.5) == []


def test_build_similarity_groups_no_group_for_two_unrelated():
    clusters = [
        _cluster("a", [1.0, 0.0, 0.0]),
        _cluster("b", [0.0, 1.0, 0.0]),
    ]
    assert clustering.build_similarity_groups(clusters, 0.5) == []


# --- interactive frame assist -------------------------------------------------
# The paused-frame assist must *offer* the nearest identity even when it is below
# the batch match/unknown thresholds, and must stay silent when nothing is close.
# Matching runs against a prebuilt matrix, so these also pin the index/meta
# alignment: a suggestion must come back with the cluster it was scored against.


def _unit(vec):
    v = np.asarray(vec, dtype=np.float32)
    return v / np.linalg.norm(v)


def _match(clusters, embedding, floor=0.5):
    matrix, meta = clustering.build_suggestion_index(clusters)
    return clustering.best_match(matrix, meta, embedding, floor)


def test_best_match_offers_below_batch_thresholds():
    # 0.45 similarity: too weak for a silent attach, still worth showing
    best, sim = _match([_cluster("far", [0.0, 1.0, 0.0], is_named=True)], _unit([1.0, 0.45, 0.0]), floor=0.4)
    assert best is not None and best["id"] == "far"
    assert 0.4 <= sim < 0.5


def test_best_match_respects_floor():
    best, sim = _match([_cluster("orthogonal", [0.0, 1.0, 0.0], is_named=True)], _unit([1.0, 0.0, 0.0]), floor=0.5)
    assert best is None
    assert sim < 0.5


def test_best_match_picks_the_closest_and_returns_its_own_cluster():
    clusters = [
        _cluster("a", [0.0, 1.0, 0.0]),
        _cluster("b", [1.0, 0.0, 0.0]),
        _cluster("c", [1.0, 0.25, 0.0]),
    ]
    best, sim = _match(clusters, _unit([1.0, 0.2, 0.0]))
    assert best["id"] == "c"
    assert sim > 0.9


def test_best_match_similarities_match_cosine():
    clusters = [_cluster("a", [1.0, 0.0, 0.0]), _cluster("b", [0.0, 1.0, 0.0])]
    emb = _unit([0.8, 0.6, 0.0])
    _, sim = _match(clusters, emb, floor=0.0)
    assert abs(sim - clustering.cosine_similarity(emb, _unit([1.0, 0.0, 0.0]))) < 1e-5


def test_build_index_skips_clusters_without_centroid():
    matrix, meta = clustering.build_suggestion_index(
        [{"id": "x", "centroid": None, "is_named": True}, _cluster("y", [1.0, 0.0, 0.0])]
    )
    assert matrix.shape == (1, 3)
    assert [c["id"] for c in meta] == ["y"]


def test_build_index_empty():
    matrix, meta = clustering.build_suggestion_index([])
    assert matrix.size == 0 and meta == []
    best, sim = clustering.best_match(matrix, meta, _unit([1.0, 0.0, 0.0]), 0.5)
    assert best is None and sim == 0.0


def test_best_match_handles_zero_query():
    matrix, meta = clustering.build_suggestion_index([_cluster("a", [1.0, 0.0, 0.0])])
    best, sim = clustering.best_match(matrix, meta, np.zeros(3, dtype=np.float32), 0.5)
    assert best is None and sim == 0.0


def test_best_match_tolerates_zero_centroid():
    _, sim = _match([_cluster("z", [0.0, 0.0, 0.0])], _unit([1.0, 0.0, 0.0]), floor=0.0)
    assert sim == 0.0  # a zero centroid must not produce a NaN match


def test_iou_identical_and_disjoint():
    box = [10.0, 10.0, 50.0, 50.0]
    assert abs(clustering.iou(box, box) - 1.0) < 1e-9
    assert clustering.iou(box, [100.0, 100.0, 120.0, 120.0]) == 0.0


def test_iou_half_overlap():
    a = [0.0, 0.0, 10.0, 10.0]
    b = [5.0, 0.0, 15.0, 10.0]
    # intersection 50, union 150
    assert abs(clustering.iou(a, b) - 50.0 / 150.0) < 1e-9


def test_iou_touching_edges_is_zero():
    # the detector and the client can disagree by a pixel; a shared edge is no hit
    assert clustering.iou([0.0, 0.0, 10.0, 10.0], [10.0, 0.0, 20.0, 10.0]) == 0.0


# --- suggestion index cache ---------------------------------------------------


def test_cache_builds_once_and_reuses():
    cache = clustering.SuggestionIndexCache(ttl_seconds=300.0)
    calls = []

    def builder():
        calls.append(1)
        return clustering.build_suggestion_index([_cluster("a", [1.0, 0.0, 0.0])])

    first = cache.get("owner", builder)
    second = cache.get("owner", builder)
    assert len(calls) == 1
    assert first is second


def test_cache_bump_forces_a_rebuild():
    cache = clustering.SuggestionIndexCache(ttl_seconds=300.0)
    calls = []

    def builder():
        calls.append(1)
        return clustering.build_suggestion_index([_cluster("a", [1.0, 0.0, 0.0])])

    cache.get("owner", builder)
    cache.bump("owner")
    cache.get("owner", builder)
    assert len(calls) == 2


def test_cache_bump_is_per_owner():
    cache = clustering.SuggestionIndexCache(ttl_seconds=300.0)
    calls = []

    def builder_a():
        calls.append("a")
        return clustering.build_suggestion_index([_cluster("a", [1.0, 0.0, 0.0])])

    def builder_b():
        calls.append("b")
        return clustering.build_suggestion_index([_cluster("b", [0.0, 1.0, 0.0])])

    cache.get("a", builder_a)
    cache.get("b", builder_b)
    cache.bump("a")
    cache.get("a", builder_a)  # a rebuilt
    cache.get("b", builder_b)  # b untouched
    assert calls == ["a", "b", "a"]


def test_cache_expires_after_ttl():
    now = [1000.0]
    cache = clustering.SuggestionIndexCache(ttl_seconds=10.0)
    cache._clock = lambda: now[0]  # deterministic clock
    calls = []

    def builder():
        calls.append(1)
        return clustering.build_suggestion_index([_cluster("a", [1.0, 0.0, 0.0])])

    cache.get("owner", builder)
    now[0] += 9.0
    cache.get("owner", builder)
    assert len(calls) == 1
    now[0] += 2.0  # past the TTL
    cache.get("owner", builder)
    assert len(calls) == 2


def test_cache_serves_a_fresh_index_after_a_write():
    # the shape db.py relies on: bump then get must not hand back the old matrix
    cache = clustering.SuggestionIndexCache(ttl_seconds=300.0)
    state = {"clusters": [_cluster("a", [1.0, 0.0, 0.0])]}

    def builder():
        return clustering.build_suggestion_index(state["clusters"])

    matrix, meta = cache.get("owner", builder)
    assert meta[0]["id"] == "a"
    # simulate a rename reaching the index
    state["clusters"] = [_cluster("a", [1.0, 0.0, 0.0], name="Alice", is_named=True)]
    cache.bump("owner")
    matrix, meta = cache.get("owner", builder)
    assert meta[0]["name"] == "Alice"
