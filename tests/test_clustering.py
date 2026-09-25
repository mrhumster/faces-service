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