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