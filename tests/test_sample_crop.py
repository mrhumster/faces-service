"""Rebuilding a sample crop from a stored occurrence: stubs only.

The frame is fetched from MinIO, the detector re-runs and the face matching the
occurrence's stored embedding is cropped. No cv2/insightface/MinIO needed here.
"""
import os
import sys
import types

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.routes import reader  # noqa: E402
from app.routes.reader import _sample_frame_index  # noqa: E402


class _Req:
    def __init__(self, store, engine):
        self.app = types.SimpleNamespace(state=types.SimpleNamespace(store=store, engine=engine))


class _Store:
    def __init__(self, frame=None):
        self.frame = frame
        self.calls: list[tuple[str, int]] = []

    def get_frame(self, prefix, index):
        self.calls.append((prefix, index))
        return self.frame


class _Engine:
    def __init__(self, detections):
        self._detections = detections
        self.cropped: list | None = None
        self.model = types.SimpleNamespace(embed=self._embed)

    def _embed(self, *_args, **_kwargs):
        return self._detections

    def _crop_jpeg(self, _img, bbox):
        self.cropped = bbox
        return b"jpeg"


def _img():
    return np.zeros((64, 64, 3), dtype=np.uint8)


class TestFrameIndex:
    def test_round_trips_batch_timestamps(self):
        for idx in range(0, 200):
            t = (idx + 0.5) * reader.SAMPLE_INTERVAL_SECONDS
            assert _sample_frame_index(t) == idx

    def test_nearest_sample_for_arbitrary_t(self):
        # t=12.3 sits nearest to the 12.5s sample (idx 2)
        assert _sample_frame_index(12.3) == 2

    def test_negative_t(self):
        assert _sample_frame_index(-1.0) == -1


class TestRenderSampleCrop:
    @pytest.fixture(autouse=True)
    def _stub_decode(self, monkeypatch):
        monkeypatch.setattr(reader, "_decode_image", lambda _raw: _img())

    def test_missing_frame_returns_none(self, monkeypatch):
        monkeypatch.setattr(
            reader.db, "get_cluster_occurrence", lambda *a, **k: {"embedding": [1.0], "confidence": 1.0}
        )
        store = _Store(frame=None)
        req = _Req(store, _Engine([]))

        assert reader._render_sample_crop(req, {"id": "c"}, "s", 2.5) is None
        assert store.calls == [("faces/s", 0)]

    def test_missing_occurrence_returns_none(self, monkeypatch):
        monkeypatch.setattr(reader.db, "get_cluster_occurrence", lambda *a, **k: None)
        store = _Store(frame=b"x")
        req = _Req(store, _Engine([]))

        assert reader._render_sample_crop(req, {"id": "c"}, "s", 2.5) is None
        # occurrence is checked before the frame is fetched
        assert store.calls == []

    def test_picks_face_matching_stored_embedding(self, monkeypatch):
        target = np.array([0.0, 1.0], dtype=np.float32)
        monkeypatch.setattr(
            reader.db,
            "get_cluster_occurrence",
            lambda *a, **k: {"embedding": target.tolist(), "confidence": 1.0},
        )
        wrong = {"bbox": [0, 0, 1, 1], "embedding": np.array([1.0, 0.0], dtype=np.float32)}
        right = {"bbox": [8, 8, 40, 40], "embedding": np.array([0.0, 1.0], dtype=np.float32)}
        engine = _Engine([wrong, right])
        req = _Req(_Store(frame=b"x"), engine)

        assert reader._render_sample_crop(req, {"id": "c"}, "s", 2.5) == b"jpeg"
        assert engine.cropped == [8, 8, 40, 40]

    def test_no_close_face_returns_none(self, monkeypatch):
        monkeypatch.setattr(
            reader.db,
            "get_cluster_occurrence",
            lambda *a, **k: {"embedding": [0.0, 1.0], "confidence": 1.0},
        )
        far = {"bbox": [0, 0, 1, 1], "embedding": np.array([1.0, 0.0], dtype=np.float32)}
        engine = _Engine([far])
        req = _Req(_Store(frame=b"x"), engine)

        assert reader._render_sample_crop(req, {"id": "c"}, "s", 2.5) is None
        assert engine.cropped is None
