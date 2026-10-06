"""Subsampling and cancellation: stubs only, no cv2/insightface/MinIO needed."""
import sys
import os

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import infer as inf  # noqa: E402
from app.infer import InferenceAborted, frame_indices  # noqa: E402


class _Store:
    """Every frame is missing, so run() walks the whole index list cheaply."""

    def __init__(self):
        self.seen: list[int] = []

    def get_frame_io(self, prefix, idx):
        self.seen.append(idx)
        return None


def _engine(store):
    engine = inf.InferEngine.__new__(inf.InferEngine)
    engine.store = store
    engine.model = None  # never reached: no frame ever decodes
    return engine


@pytest.fixture(autouse=True)
def _no_db(monkeypatch):
    monkeypatch.setattr(inf.db, "get_clusters_for_owner", lambda owner: [])
    monkeypatch.setattr(inf.db, "append_stream_occurrences", lambda owner, sid, items: 0)
    monkeypatch.setattr(inf.db, "update_cluster_centroid", lambda cid, owner, c: None)


class TestFrameIndices:
    def test_empty_count(self):
        assert frame_indices(0) == []

    def test_under_cap_is_dense(self):
        assert frame_indices(10, 500) == list(range(10))

    def test_exact_cap_is_dense(self):
        assert frame_indices(500, 500) == list(range(500))

    def test_no_cap_is_dense(self):
        assert frame_indices(690, 0) == list(range(690))

    def test_over_cap_subsamples_whole_video(self):
        # 690 frames of a 57 min recording under the 500 frame cap.
        idx = frame_indices(690, 500)

        assert len(idx) <= 500
        assert idx[0] == 0
        assert idx[-1] == 688
        assert idx == sorted(idx)
        assert len(set(idx)) == len(idx)
        # stride 2 -> every other frame, so the tail is still covered
        assert idx[1] - idx[0] == 2

    def test_sampling_never_returns_nothing(self):
        for count in range(1, 5000):
            idx = frame_indices(count, 500)
            assert idx, f"count={count} produced no indices"
            assert len(idx) <= 500, f"count={count} exceeded cap"
            assert max(idx) < count

    def test_stride_grows_with_length(self):
        # 5000 frames under the same cap needs a coarser stride
        assert len(frame_indices(5000, 500)) <= 500
        assert len(frame_indices(1000, 500)) <= 500

class TestRunSampling:
    """run() must survive the sampling branch end to end: a NameError there used
    to surface as a 500 on every long video."""

    def test_samples_and_reports(self):
        store = _Store()
        result = _engine(store).run("s1", "owner-1", "faces/s1", 690, 500)

        assert len(store.seen) == 345
        assert store.seen[0] == 0 and store.seen[-1] == 688
        assert result["frames_sampled_out"] == 345
        assert result["frames_processed"] == 0
        assert result["frames_skipped"] == 345

    def test_no_sampling_keeps_every_frame(self):
        store = _Store()
        result = _engine(store).run("s2", "owner-1", "faces/s2", 30, 500)

        assert store.seen == list(range(30))
        assert result["frames_sampled_out"] == 0


class TestRunCancellation:
    def test_aborts_before_any_work(self):
        store = _Store()
        with pytest.raises(InferenceAborted) as exc:
            _engine(store).run("s3", "owner-1", "faces/s3", 690, 500, cancel=lambda: True)

        assert exc.value.frames_done == 0
        assert store.seen == []

    def test_aborts_part_way(self):
        calls = {"n": 0}

        def cancel():
            calls["n"] += 1
            return calls["n"] > 2

        store = _Store()
        with pytest.raises(InferenceAborted) as exc:
            _engine(store).run("s4", "owner-1", "faces/s4", 500, 0, cancel=cancel, cancel_every=10)

        # polled at frames 0, 10, 20 -> gone at the third poll
        assert exc.value.frames_done == 20
        assert store.seen == list(range(20))

    def test_finishes_when_caller_stays(self):
        store = _Store()
        result = _engine(store).run("s5", "owner-1", "faces/s5", 40, 0, cancel=lambda: False)

        assert store.seen == list(range(40))
        assert result["frames_sampled_out"] == 0
