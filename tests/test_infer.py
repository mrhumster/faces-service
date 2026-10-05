"""Subsampling helper: pure stdlib, no cv2/insightface needed."""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.infer import frame_indices  # noqa: E402


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