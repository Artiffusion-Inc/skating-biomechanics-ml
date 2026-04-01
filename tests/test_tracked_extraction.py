"""Tests for H36MExtractor.extract_video_tracked().

Tests cover:
- Single-person detection (all frames valid)
- Multi-person with click-based selection
- Gaps filled with NaN when target not detected
- Pre-roll empty frames (first_detection_frame correct)
- Auto-select by most hits when no click provided
"""

from dataclasses import dataclass
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.pose_estimation.h36m_extractor import H36MExtractor
from src.types import PersonClick, TrackedExtraction, VideoMeta


# ---------------------------------------------------------------------------
# Helpers -- fake YOLO result objects that mimic torch tensors
# ---------------------------------------------------------------------------


class _NumpyLike:
    """Wrapper that provides .cpu().numpy() on a plain ndarray."""

    def __init__(self, arr: np.ndarray) -> None:
        self._arr = arr

    def cpu(self) -> "_NumpyLike":
        return self

    def numpy(self) -> np.ndarray:
        return self._arr

    def __len__(self) -> int:
        return len(self._arr)


@dataclass
class FakeKps:
    """Mimics ultralytics pose result keypoints for one frame."""

    xy: _NumpyLike  # (P, 17, 2)
    conf: _NumpyLike  # (P, 17)


@dataclass
class FakeResult:
    """Mimics a single ultralytics Results object."""

    keypoints: FakeKps | None
    orig_shape: tuple[int, int]  # (h, w)


def _make_coco_pose(x_offset: float = 0.5, y_offset: float = 0.5) -> np.ndarray:
    """Create a valid COCO 17-kp pose (17, 3) centered at (x_offset, y_offset).

    Returns pixel-space coordinates with confidence 0.9.
    """
    from src.pose_estimation.h36m_extractor import _COCOKey

    pose = np.zeros((17, 3), dtype=np.float32)
    # Use pixel-space coords (will be normalized in extract_video_tracked)
    px = x_offset * 1920
    py = y_offset * 1080
    pose[_COCOKey.NOSE] = [px, py - 324, 0.9]
    pose[_COCOKey.LEFT_EYE] = [px - 38, py - 346, 0.9]
    pose[_COCOKey.RIGHT_EYE] = [px + 38, py - 346, 0.9]
    pose[_COCOKey.LEFT_EAR] = [px - 77, py - 302, 0.9]
    pose[_COCOKey.RIGHT_EAR] = [px + 77, py - 302, 0.9]
    pose[_COCOKey.LEFT_SHOULDER] = [px - 192, py - 162, 0.9]
    pose[_COCOKey.RIGHT_SHOULDER] = [px + 192, py - 162, 0.9]
    pose[_COCOKey.LEFT_ELBOW] = [px - 288, py + 54, 0.9]
    pose[_COCOKey.RIGHT_ELBOW] = [px + 288, py + 54, 0.9]
    pose[_COCOKey.LEFT_WRIST] = [px - 384, py + 216, 0.9]
    pose[_COCOKey.RIGHT_WRIST] = [px + 384, py + 216, 0.9]
    pose[_COCOKey.LEFT_HIP] = [px - 154, py + 162, 0.9]
    pose[_COCOKey.RIGHT_HIP] = [px + 154, py + 162, 0.9]
    pose[_COCOKey.LEFT_KNEE] = [px - 154, py + 378, 0.9]
    pose[_COCOKey.RIGHT_KNEE] = [px + 154, py + 378, 0.9]
    pose[_COCOKey.LEFT_ANKLE] = [px - 154, py + 540, 0.9]
    pose[_COCOKey.RIGHT_ANKLE] = [px + 154, py + 540, 0.9]
    return pose


def _make_video_meta(num_frames: int = 100) -> VideoMeta:
    """Create a fake VideoMeta."""
    return VideoMeta(
        path=Path("/fake/video.mp4"),
        width=1920,
        height=1080,
        fps=30.0,
        num_frames=num_frames,
    )


def _wrap_kps(coco_poses: list[np.ndarray]) -> tuple[_NumpyLike, _NumpyLike]:
    """Wrap list of (17,3) coco poses into FakeKps with _NumpyLike tensors."""
    xy = np.stack([p[:, :2] for p in coco_poses])  # (P, 17, 2)
    conf = np.stack([p[:, 2] for p in coco_poses])  # (P, 17)
    return FakeKps(xy=_NumpyLike(xy), conf=_NumpyLike(conf))


def _make_single_person_results(
    num_frames: int,
    x_offset: float = 0.5,
    y_offset: float = 0.5,
    gap_frames: set[int] | None = None,
) -> list[FakeResult]:
    """Create fake YOLO results with 1 person per frame."""
    gap_frames = gap_frames or set()
    results = []
    for i in range(num_frames):
        if i in gap_frames:
            results.append(FakeResult(keypoints=None, orig_shape=(1080, 1920)))
        else:
            coco = _make_coco_pose(x_offset, y_offset)
            kps = _wrap_kps([coco])
            results.append(FakeResult(keypoints=kps, orig_shape=(1080, 1920)))
    return results


def _make_multi_person_results(
    num_frames: int,
    person_a: tuple[float, float] = (0.3, 0.5),
    person_b: tuple[float, float] = (0.7, 0.5),
    gap_frames_a: set[int] | None = None,
    gap_frames_b: set[int] | None = None,
    missing_frames: set[int] | None = None,
) -> list[FakeResult]:
    """Create fake YOLO results with 2 persons per frame."""
    gap_frames_a = gap_frames_a or set()
    gap_frames_b = gap_frames_b or set()
    missing_frames = missing_frames or set()
    results = []
    for i in range(num_frames):
        if i in missing_frames:
            results.append(FakeResult(keypoints=None, orig_shape=(1080, 1920)))
            continue

        coco_list = []
        if i not in gap_frames_a:
            coco_list.append(_make_coco_pose(person_a[0], person_a[1]))
        if i not in gap_frames_b:
            coco_list.append(_make_coco_pose(person_b[0], person_b[1]))

        if not coco_list:
            results.append(FakeResult(keypoints=None, orig_shape=(1080, 1920)))
        else:
            kps = _wrap_kps(coco_list)
            results.append(FakeResult(keypoints=kps, orig_shape=(1080, 1920)))
    return results


def _setup_extractor(
    extractor: H36MExtractor,
    fake_results: list[FakeResult],
    meta: VideoMeta,
) -> TrackedExtraction:
    """Inject fake model + meta, run extract_video_tracked, return result."""
    mock_model = MagicMock()
    mock_model.return_value = iter(fake_results)
    extractor._model = mock_model

    with patch("src.pose_estimation.h36m_extractor.get_video_meta", return_value=meta):
        return extractor.extract_video_tracked("/fake/video.mp4")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def extractor():
    """Create H36MExtractor with skip_model_check (no YOLO download)."""
    return H36MExtractor(
        model_size="n",
        conf_threshold=0.5,
        output_format="normalized",
        skip_model_check=True,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestExtractVideoTracked:
    """Tests for H36MExtractor.extract_video_tracked()."""

    def test_single_person_all_frames(self, extractor: H36MExtractor) -> None:
        """Single person detected in all frames -- all frames valid."""
        num_frames = 50
        fake_results = _make_single_person_results(num_frames)
        meta = _make_video_meta(num_frames)
        result = _setup_extractor(extractor, fake_results, meta)

        assert isinstance(result, TrackedExtraction)
        assert result.poses.shape == (num_frames, 17, 3)
        assert result.frame_indices.shape == (num_frames,)
        assert result.fps == 30.0
        assert result.video_meta == meta
        assert result.first_detection_frame == 0

        # All frames should be valid
        mask = result.valid_mask()
        assert mask.all(), f"Expected all valid, got {np.sum(~mask)} NaN frames"

        # target_track_id should be set
        assert result.target_track_id is not None

    def test_multi_person_with_click(self, extractor: H36MExtractor) -> None:
        """Two persons, click selects the nearest (person A at x=0.3)."""
        num_frames = 30
        fake_results = _make_multi_person_results(num_frames)
        meta = _make_video_meta(num_frames)

        # Click near person A (x=0.3 in normalized -> pixel 0.3*1920=576)
        click = PersonClick(x=550, y=540)

        mock_model = MagicMock()
        mock_model.return_value = iter(fake_results)
        extractor._model = mock_model

        with patch("src.pose_estimation.h36m_extractor.get_video_meta", return_value=meta):
            result = extractor.extract_video_tracked(
                "/fake/video.mp4", person_click=click
            )

        assert isinstance(result, TrackedExtraction)
        assert result.poses.shape == (num_frames, 17, 3)
        assert result.target_track_id is not None

        # The selected person's mid-hip should be near x=0.3
        valid = result.valid_mask()
        assert valid.any()
        first_valid_idx = int(np.argmax(valid))
        pose = result.poses[first_valid_idx]
        # H36Key.LHIP=4, H36Key.RHIP=1, mid-hip x should be near 0.3
        mid_hip_x = (pose[4, 0] + pose[1, 0]) / 2
        assert mid_hip_x < 0.5, (
            f"Expected person A (x~0.3), got mid_hip_x={mid_hip_x:.2f}"
        )

    def test_gaps_are_nan(self, extractor: H36MExtractor) -> None:
        """Frames where target is not detected should be NaN."""
        num_frames = 50
        gap_frames = {10, 11, 12, 25, 26}
        fake_results = _make_single_person_results(num_frames, gap_frames=gap_frames)
        meta = _make_video_meta(num_frames)
        result = _setup_extractor(extractor, fake_results, meta)

        # Gap frames should be NaN
        for gap_idx in gap_frames:
            assert np.isnan(result.poses[gap_idx, 0, 0]), (
                f"Frame {gap_idx} should be NaN (gap)"
            )

        # Non-gap frames should be valid
        for i in range(num_frames):
            if i not in gap_frames:
                assert not np.isnan(result.poses[i, 0, 0]), (
                    f"Frame {i} should be valid"
                )

    def test_preroll_empty_frames(self, extractor: H36MExtractor) -> None:
        """First N frames empty -- first_detection_frame should be N."""
        num_frames = 50
        preroll = 20
        gap_frames = set(range(preroll))  # frames 0-19 empty
        fake_results = _make_single_person_results(num_frames, gap_frames=gap_frames)
        meta = _make_video_meta(num_frames)
        result = _setup_extractor(extractor, fake_results, meta)

        assert result.first_detection_frame == preroll, (
            f"Expected first_detection_frame={preroll}, "
            f"got {result.first_detection_frame}"
        )

        # Verify NaN in pre-roll
        for i in range(preroll):
            assert np.isnan(result.poses[i, 0, 0])

        # Verify valid after pre-roll
        assert not np.isnan(result.poses[preroll, 0, 0])

    def test_auto_select_most_hits(self, extractor: H36MExtractor) -> None:
        """No click -- auto-select person with most hits.

        Person A appears in frames 0-19 (20 frames).
        Person B appears in frames 0-49 (50 frames).
        Without click, person B should be selected (more hits).
        """
        num_frames = 50
        gap_frames_a = set(range(20, num_frames))  # A absent from frame 20+
        fake_results = _make_multi_person_results(
            num_frames, gap_frames_a=gap_frames_a
        )
        meta = _make_video_meta(num_frames)
        result = _setup_extractor(extractor, fake_results, meta)

        assert result.target_track_id is not None

        # Person B (x~0.7) should be selected -- check late frame
        valid = result.valid_mask()
        late_idx = 40
        assert valid[late_idx], "Frame 40 should have a valid pose (person B)"
        pose = result.poses[late_idx]
        mid_hip_x = (pose[4, 0] + pose[1, 0]) / 2
        assert mid_hip_x > 0.5, (
            f"Expected person B (x~0.7), got mid_hip_x={mid_hip_x:.2f}"
        )

    def test_no_detections_raises(self, extractor: H36MExtractor) -> None:
        """No person detected in any frame -- ValueError."""
        num_frames = 10
        fake_results = _make_single_person_results(
            num_frames, gap_frames=set(range(num_frames))
        )
        meta = _make_video_meta(num_frames)

        mock_model = MagicMock()
        mock_model.return_value = iter(fake_results)
        extractor._model = mock_model

        with patch("src.pose_estimation.h36m_extractor.get_video_meta", return_value=meta):
            with pytest.raises(ValueError, match="No valid pose detected"):
                extractor.extract_video_tracked("/fake/video.mp4")

    def test_frame_indices_match_video_length(self, extractor: H36MExtractor) -> None:
        """frame_indices should be np.arange(num_frames)."""
        num_frames = 37
        fake_results = _make_single_person_results(num_frames)
        meta = _make_video_meta(num_frames)
        result = _setup_extractor(extractor, fake_results, meta)

        np.testing.assert_array_equal(result.frame_indices, np.arange(num_frames))

    def test_output_shape_matches_video(self, extractor: H36MExtractor) -> None:
        """Output shape is always (num_frames, 17, 3) regardless of gaps."""
        num_frames = 15
        gap_frames = {3, 7, 8, 14}
        fake_results = _make_single_person_results(num_frames, gap_frames=gap_frames)
        meta = _make_video_meta(num_frames)
        result = _setup_extractor(extractor, fake_results, meta)

        assert result.poses.shape == (num_frames, 17, 3)

    def test_valid_mask_correctness(self, extractor: H36MExtractor) -> None:
        """valid_mask() returns True for non-NaN frames."""
        num_frames = 20
        gap_frames = {5, 6, 7}
        fake_results = _make_single_person_results(num_frames, gap_frames=gap_frames)
        meta = _make_video_meta(num_frames)
        result = _setup_extractor(extractor, fake_results, meta)

        mask = result.valid_mask()
        expected_valid = num_frames - len(gap_frames)
        assert mask.sum() == expected_valid

        for i in gap_frames:
            assert not mask[i]
        for i in range(num_frames):
            if i not in gap_frames:
                assert mask[i]
