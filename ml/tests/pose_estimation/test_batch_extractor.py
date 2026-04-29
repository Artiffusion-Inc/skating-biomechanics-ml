"""Tests for BatchPoseExtractor and extract_poses_batched convenience function."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# Ensure ml/src is on path before imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from src.pose_estimation.batch_extractor import BatchPoseExtractor, extract_poses_batched
from src.types import PersonClick, TrackedExtraction, VideoMeta


@pytest.fixture
def dummy_video_meta():
    """Return a minimal VideoMeta for mocking."""
    return VideoMeta(
        path=Path("dummy.mp4"),
        width=640,
        height=480,
        fps=30.0,
        num_frames=10,
    )


@pytest.fixture
def mock_batch_rtmo(monkeypatch):
    """Mock BatchRTMO so no ONNX model is loaded."""

    class FakeBatchRTMO:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.closed = False

        def infer_batch(self, frames):
            """Return one detection per frame with COCO 17 keypoints."""
            results = []
            for _ in frames:
                # (1 person, 17 keypoints, 2 coords)
                kps = np.zeros((1, 17, 2), dtype=np.float32)
                kps[0, 0] = [320.0, 100.0]  # nose
                kps[0, 5] = [280.0, 150.0]  # left shoulder
                kps[0, 6] = [360.0, 150.0]  # right shoulder
                kps[0, 7] = [260.0, 220.0]  # left elbow
                kps[0, 8] = [380.0, 220.0]  # right elbow
                kps[0, 9] = [250.0, 290.0]  # left wrist
                kps[0, 10] = [390.0, 290.0]  # right wrist
                kps[0, 11] = [280.0, 300.0]  # left hip
                kps[0, 12] = [360.0, 300.0]  # right hip
                kps[0, 13] = [280.0, 380.0]  # left knee
                kps[0, 14] = [360.0, 380.0]  # right knee
                kps[0, 15] = [280.0, 450.0]  # left ankle
                kps[0, 16] = [360.0, 450.0]  # right ankle
                scores = np.ones((1, 17), dtype=np.float32)
                results.append((kps, scores))
            return results

        def close(self):
            self.closed = True

    monkeypatch.setattr(
        "src.pose_estimation.rtmo_batch.BatchRTMO",
        FakeBatchRTMO,
    )
    return FakeBatchRTMO


@pytest.fixture
def mock_video_capture(monkeypatch):
    """Mock cv2.VideoCapture to yield synthetic frames."""
    frames = [np.zeros((480, 640, 3), dtype=np.uint8) for _ in range(10)]

    class FakeCapture:
        def __init__(self, path):
            self._path = path
            self._idx = 0
            self._opened = True

        def isOpened(self):
            return self._opened

        def read(self):
            if self._idx < len(frames):
                frame = frames[self._idx]
                self._idx += 1
                return True, frame
            return False, None

        def release(self):
            self._opened = False

    monkeypatch.setattr(
        "src.pose_estimation.batch_extractor.cv2.VideoCapture",
        FakeCapture,
    )
    return FakeCapture


@pytest.fixture
def mock_get_video_meta(monkeypatch, dummy_video_meta):
    """Mock get_video_meta to return fixed metadata."""
    monkeypatch.setattr(
        "src.pose_estimation.batch_extractor.get_video_meta",
        lambda path: dummy_video_meta,
    )
    return dummy_video_meta


class TestBatchPoseExtractorInit:
    def test_default_init(self):
        """Should initialize with default parameters."""
        extractor = BatchPoseExtractor()
        assert extractor.batch_size == 8
        assert extractor._mode == "balanced"
        assert extractor._conf_threshold == 0.3
        assert extractor._output_format == "normalized"

    def test_custom_init(self):
        """Should accept custom parameters."""
        extractor = BatchPoseExtractor(
            batch_size=16,
            mode="performance",
            conf_threshold=0.5,
            output_format="pixels",
            device="cpu",
            backend="opencv",
        )
        assert extractor.batch_size == 16
        assert extractor._mode == "performance"
        assert extractor._conf_threshold == 0.5
        assert extractor._output_format == "pixels"
        assert extractor._device == "cpu"
        assert extractor._backend == "opencv"

    def test_batch_size_clamped_to_one(self):
        """Negative batch size should be clamped to 1."""
        extractor = BatchPoseExtractor(batch_size=-5)
        assert extractor.batch_size == 1


class TestBatchPoseExtractorExtractVideoTracked:
    def test_extract_video_tracked_returns_tracked_extraction(
        self,
        mock_batch_rtmo,
        mock_video_capture,
        mock_get_video_meta,
    ):
        """extract_video_tracked should return a TrackedExtraction."""
        extractor = BatchPoseExtractor(batch_size=4, device="cpu")
        result = extractor.extract_video_tracked("dummy.mp4")

        assert isinstance(result, TrackedExtraction)
        assert result.poses.shape == (10, 17, 3)
        assert result.frame_indices.shape == (10,)
        assert result.fps == 30.0
        assert result.video_meta.num_frames == 10

    def test_extract_video_tracked_with_person_click(
        self,
        mock_batch_rtmo,
        mock_video_capture,
        mock_get_video_meta,
    ):
        """Should accept person_click parameter without error."""
        extractor = BatchPoseExtractor(batch_size=4, device="cpu")
        click = PersonClick(x=320, y=240)
        result = extractor.extract_video_tracked(
            "dummy.mp4",
            person_click=click,
        )
        assert isinstance(result, TrackedExtraction)

    def test_extract_video_tracked_progress_cb(
        self,
        mock_batch_rtmo,
        mock_video_capture,
        mock_get_video_meta,
    ):
        """Should call progress_cb if provided."""
        extractor = BatchPoseExtractor(batch_size=4, device="cpu")
        calls = []

        def cb(fraction, msg):
            calls.append((fraction, msg))

        extractor.extract_video_tracked("dummy.mp4", progress_cb=cb)
        assert len(calls) > 0
        assert all(0.0 <= c[0] <= 1.0 for c in calls)

    def test_extract_video_tracked_video_open_failure(
        self,
        mock_get_video_meta,
        monkeypatch,
    ):
        """Should raise RuntimeError when video cannot be opened."""

        class FailingCapture:
            def isOpened(self):
                return False

            def release(self):
                pass

        monkeypatch.setattr(
            "src.pose_estimation.batch_extractor.cv2.VideoCapture",
            lambda path: FailingCapture(),
        )
        extractor = BatchPoseExtractor(device="cpu")
        with pytest.raises(RuntimeError, match="Failed to open video"):
            extractor.extract_video_tracked("bad.mp4")

    def test_extract_video_tracked_no_valid_poses(
        self,
        mock_video_capture,
        mock_get_video_meta,
        monkeypatch,
    ):
        """Should raise ValueError when no valid poses are detected."""

        class EmptyBatchRTMO:
            def __init__(self, **kwargs):
                pass

            def infer_batch(self, frames):
                # Return empty detections for every frame
                return [
                    (np.zeros((0, 17, 2), dtype=np.float32), np.zeros((0, 17), dtype=np.float32))
                    for _ in frames
                ]

            def close(self):
                pass

        monkeypatch.setattr(
            "src.pose_estimation.rtmo_batch.BatchRTMO",
            EmptyBatchRTMO,
        )
        extractor = BatchPoseExtractor(batch_size=4, device="cpu")
        with pytest.raises(ValueError, match="No valid pose detected"):
            extractor.extract_video_tracked("dummy.mp4")


class TestBatchPoseExtractorProcessBatch:
    def test_process_batch_normalized_output(
        self,
        mock_batch_rtmo,
    ):
        """_process_batch should return normalized H3.6M poses by default."""
        extractor = BatchPoseExtractor(batch_size=2, device="cpu")
        frames = [
            np.zeros((480, 640, 3), dtype=np.uint8),
            np.zeros((480, 640, 3), dtype=np.uint8),
        ]
        poses = extractor._process_batch(frames, 640, 480)

        assert len(poses) == 2
        assert poses[0].shape == (17, 3)
        # Normalized coordinates should be in [0, 1]
        assert np.all((poses[0][:, 0] >= 0.0) | np.isnan(poses[0][:, 0]))
        assert np.all((poses[0][:, 0] <= 1.0) | np.isnan(poses[0][:, 0]))

    def test_process_batch_pixels_output(
        self,
        mock_batch_rtmo,
    ):
        """_process_batch should return pixel coordinates when requested."""
        extractor = BatchPoseExtractor(batch_size=2, device="cpu", output_format="pixels")
        frames = [
            np.zeros((480, 640, 3), dtype=np.uint8),
        ]
        poses = extractor._process_batch(frames, 640, 480)

        assert len(poses) == 1
        # Pixel coordinates should be up to frame dimensions
        assert poses[0][:, 0].max() <= 640.0
        assert poses[0][:, 1].max() <= 480.0

    def test_process_batch_empty_detection(
        self,
        monkeypatch,
    ):
        """Empty detection should yield NaN pose."""

        class EmptyBatchRTMO:
            def __init__(self, **kwargs):
                pass

            def infer_batch(self, frames):
                return [
                    (np.zeros((0, 17, 2), dtype=np.float32), np.zeros((0, 17), dtype=np.float32))
                    for _ in frames
                ]

            def close(self):
                pass

        monkeypatch.setattr(
            "src.pose_estimation.rtmo_batch.BatchRTMO",
            EmptyBatchRTMO,
        )
        extractor = BatchPoseExtractor(batch_size=2, device="cpu")
        frames = [np.zeros((480, 640, 3), dtype=np.uint8)]
        poses = extractor._process_batch(frames, 640, 480)
        assert len(poses) == 1
        assert np.all(np.isnan(poses[0]))


class TestBatchPoseExtractorLifecycle:
    def test_close_releases_resources(self, mock_batch_rtmo):
        """close should clear internal references."""
        extractor = BatchPoseExtractor(batch_size=2, device="cpu")
        # Trigger lazy init of _batch_rtmo
        frames = [np.zeros((480, 640, 3), dtype=np.uint8)]
        extractor._process_batch(frames, 640, 480)
        assert extractor._batch_rtmo is not None

        extractor.close()
        assert extractor._batch_rtmo is None
        assert extractor._tracker is None

    def test_context_manager(self, mock_batch_rtmo):
        """Should work as a context manager."""
        with BatchPoseExtractor(batch_size=2, device="cpu") as extractor:
            assert isinstance(extractor, BatchPoseExtractor)
        # After exit, resources should be released
        assert extractor._batch_rtmo is None


class TestExtractPosesBatched:
    def test_convenience_function(
        self,
        mock_batch_rtmo,
        mock_video_capture,
        mock_get_video_meta,
    ):
        """extract_poses_batched should return TrackedExtraction."""
        result = extract_poses_batched("dummy.mp4", batch_size=4, mode="balanced")
        assert isinstance(result, TrackedExtraction)
        assert result.poses.shape == (10, 17, 3)

    def test_convenience_function_with_person_click(
        self,
        mock_batch_rtmo,
        mock_video_capture,
        mock_get_video_meta,
    ):
        """Should accept person_click parameter."""
        click = PersonClick(x=100, y=200)
        result = extract_poses_batched(
            "dummy.mp4",
            batch_size=4,
            person_click=click,
        )
        assert isinstance(result, TrackedExtraction)
