"""Tests for BatchPoseExtractor and extract_poses_batched convenience function."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

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
def mock_moganet_batch(monkeypatch):
    """Mock MogaNetBatch so no ONNX model is loaded."""

    class FakeMogaNetBatch:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.closed = False

        def infer_batch(self, crops, bboxes):
            if not crops:
                return np.zeros((0, 17, 2), np.float32), np.zeros((0, 17), np.float32)
            keypoints = []
            scores = []
            for crop in crops:
                h, w = crop.shape[:2]
                kp = np.zeros((1, 17, 2), dtype=np.float32)
                kp[0, :, 0] = w / 2
                kp[0, :, 1] = h / 2
                keypoints.append(kp[0])
                scores.append(np.ones(17, dtype=np.float32))
            return np.array(keypoints), np.array(scores)

        def close(self):
            self.closed = True

    monkeypatch.setattr(
        "src.pose_estimation.moganet_batch.MogaNetBatch",
        FakeMogaNetBatch,
    )
    return FakeMogaNetBatch


@pytest.fixture
def mock_person_detector(monkeypatch):
    """Mock PersonDetector to always return a fixed bbox."""

    class FakePersonDetector:
        def __init__(self, **kwargs):
            pass

        def detect_frame(self, frame):
            h, w = frame.shape[:2]
            return type(
                "BBox",
                (),
                {
                    "x1": float(w * 0.1),
                    "y1": float(h * 0.1),
                    "x2": float(w * 0.9),
                    "y2": float(h * 0.9),
                    "confidence": 0.9,
                },
            )()

    monkeypatch.setattr(
        "src.detection.person_detector.PersonDetector",
        FakePersonDetector,
    )
    return FakePersonDetector


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
    def test_default_init(self, mock_moganet_batch, mock_person_detector):
        """Should initialize with default parameters."""
        extractor = BatchPoseExtractor()
        assert extractor.batch_size == 8
        assert extractor._conf_threshold == 0.3
        assert extractor._output_format == "normalized"

    def test_custom_init(self, mock_moganet_batch, mock_person_detector):
        """Should accept custom parameters."""
        extractor = BatchPoseExtractor(
            batch_size=16,
            model_path="custom.onnx",
            conf_threshold=0.5,
            output_format="pixels",
            device="cpu",
        )
        assert extractor.batch_size == 16
        assert extractor._conf_threshold == 0.5
        assert extractor._output_format == "pixels"
        assert extractor._device == "cpu"
        assert extractor._model_path == "custom.onnx"

    def test_batch_size_clamped_to_one(self, mock_moganet_batch, mock_person_detector):
        """Negative batch size should be clamped to 1."""
        extractor = BatchPoseExtractor(batch_size=-5)
        assert extractor.batch_size == 1


class TestBatchPoseExtractorExtractVideoTracked:
    def test_extract_video_tracked_returns_tracked_extraction(
        self,
        mock_moganet_batch,
        mock_person_detector,
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
        mock_moganet_batch,
        mock_person_detector,
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
        mock_moganet_batch,
        mock_person_detector,
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
        mock_moganet_batch,
        mock_person_detector,
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

        class EmptyPersonDetector:
            def __init__(self, **kwargs):
                pass

            def detect_frame(self, frame):
                return None

        monkeypatch.setattr(
            "src.detection.person_detector.PersonDetector",
            EmptyPersonDetector,
        )

        class FakeMogaNetBatch:
            def __init__(self, **kwargs):
                pass

            def infer_batch(self, crops, bboxes):
                if not crops:
                    return np.zeros((0, 17, 2), np.float32), np.zeros((0, 17), np.float32)
                return np.zeros((len(crops), 17, 2), np.float32), np.zeros(
                    (len(crops), 17), np.float32
                )

            def close(self):
                pass

        monkeypatch.setattr(
            "src.pose_estimation.moganet_batch.MogaNetBatch",
            FakeMogaNetBatch,
        )
        extractor = BatchPoseExtractor(batch_size=4, device="cpu")
        with pytest.raises(ValueError, match="No valid pose detected"):
            extractor.extract_video_tracked("dummy.mp4")


class TestBatchPoseExtractorLifecycle:
    def test_close_releases_resources(self, mock_moganet_batch, mock_person_detector):
        """close should clear internal references."""
        extractor = BatchPoseExtractor(batch_size=2, device="cpu")
        extractor.close()
        assert extractor._moganet is None
        assert extractor._person_detector is None

    def test_context_manager(self, mock_moganet_batch, mock_person_detector):
        """Should work as a context manager."""
        with BatchPoseExtractor(batch_size=2, device="cpu") as extractor:
            assert isinstance(extractor, BatchPoseExtractor)
        # After exit, resources should be released
        assert extractor._moganet is None


class TestExtractPosesBatched:
    def test_convenience_function(
        self,
        mock_moganet_batch,
        mock_person_detector,
        mock_video_capture,
        mock_get_video_meta,
    ):
        """extract_poses_batched should return TrackedExtraction."""
        result = extract_poses_batched("dummy.mp4", batch_size=4)
        assert isinstance(result, TrackedExtraction)
        assert result.poses.shape == (10, 17, 3)

    def test_convenience_function_with_person_click(
        self,
        mock_moganet_batch,
        mock_person_detector,
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
