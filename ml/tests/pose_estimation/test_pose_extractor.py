"""Tests for PoseExtractor and extract_poses convenience function."""

import sys
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from src.pose_estimation.pose_extractor import PoseExtractor, extract_poses
from src.types import PersonClick, TrackedExtraction, VideoMeta


@pytest.fixture(autouse=True)
def evict_rtmo_batch():
    """Remove cached rtmo_batch so each test gets a fresh import."""
    for key in list(sys.modules.keys()):
        if "rtmo_batch" in key:
            del sys.modules[key]
    import src.pose_estimation

    if hasattr(src.pose_estimation, "rtmo_batch"):
        delattr(src.pose_estimation, "rtmo_batch")


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
        "src.pose_estimation.pose_extractor.cv2.VideoCapture",
        FakeCapture,
    )
    return FakeCapture


@pytest.fixture
def mock_async_frame_reader(monkeypatch):
    """Mock AsyncFrameReader to yield synthetic frames synchronously."""

    class FakeReader:
        def __init__(self, video_path, buffer_size, frame_skip):
            skip = max(1, frame_skip)
            self._frames = [
                (i, np.zeros((480, 640, 3), dtype=np.uint8)) for i in range(0, 10, skip)
            ]
            self._idx = 0

        def start(self):
            pass

        def get_frame(self):
            if self._idx < len(self._frames):
                f = self._frames[self._idx]
                self._idx += 1
                return f
            return None

        def join(self, timeout=5.0):
            pass

    monkeypatch.setattr(
        "src.pose_estimation.pose_extractor.AsyncFrameReader",
        FakeReader,
    )
    return FakeReader


@pytest.fixture
def mock_get_video_meta(monkeypatch, dummy_video_meta):
    """Mock get_video_meta to return fixed metadata."""
    monkeypatch.setattr(
        "src.pose_estimation.pose_extractor.get_video_meta",
        lambda path: dummy_video_meta,
    )
    return dummy_video_meta


@pytest.fixture
def mock_tracker(monkeypatch):
    """Mock PoseExtractor.tracker to return fake detections."""

    class FakeTracker:
        def __call__(self, frame):
            # Return one person, COCO 17 keypoints in pixel coords
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
            return (kps, scores)

    monkeypatch.setattr(
        PoseExtractor,
        "tracker",
        property(lambda self: FakeTracker()),
    )
    return FakeTracker


@pytest.fixture
def mock_batch_rtmo(monkeypatch):
    """Mock BatchRTMO for the batched inference path."""

    class FakeBatchRTMO:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def infer_batch(self, frames):
            results = []
            for _ in frames:
                kps = np.zeros((1, 17, 2), dtype=np.float32)
                kps[0, 0] = [320.0, 100.0]
                kps[0, 5] = [280.0, 150.0]
                kps[0, 6] = [360.0, 150.0]
                kps[0, 7] = [260.0, 220.0]
                kps[0, 8] = [380.0, 220.0]
                kps[0, 9] = [250.0, 290.0]
                kps[0, 10] = [390.0, 290.0]
                kps[0, 11] = [280.0, 300.0]
                kps[0, 12] = [360.0, 300.0]
                kps[0, 13] = [280.0, 380.0]
                kps[0, 14] = [360.0, 380.0]
                kps[0, 15] = [280.0, 450.0]
                kps[0, 16] = [360.0, 450.0]
                scores = np.ones((1, 17), dtype=np.float32)
                results.append((kps, scores))
            return results

    monkeypatch.setattr(
        "src.pose_estimation.rtmo_batch.BatchRTMO",
        FakeBatchRTMO,
    )
    return FakeBatchRTMO


@pytest.fixture(autouse=True)
def mock_tqdm(monkeypatch):
    """Suppress tqdm progress bars in tests."""

    class FakeTqdm:
        def __init__(self, iterable=None, *args, **kwargs):
            self._iterable = iterable

        def __iter__(self):
            if self._iterable is not None:
                return iter(self._iterable)
            return iter([])

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def update(self, n=1):
            pass

        def close(self):
            pass

    monkeypatch.setattr(
        "src.pose_estimation.pose_extractor.tqdm",
        FakeTqdm,
    )


@pytest.fixture
def mock_device_config(monkeypatch):
    """Mock DeviceConfig for auto device resolution."""

    class FakeDeviceConfig:
        def __init__(self, device="auto"):
            self.device = "cuda"

    monkeypatch.setattr(
        "src.device.DeviceConfig",
        FakeDeviceConfig,
    )
    return FakeDeviceConfig


@pytest.fixture
def mock_rtmlib_none(monkeypatch):
    """Simulate missing rtmlib by setting module-level PoseTracker to None."""
    monkeypatch.setattr(
        "src.pose_estimation.pose_extractor.PoseTracker",
        None,
    )
    monkeypatch.setattr(
        "src.pose_estimation.pose_extractor.Body",
        None,
    )


class TestPoseExtractorInit:
    def test_default_init(self):
        """Should initialize with default parameters."""
        extractor = PoseExtractor()
        assert extractor._mode == "balanced"
        assert extractor._tracking_backend == "rtmlib"
        assert extractor._tracking_mode == "auto"
        assert extractor._conf_threshold == 0.3
        assert extractor._output_format == "normalized"
        assert extractor._frame_skip == 1
        assert extractor._backend == "onnxruntime"

    def test_custom_init(self):
        """Should accept custom parameters."""
        extractor = PoseExtractor(
            mode="performance",
            tracking_backend="custom",
            tracking_mode="sports2d",
            conf_threshold=0.5,
            output_format="pixels",
            frame_skip=0,
            device="cuda",
            backend="opencv",
        )
        assert extractor._mode == "performance"
        assert extractor._tracking_backend == "custom"
        assert extractor._tracking_mode == "sports2d"
        assert extractor._conf_threshold == 0.5
        assert extractor._output_format == "pixels"
        assert extractor._frame_skip == 1  # clamped to minimum 1
        assert extractor._device == "cuda"
        assert extractor._backend == "opencv"

    def test_auto_device(self, mock_device_config):
        """Should resolve 'auto' device via DeviceConfig."""
        extractor = PoseExtractor(device="auto")
        assert extractor._device == "cuda"

    def test_import_error_when_rtmlib_missing(self, mock_rtmlib_none):
        """Should raise ImportError when rtmlib is not installed."""
        with pytest.raises(ImportError, match="rtmlib is not installed"):
            PoseExtractor()


class TestPoseExtractorTracker:
    def test_tracker_property(self, mock_tracker):
        """Should return a callable tracker instance."""
        extractor = PoseExtractor(device="cpu")
        tracker = extractor.tracker
        assert tracker is not None
        assert callable(tracker)

    def test_tracker_returns_same_instance(self, monkeypatch):
        """Should cache tracker instance on repeated access."""
        fake_rtmpose_tracker = MagicMock()

        class FakeCustom:
            def __init__(self, *args, **kwargs):
                pass

        import rtmlib

        monkeypatch.setattr(rtmlib, "PoseTracker", fake_rtmpose_tracker)
        monkeypatch.setattr(rtmlib, "Custom", FakeCustom)

        extractor = PoseExtractor(device="cpu")
        t1 = extractor.tracker
        t2 = extractor.tracker
        assert t1 is t2
        assert t1 is fake_rtmpose_tracker.return_value


class TestPoseExtractorResolveTrackingMode:
    def test_explicit_mode_returns_unchanged(self):
        """Should return the explicitly set tracking mode."""
        extractor = PoseExtractor(tracking_mode="sports2d")
        assert extractor._resolve_tracking_mode() == "sports2d"

    def test_auto_prefers_deepsort(self):
        """Auto mode should prefer deepsort when available."""
        extractor = PoseExtractor(tracking_mode="auto")
        # deep_sort_realtime is installed in this environment
        assert extractor._resolve_tracking_mode() == "deepsort"

    def test_auto_falls_back_to_sports2d(self, monkeypatch):
        """Auto mode should fall back to sports2d when deepsort unavailable."""
        monkeypatch.setitem(sys.modules, "deep_sort_realtime", None)
        # Re-import to pick up the changed module state
        from src.pose_estimation.pose_extractor import PoseExtractor as PE

        extractor = PE(tracking_mode="auto")
        assert extractor._resolve_tracking_mode() == "sports2d"


class TestPoseExtractorAssignTrackIds:
    def test_empty_poses(self):
        """Should return empty list for empty poses."""
        extractor = PoseExtractor(device="cpu")
        result = extractor._assign_track_ids(
            np.zeros((0, 17, 3), dtype=np.float32),
            {},
            0,
        )
        assert result == []

    def test_single_person(self):
        """Should assign sequential IDs starting from next_id."""
        extractor = PoseExtractor(device="cpu")
        result = extractor._assign_track_ids(
            np.zeros((1, 17, 3), dtype=np.float32),
            {},
            5,
        )
        assert result == [5]

    def test_multiple_persons(self):
        """Should assign a range of IDs for multiple persons."""
        extractor = PoseExtractor(device="cpu")
        result = extractor._assign_track_ids(
            np.zeros((3, 17, 3), dtype=np.float32),
            {},
            2,
        )
        assert result == [2, 3, 4]


class TestPoseExtractorExtractVideoTracked:
    def test_extract_video_tracked_streaming_returns_tracked_extraction(
        self,
        mock_video_capture,
        mock_async_frame_reader,
        mock_get_video_meta,
        mock_tracker,
    ):
        """Streaming path should return TrackedExtraction with valid poses."""
        extractor = PoseExtractor(device="cpu", tracking_mode="sports2d")
        result = extractor.extract_video_tracked("dummy.mp4", use_batch=False)

        assert isinstance(result, TrackedExtraction)
        assert result.poses.shape == (10, 17, 3)
        assert result.frame_indices.shape == (10,)
        assert result.fps == 30.0
        assert result.video_meta.num_frames == 10
        assert result.first_frame is not None
        assert result.target_track_id is not None

    def test_extract_video_tracked_batch_returns_tracked_extraction(
        self,
        mock_video_capture,
        mock_get_video_meta,
        mock_batch_rtmo,
    ):
        """Batch path should return TrackedExtraction with valid poses."""
        extractor = PoseExtractor(device="cpu", tracking_mode="sports2d")
        result = extractor.extract_video_tracked("dummy.mp4", use_batch=True)

        assert isinstance(result, TrackedExtraction)
        assert result.poses.shape == (10, 17, 3)
        assert result.frame_indices.shape == (10,)
        assert result.fps == 30.0
        assert result.video_meta.num_frames == 10
        assert result.first_frame is not None
        assert result.target_track_id is not None

    def test_extract_video_tracked_with_person_click_streaming(
        self,
        mock_video_capture,
        mock_async_frame_reader,
        mock_get_video_meta,
        mock_tracker,
    ):
        """Streaming path should handle person_click selection."""
        extractor = PoseExtractor(device="cpu", tracking_mode="sports2d")
        click = PersonClick(x=320, y=300)
        result = extractor.extract_video_tracked(
            "dummy.mp4",
            person_click=click,
            use_batch=False,
        )
        assert isinstance(result, TrackedExtraction)
        assert result.target_track_id is not None

    def test_extract_video_tracked_with_person_click_batch(
        self,
        mock_video_capture,
        mock_get_video_meta,
        mock_batch_rtmo,
    ):
        """Batch path should handle person_click selection."""
        extractor = PoseExtractor(device="cpu", tracking_mode="sports2d")
        click = PersonClick(x=320, y=300)
        result = extractor.extract_video_tracked(
            "dummy.mp4",
            person_click=click,
            use_batch=True,
        )
        assert isinstance(result, TrackedExtraction)
        assert result.target_track_id is not None

    def test_extract_video_tracked_no_poses_raises_value_error_streaming(
        self,
        mock_video_capture,
        mock_async_frame_reader,
        mock_get_video_meta,
        monkeypatch,
    ):
        """Streaming path should raise ValueError when no poses detected."""

        class EmptyTracker:
            def __call__(self, frame):
                return (
                    np.zeros((0, 17, 2), dtype=np.float32),
                    np.zeros((0, 17), dtype=np.float32),
                )

        monkeypatch.setattr(
            PoseExtractor,
            "tracker",
            property(lambda self: EmptyTracker()),
        )
        extractor = PoseExtractor(device="cpu", tracking_mode="sports2d")
        with pytest.raises(ValueError, match="No valid pose detected"):
            extractor.extract_video_tracked("dummy.mp4", use_batch=False)

    def test_extract_video_tracked_no_poses_raises_value_error_batch(
        self,
        mock_video_capture,
        mock_get_video_meta,
        monkeypatch,
    ):
        """Batch path should raise ValueError when no poses detected."""

        class EmptyBatchRTMO:
            def __init__(self, **kwargs):
                pass

            def infer_batch(self, frames):
                empty_kps = np.zeros((0, 17, 2), dtype=np.float32)
                empty_scores = np.zeros((0, 17), dtype=np.float32)
                return [(empty_kps, empty_scores) for _ in frames]

        monkeypatch.setattr(
            "src.pose_estimation.rtmo_batch.BatchRTMO",
            EmptyBatchRTMO,
        )
        extractor = PoseExtractor(device="cpu", tracking_mode="sports2d")
        with pytest.raises(ValueError, match="No valid pose detected"):
            extractor.extract_video_tracked("dummy.mp4", use_batch=True)

    def test_extract_video_tracked_progress_callback_streaming(
        self,
        mock_video_capture,
        mock_async_frame_reader,
        mock_get_video_meta,
        mock_tracker,
    ):
        """Streaming path should call progress_cb if provided."""
        progress_calls = []

        def progress_cb(fraction, message):
            progress_calls.append((fraction, message))

        extractor = PoseExtractor(device="cpu", tracking_mode="sports2d")
        extractor.extract_video_tracked(
            "dummy.mp4",
            progress_cb=progress_cb,
            use_batch=False,
        )
        assert len(progress_calls) > 0

    def test_extract_video_tracked_progress_callback_batch(
        self,
        mock_video_capture,
        mock_get_video_meta,
        mock_batch_rtmo,
    ):
        """Batch path should call progress_cb if provided."""
        progress_calls = []

        def progress_cb(fraction, message):
            progress_calls.append((fraction, message))

        extractor = PoseExtractor(device="cpu", tracking_mode="sports2d")
        extractor.extract_video_tracked(
            "dummy.mp4",
            progress_cb=progress_cb,
            use_batch=True,
        )
        assert len(progress_calls) > 0


class TestPoseExtractorPreviewPersons:
    def test_preview_persons_returns_list_and_path(
        self,
        mock_video_capture,
        mock_get_video_meta,
        mock_tracker,
    ):
        """Should return a list of person dicts and a preview path."""
        extractor = PoseExtractor(device="cpu", tracking_mode="sports2d")
        persons, preview_path = extractor.preview_persons("dummy.mp4", num_frames=10)

        assert isinstance(persons, list)
        assert len(persons) > 0
        assert all("track_id" in p for p in persons)
        assert all("hits" in p for p in persons)
        assert all("bbox" in p for p in persons)
        assert all("mid_hip" in p for p in persons)
        assert preview_path is not None
        assert isinstance(preview_path, str)
        assert preview_path.endswith(".jpg")

    def test_preview_persons_empty_video(
        self,
        mock_video_capture,
        mock_get_video_meta,
        monkeypatch,
    ):
        """Should return empty list when no persons detected."""

        class EmptyTracker:
            def __call__(self, frame):
                return (
                    np.zeros((0, 17, 2), dtype=np.float32),
                    np.zeros((0, 17), dtype=np.float32),
                )

        monkeypatch.setattr(
            PoseExtractor,
            "tracker",
            property(lambda self: EmptyTracker()),
        )
        extractor = PoseExtractor(device="cpu", tracking_mode="sports2d")
        persons, preview_path = extractor.preview_persons("dummy.mp4", num_frames=10)
        assert persons == []
        assert preview_path is None


class TestPoseExtractorBuildPersonGrid:
    def test_build_person_grid(self):
        """Should generate a preview image path."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        persons = [
            {
                "best_kps": np.random.rand(17, 3).astype(np.float32),
                "hits": 5,
            }
        ]
        # Ensure enough valid keypoints for bbox computation
        persons[0]["best_kps"][:, 2] = 1.0

        path = PoseExtractor._build_person_grid(frame, persons)
        assert isinstance(path, str)
        assert path.endswith(".jpg")
        assert Path(path).exists()

    def test_build_person_grid_empty(self):
        """Should return empty string for empty persons."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        path = PoseExtractor._build_person_grid(frame, [])
        assert path == ""


class TestPoseExtractorContextManager:
    def test_context_manager(self):
        """Should support with-statement."""
        with PoseExtractor(device="cpu") as extractor:
            assert isinstance(extractor, PoseExtractor)

    def test_close_releases_tracker(self, mock_tracker):
        """close() should set _tracker to None."""
        extractor = PoseExtractor(device="cpu")
        _ = extractor.tracker  # trigger lazy init via mock
        extractor.close()
        assert extractor._tracker is None


class TestExtractPoses:
    def test_convenience_function(self, monkeypatch):
        """extract_poses should create extractor and return TrackedExtraction."""
        mock_result = TrackedExtraction(
            poses=np.zeros((10, 17, 3), dtype=np.float32),
            frame_indices=np.arange(10),
            first_detection_frame=0,
            target_track_id=0,
            fps=30.0,
            video_meta=VideoMeta(Path("dummy.mp4"), 640, 480, 30.0, 10),
            first_frame=None,
        )

        def mock_extract(self, video_path, person_click=None):
            return mock_result

        monkeypatch.setattr(PoseExtractor, "extract_video_tracked", mock_extract)

        result = extract_poses("dummy.mp4")
        assert result is mock_result
