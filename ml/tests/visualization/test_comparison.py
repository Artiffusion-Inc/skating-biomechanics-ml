"""Tests for ml/src/visualization/comparison.py."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# Add ml to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.visualization.comparison import (
    ComparisonConfig,
    ComparisonMode,
    ComparisonRenderer,
    _build_layers,
)


@pytest.fixture
def sample_video(tmp_path):
    """Create a dummy video file path."""
    video = tmp_path / "test_video.mp4"
    video.write_bytes(b"dummy")
    return video


@pytest.fixture
def mock_video_meta():
    """Return a mock VideoMeta-like object."""
    meta = MagicMock()
    meta.width = 640
    meta.height = 480
    meta.fps = 30.0
    meta.num_frames = 10
    return meta


@pytest.fixture
def mock_tracked_poses():
    """Return a small sequence of tracked 3D poses."""
    poses = np.zeros((5, 17, 3), dtype=np.float32)
    for i in range(17):
        poses[:, i, 0] = (i % 5) * 0.1
        poses[:, i, 1] = (i % 7) * 0.1
    return poses


@pytest.fixture
def mock_cap():
    """Return a mock cv2.VideoCapture that yields 5 frames."""
    cap = MagicMock()
    cap.isOpened.return_value = True
    frames = [(True, np.ones((480, 640, 3), dtype=np.uint8) * i * 40) for i in range(5)] + [
        (False, None)
    ]
    cap.read.side_effect = frames
    cap.set.return_value = True
    cap.release.return_value = None
    return cap


# ---------------------------------------------------------------------------
# ComparisonConfig / ComparisonMode
# ---------------------------------------------------------------------------


class TestComparisonConfig:
    """Test configuration dataclass."""

    def test_default_values(self):
        """Default config should have sensible values."""
        cfg = ComparisonConfig()
        assert cfg.mode == ComparisonMode.SIDE_BY_SIDE
        assert cfg.overlays == ["skeleton", "axis", "angles", "timer"]
        assert cfg.resize_width == 1280
        assert cfg.reference_alpha == 0.4
        assert cfg.divider_width == 4
        assert cfg.fps == 0.0
        assert cfg.max_frames == 0
        assert cfg.start_frame == 0
        assert cfg.device == "auto"
        assert cfg.no_cache is False

    def test_overlay_mode(self):
        """Config can be set to overlay mode."""
        cfg = ComparisonConfig(mode=ComparisonMode.OVERLAY)
        assert cfg.mode == ComparisonMode.OVERLAY


class TestComparisonMode:
    """Test mode enum."""

    def test_side_by_side_value(self):
        assert ComparisonMode.SIDE_BY_SIDE.value == "side-by-side"

    def test_overlay_value(self):
        assert ComparisonMode.OVERLAY.value == "overlay"


# ---------------------------------------------------------------------------
# _build_layers
# ---------------------------------------------------------------------------


class TestBuildLayers:
    """Test layer builder helper."""

    @patch("src.visualization.comparison.SkeletonLayer")
    @patch("src.visualization.comparison.VerticalAxisLayer")
    @patch("src.visualization.comparison.JointAngleLayer")
    @patch("src.visualization.comparison.TimerLayer")
    def test_builds_all_layers(self, mock_timer, mock_angle, mock_axis, mock_skeleton):
        """All recognized overlay names should instantiate their layer classes."""
        layers = _build_layers(["skeleton", "axis", "angles", "timer"])
        assert len(layers) == 4
        mock_skeleton.assert_called_once()
        mock_axis.assert_called_once()
        mock_angle.assert_called_once()
        mock_timer.assert_called_once()

    @patch("src.visualization.comparison.SkeletonLayer")
    @patch("src.visualization.comparison.TimerLayer")
    def test_ignores_unknown_names(self, mock_timer, mock_skeleton):
        """Unknown overlay names should be silently ignored."""
        layers = _build_layers(["skeleton", "unknown_layer", "timer"])
        assert len(layers) == 2
        mock_skeleton.assert_called_once()
        mock_timer.assert_called_once()


# ---------------------------------------------------------------------------
# ComparisonRenderer.__init__
# ---------------------------------------------------------------------------


class TestComparisonRendererInit:
    """Test renderer initialization."""

    @patch("src.visualization.comparison.SkeletonLayer")
    @patch("src.visualization.comparison.VerticalAxisLayer")
    def test_default_init(self, mock_axis, mock_skeleton):
        """Renderer with default config should build default layers."""
        renderer = ComparisonRenderer()
        assert renderer.config is not None
        assert renderer.config.mode == ComparisonMode.SIDE_BY_SIDE
        assert len(renderer.layers) == 4

    def test_custom_config(self):
        """Renderer should accept a custom config."""
        cfg = ComparisonConfig(mode=ComparisonMode.OVERLAY, overlays=["timer"])
        renderer = ComparisonRenderer(config=cfg)
        assert renderer.config.mode == ComparisonMode.OVERLAY
        assert len(renderer.layers) == 1


# ---------------------------------------------------------------------------
# ComparisonRenderer._pose_cache_path
# ---------------------------------------------------------------------------


class TestPoseCachePath:
    """Test cache path generation."""

    def test_returns_npz_next_to_video(self):
        """Cache path should be video stem + '_poses.npz'."""
        renderer = ComparisonRenderer()
        video = Path("/tmp/test_video.mp4")
        cache = renderer._pose_cache_path(video)
        assert cache == Path("/tmp/test_video_poses.npz")


# ---------------------------------------------------------------------------
# ComparisonRenderer._save_pose_cache
# ---------------------------------------------------------------------------


class TestSavePoseCache:
    """Test cache writing."""

    @patch("src.visualization.comparison.np.savez_compressed")
    def test_saves_compressed_npz(self, mock_savez, tmp_path):
        """Pose cache should be saved as compressed .npz."""
        renderer = ComparisonRenderer()
        video = tmp_path / "athlete.mp4"
        video.write_bytes(b"")
        poses = [np.zeros((17, 2), dtype=np.float32) for _ in range(3)]
        renderer._save_pose_cache(video, poses)
        mock_savez.assert_called_once()
        args, kwargs = mock_savez.call_args
        assert "poses" in kwargs
        assert kwargs["poses"].shape == (3, 17, 2)

    def test_empty_poses_noop(self, tmp_path):
        """Empty pose list should not write cache."""
        renderer = ComparisonRenderer()
        video = tmp_path / "athlete.mp4"
        video.write_bytes(b"")
        with patch("src.visualization.comparison.np.savez_compressed") as mock_savez:
            renderer._save_pose_cache(video, [])
            mock_savez.assert_not_called()


# ---------------------------------------------------------------------------
# ComparisonRenderer._load_pose_cache
# ---------------------------------------------------------------------------


class TestLoadPoseCache:
    """Test cache loading."""

    def test_cache_miss_returns_none(self, tmp_path):
        """Missing cache file should return None."""
        renderer = ComparisonRenderer()
        video = tmp_path / "no_cache.mp4"
        result = renderer._load_pose_cache(video, expected_frames=10)
        assert result is None

    def test_no_cache_flag_returns_none(self, tmp_path):
        """no_cache=True should always return None."""
        cfg = ComparisonConfig(no_cache=True)
        renderer = ComparisonRenderer(config=cfg)
        video = tmp_path / "cached.mp4"
        cache = video.with_name("cached_poses.npz")
        np.savez_compressed(cache, poses=np.zeros((10, 17, 2)))
        result = renderer._load_pose_cache(video, expected_frames=10)
        assert result is None

    def test_cache_hit(self, tmp_path):
        """Valid cache should return poses as list of arrays."""
        renderer = ComparisonRenderer()
        video = tmp_path / "cached.mp4"
        cache = video.with_name("cached_poses.npz")
        np.savez_compressed(cache, poses=np.zeros((10, 17, 2)))
        result = renderer._load_pose_cache(video, expected_frames=10)
        assert result is not None
        assert len(result) == 10
        assert all(p.shape == (17, 2) for p in result)

    def test_stale_cache_returns_none(self, tmp_path):
        """Cache with large frame mismatch should return None."""
        renderer = ComparisonRenderer()
        video = tmp_path / "cached.mp4"
        cache = video.with_name("cached_poses.npz")
        np.savez_compressed(cache, poses=np.zeros((5, 17, 2)))
        result = renderer._load_pose_cache(video, expected_frames=100)
        assert result is None

    def test_corrupted_cache_returns_none(self, tmp_path):
        """Corrupted cache file should return None."""
        renderer = ComparisonRenderer()
        video = tmp_path / "bad.mp4"
        cache = video.with_name("bad_poses.npz")
        cache.write_text("not a valid npz")
        result = renderer._load_pose_cache(video, expected_frames=10)
        assert result is None


# ---------------------------------------------------------------------------
# ComparisonRenderer._create_extractor
# ---------------------------------------------------------------------------


class TestCreateExtractor:
    """Test PoseExtractor creation."""

    @patch("src.visualization.comparison.PoseExtractor")
    @patch("src.visualization.comparison.DeviceConfig")
    def test_creates_extractor_with_device(self, mock_device_cls, mock_extractor_cls):
        """Extractor should be created with device config."""
        mock_cfg = MagicMock()
        mock_cfg.device = "cuda"
        mock_device_cls.return_value = mock_cfg
        renderer = ComparisonRenderer()
        extractor = renderer._create_extractor("cuda")
        mock_device_cls.assert_called_once_with(device="cuda")
        mock_extractor_cls.assert_called_once_with(conf_threshold=0.3, device="cuda")


# ---------------------------------------------------------------------------
# ComparisonRenderer._extract_poses_streaming
# ---------------------------------------------------------------------------


class TestExtractPosesStreaming:
    """Test streaming pose extraction."""

    def test_extracts_and_converts_to_2d(self, sample_video, mock_tracked_poses):
        """Should extract 3D poses and return 2D slices."""
        renderer = ComparisonRenderer()
        extractor = MagicMock()
        result = MagicMock()
        result.poses = mock_tracked_poses
        extractor.extract_video_tracked.return_value = result

        poses = renderer._extract_poses_streaming(
            sample_video, extractor, target_w=640, target_h=480
        )
        assert len(poses) == 5
        assert all(p.ndim == 2 and p.shape == (17, 2) for p in poses)
        extractor.extract_video_tracked.assert_called_once_with(str(sample_video))

    def test_respects_max_frames(self, sample_video, mock_tracked_poses):
        """max_frames should truncate result."""
        renderer = ComparisonRenderer()
        extractor = MagicMock()
        result = MagicMock()
        result.poses = mock_tracked_poses
        extractor.extract_video_tracked.return_value = result

        poses = renderer._extract_poses_streaming(
            sample_video, extractor, target_w=640, target_h=480, max_frames=3
        )
        assert len(poses) == 3

    def test_respects_start_frame(self, sample_video, mock_tracked_poses):
        """start_frame should skip initial frames."""
        renderer = ComparisonRenderer()
        extractor = MagicMock()
        result = MagicMock()
        result.poses = mock_tracked_poses
        extractor.extract_video_tracked.return_value = result

        poses = renderer._extract_poses_streaming(
            sample_video, extractor, target_w=640, target_h=480, start_frame=2
        )
        assert len(poses) == 3  # 5 total - 2 skipped


# ---------------------------------------------------------------------------
# ComparisonRenderer.process
# ---------------------------------------------------------------------------


class TestProcess:
    """Test full rendering pipeline."""

    @patch("src.visualization.comparison.cv2.VideoCapture")
    @patch("src.visualization.comparison.get_video_meta")
    @patch("src.visualization.comparison.H264Writer")
    @patch("src.visualization.comparison.PoseExtractor")
    @patch("src.visualization.comparison.PoseSmoother")
    @patch("src.visualization.comparison.get_skating_optimized_config")
    @patch("src.visualization.comparison.draw_skeleton")
    def test_side_by_side_mode(
        self,
        mock_draw,
        mock_get_config,
        mock_smoother_cls,
        mock_extractor_cls,
        mock_writer_cls,
        mock_get_meta,
        mock_cap_cls,
        tmp_path,
        mock_video_meta,
        mock_tracked_poses,
        mock_cap,
    ):
        """Side-by-side mode should write frames with divider."""
        mock_get_meta.return_value = mock_video_meta
        mock_cap_cls.return_value = mock_cap
        mock_writer = MagicMock()
        mock_writer_cls.return_value = mock_writer

        # PoseExtractor mock
        extractor_instance = MagicMock()
        tracked = MagicMock()
        tracked.poses = mock_tracked_poses[:5]
        extractor_instance.extract_video_tracked.return_value = tracked
        mock_extractor_cls.return_value = extractor_instance

        # Smoother mock
        smoother_instance = MagicMock()
        smoother_instance.smooth.side_effect = lambda x: x
        mock_smoother_cls.return_value = smoother_instance
        mock_get_config.return_value = {}

        athlete = tmp_path / "athlete.mp4"
        reference = tmp_path / "reference.mp4"
        output = tmp_path / "output.mp4"
        athlete.write_bytes(b"")
        reference.write_bytes(b"")

        cfg = ComparisonConfig(mode=ComparisonMode.SIDE_BY_SIDE, max_frames=3)
        renderer = ComparisonRenderer(config=cfg)
        renderer.process(athlete, reference, output)

        assert mock_writer.write.call_count == 3
        mock_writer.close.assert_called_once()
        mock_cap.release.assert_called()

    @patch("src.visualization.comparison.cv2.VideoCapture")
    @patch("src.visualization.comparison.get_video_meta")
    @patch("src.visualization.comparison.H264Writer")
    @patch("src.visualization.comparison.PoseExtractor")
    @patch("src.visualization.comparison.PoseSmoother")
    @patch("src.visualization.comparison.get_skating_optimized_config")
    @patch("src.visualization.comparison.draw_skeleton")
    def test_overlay_mode(
        self,
        mock_draw,
        mock_get_config,
        mock_smoother_cls,
        mock_extractor_cls,
        mock_writer_cls,
        mock_get_meta,
        mock_cap_cls,
        tmp_path,
        mock_video_meta,
        mock_tracked_poses,
        mock_cap,
    ):
        """Overlay mode should blend reference onto athlete."""
        mock_get_meta.return_value = mock_video_meta
        mock_cap_cls.return_value = mock_cap
        mock_writer = MagicMock()
        mock_writer_cls.return_value = mock_writer

        extractor_instance = MagicMock()
        tracked = MagicMock()
        tracked.poses = mock_tracked_poses[:5]
        extractor_instance.extract_video_tracked.return_value = tracked
        mock_extractor_cls.return_value = extractor_instance

        smoother_instance = MagicMock()
        smoother_instance.smooth.side_effect = lambda x: x
        mock_smoother_cls.return_value = smoother_instance
        mock_get_config.return_value = {}

        athlete = tmp_path / "athlete.mp4"
        reference = tmp_path / "reference.mp4"
        output = tmp_path / "output.mp4"
        athlete.write_bytes(b"")
        reference.write_bytes(b"")

        cfg = ComparisonConfig(mode=ComparisonMode.OVERLAY, max_frames=3)
        renderer = ComparisonRenderer(config=cfg)
        renderer.process(athlete, reference, output)

        assert mock_writer.write.call_count == 3
        mock_writer.close.assert_called_once()

    @patch("src.visualization.comparison.cv2.VideoCapture")
    @patch("src.visualization.comparison.get_video_meta")
    @patch("src.visualization.comparison.H264Writer")
    @patch("src.visualization.comparison.PoseExtractor")
    @patch("src.visualization.comparison.PoseSmoother")
    @patch("src.visualization.comparison.get_skating_optimized_config")
    @patch("src.visualization.comparison.draw_skeleton")
    def test_no_poses_aborts_early(
        self,
        mock_draw,
        mock_get_config,
        mock_smoother_cls,
        mock_extractor_cls,
        mock_writer_cls,
        mock_get_meta,
        mock_cap_cls,
        tmp_path,
        mock_video_meta,
        mock_cap,
    ):
        """If both videos return zero poses, render_frames should be 0 and process returns."""
        mock_get_meta.return_value = mock_video_meta
        mock_cap_cls.return_value = mock_cap
        mock_writer = MagicMock()
        mock_writer_cls.return_value = mock_writer

        extractor_instance = MagicMock()
        tracked = MagicMock()
        tracked.poses = np.zeros((0, 17, 3), dtype=np.float32)
        extractor_instance.extract_video_tracked.return_value = tracked
        mock_extractor_cls.return_value = extractor_instance

        smoother_instance = MagicMock()
        mock_smoother_cls.return_value = smoother_instance
        mock_get_config.return_value = {}

        athlete = tmp_path / "athlete.mp4"
        reference = tmp_path / "reference.mp4"
        output = tmp_path / "output.mp4"
        athlete.write_bytes(b"")
        reference.write_bytes(b"")

        renderer = ComparisonRenderer()
        renderer.process(athlete, reference, output)
        mock_writer.close.assert_called_once()

    @patch("src.visualization.comparison.cv2.VideoCapture")
    @patch("src.visualization.comparison.get_video_meta")
    @patch("src.visualization.comparison.H264Writer")
    def test_cannot_open_athlete_video(
        self,
        mock_writer_cls,
        mock_get_meta,
        mock_cap_cls,
        tmp_path,
        mock_video_meta,
    ):
        """If athlete video cannot be opened, process should close writer and return."""
        mock_get_meta.return_value = mock_video_meta
        bad_cap = MagicMock()
        bad_cap.isOpened.return_value = False
        mock_cap_cls.side_effect = [bad_cap, MagicMock()]
        mock_writer = MagicMock()
        mock_writer_cls.return_value = mock_writer

        athlete = tmp_path / "athlete.mp4"
        reference = tmp_path / "reference.mp4"
        output = tmp_path / "output.mp4"
        athlete.write_bytes(b"")
        reference.write_bytes(b"")

        renderer = ComparisonRenderer()
        renderer.process(athlete, reference, output)
        mock_writer.close.assert_called_once()

    @patch("src.visualization.comparison.cv2.VideoCapture")
    @patch("src.visualization.comparison.get_video_meta")
    @patch("src.visualization.comparison.H264Writer")
    def test_cannot_open_reference_video(
        self,
        mock_writer_cls,
        mock_get_meta,
        mock_cap_cls,
        tmp_path,
        mock_video_meta,
    ):
        """If reference video cannot be opened, process should release athlete cap and return."""
        mock_get_meta.return_value = mock_video_meta
        good_cap = MagicMock()
        good_cap.isOpened.return_value = True
        bad_cap = MagicMock()
        bad_cap.isOpened.return_value = False
        mock_cap_cls.side_effect = [good_cap, bad_cap]
        mock_writer = MagicMock()
        mock_writer_cls.return_value = mock_writer

        athlete = tmp_path / "athlete.mp4"
        reference = tmp_path / "reference.mp4"
        output = tmp_path / "output.mp4"
        athlete.write_bytes(b"")
        reference.write_bytes(b"")

        renderer = ComparisonRenderer()
        renderer.process(athlete, reference, output)
        good_cap.release.assert_called_once()
        mock_writer.close.assert_called_once()

    @patch("src.visualization.comparison.cv2.VideoCapture")
    @patch("src.visualization.comparison.get_video_meta")
    @patch("src.visualization.comparison.H264Writer")
    @patch("src.visualization.comparison.PoseExtractor")
    @patch("src.visualization.comparison.PoseSmoother")
    @patch("src.visualization.comparison.get_skating_optimized_config")
    @patch("src.visualization.comparison.draw_skeleton")
    def test_uses_last_frame_when_video_ends(
        self,
        mock_draw,
        mock_get_config,
        mock_smoother_cls,
        mock_extractor_cls,
        mock_writer_cls,
        mock_get_meta,
        mock_cap_cls,
        tmp_path,
        mock_video_meta,
        mock_tracked_poses,
    ):
        """When one video ends before the other, last frame should be reused."""
        mock_get_meta.return_value = mock_video_meta

        # Athlete cap yields 3 frames then ends
        cap_a = MagicMock()
        cap_a.isOpened.return_value = True
        frames_a = [(True, np.ones((480, 640, 3), dtype=np.uint8) * i * 40) for i in range(3)] + [
            (False, None)
        ]
        cap_a.read.side_effect = frames_a

        # Reference cap yields 5 frames
        cap_r = MagicMock()
        cap_r.isOpened.return_value = True
        frames_r = [(True, np.ones((480, 640, 3), dtype=np.uint8) * i * 40) for i in range(5)] + [
            (False, None)
        ]
        cap_r.read.side_effect = frames_r

        mock_cap_cls.side_effect = [cap_a, cap_r]
        mock_writer = MagicMock()
        mock_writer_cls.return_value = mock_writer

        extractor_instance = MagicMock()
        tracked = MagicMock()
        tracked.poses = mock_tracked_poses[:5]
        extractor_instance.extract_video_tracked.return_value = tracked
        mock_extractor_cls.return_value = extractor_instance

        smoother_instance = MagicMock()
        smoother_instance.smooth.side_effect = lambda x: x
        mock_smoother_cls.return_value = smoother_instance
        mock_get_config.return_value = {}

        athlete = tmp_path / "athlete.mp4"
        reference = tmp_path / "reference.mp4"
        output = tmp_path / "output.mp4"
        athlete.write_bytes(b"")
        reference.write_bytes(b"")

        cfg = ComparisonConfig(max_frames=5)
        renderer = ComparisonRenderer(config=cfg)
        renderer.process(athlete, reference, output)

        assert mock_writer.write.call_count == 5

    @patch("src.visualization.comparison.cv2.VideoCapture")
    @patch("src.visualization.comparison.get_video_meta")
    @patch("src.visualization.comparison.H264Writer")
    @patch("src.visualization.comparison.PoseExtractor")
    @patch("src.visualization.comparison.PoseSmoother")
    @patch("src.visualization.comparison.get_skating_optimized_config")
    @patch("src.visualization.comparison.draw_skeleton")
    def test_start_frame_seek(
        self,
        mock_draw,
        mock_get_config,
        mock_smoother_cls,
        mock_extractor_cls,
        mock_writer_cls,
        mock_get_meta,
        mock_cap_cls,
        tmp_path,
        mock_video_meta,
        mock_tracked_poses,
        mock_cap,
    ):
        """start_frame > 0 should seek both captures."""
        mock_get_meta.return_value = mock_video_meta
        mock_cap_cls.return_value = mock_cap
        mock_writer = MagicMock()
        mock_writer_cls.return_value = mock_writer

        extractor_instance = MagicMock()
        tracked = MagicMock()
        tracked.poses = mock_tracked_poses[:5]
        extractor_instance.extract_video_tracked.return_value = tracked
        mock_extractor_cls.return_value = extractor_instance

        smoother_instance = MagicMock()
        smoother_instance.smooth.side_effect = lambda x: x
        mock_smoother_cls.return_value = smoother_instance
        mock_get_config.return_value = {}

        athlete = tmp_path / "athlete.mp4"
        reference = tmp_path / "reference.mp4"
        output = tmp_path / "output.mp4"
        athlete.write_bytes(b"")
        reference.write_bytes(b"")

        cfg = ComparisonConfig(start_frame=2, max_frames=3)
        renderer = ComparisonRenderer(config=cfg)
        renderer.process(athlete, reference, output)

        mock_cap.set.assert_any_call(pytest.approx(1), pytest.approx(2))  # CAP_PROP_POS_FRAMES
        mock_writer.write.assert_called()
