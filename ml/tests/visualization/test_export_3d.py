"""Tests for ml/src/visualization/export_3d.py."""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# Add ml to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.visualization.export_3d import (
    _add_ground_grid,
    _angle_color,
    poses_to_glb,
    poses_to_glb_sequence,
)


@pytest.fixture
def mock_trimesh():
    """Return a mock trimesh module."""
    mock = MagicMock()
    mock.Scene.return_value = MagicMock(geometry={})
    mock.creation.cylinder.return_value = MagicMock(
        apply_translation=MagicMock(),
        apply_transform=MagicMock(),
        visual=MagicMock(face_colors=None),
    )
    mock.creation.icosphere.return_value = MagicMock(
        apply_translation=MagicMock(),
        visual=MagicMock(face_colors=None),
    )
    mock.transformations.rotation_matrix.return_value = np.eye(4)
    mock.load_path.return_value = MagicMock()
    return mock


@pytest.fixture
def sample_3d_poses():
    """Create a minimal 2-frame 3D pose sequence."""
    poses = np.zeros((2, 17, 3), dtype=np.float32)
    # Frame 0: standing pose
    poses[0, 0] = [0.0, 0.0, 0.0]  # HIP_CENTER
    poses[0, 1] = [-0.1, 0.0, 0.0]  # RHIP
    poses[0, 2] = [-0.1, -0.4, 0.0]  # RKNEE
    poses[0, 3] = [-0.1, -0.8, 0.0]  # RFOOT
    poses[0, 4] = [0.1, 0.0, 0.0]  # LHIP
    poses[0, 5] = [0.1, -0.4, 0.0]  # LKNEE
    poses[0, 6] = [0.1, -0.8, 0.0]  # LFOOT
    poses[0, 7] = [0.0, 0.1, 0.0]  # SPINE
    poses[0, 8] = [0.0, 0.2, 0.0]  # THORAX
    poses[0, 9] = [0.0, 0.3, 0.0]  # NECK
    poses[0, 10] = [0.0, 0.4, 0.0]  # HEAD
    poses[0, 11] = [0.1, 0.2, 0.0]  # LSHOULDER
    poses[0, 12] = [0.2, 0.1, 0.0]  # LELBOW
    poses[0, 13] = [0.25, 0.0, 0.0]  # LWRIST
    poses[0, 14] = [-0.1, 0.2, 0.0]  # RSHOULDER
    poses[0, 15] = [-0.2, 0.1, 0.0]  # RELBOW
    poses[0, 16] = [-0.25, 0.0, 0.0]  # RWRIST
    # Frame 1: copy of frame 0
    poses[1] = poses[0].copy()
    return poses


# ---------------------------------------------------------------------------
# _angle_color
# ---------------------------------------------------------------------------


class TestAngleColor:
    """Test angle-based quality coloring."""

    def test_good_angle_returns_green(self):
        """Angles inside the good range should be green."""
        color = _angle_color(120.0)
        assert color == (50, 220, 50)

    def test_warn_angle_returns_yellow(self):
        """Angles in the warning range should be yellow."""
        color = _angle_color(80.0)
        assert color == (220, 220, 50)
        color = _angle_color(180.0)
        assert color == (220, 220, 50)

    def test_bad_angle_returns_red(self):
        """Angles outside both ranges should be red."""
        color = _angle_color(30.0)
        assert color == (220, 50, 50)
        color = _angle_color(200.0)
        assert color == (220, 50, 50)

    def test_boundary_values(self):
        """Exact boundary values should map correctly."""
        assert _angle_color(90.0) == (50, 220, 50)
        assert _angle_color(170.0) == (50, 220, 50)
        assert _angle_color(60.0) == (220, 220, 50)
        assert _angle_color(190.0) == (220, 220, 50)


# ---------------------------------------------------------------------------
# poses_to_glb
# ---------------------------------------------------------------------------


class TestPosesToGlb:
    """Test single-frame GLB export."""

    @patch("src.visualization.export_3d.trimesh")
    def test_valid_pose_exports_glb(self, mock_trimesh, sample_3d_poses):
        """Exporting a valid pose should return a .glb path."""
        scene_mock = MagicMock()
        scene_mock.geometry = {"a": 1, "b": 2}
        mock_trimesh.Scene.return_value = scene_mock
        mock_trimesh.creation.cylinder.return_value = MagicMock(
            apply_translation=MagicMock(),
            apply_transform=MagicMock(),
            visual=MagicMock(face_colors=None),
        )
        mock_trimesh.creation.icosphere.return_value = MagicMock(
            apply_translation=MagicMock(),
            visual=MagicMock(face_colors=None),
        )

        path = poses_to_glb(sample_3d_poses, frame_idx=0)
        assert path.endswith(".glb")
        scene_mock.export.assert_called_once()

    @patch("src.visualization.export_3d.trimesh")
    def test_frame_idx_out_of_bounds_clamped(self, mock_trimesh, sample_3d_poses):
        """If frame_idx exceeds pose length, it should clamp to last frame."""
        scene_mock = MagicMock()
        scene_mock.geometry = {"a": 1}
        mock_trimesh.Scene.return_value = scene_mock
        mock_trimesh.creation.cylinder.return_value = MagicMock(
            apply_translation=MagicMock(),
            apply_transform=MagicMock(),
            visual=MagicMock(face_colors=None),
        )
        mock_trimesh.creation.icosphere.return_value = MagicMock(
            apply_translation=MagicMock(),
            visual=MagicMock(face_colors=None),
        )

        path = poses_to_glb(sample_3d_poses, frame_idx=99)
        assert path.endswith(".glb")
        scene_mock.export.assert_called_once()

    @patch("src.visualization.export_3d.trimesh")
    def test_all_nan_returns_empty_string(self, mock_trimesh):
        """A frame where all keypoints are NaN should return empty string."""
        poses = np.full((1, 17, 3), np.nan, dtype=np.float32)
        result = poses_to_glb(poses, frame_idx=0)
        assert result == ""

    @patch("src.visualization.export_3d.trimesh")
    def test_empty_scene_returns_empty_string(self, mock_trimesh, sample_3d_poses):
        """If no valid bones/joints are drawn, return empty string."""
        scene_mock = MagicMock()
        scene_mock.geometry = {}
        mock_trimesh.Scene.return_value = scene_mock
        mock_trimesh.creation.cylinder.return_value = MagicMock(
            apply_translation=MagicMock(),
            apply_transform=MagicMock(),
            visual=MagicMock(face_colors=None),
        )
        mock_trimesh.creation.icosphere.return_value = MagicMock(
            apply_translation=MagicMock(),
            visual=MagicMock(face_colors=None),
        )

        # Make all joint positions NaN except a few isolated ones so no edges can be drawn
        poses = sample_3d_poses.copy()
        poses[0] = np.nan
        poses[0, 0] = [0.0, 0.0, 0.0]
        poses[0, 10] = [0.0, 0.4, 0.0]

        result = poses_to_glb(poses, frame_idx=0)
        assert result == ""

    @patch("src.visualization.export_3d.trimesh")
    def test_bone_radius_parameter(self, mock_trimesh, sample_3d_poses):
        """Custom bone_radius should be passed to cylinder creation."""
        scene_mock = MagicMock()
        scene_mock.geometry = {"a": 1, "b": 2}
        mock_trimesh.Scene.return_value = scene_mock
        cyl_mock = MagicMock(
            apply_translation=MagicMock(),
            apply_transform=MagicMock(),
            visual=MagicMock(face_colors=None),
        )
        mock_trimesh.creation.cylinder.return_value = cyl_mock
        mock_trimesh.creation.icosphere.return_value = MagicMock(
            apply_translation=MagicMock(),
            visual=MagicMock(face_colors=None),
        )

        poses_to_glb(sample_3d_poses, frame_idx=0, bone_radius=0.05, joint_radius=0.08)
        calls = mock_trimesh.creation.cylinder.call_args_list
        assert all(
            call.kwargs.get("radius", call.args[0] if call.args else None) == 0.05 for call in calls
        )

    @patch("src.visualization.export_3d.trimesh")
    def test_zero_length_bone_skipped(self, mock_trimesh, sample_3d_poses):
        """Bones with zero length should be skipped."""
        scene_mock = MagicMock()
        scene_mock.geometry = {"a": 1}
        mock_trimesh.Scene.return_value = scene_mock
        cyl_mock = MagicMock(
            apply_translation=MagicMock(),
            apply_transform=MagicMock(),
            visual=MagicMock(face_colors=None),
        )
        mock_trimesh.creation.cylinder.return_value = cyl_mock
        mock_trimesh.creation.icosphere.return_value = MagicMock(
            apply_translation=MagicMock(),
            visual=MagicMock(face_colors=None),
        )

        poses = sample_3d_poses.copy()
        poses[0, 1] = poses[0, 0]  # RHIP = HIP_CENTER → zero-length bone
        path = poses_to_glb(poses, frame_idx=0)
        assert path.endswith(".glb")


# ---------------------------------------------------------------------------
# poses_to_glb_sequence
# ---------------------------------------------------------------------------


class TestPosesToGlbSequence:
    """Test multi-frame GLB sequence export."""

    @patch("src.visualization.export_3d.poses_to_glb")
    def test_sequence_export(self, mock_poses_to_glb, sample_3d_poses, tmp_path):
        """Exporting a sequence should create numbered .glb files."""

        # Make poses_to_glb create actual dummy files so rename succeeds
        def _make_glb(*args, **kwargs):
            fd, path = tempfile.mkstemp(suffix=".glb")
            os.close(fd)
            return path

        mock_poses_to_glb.side_effect = _make_glb

        out_dir = poses_to_glb_sequence(sample_3d_poses, str(tmp_path))
        assert Path(out_dir).exists()
        # Should have created one file per frame
        files = list(Path(out_dir).glob("*.glb"))
        assert len(files) == len(sample_3d_poses)


# ---------------------------------------------------------------------------
# _add_ground_grid
# ---------------------------------------------------------------------------


class TestAddGroundGrid:
    """Test ground plane grid helper."""

    @patch("src.visualization.export_3d.trimesh")
    def test_adds_lines(self, mock_trimesh):
        """Ground grid should add line geometries to the scene."""
        scene_mock = MagicMock()
        pose = np.zeros((17, 3), dtype=np.float32)
        pose[:, 1] = np.arange(17) * 0.1  # varying Y values
        _add_ground_grid(scene_mock, pose)
        assert scene_mock.add_geometry.call_count > 0

    @patch("src.visualization.export_3d.trimesh")
    def test_all_nan_returns_early(self, mock_trimesh):
        """If all Y values are NaN, the function should return early."""
        scene_mock = MagicMock()
        pose = np.full((17, 3), np.nan, dtype=np.float32)
        _add_ground_grid(scene_mock, pose)
        scene_mock.add_geometry.assert_not_called()
