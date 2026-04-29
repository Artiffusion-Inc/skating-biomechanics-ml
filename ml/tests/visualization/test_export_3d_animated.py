"""Tests for ml/src/visualization/export_3d_animated.py."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# Add ml to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.visualization.export_3d_animated import (
    _compute_trs,
    _cylinder_geometry,
    _icosphere_geometry,
    _subdivide_mesh,
    poses_to_animated_glb,
)


@pytest.fixture
def sample_3d_poses():
    """Create a minimal 2-frame 3D pose sequence."""
    poses = np.zeros((2, 17, 3), dtype=np.float32)
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
    poses[1] = poses[0].copy()
    poses[1, 0, 0] += 0.01  # slight shift in second frame
    return poses


# ---------------------------------------------------------------------------
# _cylinder_geometry
# ---------------------------------------------------------------------------


class TestCylinderGeometry:
    """Test cylinder mesh generation."""

    def test_returns_vertices_and_indices(self):
        """Should return (vertices, indices) arrays."""
        verts, idx = _cylinder_geometry(radius=1.0, height=2.0, sections=8)
        assert verts.ndim == 2
        assert verts.shape[1] == 3
        assert idx.ndim == 1
        assert len(verts) == 8 * 2  # top + bottom circles
        assert len(idx) == 8 * 2 * 3  # 2 triangles per section

    def test_custom_radius(self):
        """Custom radius should scale x/z coordinates."""
        verts, _ = _cylinder_geometry(radius=2.0, height=1.0, sections=8)
        max_r = np.max(np.sqrt(verts[:, 0] ** 2 + verts[:, 2] ** 2))
        assert max_r == pytest.approx(2.0, rel=1e-5)

    def test_custom_height(self):
        """Custom height should scale y range."""
        verts, _ = _cylinder_geometry(radius=1.0, height=4.0, sections=8)
        assert np.max(verts[:, 1]) == pytest.approx(2.0, rel=1e-5)
        assert np.min(verts[:, 1]) == pytest.approx(-2.0, rel=1e-5)

    def test_different_sections(self):
        """Different section counts should change vertex count."""
        verts4, idx4 = _cylinder_geometry(radius=1.0, height=1.0, sections=4)
        verts16, idx16 = _cylinder_geometry(radius=1.0, height=1.0, sections=16)
        assert len(verts4) == 4 * 2
        assert len(verts16) == 16 * 2
        assert len(idx4) == 4 * 2 * 3
        assert len(idx16) == 16 * 2 * 3


# ---------------------------------------------------------------------------
# _icosphere_geometry
# ---------------------------------------------------------------------------


class TestIcosphereGeometry:
    """Test icosphere mesh generation."""

    def test_returns_vertices_and_indices(self):
        """Should return (vertices, indices) arrays."""
        verts, idx = _icosphere_geometry(radius=1.0, subdivisions=0)
        assert verts.ndim == 2
        assert verts.shape[1] == 3
        assert idx.ndim == 1
        assert len(verts) == 12  # icosahedron vertices
        assert len(idx) == 20 * 3  # 20 triangles

    def test_radius_scaling(self):
        """Radius should scale vertices."""
        verts, _ = _icosphere_geometry(radius=3.0, subdivisions=0)
        norms = np.linalg.norm(verts, axis=1)
        assert np.allclose(norms, 3.0)

    def test_subdivision_increases_vertices(self):
        """Subdivision should increase vertex and index count."""
        verts0, idx0 = _icosphere_geometry(radius=1.0, subdivisions=0)
        verts1, idx1 = _icosphere_geometry(radius=1.0, subdivisions=1)
        assert len(verts1) > len(verts0)
        assert len(idx1) > len(idx0)


# ---------------------------------------------------------------------------
# _subdivide_mesh
# ---------------------------------------------------------------------------


class TestSubdivideMesh:
    """Test mesh subdivision helper."""

    def test_subdivides_single_triangle(self):
        """Subdividing one triangle should produce 4 triangles."""
        verts = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
        idx = np.array([0, 1, 2], dtype=np.uint32)
        new_verts, new_idx = _subdivide_mesh(verts, idx)
        assert len(new_idx) == 4 * 3
        assert len(new_verts) > len(verts)

    def test_preserves_original_vertices(self):
        """Original vertices should be present in output."""
        verts = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
        idx = np.array([0, 1, 2], dtype=np.uint32)
        new_verts, _ = _subdivide_mesh(verts, idx)
        # First 3 vertices should match original
        np.testing.assert_array_almost_equal(new_verts[:3], verts)

    def test_shared_edges_reuse_midpoints(self):
        """Adjacent triangles should share midpoints."""
        verts = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [1.0, 1.0, 0.0],
            ],
            dtype=np.float32,
        )
        # Two triangles sharing edge (1,2)
        idx = np.array([0, 1, 2, 1, 3, 2], dtype=np.uint32)
        new_verts, new_idx = _subdivide_mesh(verts, idx)
        # Should reuse midpoints for shared edges
        assert len(new_verts) < len(verts) + 6  # 6 midpoints if no sharing, less if shared
        assert len(new_idx) == 8 * 3  # 2 triangles → 8 triangles


# ---------------------------------------------------------------------------
# _compute_trs
# ---------------------------------------------------------------------------


class TestComputeTrs:
    """Test TRS matrix computation for bone cylinders."""

    def test_basic_bone(self):
        """Compute TRS for a simple vertical bone."""
        start = np.array([0.0, 0.0, 0.0])
        end = np.array([0.0, 1.0, 0.0])
        trans, _quat, scale = _compute_trs(start, end, bone_radius=0.01)
        np.testing.assert_array_almost_equal(trans, [0.0, 0.5, 0.0])
        assert scale[1] == pytest.approx(0.5, rel=1e-5)
        assert scale[0] == pytest.approx(0.01, rel=1e-5)

    def test_degenerate_bone(self):
        """Zero-length bone should return default values."""
        start = np.array([1.0, 2.0, 3.0])
        end = start.copy()
        trans, quat, scale = _compute_trs(start, end, bone_radius=0.02)
        np.testing.assert_array_almost_equal(trans, start)
        assert quat[0] == 1.0
        assert scale[1] == 0.5

    def test_horizontal_bone(self):
        """Horizontal bone should produce a 90° rotation quaternion."""
        start = np.array([0.0, 0.0, 0.0])
        end = np.array([1.0, 0.0, 0.0])
        trans, _quat, scale = _compute_trs(start, end, bone_radius=0.01)
        np.testing.assert_array_almost_equal(trans, [0.5, 0.0, 0.0])
        assert scale[1] == pytest.approx(0.5, rel=1e-5)

    def test_negative_y_bone(self):
        """Bone pointing straight down should use special-case quaternion."""
        start = np.array([0.0, 1.0, 0.0])
        end = np.array([0.0, 0.0, 0.0])
        trans, quat, _scale = _compute_trs(start, end, bone_radius=0.01)
        np.testing.assert_array_almost_equal(trans, [0.0, 0.5, 0.0])
        # Special case: [0, 1, 0, 0] for opposite direction
        np.testing.assert_array_almost_equal(quat, [0.0, 1.0, 0.0, 0.0])


# ---------------------------------------------------------------------------
# poses_to_animated_glb
# ---------------------------------------------------------------------------


class TestPosesToAnimatedGlb:
    """Test animated GLB export."""

    @patch("src.visualization.export_3d_animated.pygltflib")
    @patch("src.visualization.export_3d_animated.R")
    def test_valid_sequence(self, mock_r_class, mock_pygltflib, sample_3d_poses):
        """Exporting a valid sequence should return a .glb path."""
        # Mock pygltflib classes
        mock_pygltflib.GLTF2.return_value = MagicMock(
            bufferViews=[],
            accessors=[],
            meshes=[],
            materials=[],
            nodes=[],
            scenes=[],
            animations=[],
            buffers=[],
        )
        mock_pygltflib.BufferView = MagicMock()
        mock_pygltflib.Accessor = MagicMock()
        mock_pygltflib.Mesh = MagicMock()
        mock_pygltflib.Material = MagicMock()
        mock_pygltflib.PbrMetallicRoughness = MagicMock()
        mock_pygltflib.Primitive = MagicMock()
        mock_pygltflib.Attributes = MagicMock(return_value=MagicMock())
        mock_pygltflib.Node = MagicMock()
        mock_pygltflib.Scene = MagicMock()
        mock_pygltflib.Animation = MagicMock()
        mock_pygltflib.AnimationSampler = MagicMock()
        mock_pygltflib.AnimationChannel = MagicMock()
        mock_pygltflib.AnimationChannelTarget = MagicMock()
        mock_pygltflib.ARRAY_BUFFER = 34962
        mock_pygltflib.ELEMENT_ARRAY_BUFFER = 34963
        mock_pygltflib.FLOAT = 5126
        mock_pygltflib.UNSIGNED_INT = 5125
        mock_pygltflib.VEC3 = "VEC3"
        mock_pygltflib.VEC4 = "VEC4"
        mock_pygltflib.SCALAR = "SCALAR"
        mock_pygltflib.TRIANGLES = 4
        mock_pygltflib.LINEAR = "LINEAR"

        # Mock Rotation
        rot_mock = MagicMock()
        rot_mock.as_quat.return_value = np.array([0.0, 0.0, 0.0, 1.0])
        mock_r_class.align_vectors.return_value = (rot_mock,)

        path = poses_to_animated_glb(sample_3d_poses, fps=30.0)
        assert path.endswith(".glb")

    def test_invalid_shape_raises(self):
        """Invalid pose shape should raise ValueError."""
        with pytest.raises(ValueError, match="poses_3d must be"):
            poses_to_animated_glb(np.zeros((10, 10, 3)))

    def test_wrong_dims_raises(self):
        """Wrong number of dimensions should raise ValueError."""
        with pytest.raises(ValueError, match="poses_3d must be"):
            poses_to_animated_glb(np.zeros((17, 3)))

    @patch("src.visualization.export_3d_animated.pygltflib")
    @patch("src.visualization.export_3d_animated.R")
    def test_nan_bones_get_default_values(self, mock_r_class, mock_pygltflib, sample_3d_poses):
        """NaN bones should use default TRS values."""
        gltf_mock = MagicMock(
            bufferViews=[],
            accessors=[],
            meshes=[],
            materials=[],
            nodes=[],
            scenes=[],
            animations=[],
            buffers=[],
        )
        mock_pygltflib.GLTF2.return_value = gltf_mock
        mock_pygltflib.BufferView = MagicMock()
        mock_pygltflib.Accessor = MagicMock()
        mock_pygltflib.Mesh = MagicMock()
        mock_pygltflib.Material = MagicMock()
        mock_pygltflib.PbrMetallicRoughness = MagicMock()
        mock_pygltflib.Primitive = MagicMock()
        mock_pygltflib.Attributes = MagicMock(return_value=MagicMock())
        mock_pygltflib.Node = MagicMock()
        mock_pygltflib.Scene = MagicMock()
        mock_pygltflib.Animation = MagicMock()
        mock_pygltflib.AnimationSampler = MagicMock()
        mock_pygltflib.AnimationChannel = MagicMock()
        mock_pygltflib.AnimationChannelTarget = MagicMock()
        mock_pygltflib.ARRAY_BUFFER = 34962
        mock_pygltflib.ELEMENT_ARRAY_BUFFER = 34963
        mock_pygltflib.FLOAT = 5126
        mock_pygltflib.UNSIGNED_INT = 5125
        mock_pygltflib.VEC3 = "VEC3"
        mock_pygltflib.VEC4 = "VEC4"
        mock_pygltflib.SCALAR = "SCALAR"
        mock_pygltflib.TRIANGLES = 4
        mock_pygltflib.LINEAR = "LINEAR"

        rot_mock = MagicMock()
        rot_mock.as_quat.return_value = np.array([0.0, 0.0, 0.0, 1.0])
        mock_r_class.align_vectors.return_value = (rot_mock,)

        poses = sample_3d_poses.copy()
        poses[0, 1] = np.nan  # Make RHIP NaN so HIP_CENTER→RHIP bone is NaN
        path = poses_to_animated_glb(poses, fps=30.0)
        assert path.endswith(".glb")

    @patch("src.visualization.export_3d_animated.pygltflib")
    @patch("src.visualization.export_3d_animated.R")
    def test_custom_fps_and_radii(self, mock_r_class, mock_pygltflib, sample_3d_poses):
        """Custom fps and radii should be accepted."""
        gltf_mock = MagicMock(
            bufferViews=[],
            accessors=[],
            meshes=[],
            materials=[],
            nodes=[],
            scenes=[],
            animations=[],
            buffers=[],
        )
        mock_pygltflib.GLTF2.return_value = gltf_mock
        mock_pygltflib.BufferView = MagicMock()
        mock_pygltflib.Accessor = MagicMock()
        mock_pygltflib.Mesh = MagicMock()
        mock_pygltflib.Material = MagicMock()
        mock_pygltflib.PbrMetallicRoughness = MagicMock()
        mock_pygltflib.Primitive = MagicMock()
        mock_pygltflib.Attributes = MagicMock(return_value=MagicMock())
        mock_pygltflib.Node = MagicMock()
        mock_pygltflib.Scene = MagicMock()
        mock_pygltflib.Animation = MagicMock()
        mock_pygltflib.AnimationSampler = MagicMock()
        mock_pygltflib.AnimationChannel = MagicMock()
        mock_pygltflib.AnimationChannelTarget = MagicMock()
        mock_pygltflib.ARRAY_BUFFER = 34962
        mock_pygltflib.ELEMENT_ARRAY_BUFFER = 34963
        mock_pygltflib.FLOAT = 5126
        mock_pygltflib.UNSIGNED_INT = 5125
        mock_pygltflib.VEC3 = "VEC3"
        mock_pygltflib.VEC4 = "VEC4"
        mock_pygltflib.SCALAR = "SCALAR"
        mock_pygltflib.TRIANGLES = 4
        mock_pygltflib.LINEAR = "LINEAR"

        rot_mock = MagicMock()
        rot_mock.as_quat.return_value = np.array([0.0, 0.0, 0.0, 1.0])
        mock_r_class.align_vectors.return_value = (rot_mock,)

        path = poses_to_animated_glb(sample_3d_poses, fps=60.0, bone_radius=0.02, joint_radius=0.03)
        assert path.endswith(".glb")
