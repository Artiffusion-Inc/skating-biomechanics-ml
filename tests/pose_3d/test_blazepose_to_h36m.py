"""Tests for BlazePose to H3.6M keypoint mapping."""

import numpy as np
import pytest

from src.pose_3d.blazepose_to_h36m import (
    H36M_KEYPOINT_NAMES,
    H36M_SKELETON_EDGES,
    H36Key,
    blazepose_to_h36m,
    h36m_to_blazepose,
)


class TestBlazePoseToH36M:
    """Tests for blazepose_to_h36m function."""

    def test_single_frame_2d(self):
        """Test converting single frame from 33 to 17 keypoints."""
        pose_33 = np.random.rand(33, 2).astype(np.float32)
        pose_17 = blazepose_to_h36m(pose_33)

        assert pose_17.shape == (17, 2)
        assert pose_17.dtype == np.float32

    def test_single_frame_3d(self):
        """Test converting single frame with confidence channel."""
        pose_33 = np.random.rand(33, 3).astype(np.float32)
        pose_17 = blazepose_to_h36m(pose_33)

        assert pose_17.shape == (17, 3)
        assert pose_17.dtype == pose_33.dtype

    def test_batch_conversion(self):
        """Test converting batch of frames."""
        poses_33 = np.random.rand(100, 33, 2).astype(np.float32)
        poses_17 = blazepose_to_h36m(poses_33)

        assert poses_17.shape == (100, 17, 2)

    def test_invalid_shape_raises_error(self):
        """Test that invalid input shape raises ValueError."""
        pose_wrong = np.random.rand(17, 2).astype(np.float32)

        with pytest.raises(ValueError, match="Expected 33 keypoints"):
            blazepose_to_h36m(pose_wrong)

    def test_mid_hip_calculation(self):
        """Test that hip center is correctly computed as midpoint."""
        pose_33 = np.zeros((33, 2), dtype=np.float32)
        pose_33[23] = [0.4, 0.5]  # LEFT_HIP
        pose_33[24] = [0.6, 0.5]  # RIGHT_HIP

        pose_17 = blazepose_to_h36m(pose_33)

        # Hip center should be midpoint
        expected = np.array([0.5, 0.5])
        np.testing.assert_array_almost_equal(pose_17[H36Key.HIP_CENTER], expected)

    def test_keypoint_names_count(self):
        """Test that we have 17 keypoint names."""
        assert len(H36M_KEYPOINT_NAMES) == 17

    def test_skeleton_edges_valid(self):
        """Test that skeleton edges reference valid keypoint indices."""
        for idx1, idx2 in H36M_SKELETON_EDGES:
            assert 0 <= idx1 < 17, f"Invalid keypoint index: {idx1}"
            assert 0 <= idx2 < 17, f"Invalid keypoint index: {idx2}"


class TestH36MToBlazePose:
    """Tests for h36m_to_blazepose function."""

    def test_single_frame_2d(self):
        """Test converting single frame from 17 to 33 keypoints."""
        pose_17 = np.random.rand(17, 2).astype(np.float32)
        pose_33 = h36m_to_blazepose(pose_17)

        assert pose_33.shape == (33, 2)
        assert pose_33.dtype == np.float32

    def test_batch_conversion(self):
        """Test converting batch of frames."""
        poses_17 = np.random.rand(50, 17, 2).astype(np.float32)
        poses_33 = h36m_to_blazepose(poses_17)

        assert poses_33.shape == (50, 33, 2)

    def test_direct_mapping_preserved(self):
        """Test that directly mapped keypoints are preserved."""
        pose_17 = np.zeros((17, 2), dtype=np.float32)

        # Set specific values
        pose_17[H36Key.HEAD] = [0.5, 0.3]
        pose_17[H36Key.LSHOULDER] = [0.4, 0.4]
        pose_17[H36Key.RSHOULDER] = [0.6, 0.4]

        pose_33 = h36m_to_blazepose(pose_17)

        # Check mapped values are preserved
        np.testing.assert_array_almost_equal(pose_33[0], pose_17[H36Key.HEAD])  # NOSE
        np.testing.assert_array_almost_equal(pose_33[11], pose_17[H36Key.LSHOULDER])
        np.testing.assert_array_almost_equal(pose_33[12], pose_17[H36Key.RSHOULDER])


class TestRoundTripConversion:
    """Tests for round-trip BlazePose -> H3.6M -> BlazePose."""

    def test_round_trip_preserves_core_joints(self):
        """Test that core joints are preserved in round-trip conversion."""
        original = np.random.rand(33, 2).astype(np.float32)

        # BlazePose -> H3.6M -> BlazePose
        intermediate = blazepose_to_h36m(original)
        recovered = h36m_to_blazepose(intermediate)

        assert recovered.shape == original.shape

        # Check that core keypoints are approximately preserved
        core_keypoints = [
            (0, H36Key.HEAD),  # NOSE
            (11, H36Key.LSHOULDER),
            (12, H36Key.RSHOULDER),
            (23, H36Key.LHIP),
            (24, H36Key.RHIP),
            (25, H36Key.LKNEE),
            (26, H36Key.RKNEE),
        ]

        for bp_idx, h36_idx in core_keypoints:
            np.testing.assert_array_almost_equal(
                original[bp_idx], recovered[bp_idx], decimal=5,
                err_msg=f"Keypoint {bp_idx} not preserved"
            )
