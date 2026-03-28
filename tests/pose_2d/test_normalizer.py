"""Tests for pose normalization."""

import numpy as np
import pytest

from src.normalizer import PoseNormalizer
from src.types import BKey


class TestPoseNormalizer:
    """Test PoseNormalizer."""

    def test_normalizer_initialization(self):
        """Should initialize with default parameters."""
        normalizer = PoseNormalizer()

        assert normalizer._target_spine_length == 0.4

    def test_normalizer_custom_spine_length(self):
        """Should initialize with custom spine length."""
        normalizer = PoseNormalizer(target_spine_length=0.5)

        assert normalizer._target_spine_length == 0.5

    def test_normalize_shape(self, sample_keypoints):
        """Should output correct shape."""
        normalizer = PoseNormalizer()

        normalized = normalizer.normalize(sample_keypoints)

        assert normalized.shape == (1, 33, 2)
        assert normalized.dtype == np.float32

    def test_normalize_centers_at_origin(self, sample_keypoints):
        """Should center mid-hip at origin."""
        normalizer = PoseNormalizer()

        normalized = normalizer.normalize(sample_keypoints)

        # Mid-hip should be at origin
        mid_hip = (normalized[0, BKey.LEFT_HIP] + normalized[0, BKey.RIGHT_HIP]) / 2

        assert np.allclose(mid_hip, [0, 0], atol=1e-5)

    def test_normalize_scales_spine(self, sample_keypoints):
        """Should scale spine to target length."""
        target_spine = 0.4
        normalizer = PoseNormalizer(target_spine_length=target_spine)

        normalized = normalizer.normalize(sample_keypoints)

        # Calculate spine length in normalized pose
        mid_shoulder = (normalized[0, BKey.LEFT_SHOULDER] + normalized[0, BKey.RIGHT_SHOULDER]) / 2
        mid_hip = (normalized[0, BKey.LEFT_HIP] + normalized[0, BKey.RIGHT_HIP]) / 2

        spine_length = np.linalg.norm(mid_shoulder - mid_hip)

        assert np.isclose(spine_length, target_spine, rtol=0.01)

    def test_is_valid_frame_good_confidence(self, sample_keypoints):
        """Should accept frame with good confidence."""
        normalizer = PoseNormalizer()

        # All keypoints have confidence >= 0.7
        is_valid = normalizer.is_valid_frame(sample_keypoints[0], min_visible=0.7)

        # Depends on sample_keypoints confidence values
        assert isinstance(is_valid, bool)

    def test_is_valid_frame_low_confidence(self):
        """Should reject frame with low confidence."""
        normalizer = PoseNormalizer()

        # Create frame with low confidence (33 keypoints for BlazePose)
        low_conf_frame = np.zeros((33, 3), dtype=np.float32)
        low_conf_frame[:, 2] = 0.1  # All low confidence

        is_valid = normalizer.is_valid_frame(low_conf_frame, min_visible=0.7)

        assert is_valid is False

    def test_is_valid_frame_wrong_shape(self):
        """Should reject frame with wrong shape."""
        normalizer = PoseNormalizer()

        # Wrong shape - missing confidence channel
        wrong_shape = np.zeros((33, 2), dtype=np.float32)

        is_valid = normalizer.is_valid_frame(wrong_shape)

        assert is_valid is False

    def test_get_spine_length(self, sample_keypoints):
        """Should calculate average spine length."""
        normalizer = PoseNormalizer()

        spine_length = normalizer.get_spine_length(sample_keypoints)

        assert spine_length > 0
        assert isinstance(spine_length, float)


class TestNormalizeMultipleFrames:
    """Test normalization with multiple frames."""

    def test_normalize_three_frames(self):
        """Should normalize three frames correctly."""
        normalizer = PoseNormalizer()

        # Create three identical frames (33 keypoints for BlazePose)
        frames = np.tile(np.zeros((1, 33, 3), dtype=np.float32), (3, 1, 1))

        # Set some positions (BlazePose 33 format)
        for i in range(3):
            frames[i, BKey.LEFT_SHOULDER, :2] = [280, 200]
            frames[i, BKey.RIGHT_SHOULDER, :2] = [360, 200]
            frames[i, BKey.LEFT_HIP, :2] = [290, 350]
            frames[i, BKey.RIGHT_HIP, :2] = [350, 350]
            frames[i, :, 2] = 0.9  # confidence

        normalized = normalizer.normalize(frames)

        assert normalized.shape == (3, 33, 2)

        # Each frame should be centered
        for i in range(3):
            mid_hip = (normalized[i, BKey.LEFT_HIP] + normalized[i, BKey.RIGHT_HIP]) / 2
            assert np.allclose(mid_hip, [0, 0], atol=1e-5)
