"""Pose normalization for camera-invariant analysis.

This module provides normalization utilities to make poses invariant to:
- Root position (centering at mid-hip)
- Scale (spine length normalization)
- Anthropometry (body proportions)
"""

import numpy as np

from .types import BKey, FrameKeypoints, NormalizedPose
from .geometry import get_mid_hip, get_mid_shoulder


class PoseNormalizer:
    """Normalize poses for camera-invariant analysis.

    Applies root-centering and scale normalization to make poses
    comparable across different videos and athletes.
    """

    def __init__(self, target_spine_length: float = 0.4) -> None:
        """Initialize pose normalizer.

        Args:
            target_spine_length: Target spine length after normalization.
                Spine is measured from mid-shoulder to mid-hip.
                Default 0.4 is typical for adult athletes.
        """
        self._target_spine_length = target_spine_length

    def normalize(self, keypoints: FrameKeypoints) -> NormalizedPose:
        """Normalize poses via root-centering and scale normalization.

        Normalization steps:
        1. Center each frame at mid-hip (root) -> origin (0, 0)
        2. Scale so spine length equals target_spine_length
        3. Drop confidence channel (33, 3) -> (33, 2)

        Args:
            keypoints: Raw keypoints (num_frames, 33, 3) with x, y, confidence.

        Returns:
            NormalizedPose (num_frames, 33, 2) with centered, scaled coordinates.

        Raises:
            ValueError: If keypoints shape is invalid.
        """
        if keypoints.ndim != 3 or keypoints.shape[1] != 33 or keypoints.shape[2] != 3:
            raise ValueError(f"Expected keypoints shape (N, 33, 3), got {keypoints.shape}")

        num_frames = keypoints.shape[0]
        normalized = np.zeros((num_frames, 33, 2), dtype=np.float32)

        # Process each frame
        for frame_idx in range(num_frames):
            frame_kp = keypoints[frame_idx]

            # Get mid-hip position
            left_hip = frame_kp[BKey.LEFT_HIP, :2]
            right_hip = frame_kp[BKey.RIGHT_HIP, :2]
            mid_hip = (left_hip + right_hip) / 2

            # 1. Root-centering: shift mid-hip to origin
            centered = frame_kp[:, :2] - mid_hip

            # 2. Scale normalization
            left_shoulder = frame_kp[BKey.LEFT_SHOULDER, :2]
            right_shoulder = frame_kp[BKey.RIGHT_SHOULDER, :2]
            mid_shoulder = (left_shoulder + right_shoulder) / 2

            spine_vector = mid_shoulder - mid_hip
            spine_length = np.linalg.norm(spine_vector)

            if spine_length < 1e-6:
                # Degenerate case: use identity scale
                scale = 1.0
            else:
                scale = self._target_spine_length / spine_length

            normalized[frame_idx] = centered * scale

        return normalized

    def get_spine_length(self, keypoints: FrameKeypoints) -> float:
        """Calculate average spine length across frames.

        Args:
            keypoints: Raw keypoints (num_frames, 17, 3).

        Returns:
            Average spine length in original coordinate units.
        """
        mid_hip = get_mid_hypot(keypoints)
        mid_shoulder = get_mid_shoulder_raw(keypoints)

        spine_lengths = np.linalg.norm(mid_shoulder - mid_hip, axis=1)
        return float(np.mean(spine_lengths))

    def is_valid_frame(
        self,
        frame_kp: np.ndarray,
        min_visible: float = 0.7,
    ) -> bool:
        """Check if frame has enough visible keypoints.

        Args:
            frame_kp: Single frame keypoints (33, 3) with x, y, confidence.
            min_visible: Minimum ratio of visible keypoints [0, 1].

        Returns:
            True if frame is valid for analysis.
        """
        if frame_kp.shape != (33, 3):
            return False

        # Count keypoints with confidence > 0.5
        visible = np.sum(frame_kp[:, 2] > 0.5)
        ratio = visible / 33

        return bool(ratio >= min_visible)


def get_mid_hypot(keypoints: FrameKeypoints) -> np.ndarray:
    """Calculate mid-hip point for each frame (raw keypoints).

    Args:
        keypoints: Raw FrameKeypoints (num_frames, 33, 3).

    Returns:
        Mid-hip coordinates (num_frames, 2).
    """
    left_hip = keypoints[:, BKey.LEFT_HIP, :2]
    right_hip = keypoints[:, BKey.RIGHT_HIP, :2]
    return (left_hip + right_hip) / 2


def get_mid_shoulder_raw(keypoints: FrameKeypoints) -> np.ndarray:
    """Calculate mid-shoulder point for each frame (raw keypoints).

    Args:
        keypoints: Raw FrameKeypoints (num_frames, 33, 3).

    Returns:
        Mid-shoulder coordinates (num_frames, 2).
    """
    left_shoulder = keypoints[:, BKey.LEFT_SHOULDER, :2]
    right_shoulder = keypoints[:, BKey.RIGHT_SHOULDER, :2]
    return (left_shoulder + right_shoulder) / 2
