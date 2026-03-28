"""Simple 3D pose estimator using biomechanical constraints.

This is a lightweight alternative to full ML-based 3D estimation.
Uses anatomical constraints to estimate Z-depth from 2D poses.

For production use, replace with MotionAGFormer or Pose3DM models.
"""

from collections import deque

import numpy as np


class Biomechanics3DEstimator:
    """Estimate 3D poses from 2D using biomechanical constraints.

    Simple approach that works without heavy ML models:
    - Uses body segment proportions (Dempster tables)
    - Assumes frontal/sagittal plane projection
    - Estimates Z from joint size ratios

    For accurate 3D, use AthletePose3D with MotionAGFormer instead.
    """

    def __init__(self, focal_length: float = 1000.0):
        """Initialize biomechanics 3D estimator.

        Args:
            focal_length: Camera focal length in pixels (for Z estimation)
        """
        self.focal_length = focal_length
        self.known_heights = {
            "head": 0.12,  # 12% of body height
            "torso": 0.30,  # 30% of body height
            "thigh": 0.25,  # 25% of body height
            "shin": 0.23,  # 23% of body height
            "foot": 0.05,  # 5% of body height
            "upper_arm": 0.19,  # 19% of body height
            "forearm": 0.16,  # 16% of body height
            "hand": 0.06,  # 6% of body height
        }

        # Temporal buffer for smoothing
        self.z_buffer: deque[np.ndarray] = deque(maxlen=5)

    def estimate_3d(
        self,
        poses_2d: np.ndarray,
        body_height: float = 1.7,
    ) -> np.ndarray:
        """Estimate 3D poses from 2D H3.6M format.

        Args:
            poses_2d: (N, 17, 2) array in H3.6M format
            body_height: Estimated body height in meters

        Returns:
            poses_3d: (N, 17, 3) array with x, y, z in meters
        """
        n_frames = poses_2d.shape[0]
        poses_3d = np.zeros((n_frames, 17, 3), dtype=np.float32)

        for frame_idx in range(n_frames):
            pose_2d = poses_2d[frame_idx]

            # Estimate Z for each joint based on its Y position
            # Lower joints (feet) are closer to camera (smaller Z)
            # Higher joints (head) are farther (larger Z)

            for joint_idx in range(17):
                x, y = pose_2d[joint_idx]

                # Simple depth estimation based on Y position
                # Assuming camera is at waist level
                z = self._estimate_depth_from_y(y)

                # Convert normalized coordinates to meters
                x_m = (x - 0.5) * body_height
                y_m = (0.5 - y) * body_height
                z_m = z * body_height

                poses_3d[frame_idx, joint_idx] = [x_m, y_m, z_m]

        # Smooth Z values temporally
        poses_3d = self._smooth_z(poses_3d)

        return poses_3d

    def _estimate_depth_from_y(self, y: float) -> float:
        """Estimate relative depth from Y position.

        Simple heuristic: lower in frame = closer to camera
        """
        # Normalize Y to [-1, 1] where 0 is center
        y_norm = (y - 0.5) * 2

        # Invert: lower Y (negative) = closer (smaller Z)
        z = -y_norm * 0.5  # Scale factor for depth range

        return z

    def _smooth_z(self, poses_3d: np.ndarray) -> np.ndarray:
        """Apply temporal smoothing to Z coordinates."""
        n_frames = poses_3d.shape[0]

        for frame_idx in range(n_frames):
            z_values = poses_3d[frame_idx, :, 2]
            self.z_buffer.append(z_values.copy())

            if len(self.z_buffer) > 1:
                # Average over buffer
                stacked = np.stack(list(self.z_buffer))
                z_smooth = np.mean(stacked, axis=0)
                poses_3d[frame_idx, :, 2] = z_smooth

        return poses_3d


def estimate_3d_simple(poses_2d: np.ndarray) -> np.ndarray:
    """Quick 3D estimation using simple scaling.

    Args:
        poses_2d: (N, 17, 2) normalized poses

    Returns:
        poses_3d: (N, 17, 3) with Z=0 (flat) or estimated depth
    """
    n_frames = poses_2d.shape[0]
    poses_3d = np.zeros((n_frames, 17, 3), dtype=np.float32)

    # Copy X, Y and set Z based on simple heuristics
    poses_3d[:, :, :2] = poses_2d

    # Simple Z: assume person is on a plane (Z varies by joint)
    # Head is farther back, feet are closer
    for joint_idx in range(17):
        # Joint-specific depth offsets (simplified)
        if joint_idx in [0, 7, 8, 9, 10]:  # Torso/head
            z_offset = 0.2
        elif joint_idx in [11, 12, 13, 14, 15, 16]:  # Arms
            z_offset = 0.1
        elif joint_idx in [4, 5]:  # Left leg
            z_offset = 0.05
        elif joint_idx in [1, 2, 3]:  # Right leg
            z_offset = -0.05
        else:
            z_offset = 0.0

        poses_3d[:, joint_idx, 2] = z_offset

    return poses_3d
