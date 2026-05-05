"""3D pose estimation."""

# Re-export H3.6M types from pose_estimation module (primary source)
from src.pose_estimation import H36M_KEYPOINT_NAMES, H36M_SKELETON_EDGES, H36Key

from .normalizer_3d import (
    Pose3DNormalizer,
    calculate_body_heights,
    get_head_center_3d,
    get_hip_center_3d,
)

__all__ = [
    "H36M_KEYPOINT_NAMES",
    "H36M_SKELETON_EDGES",
    "H36Key",
    "Pose3DNormalizer",
    "calculate_body_heights",
    "get_head_center_3d",
    "get_hip_center_3d",
]
