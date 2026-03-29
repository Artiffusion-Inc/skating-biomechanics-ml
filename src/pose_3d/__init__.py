"""3D pose estimation using AthletePose3D models."""

# Re-export H3.6M types from pose_estimation module (primary source)
from src.pose_estimation import H36M_KEYPOINT_NAMES, H36M_SKELETON_EDGES, H36Key

from .athletepose_extractor import AthletePose3DExtractor, extract_3d_poses
from .normalizer_3d import (
    Pose3DNormalizer,
    calculate_body_heights,
    get_head_center_3d,
    get_hip_center_3d,
)

__all__ = [
    "H36M_KEYPOINT_NAMES",
    "H36M_SKELETON_EDGES",
    "AthletePose3DExtractor",
    "H36Key",
    "Pose3DNormalizer",
    "calculate_body_heights",
    "extract_3d_poses",
    "get_head_center_3d",
    "get_hip_center_3d",
]
