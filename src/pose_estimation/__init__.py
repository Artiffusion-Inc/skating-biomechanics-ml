"""Pose estimation module for figure skating analysis.

This module provides H3.6M 17-keypoint pose extraction as the primary format.
The conversion from BlazePose 33kp to H3.6M 17kp is integrated into the extractor,
so we never store intermediate 33-keypoint poses.

Architecture:
    Video → H36MExtractor (BlazePose backend) → H3.6M 17kp → 3D Lifter → 3D poses

Migration Note:
    This module replaces the old blazepose_to_h36m conversion step.
    The H3.6M format is now the primary format throughout the pipeline.
"""

from src.pose_estimation.h36m_extractor import (
    BKey,
    H36MExtractor,
    H36Key,
    H36M_KEYPOINT_NAMES,
    H36M_SKELETON_EDGES,
    blazepose_to_h36m,
    extract_h36m_poses,
)

__all__ = [
    "BKey",
    "H36MExtractor",
    "H36Key",
    "H36M_KEYPOINT_NAMES",
    "H36M_SKELETON_EDGES",
    "blazepose_to_h36m",
    "extract_h36m_poses",
]
