"""Pose estimation module for figure skating analysis.

This module provides H3.6M 17-keypoint pose extraction as the primary format.
Supports multiple 2D pose estimators: H36MExtractor (BlazePose backend), YOLOPoseExtractor (YOLOv11).

Architecture:
    Video → H36MExtractor/YOLOPoseExtractor → H3.6M 17kp → 3D Lifter → 3D poses

Extractors:
    - H36MExtractor: BlazePose MediaPipe backend (33kp → 17kp integrated)
    - YOLOPoseExtractor: YOLOv11-Pose (17kp COCO, faster, no left/right confusion)
"""

from src.pose_estimation.h36m_extractor import (
    H36MExtractor,
    H36Key,
    H36M_KEYPOINT_NAMES,
    H36M_SKELETON_EDGES,
    extract_h36m_poses,
)
from src.pose_estimation.yolo_extractor import YOLOPoseExtractor

__all__ = [
    "H36MExtractor",
    "H36Key",
    "H36M_KEYPOINT_NAMES",
    "H36M_SKELETON_EDGES",
    "YOLOPoseExtractor",
    "extract_h36m_poses",
]
