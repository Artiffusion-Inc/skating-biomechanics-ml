"""Multi-person detection and tracking module."""

from .person_detector import BoundingBox, PersonDetector
from .pose_tracker import PoseTracker, Track
from .spatial_reference import CameraPose, SpatialReferenceDetector

__all__ = [
    "BoundingBox",
    "CameraPose",
    "PersonDetector",
    "PoseTracker",
    "SpatialReferenceDetector",
    "Track",
]
