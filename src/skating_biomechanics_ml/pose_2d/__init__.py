"""2D pose estimation module using MediaPipe BlazePose."""

from skating_biomechanics_ml.pose_2d.blazepose_extractor import BlazePoseExtractor
from skating_biomechanics_ml.pose_2d.normalizer import PoseNormalizer

# Export BlazePoseExtractor as PoseExtractor for backwards compatibility
PoseExtractor = BlazePoseExtractor

__all__ = ["PoseExtractor", "PoseNormalizer", "BlazePoseExtractor"]
