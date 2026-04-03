"""Алгоритмы трекинга людей для мульти-персональной ассоциации поз.

Предоставляет пос frame-to-frame реидентификацию:
- Sports2D: Венгерский алгоритм по расстояниям ключевых точек (scipy)
- DeepSORT: Appearance-based ReID (требуется deep-sort-realtime)
- SkeletalIdentity: 3D bone length biometric profiles for re-ID
- TrackletMerger: post-hoc tracklet merging for occlusion recovery
"""

from .sports2d import Sports2DTracker
from .deepsort_tracker import DeepSORTTracker
from .skeletal_identity import (
    SkeletalIdentityExtractor,
    compute_bone_lengths_3d,
    compute_identity_profile,
    compute_2d_skeletal_ratios,
    identity_similarity,
)
from .tracklet_merger import (
    Tracklet,
    TrackletMerger,
    build_tracklets,
)

__all__ = [
    "Sports2DTracker",
    "DeepSORTTracker",
    "SkeletalIdentityExtractor",
    "TrackletMerger",
    "Tracklet",
    "build_tracklets",
    "compute_bone_lengths_3d",
    "compute_identity_profile",
    "compute_2d_skeletal_ratios",
    "identity_similarity",
]
