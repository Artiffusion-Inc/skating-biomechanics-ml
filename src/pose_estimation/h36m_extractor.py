"""H3.6M 17-keypoint pose extractor.

Direct H3.6M format extraction using BlazePose backend with integrated conversion.
This is the primary 2D pose extractor for the skating analysis pipeline.

Architecture:
    BlazePose (33kp) → integrated conversion → H3.6M (17kp) output

The conversion is geometric (not learned) and happens on-the-fly during extraction,
so we never store intermediate 33-keypoint poses.
"""

from collections import deque
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np

if TYPE_CHECKING:
    pass

try:
    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision
except ImportError:
    mp = None  # type: ignore[assignment]
    python = None  # type: ignore[assignment]
    vision = None  # type: ignore[assignment]


# H3.6M keypoint indices (matching blazepose_to_h36m.py)
class H36Key:
    """H3.6M keypoint indices (17 total)."""

    HIP_CENTER = 0
    RHIP = 1
    RKNEE = 2
    RFOOT = 3
    LHIP = 4
    LKNEE = 5
    LFOOT = 6
    SPINE = 7
    THORAX = 8
    NECK = 9
    HEAD = 10
    LSHOULDER = 11
    LELBOW = 12
    LWRIST = 13
    RSHOULDER = 14
    RELBOW = 15
    RWRIST = 16


# BlazePose keypoint indices (for internal mapping)
class _BKey:
    """BlazePose keypoint indices (33 total) - internal use only."""

    NOSE = 0
    LEFT_EYE_INNER = 1
    LEFT_EYE = 2
    LEFT_EYE_OUTER = 3
    RIGHT_EYE_INNER = 4
    RIGHT_EYE = 5
    RIGHT_EYE_OUTER = 6
    LEFT_EAR = 7
    RIGHT_EAR = 8
    MOUTH_LEFT = 9
    MOUTH_RIGHT = 10
    LEFT_SHOULDER = 11
    RIGHT_SHOULDER = 12
    LEFT_ELBOW = 13
    RIGHT_ELBOW = 14
    LEFT_WRIST = 15
    RIGHT_WRIST = 16
    LEFT_PINKY = 17
    RIGHT_PINKY = 18
    LEFT_INDEX = 19
    RIGHT_INDEX = 20
    LEFT_THUMB = 21
    RIGHT_THUMB = 22
    LEFT_HIP = 23
    RIGHT_HIP = 24
    LEFT_KNEE = 25
    RIGHT_KNEE = 26
    LEFT_ANKLE = 27
    RIGHT_ANKLE = 28
    LEFT_HEEL = 29
    RIGHT_HEEL = 30
    LEFT_FOOT_INDEX = 31
    RIGHT_FOOT_INDEX = 32


# H3.6M skeleton connections for visualization
H36M_SKELETON_EDGES = [
    # Torso
    (H36Key.HIP_CENTER, H36Key.SPINE),
    (H36Key.SPINE, H36Key.THORAX),
    (H36Key.THORAX, H36Key.NECK),
    (H36Key.NECK, H36Key.HEAD),
    # Right arm
    (H36Key.THORAX, H36Key.RSHOULDER),
    (H36Key.RSHOULDER, H36Key.RELBOW),
    (H36Key.RELBOW, H36Key.RWRIST),
    # Left arm
    (H36Key.THORAX, H36Key.LSHOULDER),
    (H36Key.LSHOULDER, H36Key.LELBOW),
    (H36Key.LELBOW, H36Key.LWRIST),
    # Right leg
    (H36Key.HIP_CENTER, H36Key.RHIP),
    (H36Key.RHIP, H36Key.RKNEE),
    (H36Key.RKNEE, H36Key.RFOOT),
    # Left leg
    (H36Key.HIP_CENTER, H36Key.LHIP),
    (H36Key.LHIP, H36Key.LKNEE),
    (H36Key.LKNEE, H36Key.LFOOT),
]


# H3.6M keypoint names
H36M_KEYPOINT_NAMES = [
    "hip_center",
    "rhip",
    "rknee",
    "rfoot",
    "lhip",
    "lknee",
    "lfoot",
    "spine",
    "thorax",
    "neck",
    "head",
    "lshoulder",
    "lelbow",
    "lwrist",
    "rshoulder",
    "relbow",
    "rwrist",
]


def _blazepose_to_h36m_single(blazepose_pose: np.ndarray) -> np.ndarray:
    """Convert BlazePose 33 keypoints to H3.6M 17 keypoints (single frame).

    Args:
        blazepose_pose: (33, 2) or (33, 3) array

    Returns:
        h36m_pose: (17, 2) or (17, 3) array
    """
    has_confidence = blazepose_pose.shape[1] == 3
    n_channels = 3 if has_confidence else 2

    h36m_pose = np.zeros((17, n_channels), dtype=blazepose_pose.dtype)

    # Midpoints
    mid_hip = (blazepose_pose[_BKey.LEFT_HIP] + blazepose_pose[_BKey.RIGHT_HIP]) / 2
    mid_shoulder = (
        blazepose_pose[_BKey.LEFT_SHOULDER] + blazepose_pose[_BKey.RIGHT_SHOULDER]
    ) / 2

    # Direct mapping from BlazePose to H3.6M
    h36m_pose[H36Key.HIP_CENTER] = mid_hip
    h36m_pose[H36Key.RHIP] = blazepose_pose[_BKey.RIGHT_HIP]
    h36m_pose[H36Key.RKNEE] = blazepose_pose[_BKey.RIGHT_KNEE]
    h36m_pose[H36Key.RFOOT] = blazepose_pose[_BKey.RIGHT_ANKLE]
    h36m_pose[H36Key.LHIP] = blazepose_pose[_BKey.LEFT_HIP]
    h36m_pose[H36Key.LKNEE] = blazepose_pose[_BKey.LEFT_KNEE]
    h36m_pose[H36Key.LFOOT] = blazepose_pose[_BKey.LEFT_ANKLE]
    h36m_pose[H36Key.SPINE] = mid_shoulder * 0.5 + mid_hip * 0.5
    h36m_pose[H36Key.THORAX] = mid_shoulder
    h36m_pose[H36Key.NECK] = blazepose_pose[_BKey.NOSE]
    h36m_pose[H36Key.HEAD] = blazepose_pose[_BKey.NOSE]
    h36m_pose[H36Key.LSHOULDER] = blazepose_pose[_BKey.LEFT_SHOULDER]
    h36m_pose[H36Key.LELBOW] = blazepose_pose[_BKey.LEFT_ELBOW]
    h36m_pose[H36Key.LWRIST] = blazepose_pose[_BKey.LEFT_WRIST]
    h36m_pose[H36Key.RSHOULDER] = blazepose_pose[_BKey.RIGHT_SHOULDER]
    h36m_pose[H36Key.RELBOW] = blazepose_pose[_BKey.RIGHT_ELBOW]
    h36m_pose[H36Key.RWRIST] = blazepose_pose[_BKey.RIGHT_WRIST]

    return h36m_pose


class H36MExtractor:
    """H3.6M 17-keypoint pose extractor.

    Uses BlazePose backend with integrated H3.6M conversion.
    Outputs H3.6M format directly (17 keypoints) - no intermediate 33kp storage.

    This is the primary 2D pose extractor for the skating analysis pipeline.
    """

    # Default model path
    DEFAULT_MODEL_PATH = Path("data/models/pose_landmarker_heavy.task")
    MODEL_URL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task"

    def __init__(
        self,
        model_path: Path | str | None = None,
        min_detection_confidence: float = 0.5,
        min_presence_confidence: float = 0.5,
        num_poses: int = 1,
        output_format: str = "normalized",  # "normalized" or "pixels"
    ):
        """Initialize H3.6M extractor.

        Args:
            model_path: Path to .task model file. Defaults to data/models/pose_landmarker_heavy.task.
            min_detection_confidence: Minimum confidence for person detection [0, 1].
            min_presence_confidence: Minimum confidence for pose presence [0, 1].
            num_poses: Maximum number of poses to detect.
            output_format: "normalized" for [0,1] coords, "pixels" for absolute pixel coords.
        """
        if mp is None or vision is None:
            raise ImportError(
                "MediaPipe is not installed. Install with: uv add mediapipe"
            )

        # Determine model path
        if model_path is None:
            model_path = self.DEFAULT_MODEL_PATH
        self.model_path = Path(model_path)

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model file not found: {self.model_path}\n"
                f"Download with:\n"
                f"  mkdir -p data/models\n"
                f"  wget {self.MODEL_URL} -O {self.model_path}"
            )

        self._min_detection_confidence = min_detection_confidence
        self._min_presence_confidence = min_presence_confidence
        self._num_poses = num_poses
        self._output_format = output_format
        self._landmarker: "vision.PoseLandmarker" | None = None

    @property
    def landmarker(self) -> "vision.PoseLandmarker":
        """Lazy-load PoseLandmarker on first access."""
        if self._landmarker is None:
            base_options = python.BaseOptions(model_asset_path=str(self.model_path))

            options = vision.PoseLandmarkerOptions(
                base_options=base_options,
                running_mode=vision.RunningMode.VIDEO,
                min_pose_detection_confidence=self._min_detection_confidence,
                min_pose_presence_confidence=self._min_presence_confidence,
                num_poses=self._num_poses,
                output_segmentation_masks=False,
            )

            self._landmarker = vision.PoseLandmarker.create_from_options(options)

        return self._landmarker

    def extract_frame(
        self, frame: np.ndarray, timestamp_ms: int = 0
    ) -> np.ndarray | None:
        """Extract H3.6M pose from single frame.

        Args:
            frame: Input frame (height, width, 3) as BGR image.
            timestamp_ms: Timestamp in milliseconds for video mode.

        Returns:
            pose: (17, 3) array with x, y, confidence in H3.6M format.
                  Returns None if no person detected.
                  Coordinates are normalized [0,1] if output_format="normalized",
                  or in pixels if output_format="pixels".
        """
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Create MediaPipe Image
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

        # Detect pose
        result = self.landmarker.detect_for_video(mp_image, timestamp_ms)

        # Check if any pose detected
        if not result.pose_landmarks:
            return None

        # Get first person's landmarks
        landmarks = result.pose_landmarks[0]
        h, w = frame.shape[:2]

        # Convert BlazePose 33kp to numpy
        blazepose_kp = np.zeros((33, 3), dtype=np.float32)
        for i, landmark in enumerate(landmarks):
            blazepose_kp[i, 0] = landmark.x
            blazepose_kp[i, 1] = landmark.y
            blazepose_kp[i, 2] = landmark.presence if landmark.presence > 0 else 0.0

        # Convert to H3.6M 17kp (integrated conversion)
        h36m_kp = _blazepose_to_h36m_single(blazepose_kp)

        # Convert to pixels if requested
        if self._output_format == "pixels":
            h36m_kp[:, 0] *= w
            h36m_kp[:, 1] *= h

        return h36m_kp

    def extract_video(
        self,
        video_path: Path | str,
        fps: float | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Extract H3.6M poses from all frames of a video.

        Args:
            video_path: Path to video file.
            fps: Video FPS for timestamp calculation. If None, will be detected.

        Returns:
            poses: (N, 17, 3) array with x, y, confidence in H3.6M format.
            frame_indices: (N,) array of frame indices where poses were detected.
        """
        from src.video import extract_frames, get_video_meta

        # Get video metadata
        meta = get_video_meta(video_path)
        if fps is None:
            fps = meta.fps

        poses_list = []
        frame_indices = []

        for frame_idx, frame in enumerate(extract_frames(video_path)):
            timestamp_ms = int(frame_idx * 1000 / fps)
            pose = self.extract_frame(frame, timestamp_ms)
            if pose is not None:
                poses_list.append(pose)
                frame_indices.append(frame_idx)

        if not poses_list:
            raise ValueError(f"No valid pose detected in video: {video_path}")

        poses = np.stack(poses_list)
        frame_indices = np.array(frame_indices)

        return poses, frame_indices

    def close(self) -> None:
        """Close the landmarker and release resources."""
        if self._landmarker is not None:
            self._landmarker.close()
            self._landmarker = None

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()


# Convenience function
def extract_h36m_poses(
    video_path: Path | str,
    model_path: Path | str | None = None,
    output_format: str = "normalized",
) -> tuple[np.ndarray, np.ndarray]:
    """Extract H3.6M poses from video.

    Convenience function that creates extractor and runs extraction.

    Args:
        video_path: Path to video file.
        model_path: Path to .task model file.
        output_format: "normalized" or "pixels"

    Returns:
        poses: (N, 17, 3) array with x, y, confidence
        frame_indices: (N,) array of frame indices
    """
    extractor = H36MExtractor(model_path=model_path, output_format=output_format)
    return extractor.extract_video(video_path)


def blazepose_to_h36m(blazepose_pose: np.ndarray) -> np.ndarray:
    """Convert BlazePose 33 keypoints to H3.6M 17 keypoints.

    Public conversion function for backward compatibility.
    Handles both single frames and sequences.

    Args:
        blazepose_pose: (33, 2/3) array for single frame, or (N, 33, 2/3) for sequence

    Returns:
        h36m_pose: (17, 2/3) array for single frame, or (N, 17, 2/3) for sequence

    Raises:
        ValueError: If input shape is invalid
    """
    # Handle single frame
    if blazepose_pose.ndim == 2:
        if blazepose_pose.shape[0] != 33:
            raise ValueError(
                f"Expected 33 keypoints for BlazePose, got {blazepose_pose.shape[0]}"
            )
        return _blazepose_to_h36m_single(blazepose_pose)

    # Handle sequence
    if blazepose_pose.ndim == 3:
        if blazepose_pose.shape[1] != 33:
            raise ValueError(
                f"Expected 33 keypoints for BlazePose, got {blazepose_pose.shape[1]}"
            )
        n_frames = blazepose_pose.shape[0]
        n_channels = blazepose_pose.shape[2]
        h36m_poses = np.zeros((n_frames, 17, n_channels), dtype=blazepose_pose.dtype)
        for i in range(n_frames):
            h36m_poses[i] = _blazepose_to_h36m_single(blazepose_pose[i])
        return h36m_poses

    raise ValueError(
        f"Invalid input shape {blazepose_pose.shape}. "
        "Expected (33, 2/3) or (N, 33, 2/3)"
    )


# Public alias for backward compatibility
BKey = _BKey
