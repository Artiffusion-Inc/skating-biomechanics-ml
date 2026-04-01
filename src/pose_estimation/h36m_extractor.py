"""H3.6M 17-keypoint pose extractor.

Direct H3.6M format extraction using YOLO26-Pose backend with integrated conversion.
This is the primary 2D pose extractor for the skating analysis pipeline.

Architecture:
    YOLO26-Pose (17kp COCO) → geometric conversion → H3.6M (17kp) output

The conversion is geometric (not learned) and happens on-the-fly during extraction.
"""

from pathlib import Path

import numpy as np

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None  # type: ignore[assignment]

from ..detection.pose_tracker import PoseTracker
from ..types import PersonClick, TrackedExtraction, VideoMeta
from ..utils.video import get_video_meta


# H3.6M keypoint indices
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


# YOLO26-Pose COCO keypoint indices (for internal mapping)
class _COCOKey:
    """YOLO26-Pose COCO keypoint indices (17 total) - internal use only."""

    NOSE = 0
    LEFT_EYE = 1
    RIGHT_EYE = 2
    LEFT_EAR = 3
    RIGHT_EAR = 4
    LEFT_SHOULDER = 5
    RIGHT_SHOULDER = 6
    LEFT_ELBOW = 7
    RIGHT_ELBOW = 8
    LEFT_WRIST = 9
    RIGHT_WRIST = 10
    LEFT_HIP = 11
    RIGHT_HIP = 12
    LEFT_KNEE = 13
    RIGHT_KNEE = 14
    LEFT_ANKLE = 15
    RIGHT_ANKLE = 16


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


def _coco_to_h36m_single(coco_pose: np.ndarray) -> np.ndarray:
    """Convert YOLO26-Pose COCO 17 keypoints to H3.6M 17 keypoints (single frame).

    Args:
        coco_pose: (17, 2) or (17, 3) array in COCO format

    Returns:
        h36m_pose: (17, 2) or (17, 3) array in H3.6M format
    """
    has_confidence = coco_pose.shape[1] == 3
    n_channels = 3 if has_confidence else 2

    h36m_pose = np.zeros((17, n_channels), dtype=coco_pose.dtype)

    # Midpoints
    mid_hip = (coco_pose[_COCOKey.LEFT_HIP] + coco_pose[_COCOKey.RIGHT_HIP]) / 2
    mid_shoulder = (coco_pose[_COCOKey.LEFT_SHOULDER] + coco_pose[_COCOKey.RIGHT_SHOULDER]) / 2

    # Direct mapping from COCO to H3.6M
    h36m_pose[H36Key.HIP_CENTER] = mid_hip
    h36m_pose[H36Key.RHIP] = coco_pose[_COCOKey.RIGHT_HIP]
    h36m_pose[H36Key.RKNEE] = coco_pose[_COCOKey.RIGHT_KNEE]
    h36m_pose[H36Key.RFOOT] = coco_pose[_COCOKey.RIGHT_ANKLE]
    h36m_pose[H36Key.LHIP] = coco_pose[_COCOKey.LEFT_HIP]
    h36m_pose[H36Key.LKNEE] = coco_pose[_COCOKey.LEFT_KNEE]
    h36m_pose[H36Key.LFOOT] = coco_pose[_COCOKey.LEFT_ANKLE]
    h36m_pose[H36Key.SPINE] = mid_shoulder * 0.5 + mid_hip * 0.5
    h36m_pose[H36Key.THORAX] = mid_shoulder
    h36m_pose[H36Key.NECK] = coco_pose[_COCOKey.NOSE]
    h36m_pose[H36Key.HEAD] = coco_pose[_COCOKey.NOSE]
    h36m_pose[H36Key.LSHOULDER] = coco_pose[_COCOKey.LEFT_SHOULDER]
    h36m_pose[H36Key.LELBOW] = coco_pose[_COCOKey.LEFT_ELBOW]
    h36m_pose[H36Key.LWRIST] = coco_pose[_COCOKey.LEFT_WRIST]
    h36m_pose[H36Key.RSHOULDER] = coco_pose[_COCOKey.RIGHT_SHOULDER]
    h36m_pose[H36Key.RELBOW] = coco_pose[_COCOKey.RIGHT_ELBOW]
    h36m_pose[H36Key.RWRIST] = coco_pose[_COCOKey.RIGHT_WRIST]

    return h36m_pose


class H36MExtractor:
    """H3.6M 17-keypoint pose extractor.

    Uses YOLO26-Pose backend with integrated H3.6M conversion.
    Outputs H3.6M format directly (17 keypoints) - no intermediate COCO storage.

    This is the primary 2D pose extractor for the skating analysis pipeline.

    Advantages over BlazePose:
    - Single-stage detection + pose (faster)
    - No left/right confusion (better tracking)
    - Easy API with ultralytics
    """

    def __init__(
        self,
        model_size: str = "n",
        model_path: Path | str | None = None,
        device: str = "0",
        conf_threshold: float = 0.5,
        output_format: str = "normalized",  # "normalized" or "pixels"
        skip_model_check: bool = False,
    ):
        """Initialize H3.6M extractor with YOLO26-Pose backend.

        Args:
            model_size: Model size - 'n' (nano), 's' (small), 'm' (medium)
            model_path: Path to custom model weights, or None for default
            conf_threshold: Minimum confidence for pose detection [0, 1]
            output_format: "normalized" for [0,1] coords, "pixels" for absolute pixel coords
            skip_model_check: If True, don't validate model exists (for testing)
        """
        if YOLO is None:
            raise ImportError("Ultralytics not installed. Install with: uv add ultralytics")

        self.model_size = model_size
        self._model_path = Path(model_path) if model_path else None
        self._conf_threshold = conf_threshold
        self._output_format = output_format
        self._skip_model_check = skip_model_check

        self._device = device

        # Lazy-load model on first access
        self._model: YOLO | None = None

    @property
    def model(self) -> "YOLO":
        """Lazy-load YOLO model on first access."""
        if self._model is None:
            if self._model_path is not None:
                self._model = YOLO(str(self._model_path))
            else:
                # YOLO26-Pose (NMS-free, better occlusion handling)
                model_name = f"yolo26{self.model_size}-pose.pt"
                self._model = YOLO(model_name)
        return self._model

    def extract_video_tracked(
        self,
        video_path: Path | str,
        person_click: PersonClick | None = None,
    ) -> TrackedExtraction:
        """Extract H3.6M poses from video with multi-person tracking.

        Runs YOLO26-Pose on every frame, tracks all detected persons via
        PoseTracker (OC-SORT + biometric Re-ID), and selects a single
        target person for output.

        Args:
            video_path: Path to video file.
            person_click: Optional user click to select target person.
                If provided, the nearest detection's mid-hip to the click
                point (in the first min_hits*2 frames) is locked as target.

        Returns:
            TrackedExtraction with poses (N, 17, 3), frame indices,
            first_detection_frame, target_track_id, fps, and video_meta.
            Missing frames are filled with NaN.

        Raises:
            ValueError: If no pose is detected in any frame.
        """
        video_path = Path(video_path)
        video_meta = get_video_meta(video_path)
        num_frames = video_meta.num_frames

        # Pre-allocate with NaN
        all_poses = np.full(
            (num_frames, 17, 3), np.nan, dtype=np.float32
        )

        tracker = PoseTracker(
            max_disappeared=30,
            min_hits=3,
            fps=video_meta.fps,
        )

        # Per-frame mapping: track_id → pose for current frame
        target_track_id: int | None = None
        click_lock_window = tracker.min_hits * 2
        click_norm: tuple[float, float] | None = None
        if person_click is not None:
            click_norm = person_click.to_normalized(
                video_meta.width, video_meta.height
            )

        # Track hit counts for auto-select (track_id → hits)
        track_hit_counts: dict[int, int] = {}

        # Per-frame track_id→pose mapping for retroactive fill
        frame_track_poses: dict[int, dict[int, np.ndarray]] = {}

        # Run YOLO stream
        results = self.model(
            str(video_path), verbose=False, conf=self._conf_threshold, stream=True
        )

        for frame_idx, result in enumerate(results):
            if frame_idx >= num_frames:
                break

            if result.keypoints is None or len(result.keypoints.xy) == 0:
                # No detections — still update tracker so it ages tracks
                tracker.update(np.empty((0, 17, 2), dtype=np.float32))
                continue

            h, w = result.orig_shape
            kps_all = result.keypoints.xy.cpu().numpy()   # (P, 17, 2)
            confs_all = result.keypoints.conf.cpu().numpy()   # (P, 17)

            n_persons = kps_all.shape[0]
            h36m_poses = np.zeros((n_persons, 17, 3), dtype=np.float32)

            for p in range(n_persons):
                kp = kps_all[p]        # (17, 2) pixel
                conf = confs_all[p]    # (17,)

                # Normalize to [0, 1]
                kp_norm = kp.copy()
                kp_norm[:, 0] /= w
                kp_norm[:, 1] /= h

                # Combine x, y, confidence
                coco_kp = np.zeros((17, 3), dtype=np.float32)
                coco_kp[:, :2] = kp_norm
                coco_kp[:, 2] = conf

                # Convert to H3.6M 17kp
                h36m_kp = _coco_to_h36m_single(coco_kp)

                # Convert to pixels if requested
                if self._output_format == "pixels":
                    h36m_kp[:, 0] *= w
                    h36m_kp[:, 1] *= h

                h36m_poses[p] = h36m_kp

            # Feed to tracker (xy coords only for tracking)
            track_ids = tracker.update(h36m_poses[:, :, :2], h36m_poses[:, :, 2])

            # Store per-track poses for retroactive fill
            frame_track_poses[frame_idx] = {
                tid: h36m_poses[p].copy()
                for p, tid in enumerate(track_ids)
            }

            # Update hit counts
            for tid in track_ids:
                track_hit_counts[tid] = track_hit_counts.get(tid, 0) + 1

            # --- Target selection ---
            # Phase 1: Click-based selection (within lock window)
            if (
                target_track_id is None
                and click_norm is not None
                and frame_idx < click_lock_window
            ):
                best_dist = float("inf")
                best_tid: int | None = None
                for p, tid in enumerate(track_ids):
                    # Mid-hip in normalized coords
                    mid_hip_x = (h36m_poses[p, H36Key.LHIP, 0] + h36m_poses[p, H36Key.RHIP, 0]) / 2
                    mid_hip_y = (h36m_poses[p, H36Key.LHIP, 1] + h36m_poses[p, H36Key.RHIP, 1]) / 2
                    dist = (
                        (mid_hip_x - click_norm[0]) ** 2
                        + (mid_hip_y - click_norm[1]) ** 2
                    )
                    if dist < best_dist:
                        best_dist = dist
                        best_tid = tid
                if best_tid is not None:
                    target_track_id = best_tid

            # Fill target pose for current frame (click path only — auto deferred)
            if target_track_id is not None:
                for p, tid in enumerate(track_ids):
                    if tid == target_track_id:
                        all_poses[frame_idx] = h36m_poses[p]
                        break

        # Phase 2 (deferred): Auto-select by most hits — after full loop
        # This ensures we pick the track with the most total detections across
        # the entire video, not just an early snapshot.
        if target_track_id is None and track_hit_counts:
            target_track_id = max(
                track_hit_counts, key=lambda k: track_hit_counts[k]  # type: ignore[arg-type]
            )
            # Retroactive fill: all frames where this track appeared
            for fidx, tmap in frame_track_poses.items():
                if target_track_id in tmap:
                    all_poses[fidx] = tmap[target_track_id]

        # Determine first_detection_frame
        valid_mask = ~np.isnan(all_poses[:, 0, 0])
        if not np.any(valid_mask):
            raise ValueError(f"No valid pose detected in video: {video_path}")
        first_detection_frame = int(np.argmax(valid_mask))

        return TrackedExtraction(
            poses=all_poses,
            frame_indices=np.arange(num_frames),
            first_detection_frame=first_detection_frame,
            target_track_id=target_track_id,
            fps=video_meta.fps,
            video_meta=video_meta,
        )

    def close(self) -> None:
        """Close the extractor and release resources.

        Note: YOLO model doesn't require explicit cleanup, but method kept for API compatibility.
        """
        pass

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()


# Convenience function
def extract_h36m_poses(
    video_path: Path | str,
    model_size: str = "n",
    model_path: Path | str | None = None,
    output_format: str = "normalized",
    person_click: PersonClick | None = None,
) -> TrackedExtraction:
    """Extract H3.6M poses from video with multi-person tracking.

    Convenience function that creates extractor and runs tracked extraction.

    Args:
        video_path: Path to video file.
        model_size: Model size - 'n' (nano), 's' (small), 'm' (medium)
        model_path: Path to custom model weights (deprecated, use model_size)
        output_format: "normalized" or "pixels"
        person_click: Optional click to select target person.

    Returns:
        TrackedExtraction with poses and tracking metadata.
    """
    extractor = H36MExtractor(
        model_size=model_size, model_path=model_path, output_format=output_format
    )
    return extractor.extract_video_tracked(video_path, person_click=person_click)


def blazepose_to_h36m(_blazepose_pose: np.ndarray) -> np.ndarray:
    """Convert BlazePose 33 keypoints to H3.6M 17 keypoints.

    .. deprecated::
        BlazePose is no longer supported. Use YOLO26-Pose via H36MExtractor instead.
        This function provides YOLO-based conversion for backward compatibility.

    Args:
        blazepose_pose: (33, 2/3) array for single frame, or (N, 33, 2/3) for sequence

    Returns:
        h36m_pose: (17, 2/3) array for single frame, or (N, 17, 2/3) for sequence

    Raises:
        ValueError: If input shape is invalid
    """
    import warnings  # noqa: PLC0415

    warnings.warn(
        "blazepose_to_h36m is deprecated and will be removed in a future version. "
        "Use H36MExtractor with YOLO26-Pose backend instead.",
        DeprecationWarning,
        stacklevel=2,
    )

    # For backward compatibility: process via YOLO pose estimation
    # This is NOT a direct conversion - it's a YOLO-based fallback
    # The result will be YOLO H3.6M keypoints, not converted BlazePose
    raise NotImplementedError(
        "Direct BlazePose to H3.6M conversion is no longer supported. "
        "Use H36MExtractor with YOLO26-Pose backend for new pose extraction. "
        "For existing BlazePose data, you must re-extract using H36MExtractor."
    )


# Public alias for backward compatibility (now maps to COCO)
BKey = _COCOKey
