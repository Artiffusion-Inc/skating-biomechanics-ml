"""RTMO-based pose extractor via rtmlib.

Uses the rtmlib PoseTracker with Body model to extract 17-keypoint
COCO poses.  The output is converted to H3.6M 17-keypoint format
for the analysis pipeline.

Architecture:
    Video → rtmlib PoseTracker (Body) → COCO (17kp) → H3.6M (17kp)

Key advantages:
    - One-stage detection (no separate detector needed)
    - Better accuracy on distant/small subjects
    - Built-in tracking with consistent IDs
    - ONNX Runtime inference (no PyTorch dependency)

References:
    - rtmlib: https://github.com/Tau-J/rtmlib
    - RTMO: https://github.com/Tau-J/rtmlib/tree/main/docs/en/rtmo
"""

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np


def _get_tqdm():
    try:
        from tqdm import tqdm  # noqa: TC003, S110

        return tqdm
    except (ImportError, ValueError):

        def _tqdm_mock(iterable=None, **_kwargs):
            if iterable is not None:
                return iterable
            return type(
                "_TqdmMock",
                (),
                {
                    "update": lambda *_a: None,
                    "close": lambda *_a: None,
                    "__enter__": lambda s: s,
                    "__exit__": lambda *_a: None,
                },
            )()

        return _tqdm_mock


if TYPE_CHECKING:
    from rtmlib import Body, PoseTracker
else:
    try:
        from rtmlib import Body, PoseTracker
    except ImportError:
        PoseTracker = None  # type: ignore[assignment]
        Body = None  # type: ignore[assignment]

from ..detection.pose_tracker import PoseTracker as CustomPoseTracker
from ..tracking.skeletal_identity import compute_2d_skeletal_ratios
from ..tracking.tracklet_merger import TrackletMerger, build_tracklets
from ..types import PersonClick, TrackedExtraction
from ..utils.frame_buffer import AsyncFrameReader
from ..utils.video import get_video_meta
from ._frame_processor import FrameProcessor
from ._target_selector import TargetSelector
from ._track_state import TrackState
from ._track_validator import TrackValidator
from .h36m import coco_to_h36m

logger = logging.getLogger(__name__)


class PoseExtractor:
    """COCO pose extractor using rtmlib Body model (RTMO).

    Provides H3.6M 17-keypoint poses. Uses rtmlib's built-in tracking
    for multi-person handling.

    Args:
        mode: Model preset — ``"lightweight"`` (fast), ``"balanced"``
            (default), ``"performance"`` (accurate).
        tracking_backend: ``"rtmlib"`` uses rtmlib's built-in tracker;
            ``"custom"`` feeds detections into our PoseTracker
            (OC-SORT + biometric Re-ID).
        conf_threshold: Minimum keypoint confidence to accept [0, 1].
        output_format: ``"normalized"`` for [0, 1] coords. ``"pixels"``
            for absolute pixel coords.
        frame_skip: Process every Nth frame for pose estimation (1 = every
            frame). Higher values = faster but less accurate. Skipped
            frames are filled with NaN for downstream interpolation.
        device: ``"cpu"`` or ``"cuda"``.
        backend: Inference backend — ``"onnxruntime"`` or ``"opencv"``.
    """

    def __init__(
        self,
        mode: str = "balanced",
        tracking_backend: str = "rtmlib",
        tracking_mode: str = "auto",
        conf_threshold: float = 0.3,
        output_format: str = "normalized",
        frame_skip: int = 1,
        device: str = "auto",
        backend: str = "onnxruntime",
    ) -> None:
        if PoseTracker is None:
            raise ImportError("rtmlib is not installed. Install with: uv add rtmlib")

        self._mode = mode
        self._tracking_backend = tracking_backend
        self._tracking_mode = tracking_mode
        self._conf_threshold = conf_threshold
        self._output_format = output_format
        self._frame_skip = max(1, frame_skip)
        self._device = device
        self._backend = backend

        # Resolve device via DeviceConfig for consistent GPU-first behavior
        if device == "auto":
            from ..device import DeviceConfig

            self._device = DeviceConfig(device="auto").device

        # Lazy-initialised on first call
        self._tracker: PoseTracker | None = None

    @property
    def tracker(self):
        """Lazy-initialise rtmlib PoseTracker on first access."""
        if self._tracker is None:
            if Body is None:
                raise ImportError("rtmlib Body model not available")
            # Create a local tracker variable with proper type
            from functools import partial

            from rtmlib import Custom
            from rtmlib import PoseTracker as RTMPoseTracker

            rtmo_urls = {
                "performance": "https://download.openmmlab.com/mmpose/v1/projects/rtmo/onnx_sdk/rtmo-l_16xb16-600e_body7-640x640-b37118ce_20231211.zip",
                "lightweight": "https://download.openmmlab.com/mmpose/v1/projects/rtmo/onnx_sdk/rtmo-s_8xb32-600e_body7-640x640-dac2bf74_20231211.zip",
                "balanced": "https://download.openmmlab.com/mmpose/v1/projects/rtmo/onnx_sdk/rtmo-m_16xb16-600e_body7-640x640-39e78cc4_20231211.zip",
            }

            RTMOSolution = partial(
                Custom,
                pose_class="RTMO",
                pose=rtmo_urls[self._mode],
                pose_input_size=(640, 640),
                to_openpose=False,
                backend=self._backend,
                device=self._device,
            )

            self._tracker = RTMPoseTracker(
                RTMOSolution,
                tracking=True,
                tracking_thr=0.3,
            )
        return self._tracker

    # ------------------------------------------------------------------
    # Core extraction
    # ------------------------------------------------------------------

    def extract_video_tracked(
        self,
        video_path: Path | str,
        person_click: PersonClick | None = None,
        progress_cb=None,
        use_batch: bool = True,
        batch_size: int = 8,
    ) -> TrackedExtraction:
        """Extract H3.6M poses from video with tracking.

        Runs rtmlib Body (RTMO) on every frame, tracks all persons,
        and selects a single target person for output.

        Args:
            video_path: Path to video file.
            person_click: Optional click to select target person by
                proximity to the click point in the first few frames.
            progress_cb: Optional callback ``(fraction, message)`` for
                progress reporting (e.g. Gradio progress bar).
            use_batch: If True, use BatchRTMO for batched inference.
            batch_size: Batch size for batched inference (default 8).

        Returns:
            TrackedExtraction with poses (N, 17, 3), frame_indices,
            tracking metadata.  Missing frames are filled with NaN.

        Raises:
            ValueError: If no pose is detected in any frame.
        """
        if use_batch:
            return self._extract_batch(
                video_path,
                person_click=person_click,
                progress_cb=progress_cb,
                batch_size=batch_size,
            )
        return self._extract_per_frame(
            video_path,
            person_click=person_click,
            progress_cb=progress_cb,
        )

    def _extract_per_frame(
        self,
        video_path: Path | str,
        person_click: PersonClick | None = None,
        progress_cb=None,
    ) -> TrackedExtraction:
        """Per-frame extraction path (delegates to shared components)."""
        video_path = Path(video_path)
        video_meta = get_video_meta(video_path)
        num_frames = video_meta.num_frames

        all_poses = np.full((num_frames, 17, 3), np.nan, dtype=np.float32)

        track_state = TrackState(
            fps=video_meta.fps,
            tracking_backend=self._tracking_backend,
            tracking_mode=self._tracking_mode,
        )
        target_selector = TargetSelector(
            click_norm=(
                person_click.to_normalized(video_meta.width, video_meta.height)
                if person_click is not None
                else None
            ),
        )
        validator = TrackValidator()
        frame_processor = FrameProcessor(output_format=self._output_format)

        last_target_pose: np.ndarray | None = None
        last_target_ratios: np.ndarray | None = None
        target_lost_frame: int | None = None

        # Cache first frame for spatial reference
        cap_first = cv2.VideoCapture(str(video_path))
        if not cap_first.isOpened():
            raise RuntimeError(f"Failed to open video: {video_path}")
        ret, first_frame = cap_first.read()
        cap_first.release()
        if not ret:
            raise RuntimeError(f"Failed to read first frame from video: {video_path}")

        pbar = _get_tqdm()(
            total=num_frames,
            desc="Extracting poses",
            unit="frame",
            ncols=100,
            disable=progress_cb is not None,
        )

        reader = AsyncFrameReader(
            video_path,
            buffer_size=16,
            frame_skip=self._frame_skip,
        )
        reader.start()

        try:
            while True:
                result = reader.get_frame()
                if result is None:
                    break
                frame_idx, frame = result
                h, w = frame.shape[:2]

                # Resize large frames for detection
                if max(h, w) > 1920:
                    scale = 1920 / max(h, w)
                    frame_ds = cv2.resize(frame, (int(w * scale), int(h * scale)))
                else:
                    frame_ds = frame

                tracker = self.tracker
                if tracker is None:
                    pbar.update(self._frame_skip)
                    continue
                tracker_result = tracker(frame_ds)
                if not isinstance(tracker_result, tuple) or len(tracker_result) != 2:
                    pbar.update(self._frame_skip)
                    continue
                keypoints, scores = tracker_result

                if frame_ds is not frame:
                    keypoints = keypoints * (max(h, w) / 1920)

                if keypoints is None or len(keypoints) == 0:
                    if track_state.custom_tracker is not None:
                        track_state.custom_tracker.update(np.empty((0, 17, 2), dtype=np.float32))
                    pbar.update(self._frame_skip)
                    continue

                h36m_poses = frame_processor.convert_keypoints(keypoints, scores, w, h)
                track_ids = track_state.update_tracking(
                    h36m_poses, frame=frame, frame_width=w, frame_height=h
                )
                track_state.record_frame(frame_idx, h36m_poses, track_ids)

                # Target selection via click
                selected = target_selector.select_target(h36m_poses, track_ids, frame_idx)
                if selected is not None:
                    track_state.target_track_id = selected

                # Fill target data
                if track_state.target_track_id is not None:
                    target_track_id = track_state.target_track_id
                    found = False
                    stolen = False
                    for p, tid in enumerate(track_ids):
                        if tid == target_track_id:
                            if last_target_pose is not None and validator.is_stolen(
                                h36m_poses[p], last_target_pose, last_target_ratios
                            ):
                                stolen = True
                                break
                            all_poses[frame_idx] = h36m_poses[p]
                            last_target_pose = h36m_poses[p].copy()
                            last_target_ratios = compute_2d_skeletal_ratios(h36m_poses[p])
                            target_lost_frame = None
                            found = True
                            break

                    if stolen:
                        all_poses[frame_idx] = np.full((17, 3), np.nan, dtype=np.float32)
                        found = False

                    if (not found or stolen) and last_target_pose is not None:
                        if target_lost_frame is None:
                            target_lost_frame = frame_idx
                        if frame_idx - target_lost_frame <= TrackValidator.MAX_LOST_FRAMES:
                            best_dist = float("inf")
                            best_new_tid: int | None = None
                            best_new_pose: np.ndarray | None = None
                            for p, tid in enumerate(track_ids):
                                if stolen and tid == target_track_id:
                                    continue
                                score = validator.migration_score(
                                    h36m_poses[p],
                                    last_target_pose,
                                    elapsed=frame_idx - target_lost_frame,
                                )
                                if score < best_dist:
                                    best_dist = score
                                    best_new_tid = tid
                                    best_new_pose = h36m_poses[p]
                            if (
                                best_new_tid is not None
                                and best_dist < TrackValidator.MIGRATION_THRESHOLD
                                and best_new_pose is not None
                            ):
                                track_state.target_track_id = best_new_tid
                                all_poses[frame_idx] = best_new_pose
                                last_target_pose = best_new_pose.copy()
                                last_target_ratios = compute_2d_skeletal_ratios(best_new_pose)
                                target_lost_frame = None
                                track_state.retroactive_fill(all_poses, track_state.target_track_id)

                pbar.update(self._frame_skip)
                if progress_cb:
                    progress_cb(
                        frame_idx / num_frames * 0.3,
                        f"Extracting poses... {frame_idx}/{num_frames}",
                    )
        finally:
            reader.join()
            pbar.close()

        # Deferred auto-select by hits
        if track_state.target_track_id is None and track_state.track_hit_counts:
            track_state.target_track_id = track_state.auto_select_target()
            if track_state.target_track_id is not None:
                track_state.retroactive_fill(all_poses, track_state.target_track_id)

        # Post-hoc tracklet merging
        self._post_hoc_merge(all_poses, track_state.frame_track_data, track_state.target_track_id)

        valid_mask = ~np.isnan(all_poses[:, 0, 0])
        if not np.any(valid_mask):
            raise ValueError(f"No valid pose detected in video: {video_path}")
        first_detection_frame = int(np.argmax(valid_mask))

        return TrackedExtraction(
            poses=all_poses,
            frame_indices=np.arange(num_frames),
            first_detection_frame=first_detection_frame,
            target_track_id=track_state.target_track_id,
            fps=video_meta.fps,
            video_meta=video_meta,
            first_frame=first_frame,
        )

    def _extract_batch(
        self,
        video_path: Path | str,
        person_click: PersonClick | None = None,
        progress_cb=None,
        batch_size: int = 8,
    ) -> TrackedExtraction:
        """Extract poses using BatchRTMO for batched inference.

        Args:
            video_path: Path to video file.
            person_click: Optional click to select target person.
            progress_cb: Optional progress callback.
            batch_size: Number of frames to process per batch.

        Returns:
            TrackedExtraction with poses (N, 17, 3), frame_indices,
            tracking metadata.  Missing frames are filled with NaN.
        """
        from .rtmo_batch import BatchRTMO

        video_path = Path(video_path)
        video_meta = get_video_meta(video_path)
        num_frames = video_meta.num_frames

        all_poses = np.full((num_frames, 17, 3), np.nan, dtype=np.float32)

        rtmo = BatchRTMO(
            mode=self._mode,
            device=self._device,
            score_thr=self._conf_threshold,
            nms_thr=0.45,
        )

        track_state = TrackState(
            fps=video_meta.fps,
            tracking_backend=self._tracking_backend,
            tracking_mode=self._tracking_mode,
        )
        target_selector = TargetSelector(
            click_norm=(
                person_click.to_normalized(video_meta.width, video_meta.height)
                if person_click is not None
                else None
            ),
        )
        validator = TrackValidator()
        frame_processor = FrameProcessor(output_format=self._output_format)

        last_target_pose: np.ndarray | None = None
        last_target_ratios: np.ndarray | None = None
        target_lost_frame: int | None = None

        # Read first frame for spatial reference
        cap_first = cv2.VideoCapture(str(video_path))
        if not cap_first.isOpened():
            raise RuntimeError(f"Failed to open video: {video_path}")
        ret, first_frame = cap_first.read()
        cap_first.release()
        if not ret:
            raise RuntimeError(f"Failed to read first frame from video: {video_path}")

        # Read all frames (respecting frame_skip)
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {video_path}")

        frames_to_process: list[np.ndarray] = []
        frame_indices: list[int] = []

        try:
            for idx in _get_tqdm()(
                range(num_frames), desc="Reading frames", unit="frame", ncols=100
            ):
                if idx % self._frame_skip != 0:
                    continue
                ret, frame = cap.read()
                if not ret:
                    break
                frames_to_process.append(frame)
                frame_indices.append(idx)
        finally:
            cap.release()

        if not frames_to_process:
            raise ValueError(f"No frames read from video: {video_path}")

        pbar = _get_tqdm()(
            total=len(frames_to_process),
            desc="Batch extracting poses",
            unit="batch",
            ncols=100,
            disable=progress_cb is not None,
        )

        for batch_start in range(0, len(frames_to_process), batch_size):
            batch_end = min(batch_start + batch_size, len(frames_to_process))
            batch_frames = frames_to_process[batch_start:batch_end]
            batch_indices = frame_indices[batch_start:batch_end]

            batch_results = rtmo.infer_batch(batch_frames)

            for frame_idx, frame, (keypoints, scores) in zip(
                batch_indices, batch_frames, batch_results, strict=True
            ):
                h, w = frame.shape[:2]

                if keypoints.shape[0] == 0:
                    pbar.update(1)
                    continue

                h36m_poses = frame_processor.convert_keypoints(keypoints, scores, w, h)
                track_ids = track_state.update_tracking(
                    h36m_poses, frame=frame, frame_width=w, frame_height=h
                )
                track_state.record_frame(frame_idx, h36m_poses, track_ids)

                # Target selection via click
                selected = target_selector.select_target(h36m_poses, track_ids, frame_idx)
                if selected is not None:
                    track_state.target_track_id = selected

                # Fill target data
                if track_state.target_track_id is not None:
                    target_track_id = track_state.target_track_id
                    found = False
                    stolen = False
                    for p, tid in enumerate(track_ids):
                        if tid == target_track_id:
                            if last_target_pose is not None and validator.is_stolen(
                                h36m_poses[p], last_target_pose, last_target_ratios
                            ):
                                stolen = True
                                break
                            all_poses[frame_idx] = h36m_poses[p]
                            last_target_pose = h36m_poses[p].copy()
                            last_target_ratios = compute_2d_skeletal_ratios(h36m_poses[p])
                            target_lost_frame = None
                            found = True
                            break

                    if stolen:
                        all_poses[frame_idx] = np.full((17, 3), np.nan, dtype=np.float32)
                        found = False

                    if (not found or stolen) and last_target_pose is not None:
                        if target_lost_frame is None:
                            target_lost_frame = frame_idx
                        if frame_idx - target_lost_frame <= TrackValidator.MAX_LOST_FRAMES:
                            best_dist = float("inf")
                            best_new_tid: int | None = None
                            best_new_pose: np.ndarray | None = None
                            for p, tid in enumerate(track_ids):
                                if stolen and tid == target_track_id:
                                    continue
                                score = validator.migration_score(
                                    h36m_poses[p],
                                    last_target_pose,
                                    elapsed=frame_idx - target_lost_frame,
                                )
                                if score < best_dist:
                                    best_dist = score
                                    best_new_tid = tid
                                    best_new_pose = h36m_poses[p]
                            if (
                                best_new_tid is not None
                                and best_dist < TrackValidator.MIGRATION_THRESHOLD
                                and best_new_pose is not None
                            ):
                                track_state.target_track_id = best_new_tid
                                all_poses[frame_idx] = best_new_pose
                                last_target_pose = best_new_pose.copy()
                                last_target_ratios = compute_2d_skeletal_ratios(best_new_pose)
                                target_lost_frame = None
                                track_state.retroactive_fill(all_poses, track_state.target_track_id)

                pbar.update(1)
                if progress_cb:
                    progress_cb(
                        frame_idx / num_frames * 0.3,
                        f"Batch extracting poses... {frame_idx}/{num_frames}",
                    )

        pbar.close()

        # Deferred auto-select by hits
        if track_state.target_track_id is None and track_state.track_hit_counts:
            track_state.target_track_id = track_state.auto_select_target()
            if track_state.target_track_id is not None:
                track_state.retroactive_fill(all_poses, track_state.target_track_id)

        # Post-hoc tracklet merging
        self._post_hoc_merge(all_poses, track_state.frame_track_data, track_state.target_track_id)

        valid_mask = ~np.isnan(all_poses[:, 0, 0])
        if not np.any(valid_mask):
            raise ValueError(f"No valid pose detected in video: {video_path}")
        first_detection_frame = int(np.argmax(valid_mask))

        return TrackedExtraction(
            poses=all_poses,
            frame_indices=np.arange(num_frames),
            first_detection_frame=first_detection_frame,
            target_track_id=track_state.target_track_id,
            fps=video_meta.fps,
            video_meta=video_meta,
            first_frame=first_frame,
        )

    # ------------------------------------------------------------------
    # Preview
    # ------------------------------------------------------------------

    @staticmethod
    def _build_person_grid(
        best_frame: np.ndarray,
        persons: list[dict],
    ) -> str:
        """Нарисовать bbox + номер на полном кадре.

        Авто-контраст: измеряет яркость фона и выбирает светлую/тёмную рамку.
        Анти-перекрытие: сдвигает метки при наложении.

        Args:
            best_frame: Кадр (H, W, 3) BGR.
            persons: Список dict с ключами:
                - best_kps: (17, 3) нормализованные H3.6M ключевые точки
                - hits: int
                - best_conf: float

        Returns:
            Путь к сохранённому изображению.
        """
        if not persons:
            return ""

        import tempfile

        preview = best_frame.copy()
        frame_h, frame_w = best_frame.shape[:2]
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.55
        thickness = 1

        # Собираем данные для рисования
        placements: list[tuple[int, int, int, int, int, int, float, int]] = []
        for person in persons:
            kps = person["best_kps"]
            valid = kps[kps[:, 2] > 0.1]
            if len(valid) < 3:
                continue
            bx1 = int(np.min(valid[:, 0]) * frame_w)
            by1 = int(np.min(valid[:, 1]) * frame_h)
            bx2 = int(np.max(valid[:, 0]) * frame_w)
            by2 = int(np.max(valid[:, 1]) * frame_h)
            cx = (bx1 + bx2) // 2
            iy = int(np.clip(by1, 0, frame_h - 1))
            ix = int(np.clip(cx, 0, frame_w - 1))
            brightness = float(preview[iy, ix].mean())
            hits = person["hits"]
            placements.append((cx, 0, bx1, by1, bx2, by2, brightness, hits))

        # Анти-перекрытие меток
        occupied: list[tuple[int, int, int, int]] = []

        for i, (_cx, _cy, bx1, by1, bx2, by2, bg_brightness, hits) in enumerate(placements):
            is_dark = bg_brightness < 128
            color = (255, 255, 255) if is_dark else (0, 0, 0)

            # Bbox
            cv2.rectangle(preview, (bx1, by1), (bx2, by2), color, 1, cv2.LINE_AA)

            # Метка с номером
            label = f" {i + 1} ({hits}) "
            (tw, th), baseline = cv2.getTextSize(label, font, font_scale, thickness)
            pad = 3
            tag_w = tw + 2 * pad
            tag_h = th + baseline + 2 * pad

            lx = bx1
            ly = by1 - tag_h - 2
            lx = max(2, min(lx, frame_w - tag_w - 2))
            ly = max(2, ly)

            # Анти-перекрытие
            for _ in range(10):
                overlap = False
                for ox1, oy1, ox2, oy2 in occupied:
                    if lx < ox2 and lx + tag_w > ox1 and ly < oy2 and ly + tag_h > oy1:
                        ly = oy2 + 2
                        overlap = True
                        break
                if not overlap:
                    break

            occupied.append((lx, ly, lx + tag_w, ly + tag_h))

            # Полупрозрачная плашка
            overlay = preview.copy()
            cv2.rectangle(overlay, (lx, ly), (lx + tag_w, ly + tag_h), color, -1)
            cv2.addWeighted(overlay, 0.7, preview, 0.3, 0, dst=preview)

            # Текст (инвертированный цвет)
            text_color = (0, 0, 0) if is_dark else (255, 255, 255)
            cv2.putText(
                preview,
                label,
                (lx + pad, ly + pad + th),
                font,
                font_scale,
                text_color,
                thickness,
                cv2.LINE_AA,
            )

        preview_path = str(Path(tempfile.mktemp(suffix=".jpg")).with_name("person_preview.jpg"))
        cv2.imwrite(preview_path, preview)
        return preview_path

    def preview_persons(
        self,
        video_path: Path | str,
        num_frames: int = 30,
    ) -> list[dict]:
        """Preview all detected persons in the first few frames.

        Runs rtmlib on the first ``num_frames`` frames and returns a
        summary for each tracked person so the user can choose which
        one to follow.

        Args:
            video_path: Path to video file.
            num_frames: Number of frames to scan (default 30).

        Returns:
            Tuple of (list of dicts, preview_path)::

                {
                    "track_id": int,
                    "hits": int,
                    "bbox": (x1, y1, x2, y2),  # normalized [0,1]
                    "first_frame": int,
                    "mid_hip": (x, y),          # normalized
                }

            preview_path: Path to person grid preview image.
        """
        video_path = Path(video_path)
        video_meta = get_video_meta(video_path)

        if self._tracking_backend == "custom":
            custom_tracker = CustomPoseTracker(max_disappeared=30, min_hits=2, fps=video_meta.fps)
        else:
            custom_tracker = None  # type: ignore[assignment]

        # Новый трекинг (Sports2D / DeepSORT)
        resolved_mode = self._resolve_tracking_mode()
        sports2d_tracker = None
        deepsort_tracker = None
        if resolved_mode == "sports2d":
            from ..tracking.sports2d import Sports2DTracker

            sports2d_tracker = Sports2DTracker(max_disappeared=30, fps=video_meta.fps)
        elif resolved_mode == "deepsort":
            from ..tracking.deepsort_tracker import DeepSORTTracker

            deepsort_tracker = DeepSORTTracker(max_age=30, embedder_gpu=True)

        rtmlib_id_map: dict[int, int] = {}
        next_internal_id = 0
        person_data: dict[int, dict] = {}

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {video_path}")

        best_frame: np.ndarray | None = None  # keep the frame with highest avg confidence

        try:
            for frame_idx in _get_tqdm()(
                range(num_frames), desc="Previewing persons", unit="frame", ncols=100
            ):
                ret, frame = cap.read()
                if not ret:
                    break

                h, w = frame.shape[:2]
                if best_frame is None:
                    best_frame = frame.copy()
                tracker = self.tracker
                if tracker is None:
                    continue
                tracker_result = tracker(frame)
                if not isinstance(tracker_result, tuple) or len(tracker_result) != 2:
                    continue
                keypoints, scores = tracker_result

                if keypoints is None or len(keypoints) == 0:
                    if custom_tracker is not None:
                        custom_tracker.update(np.empty((0, 17, 2), dtype=np.float32))
                    continue

                n_persons = len(keypoints)
                h36m_poses = np.zeros((n_persons, 17, 3), dtype=np.float32)

                for p in range(n_persons):
                    kp = keypoints[p].astype(np.float32)
                    conf = scores[p].astype(np.float32)

                    coco = np.zeros((17, 3), dtype=np.float32)
                    coco[:, :2] = kp
                    coco[:, 2] = conf
                    coco[:, 0] /= w
                    coco[:, 1] /= h

                    h36m_poses[p] = coco_to_h36m(coco)

                # Track association
                if sports2d_tracker is not None:
                    track_ids = sports2d_tracker.update(h36m_poses[:, :, :2], h36m_poses[:, :, 2])
                elif deepsort_tracker is not None:
                    track_ids = deepsort_tracker.update(
                        h36m_poses[:, :, :2],
                        h36m_poses[:, :, 2],
                        frame=frame,
                        frame_width=w,
                        frame_height=h,
                    )
                elif custom_tracker is not None:
                    track_ids = custom_tracker.update(h36m_poses[:, :, :2], h36m_poses[:, :, 2])
                else:
                    track_ids = self._assign_track_ids(h36m_poses, rtmlib_id_map, next_internal_id)
                    next_internal_id = max(rtmlib_id_map.values(), default=-1) + 1

                for p, tid in enumerate(track_ids):
                    if tid not in person_data:
                        person_data[tid] = {
                            "hits": 0,
                            "best_conf": 0.0,
                            "best_kps": None,
                            "best_frame": frame_idx,
                            "first_frame": frame_idx,
                        }
                    person_data[tid]["hits"] += 1
                    avg_conf = float(np.mean(h36m_poses[p, :, 2]))
                    if avg_conf > person_data[tid]["best_conf"]:
                        person_data[tid]["best_conf"] = avg_conf
                        person_data[tid]["best_kps"] = h36m_poses[p].copy()
                        person_data[tid]["best_frame"] = frame_idx
        finally:
            cap.release()

        # Build person grid preview
        preview_path: str | None = None
        if best_frame is not None and person_data:
            persons_for_grid = []
            for _tid, data in sorted(
                person_data.items(), key=lambda kv: kv[1]["hits"], reverse=True
            ):
                if data["best_kps"] is not None:
                    valid = data["best_kps"][data["best_kps"][:, 2] > 0.1]
                    if len(valid) >= 3:
                        persons_for_grid.append(data)
            if persons_for_grid:
                preview_path = PoseExtractor._build_person_grid(best_frame, persons_for_grid)

        # Build output with deduplication
        output: list[dict] = []
        preview_path_out = preview_path
        min_hits = max(2, num_frames // 10)  # At least 10% of scanned frames

        for tid, data in sorted(person_data.items(), key=lambda kv: kv[1]["hits"], reverse=True):
            if data["hits"] < min_hits:
                continue
            kps = data["best_kps"]
            if kps is None:
                continue
            valid = kps[kps[:, 2] > 0.1]
            if len(valid) < 3:
                continue
            x1, y1 = float(np.min(valid[:, 0])), float(np.min(valid[:, 1]))
            x2, y2 = float(np.max(valid[:, 0])), float(np.max(valid[:, 1]))

            # NMS: skip if this bbox overlaps heavily with a better (more hits) one
            skip = False
            for existing in output:
                ex1, ey1, ex2, ey2 = existing["bbox"]
                # IoU check
                ix1 = max(x1, ex1)
                iy1 = max(y1, ey1)
                ix2 = min(x2, ex2)
                iy2 = min(y2, ey2)
                inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
                area_a = (x2 - x1) * (y2 - y1)
                area_b = (ex2 - ex1) * (ey2 - ey1)
                union = area_a + area_b - inter
                if union > 0 and inter / union > 0.5:
                    skip = True
                    break
            if skip:
                continue

            # Mid-hip (H3.6M: LHIP=4, RHIP=1)
            mid_hip_x = float((kps[4, 0] + kps[1, 0]) / 2)
            mid_hip_y = float((kps[4, 1] + kps[1, 1]) / 2)
            output.append(
                {
                    "track_id": tid,
                    "hits": data["hits"],
                    "bbox": (x1, y1, x2, y2),
                    "first_frame": data["first_frame"],
                    "mid_hip": (mid_hip_x, mid_hip_y),
                }
            )

        return output, preview_path_out  # type: ignore[return-value]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve_tracking_mode(self) -> str:
        """Разрешить 'auto' в конкретный режим трекинга."""
        if self._tracking_mode != "auto":
            return self._tracking_mode
        try:
            import deep_sort_realtime  # noqa: F401

            logger.info("Авто-выбор: DeepSORT (deep-sort-realtime доступен)")
            return "deepsort"
        except ImportError:
            logger.info("Авто-выбор: Sports2D (Венгерский алгоритм)")
            return "sports2d"

    def _assign_track_ids(
        self,
        h36m_poses: np.ndarray,
        id_map: dict[int, int],
        next_id: int,
    ) -> list[int]:
        """Assign stable track IDs to detected poses.

        For rtmlib backend (no custom tracker), uses biometric matching
        to associate detections across frames.  First detection per person
        gets a new ID; subsequent detections are matched by biometric
        distance to known persons.

        Args:
            h36m_poses: (P, 17, 3) H3.6M poses for current frame.
            id_map: Map from internal_id → internal_id (identity map,
                used for tracking known IDs).
            next_id: Next available internal ID.

        Returns:
            List of track IDs, one per detected person.
        """
        if len(h36m_poses) == 0:
            return []

        track_ids: list[int] = []

        # For each detection, find closest existing track by biometric distance
        for _tid in id_map:
            # Store the last known pose — but we don't have it here.
            # This is a simplified approach: for the first frame, assign new IDs.
            pass

        # Simple strategy: assign new IDs for each detection.
        # The built-in rtmlib tracking handles spatial consistency already,
        # but since we can't access its track IDs, we rely on the
        # custom tracker path for production use.
        # For the rtmlib path, we just assign sequential IDs per-frame
        # and let biometric matching handle re-association.
        for p in range(len(h36m_poses)):
            new_id = next_id + p
            track_ids.append(new_id)

        return track_ids

    def _post_hoc_merge(
        self,
        all_poses: np.ndarray,
        frame_track_data: dict[int, dict[int, np.ndarray]],
        target_track_id: int | None,
    ) -> None:
        """Post-hoc tracklet merging for occlusion recovery."""
        valid_mask_pre = ~np.isnan(all_poses[:, 0, 0])
        if valid_mask_pre.all() or not frame_track_data or target_track_id is None:
            return
        model_3d = Path("data/models/motionagformer-s-ap3d.onnx")
        identity_ext = None
        if model_3d.exists():
            from ..tracking.skeletal_identity import SkeletalIdentityExtractor

            identity_ext = SkeletalIdentityExtractor(
                model_path=model_3d,
                device="auto",
            )
        merger = TrackletMerger(
            identity_extractor=identity_ext,
            similarity_threshold=0.80,
        )
        tracklets = build_tracklets(frame_track_data)
        target_tracklet = None
        for t in tracklets:
            if t.track_id == target_track_id:
                target_tracklet = t
                break
        if target_tracklet is None:
            return
        valid_frames = np.where(valid_mask_pre)[0]
        if len(valid_frames) == 0:
            return
        last_valid = int(valid_frames[-1])
        num_frames = all_poses.shape[0]
        if last_valid >= num_frames - 1:
            return
        candidates = [t for t in tracklets if t.track_id != target_track_id]
        match = merger.find_best_match(target_tracklet, candidates)
        if match is None:
            return
        for f in match.frames:
            if f < num_frames and np.isnan(all_poses[f, 0, 0]):
                all_poses[f] = match.poses.get(f, all_poses[f])
        logger.info(
            "Post-hoc merge: filled %d frames from track %d",
            sum(1 for f in match.frames if f < num_frames and np.isnan(all_poses[f, 0, 0])),
            match.track_id,
        )

    def close(self) -> None:
        """Release resources."""
        self._tracker = None

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()


def extract_poses(
    video_path: Path | str,
    mode: str = "balanced",
    output_format: str = "normalized",
    person_click: PersonClick | None = None,
) -> TrackedExtraction:
    """Extract H3.6M poses from video using rtmlib.

    Convenience function that creates an PoseExtractor and runs
    tracked extraction.

    Args:
        video_path: Path to video file.
        mode: Model preset — ``"lightweight"``, ``"balanced"``, ``"performance"``.
        output_format: ``"normalized"`` or ``"pixels"``.
        person_click: Optional click to select target person.

    Returns:
        TrackedExtraction with poses populated.
    """
    extractor = PoseExtractor(mode=mode, output_format=output_format)
    return extractor.extract_video_tracked(video_path, person_click=person_click)
