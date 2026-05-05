from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

logger = logging.getLogger(__name__)


class TrackState:
    """Manages tracker instances, hit counts, and per-frame track data."""

    def __init__(
        self,
        fps: float,
        tracking_backend: str = "rtmlib",
        tracking_mode: str = "auto",
    ) -> None:
        self.fps = fps
        self.tracking_backend = tracking_backend
        self.tracking_mode = tracking_mode
        self.target_track_id: int | None = None
        self.track_hit_counts: dict[int, int] = {}
        self.frame_track_data: dict[int, dict[int, NDArray[np.float32]]] = {}
        self._next_internal_id = 0

        self.sports2d_tracker = None
        self.deepsort_tracker = None
        self.custom_tracker = None
        self._init_trackers()

    def _init_trackers(self) -> None:
        if self.tracking_backend == "custom":
            from ..tracking.sports2d import Sports2DTracker

            self.custom_tracker = Sports2DTracker(max_disappeared=30, fps=self.fps)
            return

        resolved = self._resolve_tracking_mode()
        if resolved == "sports2d":
            from ..tracking.sports2d import Sports2DTracker

            self.sports2d_tracker = Sports2DTracker(max_disappeared=30, fps=self.fps)
        elif resolved == "deepsort":
            from ..tracking.deepsort_tracker import DeepSORTTracker

            self.deepsort_tracker = DeepSORTTracker(max_age=30, embedder_gpu=True)

    def _resolve_tracking_mode(self) -> str:
        if self.tracking_mode == "auto":
            import importlib.util

            if importlib.util.find_spec("src.tracking.deepsort_tracker") is not None:
                return "deepsort"
            return "sports2d"
        return self.tracking_mode

    @property
    def tracker_instances(self) -> tuple:
        return (self.sports2d_tracker, self.deepsort_tracker, self.custom_tracker)

    def update_tracking(
        self,
        h36m_poses: NDArray[np.float32],
        frame: NDArray[np.uint8] | None = None,
        frame_width: int = 0,
        frame_height: int = 0,
    ) -> list[int]:
        """Run trackers on current frame detections. Returns track IDs."""
        n_persons = h36m_poses.shape[0]
        if n_persons == 0:
            return []

        if self.sports2d_tracker is not None:
            return self.sports2d_tracker.update(h36m_poses[:, :, :2], h36m_poses[:, :, 2])
        if self.deepsort_tracker is not None:
            return self.deepsort_tracker.update(
                h36m_poses[:, :, :2],
                h36m_poses[:, :, 2],
                frame=frame,
                frame_width=frame_width,
                frame_height=frame_height,
            )
        if self.custom_tracker is not None:
            return self.custom_tracker.update(h36m_poses[:, :, :2], h36m_poses[:, :, 2])

        # Fallback: sequential IDs
        track_ids = list(range(self._next_internal_id, self._next_internal_id + n_persons))
        self._next_internal_id += n_persons
        return track_ids

    def record_frame(
        self,
        frame_idx: int,
        h36m_poses: NDArray[np.float32],
        track_ids: list[int],
    ) -> None:
        self.frame_track_data[frame_idx] = {
            tid: h36m_poses[p].copy() for p, tid in enumerate(track_ids)
        }
        for tid in track_ids:
            self.track_hit_counts[tid] = self.track_hit_counts.get(tid, 0) + 1

    def auto_select_target(self) -> int | None:
        if not self.track_hit_counts:
            return None
        return max(self.track_hit_counts, key=lambda k: self.track_hit_counts[k])

    def retroactive_fill(
        self,
        all_poses: NDArray[np.float32],
        target_track_id: int,
    ) -> None:
        for fidx, tmap in self.frame_track_data.items():
            if target_track_id in tmap and np.isnan(all_poses[fidx, 0, 0]):
                all_poses[fidx] = tmap[target_track_id]
