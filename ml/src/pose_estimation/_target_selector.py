from __future__ import annotations

import numpy as np  # noqa: TC002
from numpy.typing import NDArray  # noqa: TC002


class TargetSelector:
    """Selects target person via click proximity or auto-select by detection frequency."""

    def __init__(
        self,
        click_norm: tuple[float, float] | None = None,
        click_lock_window: int = 6,
    ) -> None:
        self.click_norm = click_norm
        self.click_lock_window = click_lock_window
        self._target_track_id: int | None = None

    @property
    def target_track_id(self) -> int | None:
        return self._target_track_id

    def select_target(
        self,
        h36m_poses: NDArray[np.float32],
        track_ids: list[int],
        frame_idx: int,
    ) -> int | None:
        """Try to select target via click proximity. Returns selected track_id or None."""
        if self._target_track_id is not None:
            return None
        if self.click_norm is None:
            return None
        if frame_idx >= self.click_lock_window:
            return None

        best_dist = float("inf")
        best_tid = None
        cx_click, cy_click = self.click_norm

        for p, tid in enumerate(track_ids):
            mid_hip_x = (h36m_poses[p, 4, 0] + h36m_poses[p, 1, 0]) / 2
            mid_hip_y = (h36m_poses[p, 4, 1] + h36m_poses[p, 1, 1]) / 2
            dist = (mid_hip_x - cx_click) ** 2 + (mid_hip_y - cy_click) ** 2
            if dist < best_dist:
                best_dist = dist
                best_tid = tid

        if best_tid is not None:
            self._target_track_id = best_tid
        return best_tid

    @staticmethod
    def auto_select_by_hits(track_hit_counts: dict[int, int]) -> int | None:
        if not track_hit_counts:
            return None
        return max(track_hit_counts, key=lambda k: track_hit_counts[k])
