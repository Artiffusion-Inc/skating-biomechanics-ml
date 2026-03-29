"""Процессор рендеринга отдельного кадра.

Single frame renderer with layer composition.
Uses H3.6M 17-keypoint format as primary.
"""

from collections import deque

import numpy as np

from src.pose_estimation import H36M_SKELETON_EDGES, H36Key
from src.ui.core.events import EventBus
from src.ui.types import LayerSettings, ProcessedPoses
from src.visualization import (
    draw_blade_state_3d_hud,
    draw_skeleton,
    draw_skeleton_3d_pip,
    draw_trails,
    draw_velocity_vectors,
)


class FrameRenderer:
    """Рендеринг отдельного кадра с выбранными слоями.

    Render single frame with selected visualization layers.
    Uses H3.6M 17-keypoint format as primary.
    """

    def __init__(self, events: EventBus | None = None) -> None:
        """Инициализация рендерера.

        Args:
            events: Шина событий для оповещений.
        """
        self._events = events
        self._trail_history_left: deque = deque(maxlen=50)
        self._trail_history_right: deque = deque(maxlen=50)

    def process(
        self,
        frame: np.ndarray,
        frame_idx: int,
        poses: ProcessedPoses,
        settings: LayerSettings,
    ) -> np.ndarray:
        """Отрисовать кадр с выбранными слоями.

        Args:
            frame: Исходный кадр (H, W, 3) BGR.
            frame_idx: Индекс кадра в видео.
            poses: Обработанные позы (H3.6M format).
            settings: Настройки слоёв.

        Returns:
            Кадр с визуализацией.
        """
        # Find corresponding pose index
        pose_idx = self._find_pose_index(frame_idx, poses.pose_frame_indices)
        if pose_idx is None:
            return frame

        # Layer 0: Skeleton
        if settings.skeleton:
            if settings.enable_3d and poses.has_3d:
                # 3D skeleton in PIP (use H3.6M 3D poses)
                pose_3d = poses.poses_3d[pose_idx]
                frame = draw_skeleton_3d_pip(
                    frame,
                    pose_3d,
                    H36M_SKELETON_EDGES,
                    poses.height,
                    poses.width,
                    camera_z=settings.d_3d_scale,
                    auto_scale=not settings.no_3d_autoscale,
                )
            else:
                # 2D skeleton (use H3.6M 2D poses)
                pose_h36m = poses.poses_h36m[pose_idx]
                # Convert normalized to pixels for draw_skeleton
                pose_h36m_px = pose_h36m[:, :2] * np.array([poses.width, poses.height])
                frame = draw_skeleton(
                    frame, pose_h36m_px, poses.height, poses.width
                )

        # Layer 1: Kinematics (use H3.6M 17kp normalized)
        if settings.velocity and pose_idx < len(poses.poses_h36m):
            frame = draw_velocity_vectors(
                frame,
                poses.poses_h36m,
                pose_idx,
                poses.fps,
                poses.height,
                poses.width,
            )

        if settings.trails and pose_idx < len(poses.poses_h36m):
            # Update trail history for both feet
            current_pose_h36m = poses.poses_h36m[pose_idx].copy()
            self._trail_history_left.append(current_pose_h36m)
            self._trail_history_right.append(current_pose_h36m)

            # Trim to specified length
            while len(self._trail_history_left) > settings.trail_length:
                self._trail_history_left.popleft()
            while len(self._trail_history_right) > settings.trail_length:
                self._trail_history_right.popleft()

            # Draw trails for both feet
            if len(self._trail_history_left) > 1:
                frame = draw_trails(
                    frame,
                    self._trail_history_left,
                    H36Key.LFOOT,
                    poses.height,
                    poses.width,
                )
            if len(self._trail_history_right) > 1:
                frame = draw_trails(
                    frame,
                    self._trail_history_right,
                    H36Key.RFOOT,
                    poses.height,
                    poses.width,
                )

        # Layer 2: Technical
        if settings.edge_indicators and poses.has_blade_states:
            if poses.blade_states_left and pose_idx < len(poses.blade_states_left):
                state_left = poses.blade_states_left[pose_idx]
                state_right = (
                    poses.blade_states_right[pose_idx]
                    if poses.blade_states_right
                    else None
                )
                frame = draw_blade_state_3d_hud(
                    frame,
                    state_left,
                    state_right,
                    poses.height,
                    poses.width,
                )

        # Layer 3: Subtitles (would need VTT file)
        # TODO: Add subtitle support if VTT file provided

        # CoM trajectory (requires PhysicsEngine for CoM calculation)
        # TODO: Implement CoM trajectory visualization
        # if settings.com_trajectory and poses.has_3d and pose_idx > 0:
        #     from src.analysis.physics_engine import PhysicsEngine
        #     engine = PhysicsEngine()
        #     com_points = []
        #     for i in range(pose_idx + 1):
        #         com = engine.calculate_center_of_mass(poses.poses_3d[i:i+1])[0]
        #         com_points.append(com)
        #     com_trajectory = np.array(com_points)
        #     frame = draw_3d_trajectory(
        #         frame, com_trajectory, poses.height, poses.width
        #     )

        return frame

    def _find_pose_index(
        self,
        frame_idx: int,
        pose_frame_indices: np.ndarray | None,
    ) -> int | None:
        """Найти индекс позы для кадра.

        Args:
            frame_idx: Индекс кадра.
            pose_frame_indices: Массив индексов кадров поз.

        Returns:
            Индекс позы или None.
        """
        if pose_frame_indices is None:
            # Assume sequential mapping
            return frame_idx

        # Find closest pose frame index
        for i, pose_frame_idx in enumerate(pose_frame_indices):
            if pose_frame_idx == frame_idx:
                return i
            if pose_frame_idx > frame_idx:
                return i - 1 if i > 0 else 0

        return len(pose_frame_indices) - 1 if len(pose_frame_indices) > 0 else None

    def clear_trails(self) -> None:
        """Очистить историю траекторий."""
        self._trail_history_left.clear()
        self._trail_history_right.clear()
