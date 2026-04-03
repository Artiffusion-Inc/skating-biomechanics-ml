"""Трекинг людей Sports2D через Венгерский алгоритм.

Адаптация Pose2Sim/common.py sort_people_sports2d().
Попарные расстояния ключевых точек + scipy.optimize.linear_sum_assignment
для оптимального однозначного сопоставления между кадрами.

Reference:
    - Pose2Sim: https://github.com/Pose2Sim/Pose2Sim
    - scipy.optimize.linear_sum_assignment (Венгерский алгоритм)
"""

import logging

import numpy as np
from scipy.optimize import linear_sum_assignment

logger = logging.getLogger(__name__)


class Sports2DTracker:
    """Попарный трекер на Венгерском алгоритме.

    Хранит ключевые точки предыдущего кадра и назначает стабильные ID,
    находя оптимальное однозначное сопоставление, минимизирующее суммарное
    попарное расстояние ключевых точек.

    Args:
        max_dist: Максимальное допустимое расстояние для ассоциации
            (нормализованные координаты). Если None, автоматически вычисляется
            как 1.5 * средняя диагональ bbox.
        max_disappeared: Кадров без детекции перед удалением трека.
    """

    def __init__(
        self,
        max_dist: float | None = None,
        max_disappeared: int = 30,
    ) -> None:
        self._max_dist = max_dist
        self._max_disappeared = max_disappeared

        # Состояние
        self._prev_keypoints: np.ndarray | None = None  # (P_prev, 17, 2)
        self._prev_scores: np.ndarray | None = None      # (P_prev, 17)
        self._prev_track_ids: list[int] = []
        self._track_last_seen: dict[int, int] = {}
        self._track_keypoints: dict[int, np.ndarray] = {}  # Исторические keypoints для ReID
        self._frame_count: int = 0
        self._next_id: int = 0

    def update(
        self,
        keypoints: np.ndarray,
        scores: np.ndarray,
    ) -> list[int]:
        """Обновить трекер детекциями текущего кадра.

        Args:
            keypoints: (P, 17, 2) ключевые точки H3.6M (только xy).
            scores: (P, 17) confidence для каждого ключа.

        Returns:
            Список track ID, по одному на каждого обнаруженного человека.
        """
        n_curr = len(keypoints)

        # Пустой кадр
        if n_curr == 0:
            self._prev_keypoints = None
            self._prev_scores = None
            self._prev_track_ids = []
            self._frame_count += 1
            # Очистить старые треки
            valid_tracks = set()
            for tid, last in self._track_last_seen.items():
                if self._frame_count - last <= self._max_disappeared:
                    valid_tracks.add(tid)
            self._track_last_seen = {
                tid: last for tid, last in self._track_last_seen.items()
                if tid in valid_tracks
            }
            self._track_keypoints = {
                tid: kps for tid, kps in self._track_keypoints.items()
                if tid in valid_tracks
            }
            return []

        # Первый кадр — новые ID
        if self._prev_keypoints is None or len(self._prev_keypoints) == 0:
            track_ids = list(range(self._next_id, self._next_id + n_curr))
            self._next_id += n_curr
            self._prev_keypoints = keypoints.copy()
            self._prev_scores = scores.copy()
            self._prev_track_ids = track_ids.copy()
            for i, tid in enumerate(track_ids):
                self._track_last_seen[tid] = self._frame_count
                self._track_keypoints[tid] = keypoints[i].copy()
            self._frame_count += 1
            return track_ids

        # Матрица расстояний: (n_prev, n_curr)
        prev_expanded = self._prev_keypoints[:, np.newaxis, :, :]  # (n_prev, 1, 17, 2)
        curr_expanded = keypoints[np.newaxis, :, :, :]             # (1, n_curr, 17, 2)
        diff = curr_expanded - prev_expanded
        distances_per_kp = np.sqrt(np.nansum(diff ** 2, axis=3))   # (n_prev, n_curr, 17)
        dist_matrix = np.nanmean(distances_per_kp, axis=2)         # (n_prev, n_curr)
        dist_matrix = np.nan_to_num(dist_matrix, nan=1e10, posinf=1e10)

        # Авто max_dist из bbox
        max_dist = self._max_dist
        if max_dist is None:
            all_kps = np.concatenate([self._prev_keypoints, keypoints], axis=0)
            x_min = np.nanmin(all_kps[:, :, 0], axis=1)
            x_max = np.nanmax(all_kps[:, :, 0], axis=1)
            y_min = np.nanmin(all_kps[:, :, 1], axis=1)
            y_max = np.nanmax(all_kps[:, :, 1], axis=1)
            widths = x_max - x_min
            heights = y_max - y_min
            diagonals = np.sqrt(widths ** 2 + heights ** 2)
            valid_diags = diagonals[diagonals > 0]
            if len(valid_diags) > 0:
                max_dist = float(1.5 * np.mean(valid_diags))
            else:
                max_dist = 1.0

        # Венгерский алгоритм
        pre_ids, curr_ids = linear_sum_assignment(dist_matrix)

        # Фильтр по порогу
        valid_associations: list[tuple[int, int]] = []
        for pre_id, curr_id in zip(pre_ids, curr_ids):
            if dist_matrix[pre_id, curr_id] <= max_dist:
                valid_associations.append((pre_id, curr_id))

        # Построить результат
        associated_curr = {curr_id for _, curr_id in valid_associations}
        unassociated_curr = [i for i in range(n_curr) if i not in associated_curr]

        track_ids: list[int] = [0] * n_curr
        for prev_idx, curr_idx in valid_associations:
            track_ids[curr_idx] = self._prev_track_ids[prev_idx]

        # Попытка сопоставить неподошедшие детекции с исчезнувшими треками
        disappeared_tracks = [
            tid for tid, last_seen in self._track_last_seen.items()
            if tid not in self._prev_track_ids and
            self._frame_count - last_seen <= self._max_disappeared
        ]

        if disappeared_tracks and unassociated_curr:
            # Матрица расстояний к исчезнувшим трекам
            disappeared_kps = np.array([
                self._track_keypoints[tid] for tid in disappeared_tracks
            ])  # (n_disappeared, 17, 2)

            unassociated_kps = keypoints[unassociated_curr]  # (n_unassoc, 17, 2)

            # Расчет расстояний
            disp_expanded = disappeared_kps[:, np.newaxis, :, :]  # (n_disp, 1, 17, 2)
            unassoc_expanded = unassociated_kps[np.newaxis, :, :, :]  # (1, n_unassoc, 17, 2)
            diff = unassoc_expanded - disp_expanded
            dists_per_kp = np.sqrt(np.nansum(diff ** 2, axis=3))
            dist_matrix_disp = np.nanmean(dists_per_kp, axis=2)
            dist_matrix_disp = np.nan_to_num(dist_matrix_disp, nan=1e10, posinf=1e10)

            # Венгерский алгоритм для исчезнувших
            disp_ids, unassoc_ids = linear_sum_assignment(dist_matrix_disp)

            # Сопоставить валидные пары
            for disp_idx, unassoc_idx in zip(disp_ids, unassoc_ids):
                if dist_matrix_disp[disp_idx, unassoc_idx] <= max_dist:
                    # Нашли сопоставление с исчезнувшим треком
                    curr_idx = unassociated_curr[unassoc_idx]
                    track_id = disappeared_tracks[disp_idx]
                    track_ids[curr_idx] = track_id
                    # Убрать из списка неподошедших
                    unassociated_curr.remove(curr_idx)

        # Новые ID для оставшихся
        for curr_idx in unassociated_curr:
            track_ids[curr_idx] = self._next_id
            self._next_id += 1

        # Обновить состояние
        self._prev_keypoints = keypoints.copy()
        self._prev_scores = scores.copy()
        self._prev_track_ids = track_ids.copy()

        for i, tid in enumerate(track_ids):
            self._track_last_seen[tid] = self._frame_count
            self._track_keypoints[tid] = keypoints[i].copy()

        # Удалить старые треки
        valid_tracks = set()
        for tid, last in self._track_last_seen.items():
            if self._frame_count - last <= self._max_disappeared:
                valid_tracks.add(tid)

        self._track_last_seen = {
            tid: last for tid, last in self._track_last_seen.items()
            if tid in valid_tracks
        }
        self._track_keypoints = {
            tid: kps for tid, kps in self._track_keypoints.items()
            if tid in valid_tracks
        }

        self._frame_count += 1
        return track_ids

    def reset(self) -> None:
        """Сбросить состояние трекера."""
        self._prev_keypoints = None
        self._prev_scores = None
        self._prev_track_ids = []
        self._track_last_seen = {}
        self._track_keypoints = {}
        self._frame_count = 0
        self._next_id = 0
