"""Тесты Sports2DTracker."""

import numpy as np
import pytest

from src.tracking.sports2d import Sports2DTracker
from src.types import H36Key


def _make_person_pose(
    cx: float, cy: float, scale: float = 0.1
) -> np.ndarray:
    """Создать простую позу стоящего человека по центру (cx, cy)."""
    pose = np.zeros((17, 2), dtype=np.float32)
    s = scale / 0.1  # нормализация масштаба
    pose[H36Key.HIP_CENTER] = [cx, cy]
    pose[H36Key.RHIP] = [cx - 0.04 * s, cy]
    pose[H36Key.LHIP] = [cx + 0.04 * s, cy]
    pose[H36Key.RKNEE] = [cx - 0.04 * s, cy + 0.20 * s]
    pose[H36Key.LKNEE] = [cx + 0.04 * s, cy + 0.20 * s]
    pose[H36Key.RFOOT] = [cx - 0.04 * s, cy + 0.40 * s]
    pose[H36Key.LFOOT] = [cx + 0.04 * s, cy + 0.40 * s]
    pose[H36Key.SPINE] = [cx, cy - 0.15 * s]
    pose[H36Key.THORAX] = [cx, cy - 0.25 * s]
    pose[H36Key.NECK] = [cx, cy - 0.30 * s]
    pose[H36Key.HEAD] = [cx, cy - 0.35 * s]
    pose[H36Key.LSHOULDER] = [cx + 0.08 * s, cy - 0.25 * s]
    pose[H36Key.RSHOULDER] = [cx - 0.08 * s, cy - 0.25 * s]
    pose[H36Key.LELBOW] = [cx + 0.12 * s, cy - 0.15 * s]
    pose[H36Key.RELBOW] = [cx - 0.12 * s, cy - 0.15 * s]
    pose[H36Key.LWRIST] = [cx + 0.14 * s, cy - 0.05 * s]
    pose[H36Key.RWRIST] = [cx - 0.14 * s, cy - 0.05 * s]
    return pose


def _make_scores(n_persons: int, base: float = 0.8) -> np.ndarray:
    """Создать confidence scores."""
    return np.full((n_persons, 17), base, dtype=np.float32)


class TestFirstFrame:
    def test_first_frame_assigns_sequential_ids(self):
        """Первый кадр получает ID [0, 1, ...]."""
        tracker = Sports2DTracker()
        kps = np.array([_make_person_pose(0.3, 0.5),
                         _make_person_pose(0.7, 0.5)])
        scores = _make_scores(2)

        ids = tracker.update(kps, scores)

        assert ids == [0, 1]


class TestStableTracking:
    def test_stable_ids_small_movement(self):
        """Тот же человек с небольшим смещением получает тот же ID."""
        tracker = Sports2DTracker()
        person_a = _make_person_pose(0.3, 0.5)

        # Кадр 1
        ids1 = tracker.update(
            np.array([person_a, _make_person_pose(0.7, 0.5)]),
            _make_scores(2),
        )

        # Кадр 2 — оба чуть сдвинулись
        ids2 = tracker.update(
            np.array([_make_person_pose(0.31, 0.51),
                       _make_person_pose(0.69, 0.49)]),
            _make_scores(2),
        )

        assert ids1 == ids2 == [0, 1]

    def test_two_people_swap_order(self):
        """Люди меняются порядком в списке — ID остаются стабильными."""
        tracker = Sports2DTracker()
        person_a = _make_person_pose(0.3, 0.5)
        person_b = _make_person_pose(0.7, 0.5)

        # Кадр 1: A первым
        tracker.update(np.array([person_a, person_b]), _make_scores(2))

        # Кадр 2: B первым (порядок меняется)
        ids2 = tracker.update(np.array([person_b, person_a]), _make_scores(2))

        # A = ID 0, B = ID 1. Во втором кадре B (индекс 0) = ID 1,
        # A (индекс 1) = ID 0.
        assert ids2[0] == 1  # B
        assert ids2[1] == 0  # A


class TestNewPerson:
    def test_extra_person_gets_new_id(self):
        """Новый человек на кадре 2 получает новый ID."""
        tracker = Sports2DTracker()
        person_a = _make_person_pose(0.3, 0.5)

        # Кадр 1: 1 человек
        ids1 = tracker.update(np.array([person_a]), _make_scores(1))

        # Кадр 2: 2 человека
        ids2 = tracker.update(
            np.array([_make_person_pose(0.31, 0.51),
                       _make_person_pose(0.7, 0.5)]),
            _make_scores(2),
        )

        assert ids1 == [0]
        # ID 0 = первый человек (сопоставлен), ID 1 = новый
        assert 0 in ids2
        assert 1 in ids2


class TestPersonLeaves:
    def test_person_disappears_id_persists(self):
        """Если человек пропал на кадр, его ID сохраняется при возвращении."""
        tracker = Sports2DTracker(max_disappeared=30)
        person_a = _make_person_pose(0.3, 0.5)
        person_b = _make_person_pose(0.7, 0.5)

        # Кадр 1: 2 человека
        ids1 = tracker.update(np.array([person_a, person_b]), _make_scores(2))

        # Кадр 2: только 1 человек
        ids2 = tracker.update(np.array([person_a]), _make_scores(1))

        # Кадр 3: снова 2 человека
        ids3 = tracker.update(np.array([person_a, person_b]), _make_scores(2))

        assert ids1[0] == 0  # A
        assert ids1[1] == 1  # B
        assert ids2 == [0]    # A остался
        assert ids3[0] == 0  # A
        assert ids3[1] == 1  # B вернулся с тем же ID


class TestNaNHandling:
    def test_nan_keypoints_no_crash(self):
        """NaN в ключевых точках не вызывает краш."""
        tracker = Sports2DTracker()
        person_a = _make_person_pose(0.3, 0.5)
        person_b = _make_person_pose(0.7, 0.5)

        # Кадр 1 — нормальный
        tracker.update(np.array([person_a, person_b]), _make_scores(2))

        # Кадр 2 — с NaN
        person_b_nan = person_b.copy()
        person_b_nan[3, :] = np.nan  # RFOOT = NaN
        ids2 = tracker.update(
            np.array([_make_person_pose(0.31, 0.51), person_b_nan]),
            _make_scores(2),
        )

        assert len(ids2) == 2
        assert all(isinstance(x, int) for x in ids2)


class TestEmptyFrame:
    def test_empty_then_normal(self):
        """Пустой кадр, затем нормальный — не крашится."""
        tracker = Sports2DTracker()
        person_a = _make_person_pose(0.3, 0.5)

        # Кадр 1: пустой
        ids1 = tracker.update(np.zeros((0, 17, 2)), np.zeros((0, 17)))
        assert ids1 == []

        # Кадр 2: нормальный
        ids2 = tracker.update(np.array([person_a]), _make_scores(1))
        assert ids2 == [0]


class TestMaxDist:
    def test_large_jump_creates_new_track(self):
        """Слишком большое перемещение → новый трек."""
        tracker = Sports2DTracker(max_dist=0.01)  # очень маленький порог
        person_a = _make_person_pose(0.3, 0.5)

        # Кадр 1
        ids1 = tracker.update(np.array([person_a]), _make_scores(1))

        # Кадр 2: человек далеко (0.3 → 0.8)
        ids2 = tracker.update(
            np.array([_make_person_pose(0.8, 0.5)]),
            _make_scores(1),
        )

        # Слишком далеко для max_dist=0.01 → новый ID
        assert ids1 == [0]
        assert ids2 == [1]


class TestAutoMaxDist:
    def test_auto_max_dist_works(self):
        """Авто-вычисленный max_dist из диагонали bbox."""
        tracker = Sports2DTracker(max_dist=None)
        person_a = _make_person_pose(0.3, 0.5)
        person_b = _make_person_pose(0.7, 0.5)

        # Кадр 1
        tracker.update(np.array([person_a, person_b]), _make_scores(2))

        # Кадр 2: небольшое смещение (должно сопоставиться)
        ids2 = tracker.update(
            np.array([_make_person_pose(0.31, 0.51),
                       _make_person_pose(0.69, 0.49)]),
            _make_scores(2),
        )

        assert ids2[0] == 0
        assert ids2[1] == 1


class TestTrackPurge:
    def test_old_tracks_purged(self):
        """Старые треки удаляются после max_disappeared."""
        tracker = Sports2DTracker(max_disappeared=2)
        person_a = _make_person_pose(0.3, 0.5)

        # Кадр 1
        ids1 = tracker.update(np.array([person_a]), _make_scores(1))
        assert ids1 == [0]

        # 3 пустых кадра (превышает max_disappeared=2)
        for _ in range(3):
            tracker.update(np.zeros((0, 17, 2)), np.zeros((0, 17)))

        # Новый человек → новый ID (старый 0 удалён)
        ids5 = tracker.update(np.array([person_a]), _make_scores(1))
        assert ids5 == [1]
