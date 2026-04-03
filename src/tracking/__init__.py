"""Алгоритмы трекинга людей для мульти-персональной ассоциации поз.

Предоставляет пос frame-to-frame реидентификацию:
- Sports2D: Венгерский алгоритм по расстояниям ключевых точек (scipy)
- DeepSORT: Appearance-based ReID (требуется deep-sort-realtime)
"""

from .sports2d import Sports2DTracker
from .deepsort_tracker import DeepSORTTracker

__all__ = ["Sports2DTracker", "DeepSORTTracker"]
