"""Базовые модули UI: состояние, конфигурация, события.

Core modules: session state, settings persistence, event bus.
"""

from .config import UIConfig
from .events import EventBus
from .state import UIState

__all__ = ["EventBus", "UIConfig", "UIState"]
