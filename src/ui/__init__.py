"""Модуль интерактивного веб-UI для анализа фигурного катания.

Streamlit-based UI with modular architecture:
- core: Session state, config persistence, event bus
- components: Reusable UI widgets
- processors: Data processing pipeline
- layouts: Page composition
"""

__version__ = "0.1.0"

# Re-export core components for convenience
from src.ui.core.config import UIConfig
from src.ui.core.events import EventBus
from src.ui.core.state import UIState
from src.ui.types import LayerSettings, ProcessedPoses, VideoSource

__all__ = [
    "EventBus",
    "LayerSettings",
    "ProcessedPoses",
    "UIConfig",
    "UIState",
    "VideoSource",
]
