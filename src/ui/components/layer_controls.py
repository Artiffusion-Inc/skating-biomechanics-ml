"""Элементы управления слоями визуализации.

Layer toggle controls for visualization settings.
"""


import streamlit as st

from src.ui.core.config import UIConfig
from src.ui.core.events import EventBus
from src.ui.types import LayerSettings


def render_layer_controls(
    config: UIConfig,
    events: EventBus,
) -> LayerSettings:
    """Отрисовать элементы управления слоями.

    Args:
        config: Менеджер настроек.
        events: Шина событий.

    Returns:
        LayerSettings с текущими значениями.
    """
    # This function is now integrated into render_sidebar
    # Kept for backward compatibility
    return render_layer_controls(config, events)


def _render_layer_checkbox(name: str, default: bool, key: str) -> bool:
    """Отрисовать один чекбокс слоя.

    Args:
        name: Название слоя.
        default: Значение по умолчанию.
        key: Уникальный ключ для Streamlit.

    Returns:
        Значение чекбокса.
    """
    return st.checkbox(name, value=default, key=key)
