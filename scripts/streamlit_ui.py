#!/usr/bin/env python3
"""Интерактивный веб-UI для анализа фигурного катания.

Interactive web UI for figure skating analysis.
Based on modular architecture with Streamlit.
"""


from src.ui.core.config import UIConfig
from src.ui.core.events import EventBus
from src.ui.layouts.main_layout import MainLayout


def main() -> None:
    """Главная функция."""
    # Initialize core components
    config = UIConfig()
    events = EventBus()

    # Build and render layout
    layout = MainLayout.from_config(config, events)
    layout.render()


if __name__ == "__main__":
    main()
