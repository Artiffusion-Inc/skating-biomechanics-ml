"""Шина событий для коммуникации между компонентами.

Event bus for pub/sub communication between UI components.
"""

from collections import defaultdict
from collections.abc import Callable
from typing import Any


class EventBus:
    """Простая шина событий для подписки/публикации.

    Simple event bus for component communication without coupling.
    """

    # Standard event names
    EVENT_VIDEO_LOADED = "video:loaded"
    EVENT_POSES_EXTRACTED = "poses:extracted"
    EVENT_LAYER_TOGGLED = "layer:toggled"
    EVENT_EXPORT_STARTED = "export:started"
    EVENT_EXPORT_COMPLETE = "export:complete"
    EVENT_FRAME_CHANGED = "frame:changed"

    def __init__(self) -> None:
        """Инициализация шины событий."""
        self._subscribers: defaultdict[str, list[Callable]] = defaultdict(list)

    def subscribe(self, event: str, handler: Callable[[Any], None]) -> None:
        """Подписаться на событие.

        Args:
            event: Имя события.
            handler: Функция-обработчик (получает данные события).
        """
        if handler not in self._subscribers[event]:
            self._subscribers[event].append(handler)

    def unsubscribe(self, event: str, handler: Callable[[Any], None]) -> None:
        """Отписаться от события.

        Args:
            event: Имя события.
            handler: Функция-обработчик для удаления.
        """
        if event in self._subscribers and handler in self._subscribers[event]:
            self._subscribers[event].remove(handler)

    def publish(self, event: str, data: Any = None) -> None:
        """Опубликовать событие.

        Args:
            event: Имя события.
            data: Данные для передачи подписчикам.
        """
        for handler in self._subscribers.get(event, []):
            try:
                handler(data)
            except Exception:
                # Don't let one handler break others
                pass

    def clear(self, event: str | None = None) -> None:
        """Очистить подписчиков.

        Args:
            event: Имя события для очистки. Если None, очищает все.
        """
        if event:
            self._subscribers[event].clear()
        else:
            self._subscribers.clear()
