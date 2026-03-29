"""Менеджер состояния сессии Streamlit.

Session state manager for centralized state handling.
"""

from pathlib import Path
from typing import Any

import streamlit as st


class UIState:
    """Централизованный менеджер состояния сессии Streamlit.

    Centralized session state manager with typed accessors.
    """

    # Session state keys
    KEY_VIDEO_PATH = "video_path"
    KEY_POSES = "poses"
    KEY_POSES_3D = "poses_3d"
    KEY_BLADE_STATES_LEFT = "blade_states_left"
    KEY_BLADE_STATES_RIGHT = "blade_states_right"
    KEY_METADATA = "metadata"
    KEY_CURRENT_FRAME = "current_frame"
    KEY_PROCESSED = "processed"

    def __init__(self) -> None:
        """Инициализация менеджера состояния."""
        self._ensure_initialized()

    def _ensure_initialized(self) -> None:
        """Убедиться что все ключи инициализированы."""
        defaults = {
            self.KEY_VIDEO_PATH: None,
            self.KEY_POSES: None,
            self.KEY_POSES_3D: None,
            self.KEY_BLADE_STATES_LEFT: None,
            self.KEY_BLADE_STATES_RIGHT: None,
            self.KEY_METADATA: {},
            self.KEY_CURRENT_FRAME: 0,
            self.KEY_PROCESSED: False,
        }
        for key, value in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = value

    def get(self, key: str, default: Any = None) -> Any:
        """Получить значение из состояния сессии.

        Args:
            key: Ключ состояния.
            default: Значение по умолчанию если ключ не найден.

        Returns:
            Значение из состояния или default.
        """
        return st.session_state.get(key, default)

    def set(self, key: str, value: Any) -> None:
        """Установить значение в состоянии сессии.

        Args:
            key: Ключ состояния.
            value: Значение для сохранения.
        """
        st.session_state[key] = value

    def reset(self) -> None:
        """Сбросить состояние сессии в начальные значения."""
        keys_to_clear = [
            self.KEY_VIDEO_PATH,
            self.KEY_POSES,
            self.KEY_POSES_3D,
            self.KEY_BLADE_STATES_LEFT,
            self.KEY_BLADE_STATES_RIGHT,
            self.KEY_METADATA,
            self.KEY_CURRENT_FRAME,
            self.KEY_PROCESSED,
        ]
        for key in keys_to_clear:
            st.session_state[key] = None
        st.session_state[self.KEY_CURRENT_FRAME] = 0
        st.session_state[self.KEY_PROCESSED] = False

    @property
    def video_path(self) -> Path | None:
        """Путь к загруженному видео."""
        path = self.get(self.KEY_VIDEO_PATH)
        return Path(path) if path else None

    @video_path.setter
    def video_path(self, value: Path | None) -> None:
        self.set(self.KEY_VIDEO_PATH, str(value) if value else None)

    @property
    def poses(self) -> Any:
        """Извлечённые позы (numpy array или None)."""
        return self.get(self.KEY_POSES)

    @poses.setter
    def poses(self, value: Any) -> None:
        self.set(self.KEY_POSES, value)

    @property
    def poses_3d(self) -> Any:
        """3D позы или None."""
        return self.get(self.KEY_POSES_3D)

    @poses_3d.setter
    def poses_3d(self, value: Any) -> None:
        self.set(self.KEY_POSES_3D, value)

    @property
    def blade_states_left(self) -> list | None:
        """Состояния левого конька."""
        return self.get(self.KEY_BLADE_STATES_LEFT)

    @blade_states_left.setter
    def blade_states_left(self, value: list | None) -> None:
        self.set(self.KEY_BLADE_STATES_LEFT, value)

    @property
    def blade_states_right(self) -> list | None:
        """Состояния правого конька."""
        return self.get(self.KEY_BLADE_STATES_RIGHT)

    @blade_states_right.setter
    def blade_states_right(self, value: list | None) -> None:
        self.set(self.KEY_BLADE_STATES_RIGHT, value)

    @property
    def metadata(self) -> dict:
        """Метаданные видео."""
        return self.get(self.KEY_METADATA, {})

    @metadata.setter
    def metadata(self, value: dict) -> None:
        self.set(self.KEY_METADATA, value)

    @property
    def current_frame(self) -> int:
        """Текущий кадр для просмотра."""
        return self.get(self.KEY_CURRENT_FRAME, 0)

    @current_frame.setter
    def current_frame(self, value: int) -> None:
        self.set(self.KEY_CURRENT_FRAME, max(0, value))

    @property
    def is_processed(self) -> bool:
        """Было ли видео обработано (позы извлечены)."""
        return self.get(self.KEY_PROCESSED, False)

    @is_processed.setter
    def is_processed(self, value: bool) -> None:
        self.set(self.KEY_PROCESSED, value)
