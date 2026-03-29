"""Компонент видеоплеера с перемоткой кадров.

Video player component with frame scrubbing.
"""

from typing import Any

import streamlit as st


def render_frame_slider(num_frames: int, current: int = 0) -> int:
    """Отрисовать слайдер для перемотки кадров.

    Args:
        num_frames: Общее количество кадров.
        current: Текущий кадр.

    Returns:
        Новый выбранный кадр.
    """
    if num_frames <= 0:
        return 0

    return st.slider(
        "Кадр",
        min_value=0,
        max_value=num_frames - 1,
        value=current,
        step=1,
        help="Перемотайте для просмотра отдельных кадров",
    )


def render_video_frame(frame: Any, key: str = "video_frame") -> None:
    """Отрисовать кадр в интерфейсе.

    Args:
        frame: Кадр (numpy array или похожий объект).
        key: Уникальный ключ для Streamlit.
    """
    st.image(frame, channels="BGR", width="stretch")


def render_video_info(
    width: int,
    height: int,
    fps: float,
    num_frames: int,
    duration: float,
) -> None:
    """Отрисовать информацию о видео.

    Args:
        width: Ширина видео.
        height: Высота видео.
        fps: Кадров в секунду.
        num_frames: Количество кадров.
        duration: Длительность в секундах.
    """
    st.caption(
        f"📹 {width}x{height} • {fps:.1f} FPS • "
        f"{num_frames} кадров • {duration:.1f} сек"
    )
