"""Процессор экспорта полного видео.

Full video export processor with progress tracking.
"""

from collections.abc import Callable
from pathlib import Path

import cv2
import numpy as np
import streamlit as st

from src.ui.core.events import EventBus
from src.ui.types import LayerSettings, ProcessedPoses
from src.video import extract_frames


class VideoExporter:
    """Экспорт полного видео с выбранными настройками.

    Export full video with current visualization settings.
    """

    def __init__(self, events: EventBus | None = None) -> None:
        """Инициализация экспортёра.

        Args:
            events: Шина событий для оповещений.
        """
        self._events = events
        # Import here to avoid circular dependency
        from src.ui.processors.layer_composer import LayerComposer
        self._layer_composer = LayerComposer()

    def process(
        self,
        video_path: Path,
        poses: ProcessedPoses,
        settings: LayerSettings,
        output_path: Path,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> Path:
        """Экспортировать видео с визуализацией.

        Args:
            video_path: Путь к исходному видео.
            poses: Обработанные позы.
            settings: Настройки визуализации.
            output_path: Путь для сохранения результата.
            progress_callback: Функция для отчёта о прогрессе (current, total).

        Returns:
            Путь к сохранённому файлу.
        """
        # Setup video writer
        cap = cv2.VideoCapture(str(video_path))
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(
            str(output_path),
            fourcc,
            poses.fps,
            (poses.width, poses.height),
        )

        # Clear trail history for clean export
        self._layer_composer.clear_trails()

        total_frames = poses.num_frames
        frame_count = 0

        # Status text
        status_text = st.empty()
        progress_bar = st.progress(0)

        try:
            for frame_idx, frame in enumerate(extract_frames(video_path)):
                # Generate layer mask
                mask = self._layer_composer.compose_mask(
                    frame.shape,
                    frame_idx,
                    poses,
                    settings,
                )

                # Blend frame with mask (alpha=0.8 for visibility)
                # mask has black background, only non-black pixels are layers
                mask_alpha = (mask > 0).any(axis=2, keepdim=True).astype(float) * 0.8
                rendered = (frame.astype(float) * (1 - mask_alpha) +
                           mask.astype(float) * mask_alpha).astype(np.uint8)

                # Write frame
                writer.write(rendered)
                frame_count += 1

                # Update progress
                if frame_count % 10 == 0:
                    progress = frame_count / total_frames
                    progress_bar.progress(progress)
                    status_text.text(f"Экспорт: {frame_count}/{total_frames} кадров ({progress*100:.0f}%)")

                if progress_callback:
                    progress_callback(frame_count, total_frames)

        finally:
            cap.release()
            writer.release()
            progress_bar.empty()
            status_text.empty()

        # Publish event
        if self._events:
            self._events.publish("export:complete", output_path)

        return output_path
