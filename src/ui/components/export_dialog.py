"""Диалог экспорта видео с прогрессом.

Export dialog with progress tracking.
"""

from pathlib import Path

import streamlit as st

from src.ui.core.events import EventBus
from src.ui.types import LayerSettings, ProcessedPoses


def export_dialog(
    video_path: Path,
    poses: ProcessedPoses,
    settings: LayerSettings,
    events: EventBus | None = None,
) -> Path | None:
    """Диалог экспорта видео с прогрессом.

    Args:
        video_path: Путь к исходному видео.
        poses: Обработанные позы.
        settings: Настройки визуализации.
        events: Шина событий.

    Returns:
        Путь к сохранённому файлу или None если отменено.
    """
    from src.ui.processors.video_exporter import VideoExporter

    st.info("Подготовка к экспорту...")

    # Get output filename
    output_name = st.text_input(
        "Имя выходного файла",
        value=video_path.stem + "_annotated.mp4",
        help="Имя файла для сохранения (без пути)",
    )

    col1, col2 = st.columns(2)

    with col1:
        if st.button("Экспортировать", type="primary"):
            output_path = video_path.parent / output_name

            # Create exporter
            exporter = VideoExporter(events)

            # Progress placeholder
            progress_placeholder = st.empty()
            status_placeholder = st.empty()

            def progress_callback(current: int, total: int) -> None:
                """Обновить прогресс."""
                progress = current / total
                progress_placeholder.progress(progress)
                status_placeholder.text(f"Экспорт: {current}/{total} кадров")

            try:
                # Export
                result_path = exporter.process(
                    video_path,
                    poses,
                    settings,
                    output_path,
                    progress_callback=progress_callback,
                )

                progress_placeholder.empty()
                status_placeholder.empty()

                st.success(f"✅ Экспорт завершён: {result_path}")

                # Download button
                with open(result_path, "rb") as f:
                    st.download_button(
                        label="📥 Скачать видео",
                        data=f,
                        file_name=output_name,
                        mime="video/mp4",
                        width="stretch",
                    )

                return result_path

            except Exception as e:
                st.error(f"❌ Ошибка экспорта: {e}")
                return None

    with col2:
        if st.button("Отмена"):
            st.info("Экспорт отменён")
            return None

    return None
