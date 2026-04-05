#!/usr/bin/env python3
"""Gradio web UI for figure skating biomechanics analysis.

Provides a browser-based interface for:
- Video upload and preview
- Interactive person selection (click on image or radio buttons)
- Real-time analysis with configurable options
- Side-by-side video + animated 3D skeleton viewer
- Downloadable results (video, poses, CSV)
"""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import gradio as gr

from src.device import DeviceConfig
from src.gradio_helpers import (
    choice_to_person_click,
    match_click_to_person,
    persons_to_choices,
    process_video_pipeline,
    render_person_preview,
)
from src.pose_estimation.rtmlib_extractor import RTMPoseExtractor
from src.types import PersonClick
from src.utils.video import get_video_meta

logger = logging.getLogger(__name__)


def _create_extractor(tracking: str) -> RTMPoseExtractor:
    """Create RTMPoseExtractor with GPU->CPU fallback."""
    cfg = DeviceConfig.default()
    return RTMPoseExtractor(
        mode="balanced",
        tracking_backend="rtmlib",
        tracking_mode=tracking,
        conf_threshold=0.3,
        output_format="normalized",
        device=cfg.device,
    )


def _detect_persons(
    video_path: str,
    tracking: str,
) -> tuple[gr.Image, list[str], list[dict], str]:
    """Detect all persons in the video and show annotated preview.

    Args:
        video_path: Path to uploaded video file.
        tracking: Tracking mode ("auto", "sports2d", "deepsort").

    Returns:
        (annotated_image, radio_choices, persons_state, status)
    """
    if not video_path:
        return None, gr.update(choices=[], value=None), [], "⚠️ Загрузите видео."

    try:
        extractor = _create_extractor(tracking)

        persons = extractor.preview_persons(Path(video_path), num_frames=30)

        if not persons:
            return (
                None,
                gr.update(choices=[], value=None),
                [],
                "⚠️ Люди не найдены. Попробуйте другое видео.",
            )

        # Load the preview frame (first frame with detections)
        cap = cv2.VideoCapture(video_path)
        ret, frame = cap.read()
        cap.release()

        if not ret:
            return None, gr.update(choices=[], value=None), [], "⚠️ Не удалось прочитать кадр видео."

        # Render annotated preview with numbered bboxes
        annotated = render_person_preview(frame, persons, selected_idx=None)

        # Convert BGR to RGB for Gradio
        annotated = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

        choices = persons_to_choices(persons)
        status = f"✅ Обнаружено {len(persons)} чел. Нажмите на человека на изображении или выберите из списка."

        return (
            annotated,
            gr.update(choices=choices, value=choices[0] if len(choices) == 1 else None),
            persons,
            status,
        )

    except Exception as e:
        return None, gr.update(choices=[], value=None), [], f"❌ Ошибка: {e}"


def _on_image_select(
    evt: gr.SelectData,
    persons_state: list[dict],
    video_path: str,
) -> tuple[str, gr.Image, PersonClick]:
    """Handle click on preview image to select a person.

    Args:
        evt: Gradio SelectData event with .index (pixel coords).
        persons_state: List of detected person dicts.
        video_path: Path to video for frame dimensions.

    Returns:
        (status_text, annotated_image, person_click)
    """
    if not persons_state:
        return "⚠️ Сначала обнаружьте людей в видео.", None, None

    # Get video dimensions for coordinate normalization
    meta = get_video_meta(Path(video_path))
    w, h = meta.width, meta.height

    # Convert pixel click to normalized coordinates
    x_norm = evt.index[0] / w
    y_norm = evt.index[1] / h

    # Find closest person
    matched = match_click_to_person(persons_state, x_norm, y_norm)

    if matched is None:
        return "⚠️ Нажатие мимо. Попробуйте попасть на человека.", None, None

    # Find the index of the matched person
    idx = persons_state.index(matched)

    # Re-render preview with green highlight
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        return "⚠️ Не удалось прочитать кадр видео.", None, None

    annotated = render_person_preview(frame, persons_state, selected_idx=idx)
    annotated = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

    # Create PersonClick from mid_hip
    mid_hip = matched["mid_hip"]
    person_click = PersonClick(
        x=int(mid_hip[0] * w),
        y=int(mid_hip[1] * h),
    )

    status = f"✅ Выбран #{idx + 1} (трек {matched['track_id']}, {matched['hits']} кадров)"

    return status, annotated, person_click


def _on_person_select(
    choice: str,
    persons_state: list[dict],
    video_path: str,
) -> tuple[gr.Image, PersonClick]:
    """Handle selection from Radio dropdown.

    Args:
        choice: Selected choice string (e.g., "Person #1 (10 hits, track 0)").
        persons_state: List of detected person dicts.
        video_path: Path to video for frame dimensions.

    Returns:
        (annotated_image, person_click)
    """
    if not choice or not persons_state:
        return None, None

    meta = get_video_meta(Path(video_path))
    person_click = choice_to_person_click(choice, persons_state, meta.width, meta.height)

    # Find the index
    idx = int(choice.split("#")[1].split(" ", maxsplit=1)[0]) - 1

    # Re-render preview with green highlight
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        return None, None

    annotated = render_person_preview(frame, persons_state, selected_idx=idx)
    annotated = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

    return annotated, person_click


def _run_pipeline(
    video_path: str,
    person_click_state: PersonClick | None,
    persons_state: list[dict],
    person_choice: str,
    frame_skip: int,
    layer: int,
    tracking: str,
    export: bool,
    progress=gr.Progress(),  # noqa: B008
) -> tuple[str, str, str, str, str]:
    """Run the full analysis pipeline.

    Args:
        video_path: Path to input video.
        person_click_state: Selected person click (from image click).
        persons_state: List of detected persons.
        person_choice: Selected person from Radio (fallback).
        frame_skip: Frame skip for pose extraction.
        layer: HUD layer (0-3).
        tracking: Tracking mode.
        export: Export poses/CSV.
        progress: Gradio progress callback.

    Returns:
        (output_video_path, poses_path, csv_path, status_text, glb_path)
    """
    if not video_path:
        return None, None, None, "⚠️ Загрузите видео.", None

    # Resolve PersonClick (prefer image click, fallback to radio)
    person_click = person_click_state
    if person_click is None and person_choice and persons_state:
        meta = get_video_meta(Path(video_path))
        person_click = choice_to_person_click(person_choice, persons_state, meta.width, meta.height)

    if person_click is None:
        return (
            None,
            None,
            None,
            "⚠️ Выберите человека (нажмите на изображение или выберите из списка).",
            None,
        )

    # Generate output path
    input_path = Path(video_path)
    output_dir = input_path.parent / "gradio_outputs"
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / f"{input_path.stem}_analyzed.mp4"

    try:
        result = process_video_pipeline(
            video_path=video_path,
            person_click=person_click,
            frame_skip=frame_skip,
            layer=layer,
            tracking=tracking,
            blade_3d=False,  # Disabled for now
            export=export,
            output_path=str(output_path),
            progress_cb=lambda p, msg: progress(p, desc=msg),
        )

        stats = result["stats"]
        glb_path = result.get("glb_path")
        status = (
            f"✅ Анализ завершён!\n"
            f"   Разрешение: {stats['resolution']}\n"
            f"   Кадров: {stats['valid_frames']}/{stats['total_frames']} валидных\n"
            f"   FPS: {stats['fps']:.1f}\n"
            f"   Результат: {result['video_path']}"
        )

        return (
            result["video_path"],
            result["poses_path"],
            result["csv_path"],
            status,
            glb_path,
        )

    except Exception as e:
        return None, None, None, f"❌ Ошибка обработки: {e}", None


def build_app() -> gr.Blocks:
    """Build and return the Gradio app interface."""

    # HTML template for animated 3D skeleton viewer
    model_viewer_template = """
    <div class="viewer-container">
        <model-viewer
            id="mv-3d"
            src="${value}"
            autoplay
            camera-controls
            shadow-intensity="0"
            interaction-prompt="none"
            style="width:100%; height:500px; background:#1a1a2e; border-radius:8px;">
        </model-viewer>
        <div class="viewer-controls">
            <button id="mv-play" onclick="document.getElementById('mv-3d').play()">▶ Play</button>
            <button id="mv-pause" onclick="document.getElementById('mv-3d').pause()">⏸ Pause</button>
            <input
                type="range"
                id="mv-seek"
                min="0"
                max="1"
                step="0.001"
                value="0"
                style="flex:1;">
            <span id="mv-time" style="font-family:monospace; font-size:13px; color:#aaa; min-width:80px;">0:00.000</span>
        </div>
    </div>
    """

    # CSS for 3D viewer controls
    viewer_css = """
    .viewer-container { display:flex; flex-direction:column; gap:8px; }
    .viewer-controls { display:flex; align-items:center; gap:8px; padding:4px 0; }
    .viewer-controls button {
        width:80px; height:36px; border-radius:6px; border:none; cursor:pointer;
        font-size:14px; background:#333; color:#fff; transition:background 0.2s;
    }
    .viewer-controls button:hover { background:#555; }
    .viewer-controls input[type=range] { flex:1; cursor:pointer; }
    """

    # JavaScript for syncing seek bar with model-viewer playback
    viewer_js = """
        const viewer = element.querySelector('#mv-3d');
        const seek = element.querySelector('#mv-seek');
        const timeDisplay = element.querySelector('#mv-time');

        if (viewer) {
            // Wait for model to load
            viewer.addEventListener('load', () => {
                const dur = viewer.duration || 1;
                seek.max = dur;

                // Update seek bar during playback
                const updateSeek = () => {
                    if (viewer.currentTime !== undefined) {
                        seek.value = viewer.currentTime;
                        const m = Math.floor(viewer.currentTime / 60);
                        const s = (viewer.currentTime % 60).toFixed(3);
                        timeDisplay.textContent = m + ':' + s.padStart(6, '0');
                    }
                    requestAnimationFrame(updateSeek);
                };
                updateSeek();
            });

            // Seek bar input
            seek.addEventListener('input', () => {
                viewer.currentTime = parseFloat(seek.value);
            });
        }
    """

    with gr.Blocks(
        title="AI Тренер по фигурному катанию",
        head="""
        <script type="module" src="https://ajax.googleapis.com/ajax/libs/model-viewer/3.5.0/model-viewer.min.js"></script>
        """,
    ) as app:
        gr.Markdown("# AI Тренер по фигурному катанию")
        gr.Markdown("Загрузите видео, выберите фигуриста и получите биомеханический анализ.")

        # State
        persons_state = gr.State()
        person_click_state = gr.State(None)

        with gr.Row():
            # Left column: Controls
            with gr.Column(scale=1):
                video_input = gr.Video(
                    label="Загрузите видео",
                    sources=["upload"],
                )

                tracking_dropdown = gr.Dropdown(
                    label="Режим трекинга",
                    choices=["auto", "sports2d", "deepsort"],
                    value="auto",
                    info="Auto — DeepSORT при наличии, иначе Sports2D",
                )

                detect_btn = gr.Button("Обнаружить людей", variant="primary", size="lg")

                preview_image = gr.Image(
                    label="Превью (нажмите на фигуриста)",
                    interactive=False,
                    type="numpy",
                    height=400,
                )

                person_radio = gr.Radio(
                    label="Или выберите из списка",
                    choices=[],
                    interactive=True,
                )

                selection_status = gr.Textbox(
                    label="Статус выбора",
                    value="Загрузите видео и нажмите «Обнаружить людей»",
                    interactive=False,
                )

                with gr.Accordion("Расширенные настройки", open=False):
                    frame_skip_slider = gr.Slider(
                        label="Пропуск кадров",
                        minimum=1,
                        maximum=8,
                        step=1,
                        value=4,
                        info="1=все кадры (медленно), 4=каждый 4-й (рекомендуется), 8=максимальная скорость",
                    )

                    layer_slider = gr.Slider(
                        label="Уровень HUD",
                        minimum=0,
                        maximum=3,
                        step=1,
                        value=3,
                        info="0=скелет, 1=скорость+следы+углы, 2=+ось, 3=полный HUD",
                    )

                    export_checkbox = gr.Checkbox(
                        label="Экспорт поз и CSV",
                        value=True,
                        info="Скачать данные поз и биомеханики",
                    )

                process_btn = gr.Button("Обработать видео", variant="primary", size="lg")

            # Right column: Outputs (side-by-side video + 3D)
            with gr.Column(scale=1):
                output_video = gr.Video(
                    label="Видео с анализом",
                    autoplay=True,
                    height=300,
                )

                model_3d_viewer = gr.HTML(
                    value=None,
                    label="3D Анимированный скелет",
                    html_template=model_viewer_template,
                    css_template=viewer_css,
                    js_on_load=viewer_js,
                )

                poses_download = gr.File(
                    label="Скачать позы (.npy)",
                    visible=False,
                )

                csv_download = gr.File(
                    label="Скачать биомеханику (.csv)",
                    visible=False,
                )

                output_status = gr.Textbox(
                    label="Статус",
                    value="Ожидание видео...",
                    lines=4,
                    interactive=False,
                )

        # Event wiring
        detect_btn.click(
            fn=_detect_persons,
            inputs=[video_input, tracking_dropdown],
            outputs=[preview_image, person_radio, persons_state, selection_status],
        )

        preview_image.select(
            fn=_on_image_select,
            inputs=[persons_state, video_input],
            outputs=[selection_status, preview_image, person_click_state],
        )

        person_radio.change(
            fn=_on_person_select,
            inputs=[person_radio, persons_state, video_input],
            outputs=[preview_image, person_click_state],
        )

        process_btn.click(
            fn=_run_pipeline,
            inputs=[
                video_input,
                person_click_state,
                persons_state,
                person_radio,
                frame_skip_slider,
                layer_slider,
                tracking_dropdown,
                export_checkbox,
            ],
            outputs=[
                output_video,
                poses_download,
                csv_download,
                output_status,
                model_3d_viewer,
            ],
        )

    return app


def main() -> None:
    """Launch the Gradio app."""
    app = build_app()
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
        theme=gr.themes.Soft(),
        css="""
        .gradio-container {
            max-width: 1400px !important;
        }
        """,
    )


if __name__ == "__main__":
    main()
