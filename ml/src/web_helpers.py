"""Pure helper functions for the Gradio UI.

All functions are stateless and testable without a running Gradio server.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import cv2

from src.types import PersonClick
from src.utils.video_writer import H264Writer

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray


logger = logging.getLogger(__name__)


class PipelineCancelled(Exception):
    """Raised when the user cancels video processing."""


def match_click_to_person(
    persons: list[dict],
    x: float,
    y: float,
) -> dict | None:
    """Match a normalized click coordinate to the closest person bbox."""
    if not persons:
        return None

    best: dict | None = None
    best_dist = float("inf")

    for p in persons:
        x1, y1, x2, y2 = p["bbox"]
        if x1 <= x <= x2 and y1 <= y <= y2:
            mx, my = p["mid_hip"]
            dist = (x - mx) ** 2 + (y - my) ** 2
            if dist < best_dist:
                best_dist = dist
                best = p

    return best


def render_person_preview(
    frame: NDArray[np.uint8],
    persons: list[dict],
    selected_idx: int | None = None,
) -> NDArray[np.uint8]:
    """Draw numbered bounding boxes for each detected person."""
    if not persons:
        return frame

    annotated = frame.copy()
    h, w = frame.shape[:2]

    colors = [
        (255, 165, 0),  # Blue (OpenCV BGR)
        (0, 200, 200),  # Yellow
        (200, 100, 0),  # Cyan
        (200, 0, 200),  # Magenta
        (0, 180, 255),  # Orange
    ]

    for i, p in enumerate(persons):
        x1, y1, x2, y2 = p["bbox"]
        px1, py1 = int(x1 * w), int(y1 * h)
        px2, py2 = int(x2 * w), int(y2 * h)

        if selected_idx is not None and i == selected_idx:
            color = (0, 255, 0)  # Green for selected
            thickness = 3
        else:
            color = colors[i % len(colors)]
            thickness = 2

        cv2.rectangle(annotated, (px1, py1), (px2, py2), color, thickness)

        label = f"#{i + 1} (hits: {p['hits']})"
        cv2.rectangle(annotated, (px1, py1 - 28), (px1 + len(label) * 10 + 10, py1), color, -1)
        cv2.putText(
            annotated,
            label,
            (px1 + 5, py1 - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    return annotated


def persons_to_choices(persons: list[dict]) -> list[str]:
    """Convert person list to Gradio Radio choices."""
    return [
        f"Person #{i + 1} ({p['hits']} hits, track {p['track_id']})" for i, p in enumerate(persons)
    ]


def choice_to_person_click(
    choice: str,
    persons: list[dict],
    width: int,
    height: int,
) -> PersonClick:
    """Convert a Gradio Radio selection to a PersonClick."""
    idx = int(choice.split("#")[1].split(" ", maxsplit=1)[0]) - 1
    mid_hip = persons[idx]["mid_hip"]
    return PersonClick(
        x=int(mid_hip[0] * width),
        y=int(mid_hip[1] * height),
    )


def process_video_pipeline(  # noqa: PLR0913
    video_path: str | Path,
    person_click: PersonClick | None,
    frame_skip: int,
    layer: int,
    tracking: str,
    output_path: str | Path,
    progress_cb=None,
    cancel_event=None,
) -> dict:
    """Run the full visualization pipeline (mirrors visualize_with_skeleton.py)."""
    from src.visualization.pipeline import VizPipeline, prepare_poses

    video_path = Path(video_path) if isinstance(video_path, str) else video_path
    output_path = Path(output_path) if isinstance(output_path, str) else output_path

    # --- Unified pose preparation ---
    prepared = prepare_poses(
        video_path,
        person_click=person_click,
        frame_skip=frame_skip,
        tracking=tracking,
        progress_cb=progress_cb,
    )

    if progress_cb:
        progress_cb(0.6, "Рендеринг...")

    # --- Build rendering pipeline ---
    pipe = VizPipeline(
        meta=prepared.meta,
        poses_norm=prepared.poses_norm,
        poses_px=prepared.poses_px,
        poses_3d=prepared.poses_3d,
        layer=layer,
        confs=prepared.confs,
        frame_indices=prepared.frame_indices,
    )

    meta = prepared.meta
    writer = H264Writer(output_path, meta.width, meta.height, meta.fps)

    # Use AsyncFrameReader for decode-inference overlap
    from src.utils.frame_buffer import AsyncFrameReader

    reader = AsyncFrameReader(video_path, buffer_size=16, frame_skip=1)
    reader.start()

    # --- Render loop ---
    frame_idx = 0
    pose_idx = 0
    total = meta.num_frames

    while True:
        if cancel_event is not None and cancel_event.is_set():
            reader.join(timeout=1)
            writer.close()
            raise PipelineCancelled("Processing cancelled by user")

        result = reader.get_frame()
        if result is None:
            break
        frame_idx, frame = result

        current_pose_idx, pose_idx = pipe.find_pose_idx(frame_idx, pose_idx)

        frame, _ = pipe.render_frame(frame, frame_idx, current_pose_idx)
        pipe.draw_frame_counter(frame, frame_idx)
        writer.write(frame)
        frame_idx += 1

        if progress_cb and frame_idx % 50 == 0:
            progress_cb(0.6 + 0.3 * frame_idx / total, f"Rendering frame {frame_idx}/{total}")

    reader.join(timeout=5)
    writer.close()

    if progress_cb:
        progress_cb(0.95, "Saving exports...")

    export_result = {"poses_path": None, "csv_path": None}

    return {
        "video_path": str(output_path),
        "poses_path": export_result["poses_path"],
        "csv_path": export_result["csv_path"],
        "stats": {
            "total_frames": total,
            "valid_frames": prepared.n_valid,
            "fps": meta.fps,
            "resolution": f"{meta.width}x{meta.height}",
        },
        "metrics": [],
        "phases": None,
        "recommendations": [],
    }
