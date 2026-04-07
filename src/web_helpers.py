"""Pure helper functions for the Gradio UI.

All functions are stateless and testable without a running Gradio server.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np

from src.types import PersonClick
from src.utils.video_writer import H264Writer

if TYPE_CHECKING:
    from numpy.typing import NDArray


logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def _find_model(filename: str) -> str:
    """Find model file in data/models/."""
    path = _PROJECT_ROOT / "data" / "models" / filename
    if path.exists():
        return str(path)
    raise FileNotFoundError(f"Model not found: {path}")


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
    blade_3d: bool,
    export: bool,
    output_path: str | Path,
    progress_cb=None,
    # ML flags (all default False)
    depth: bool = False,
    optical_flow: bool = False,
    segment: bool = False,
    foot_track: bool = False,
    matting: bool = False,
) -> dict:
    """Run the full visualization pipeline (mirrors visualize_with_skeleton.py)."""
    from src.visualization import render_layers
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

    # --- Initialize ML models (lazy-loaded) ---
    registry = None
    depth_est = None
    flow_est = None
    seg_est = None
    matting_est = None
    ml_layers: list = []

    any_ml = depth or optical_flow or segment or foot_track or matting
    if any_ml:
        from src.ml.model_registry import ModelRegistry

        registry = ModelRegistry(device="auto")

        if depth:
            try:
                registry.register(
                    "depth_anything", vram_mb=200, path=_find_model("depth_anything_v2_small.onnx")
                )
            except FileNotFoundError as e:
                logger.warning("Depth model not found: %s", e)
        if optical_flow:
            try:
                registry.register(
                    "optical_flow", vram_mb=80, path=_find_model("neuflowv2_mixed.onnx")
                )
            except FileNotFoundError as e:
                logger.warning("Optical flow model not found: %s", e)
        if segment:
            try:
                registry.register(
                    "segment_anything", vram_mb=200, path=_find_model("sam2_tiny.onnx")
                )
            except FileNotFoundError as e:
                logger.warning("Segmentation model not found: %s", e)
        if foot_track:
            try:
                registry.register("foot_tracker", vram_mb=30, path=_find_model("foot_tracker.onnx"))
            except FileNotFoundError as e:
                logger.warning("Foot tracker model not found: %s", e)
        if matting:
            try:
                registry.register(
                    "video_matting", vram_mb=40, path=_find_model("rvm_mobilenetv3.onnx")
                )
            except FileNotFoundError as e:
                logger.warning("Video matting model not found: %s", e)

        # Load models that will be used
        if depth and registry.is_registered("depth_anything"):
            from src.ml.depth_anything import DepthEstimator
            from src.visualization.layers.depth_layer import DepthMapLayer

            try:
                depth_est = DepthEstimator(registry)
                ml_layers.append(DepthMapLayer(opacity=0.4))
            except Exception as e:
                logger.warning("Failed to load depth model: %s", e)
        if optical_flow and registry.is_registered("optical_flow"):
            from src.ml.optical_flow import OpticalFlowEstimator
            from src.visualization.layers.optical_flow_layer import OpticalFlowLayer

            try:
                flow_est = OpticalFlowEstimator(registry)
                ml_layers.append(OpticalFlowLayer(opacity=0.5))
            except Exception as e:
                logger.warning("Failed to load optical flow model: %s", e)
        if segment and registry.is_registered("segment_anything"):
            from src.ml.segment_anything import SegmentAnything
            from src.visualization.layers.segmentation_layer import SegmentationMaskLayer

            try:
                seg_est = SegmentAnything(registry)
                ml_layers.append(SegmentationMaskLayer(opacity=0.3))
            except Exception as e:
                logger.warning("Failed to load segmentation model: %s", e)
        if matting and registry.is_registered("video_matting"):
            from src.ml.video_matting import VideoMatting
            from src.visualization.layers.matting_layer import MattingLayer

            try:
                matting_est = VideoMatting(registry)
                ml_layers.append(MattingLayer())
            except Exception as e:
                logger.warning("Failed to load video matting model: %s", e)

    if progress_cb:
        progress_cb(0.6, "Rendering...")

    # --- Build rendering pipeline ---
    pipe = VizPipeline(
        meta=prepared.meta,
        poses_norm=prepared.poses_norm,
        poses_px=prepared.poses_px,
        foot_kps=prepared.foot_kps,
        poses_3d=prepared.poses_3d,
        layer=layer,
        confs=prepared.confs,
        frame_indices=prepared.frame_indices,
    )
    pipe.add_ml_layers(ml_layers)

    meta = prepared.meta
    cap = cv2.VideoCapture(str(video_path))
    writer = H264Writer(output_path, meta.width, meta.height, meta.fps)

    # --- Render loop ---
    frame_idx = 0
    pose_idx = 0
    total = meta.num_frames

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        current_pose_idx, pose_idx = pipe.find_pose_idx(frame_idx, pose_idx)

        # Per-frame ML inference
        if (
            depth_est is not None
            or flow_est is not None
            or seg_est is not None
            or matting_est is not None
        ):
            _, context = pipe.render_frame(frame, frame_idx, current_pose_idx)
            if depth_est is not None:
                depth_map = depth_est.estimate(frame)
                context.custom_data["depth_map"] = depth_map
            if flow_est is not None:
                flow = flow_est.estimate_from_previous(frame)
                if flow is not None:
                    context.custom_data["flow_field"] = flow
            if seg_est is not None and current_pose_idx is not None:
                # Use mid-hip (index 11 in H3.6M) as SAM2 point prompt
                mid_hip = prepared.poses_px[current_pose_idx, 11, :2]
                if not np.any(np.isnan(mid_hip)):
                    mask = seg_est.segment(frame, point=(int(mid_hip[0]), int(mid_hip[1])))
                    if mask is not None:
                        context.custom_data["seg_mask"] = mask
            if matting_est is not None:
                alpha = matting_est.matting(frame)
                context.custom_data["alpha_matte"] = alpha
            # Re-render ML layers with data
            if layer >= 1 and current_pose_idx is not None:
                frame = render_layers(frame, ml_layers, context)
        else:
            frame, _ = pipe.render_frame(frame, frame_idx, current_pose_idx)

        pipe.draw_frame_counter(frame, frame_idx)

        if export:
            pipe.collect_export_data(frame_idx, current_pose_idx)

        writer.write(frame)
        frame_idx += 1

        if progress_cb and frame_idx % 50 == 0:
            progress_cb(0.6 + 0.3 * frame_idx / total, f"Rendering frame {frame_idx}/{total}")

    cap.release()
    writer.close()

    if progress_cb:
        progress_cb(0.95, "Saving exports...")

    export_result = (
        pipe.save_exports(output_path) if export else {"poses_path": None, "csv_path": None}
    )

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
    }
