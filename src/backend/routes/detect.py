"""POST /api/detect — detect persons in an uploaded video."""

from __future__ import annotations

import base64
import tempfile
import uuid
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
from fastapi import APIRouter, HTTPException, UploadFile

from src.backend.schemas import DetectResponse, PersonClick, PersonInfo
from src.device import DeviceConfig
from src.pose_estimation.rtmlib_extractor import RTMPoseExtractor
from src.storage import upload_bytes
from src.utils.video import get_video_meta
from src.web_helpers import (
    render_person_preview,
)

if TYPE_CHECKING:
    import numpy as np

router = APIRouter()


def _create_extractor(tracking: str) -> RTMPoseExtractor:
    cfg = DeviceConfig.default()
    return RTMPoseExtractor(
        mode="balanced",
        tracking_backend="rtmlib",
        tracking_mode=tracking,
        conf_threshold=0.3,
        output_format="normalized",
        device=cfg.device,
    )


def _encode_frame_bgr(frame: np.ndarray) -> str:
    """Encode BGR frame to base64 PNG string."""
    success, buf = cv2.imencode(".png", frame)
    if not success:
        raise RuntimeError("Failed to encode preview image")
    return base64.b64encode(buf).decode("ascii")


@router.post("/detect", response_model=DetectResponse)
async def detect_persons(
    video: UploadFile,
    tracking: str = "auto",
) -> DetectResponse:
    """Detect all persons in the uploaded video and return annotated preview."""
    suffix = Path(video.filename or "video.mp4").suffix
    video_key = f"input/{uuid.uuid4().hex}{suffix}"

    content = await video.read()

    # Upload to R2
    upload_bytes(content, video_key)

    # Save to /tmp for local RTMPose processing
    tmp_file = None
    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
            f.write(content)
            tmp_file = f.name

        video_path = Path(tmp_file)

        extractor = _create_extractor(tracking)
        persons, _ = extractor.preview_persons(video_path, num_frames=30)

        if not persons:
            return DetectResponse(
                persons=[],
                preview_image="",
                video_key=video_key,
                status="Люди не найдены. Попробуйте другое видео.",
            )

        # Read first frame for annotated preview
        cap = cv2.VideoCapture(str(video_path))
        ret, frame = cap.read()
        cap.release()

        if not ret:
            raise HTTPException(status_code=500, detail="Failed to read video frame")

        meta = get_video_meta(video_path)
        w, h = meta.width, meta.height

        annotated = render_person_preview(frame, persons, selected_idx=None)
        preview_b64 = _encode_frame_bgr(annotated)

        # Auto-select if only one person
        auto_click = None
        status: str
        if len(persons) == 1:
            mid_hip = persons[0]["mid_hip"]
            auto_click = PersonClick(
                x=int(mid_hip[0] * w),
                y=int(mid_hip[1] * h),
            )
            status = "Обнаружен 1 человек — выбран автоматически"
        else:
            status = f"Обнаружено {len(persons)} человек. Выберите на превью или из списка."

        persons_out = [
            PersonInfo(
                track_id=p["track_id"],
                hits=p["hits"],
                bbox=p["bbox"],
                mid_hip=p["mid_hip"],
            )
            for p in persons
        ]

        return DetectResponse(
            persons=persons_out,
            preview_image=preview_b64,
            video_key=video_key,
            auto_click=auto_click,
            status=status,
        )

    finally:
        # Cleanup /tmp file
        if tmp_file:
            Path(tmp_file).unlink(missing_ok=True)
