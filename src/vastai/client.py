"""Client for calling Vast.ai Serverless GPU endpoint.

Flow:
  1. Upload input video to R2
  2. POST /route to get worker URL from Vast.ai
  3. POST /process to the worker with R2 key + credentials
  4. Download result video from R2
  5. Cleanup R2 input object
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path

import httpx

from src.config import get_settings
from src.storage import delete_object, download_file, upload_file

logger = logging.getLogger(__name__)

ROUTE_URL = "https://run.vast.ai/route/"
REQUEST_TIMEOUT = 600  # 10 min for video processing
ROUTE_TIMEOUT = 30


@dataclass
class VastResult:
    video_path: str
    poses_path: str | None
    csv_path: str | None
    stats: dict


def _get_worker_url(endpoint_name: str, api_key: str) -> str:
    """Route request to get a ready worker URL."""
    resp = httpx.post(
        ROUTE_URL,
        headers={"Authorization": f"Bearer {api_key}"},
        json={"endpoint": endpoint_name},
        timeout=ROUTE_TIMEOUT,
    )
    resp.raise_for_status()
    data = resp.json()
    return data["url"]


def process_video_remote(
    video_path: str,
    person_click: dict[str, int] | None = None,
    frame_skip: int = 1,
    layer: int = 3,
    tracking: str = "auto",
    export: bool = True,
    ml_flags: dict[str, bool] | None = None,
    output_path: str = "",
) -> VastResult:
    """Send video processing to Vast.ai Serverless GPU.

    Raises httpx.HTTPStatusError on routing/processing failures.
    """
    settings = get_settings()
    if ml_flags is None:
        ml_flags = {}

    api_key = settings.vastai_api_key
    endpoint_name = settings.vastai_endpoint_name

    # 1. Upload input video to R2
    r2_key = f"input/{Path(video_path).stem}_{int(time.time())}.mp4"
    logger.info("Uploading video to R2: %s", r2_key)
    upload_file(video_path, r2_key)

    try:
        # 2. Route to worker
        logger.info("Routing to Vast.ai endpoint: %s", endpoint_name)
        worker_url = _get_worker_url(endpoint_name, api_key)
        logger.info("Worker URL: %s", worker_url)

        # 3. Send processing request
        payload = {
            "video_r2_key": r2_key,
            "person_click": person_click,
            "frame_skip": frame_skip,
            "layer": layer,
            "tracking": tracking,
            "export": export,
            "ml_flags": ml_flags,
            "r2_endpoint_url": settings.cf_r2_endpoint_url,
            "r2_access_key_id": settings.cf_r2_access_key_id,
            "r2_secret_access_key": settings.cf_r2_secret_access_key,
            "r2_bucket": settings.cf_r2_bucket,
        }
        resp = httpx.post(
            f"{worker_url}/process",
            json=payload,
            timeout=REQUEST_TIMEOUT,
        )
        resp.raise_for_status()
        result = resp.json()

        # 4. Download results from R2
        output_dir = Path(output_path).parent if output_path else Path(settings.outputs_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        stem = Path(video_path).stem

        result_video = str(output_dir / f"{stem}_analyzed.mp4")
        download_file(result["video_r2_key"], result_video)

        poses_path = None
        if result.get("poses_r2_key"):
            poses_path = str(output_dir / f"{stem}_poses.npy")
            download_file(result["poses_r2_key"], poses_path)

        csv_path = None
        if result.get("csv_r2_key"):
            csv_path = str(output_dir / f"{stem}_biomechanics.csv")
            download_file(result["csv_r2_key"], csv_path)

        return VastResult(
            video_path=result_video,
            poses_path=poses_path,
            csv_path=csv_path,
            stats=result["stats"],
        )

    finally:
        # Cleanup input from R2
        try:
            delete_object(r2_key)
        except Exception:
            logger.warning("Failed to cleanup R2 key: %s", r2_key)
