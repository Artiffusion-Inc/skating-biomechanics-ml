"""arq worker for video processing pipeline.

Run with: uv run python -m src.worker
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import threading
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, ClassVar

from arq import Retry
from arq.connections import RedisSettings

from src.config import get_settings
from src.task_manager import (
    TaskStatus,
    get_valkey_client,
    is_cancelled,
    mark_cancelled,
    store_error,
    store_result,
    update_progress,
)

logger = logging.getLogger(__name__)

_cancel_events: dict[str, threading.Event] = {}


@dataclass
class MLModelFlags:
    """ML model feature flags for video processing."""

    depth: bool = False
    optical_flow: bool = False
    segment: bool = False
    foot_track: bool = False
    matting: bool = False
    inpainting: bool = False


async def startup(ctx: dict[str, Any]) -> None:
    logger.info("Video processing worker starting up")


async def shutdown(ctx: dict[str, Any]) -> None:
    logger.info("Video processing worker shutting down")
    _cancel_events.clear()


async def _poll_cancel(task_id: str, cancel_event: threading.Event) -> None:
    """Poll Valkey cancel signal and set threading event when detected."""
    valkey = await get_valkey_client()
    try:
        while not cancel_event.is_set():
            if await is_cancelled(task_id, valkey=valkey):
                cancel_event.set()
                break
            await asyncio.sleep(0.5)
    finally:
        await valkey.close()


async def _async_update_progress(task_id: str, fraction: float, message: str) -> None:
    valkey = await get_valkey_client()
    try:
        await update_progress(task_id, fraction, message, valkey=valkey)
    finally:
        await valkey.close()


async def process_video_task(
    ctx: dict[str, Any],
    *,
    task_id: str,
    video_path: str,
    person_click: dict[str, int],
    frame_skip: int = 1,
    layer: int = 3,
    tracking: str = "auto",
    export: bool = True,
    ml_flags: MLModelFlags | None = None,
) -> dict[str, Any]:
    """arq task: run process_video_pipeline() with Valkey state tracking."""
    if ml_flags is None:
        ml_flags = MLModelFlags()
    settings = get_settings()
    valkey = await get_valkey_client()

    try:
        now = datetime.now(UTC).isoformat()
        await valkey.hset(
            f"task:{task_id}",
            mapping={"status": TaskStatus.RUNNING, "started_at": now},
        )

        cancel_event = threading.Event()
        _cancel_events[task_id] = cancel_event

        outputs_dir = Path(settings.outputs_dir)
        outputs_dir.mkdir(parents=True, exist_ok=True)
        output_path = str(outputs_dir / f"{Path(video_path).stem}_analyzed.mp4")

        def progress_cb(fraction: float, message: str) -> None:
            try:
                loop = asyncio.get_event_loop()
                loop.call_soon_threadsafe(_async_update_progress, task_id, fraction, message)
            except RuntimeError:
                pass

        poll_task = asyncio.create_task(_poll_cancel(task_id, cancel_event))

        try:
            from src.types import PersonClick
            from src.web_helpers import PipelineCancelled, process_video_pipeline

            click = PersonClick(x=person_click["x"], y=person_click["y"])
            result = await asyncio.to_thread(
                process_video_pipeline,
                video_path=video_path,
                person_click=click,
                frame_skip=frame_skip,
                layer=layer,
                tracking=tracking,
                blade_3d=False,
                export=export,
                output_path=output_path,
                progress_cb=progress_cb,
                cancel_event=cancel_event,
                depth=ml_flags.depth,
                optical_flow=ml_flags.optical_flow,
                segment=ml_flags.segment,
                foot_track=ml_flags.foot_track,
                matting=ml_flags.matting,
                inpainting=ml_flags.inpainting,
            )
        finally:
            poll_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await poll_task

        stats = result["stats"]
        out_path = Path(output_path)
        video_rel = (
            str(out_path.relative_to(outputs_dir))
            if out_path.is_relative_to(outputs_dir)
            else out_path.name
        )
        poses_rel = None
        csv_rel = None
        if result.get("poses_path"):
            pp = Path(result["poses_path"])
            poses_rel = (
                str(pp.relative_to(outputs_dir)) if pp.is_relative_to(outputs_dir) else pp.name
            )
        if result.get("csv_path"):
            cp = Path(result["csv_path"])
            csv_rel = (
                str(cp.relative_to(outputs_dir)) if cp.is_relative_to(outputs_dir) else cp.name
            )

        response_data = {
            "video_path": video_rel,
            "poses_path": poses_rel,
            "csv_path": csv_rel,
            "stats": stats,
            "status": "Analysis complete!",
        }
        await store_result(task_id, response_data, valkey=valkey)
        return response_data

    except PipelineCancelled:
        await mark_cancelled(task_id, valkey=valkey)
        return {"status": "cancelled", "task_id": task_id}

    except Exception as e:
        logger.exception("Pipeline task %s failed", task_id)
        await store_error(task_id, str(e), valkey=valkey)
        error_msg = str(e).lower()
        if any(term in error_msg for term in ["timeout", "connection", "network"]):
            raise Retry(defer=ctx.get("job_try", 1) * 10) from e
        raise

    finally:
        _cancel_events.pop(task_id, None)
        await valkey.close()


_settings = get_settings()


class WorkerSettings:
    """arq worker configuration."""

    queue_name: str = "skating:queue"
    max_jobs: int = _settings.worker_max_jobs
    retry_jobs: bool = True
    retry_delays: ClassVar[list[int]] = _settings.worker_retry_delays

    on_startup = startup
    on_shutdown = shutdown

    functions: ClassVar[list] = [process_video_task]
    cron_jobs: ClassVar[list] = []

    redis_settings = RedisSettings(
        host=_settings.valkey_host,
        port=_settings.valkey_port,
        database=_settings.valkey_db,
        password=_settings.valkey_password,
    )
