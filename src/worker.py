"""arq worker for video processing pipeline.

Run with: uv run python -m src.worker

Dispatches all processing to Vast.ai Serverless GPU.
No local GPU fallback.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, ClassVar

from arq import Retry
from arq.connections import RedisSettings

from src.config import get_settings
from src.task_manager import (
    TaskStatus,
    get_valkey_client,
    store_error,
    store_result,
)

logger = logging.getLogger(__name__)


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


async def process_video_task(
    ctx: dict[str, Any],
    *,
    task_id: str,
    video_key: str,
    person_click: dict[str, int],
    frame_skip: int = 1,
    layer: int = 3,
    tracking: str = "auto",
    export: bool = True,
    ml_flags: MLModelFlags | None = None,
    session_id: str | None = None,
) -> dict[str, Any]:
    """arq task: dispatch video processing to Vast.ai Serverless GPU."""
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

        from src.backend.database import async_session
        from src.backend.crud.session import get_by_id
        from src.vastai.client import process_video_remote

        # Fetch element_type from session if session_id provided
        element_type = None
        if session_id:
            async with async_session() as db:
                session = await get_by_id(db, session_id)
                if session:
                    element_type = session.element_type

        logger.info("Dispatching task %s to Vast.ai (video_key=%s)", task_id, video_key)
        vast_result = await asyncio.to_thread(
            process_video_remote,
            video_key=video_key,
            person_click={"x": person_click["x"], "y": person_click["y"]},
            frame_skip=frame_skip,
            layer=layer,
            tracking=tracking,
            export=export,
            ml_flags={
                "depth": ml_flags.depth,
                "optical_flow": ml_flags.optical_flow,
                "segment": ml_flags.segment,
                "foot_track": ml_flags.foot_track,
                "matting": ml_flags.matting,
                "inpainting": ml_flags.inpainting,
            },
            element_type=element_type,
        )
        logger.info("Vast.ai processing complete for task %s", task_id)

        response_data = {
            "video_path": vast_result.video_key,
            "poses_path": vast_result.poses_key,
            "csv_path": vast_result.csv_key,
            "stats": vast_result.stats,
            "status": "Analysis complete!",
        }
        await store_result(task_id, response_data, valkey=valkey)

        # Save analysis results to Postgres if session_id was provided
        if session_id and vast_result.metrics:
            try:
                from src.backend.database import async_session
                from src.backend.services.session_saver import save_analysis_results

                async with async_session() as db:
                    await save_analysis_results(
                        db,
                        session_id=session_id,
                        metrics=vast_result.metrics,
                        phases=vast_result.phases,
                        recommendations=vast_result.recommendations or [],
                    )
                    await db.commit()
            except Exception as save_err:
                logger.warning("Failed to save session results: %s", save_err)

        return response_data

    except Exception as e:
        logger.exception("Pipeline task %s failed", task_id)
        await store_error(task_id, str(e), valkey=valkey)
        error_msg = str(e).lower()
        if any(term in error_msg for term in ["timeout", "connection", "network"]):
            raise Retry(defer=ctx.get("job_try", 1) * 10) from e
        raise

    finally:
        await valkey.close()


_settings = get_settings()


class WorkerSettings:
    """arq worker configuration."""

    queue_name: str = "skating:queue"
    max_jobs: int = _settings.app.worker_max_jobs
    retry_jobs: bool = True
    retry_delays: ClassVar[list[int]] = _settings.app.worker_retry_delays

    on_startup = startup
    on_shutdown = shutdown

    functions: ClassVar[list] = [process_video_task]
    cron_jobs: ClassVar[list] = []

    redis_settings = RedisSettings(
        host=_settings.valkey.host,
        port=_settings.valkey.port,
        database=_settings.valkey.db,
        password=_settings.valkey.password.get_secret_value(),
    )
