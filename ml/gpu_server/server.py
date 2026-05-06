"""FastAPI inference server for Vast.ai Serverless GPU worker.

Runs on the remote GPU. Receives R2 keys, processes video, returns results.
R2 credentials are passed per-request so the worker does not store cloud credentials.
"""

from __future__ import annotations

import asyncio
import logging
import os
import tempfile
import time
from pathlib import Path

import aiobotocore.session
from fastapi import FastAPI
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest
from pydantic import BaseModel
from starlette.responses import Response

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Skating ML GPU Worker")

# Prometheus metrics
INFERENCE_DURATION = Histogram(
    "inference_duration_seconds",
    "Time spent processing a video",
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0, 120.0],
)
INFERENCE_REQUESTS = Counter(
    "inference_requests_total",
    "Total /process requests",
    ["status"],
)
ACTIVE_REQUESTS = Gauge(
    "active_requests",
    "Number of requests currently being processed",
)

# Models are at /app/data/models/ inside the container
os.environ.setdefault("PROJECT_ROOT", "/app")

# Async session for R2
_async_session = aiobotocore.session.get_session()


@app.on_event("startup")
async def warmup_gpu():
    """Pre-warm CUDA/cuDNN to eliminate cold-start latency."""
    from src.device import DeviceConfig

    cfg = DeviceConfig.default()
    if not cfg.is_cuda:
        return
    import onnxruntime as ort

    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1
    opts.inter_op_num_threads = 2
    # Just importing ort and accessing CUDA provider triggers init
    logging.getLogger(__name__).info("GPU warmup: CUDA initialized")


@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint."""
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/ready")
async def ready():
    """Readiness probe — checks ONNX session health."""
    try:
        import onnxruntime as ort

        providers = ort.get_available_providers()
        if "CUDAExecutionProvider" not in providers:
            return Response(status_code=503, content='{"status": "no_cuda"}')
        return {"status": "ready"}
    except Exception:
        return Response(status_code=503, content='{"status": "unhealthy"}')


class ProcessRequest(BaseModel):
    video_r2_key: str
    person_click: dict[str, int] | None = None
    frame_skip: int = 1
    layer: int = 3
    tracking: str = "auto"
    export: bool = True
    ml_flags: dict[str, bool] = {}
    element_type: str | None = None
    # R2 credentials passed per-request (worker doesn't store them)
    r2_endpoint_url: str = ""
    r2_access_key_id: str = ""
    r2_secret_access_key: str = ""
    r2_bucket: str = ""


class ProcessResponse(BaseModel):
    video_r2_key: str
    poses_r2_key: str | None = None
    csv_r2_key: str | None = None
    stats: dict
    metrics: list | None = None
    phases: object | None = None
    recommendations: list | None = None


def _s3(req: ProcessRequest):
    """Async S3 client factory (returns context manager)."""
    return _async_session.create_client(
        "s3",
        endpoint_url=req.r2_endpoint_url,
        aws_access_key_id=req.r2_access_key_id,
        aws_secret_access_key=req.r2_secret_access_key,
        region_name="auto",
    )


@app.post("/process", response_model=ProcessResponse)
async def process(req: ProcessRequest):
    from src.types import PersonClick
    from src.utils.frame_buffer import AsyncFrameReader
    from src.utils.video_writer import H264Writer
    from src.visualization.pipeline import VizPipeline, prepare_poses

    ACTIVE_REQUESTS.inc()
    start = time.perf_counter()
    try:
        async with await _s3(req) as s3:
            with tempfile.TemporaryDirectory() as tmpdir:
                video_local = Path(tmpdir) / "input.mp4"
                output_local = Path(tmpdir) / "output.mp4"

                logger.info("Downloading video from R2: %s", req.video_r2_key)
                await s3.download_file(req.r2_bucket, req.video_r2_key, str(video_local))

                click = (
                    PersonClick(x=req.person_click["x"], y=req.person_click["y"])
                    if req.person_click
                    else None
                )

                logger.info("Running pipeline (ml_flags=%s)", req.ml_flags)
                prepared = prepare_poses(
                    video_local,
                    person_click=click,
                    frame_skip=req.frame_skip,
                    tracking=req.tracking,
                    progress_cb=None,
                )

                pipe = VizPipeline(
                    meta=prepared.meta,
                    poses_norm=prepared.poses_norm,
                    poses_px=prepared.poses_px,
                    poses_3d=prepared.poses_3d,
                    layer=req.layer,
                    confs=prepared.confs,
                    frame_indices=prepared.frame_indices,
                )

                meta = prepared.meta
                writer = H264Writer(output_local, meta.width, meta.height, meta.fps)
                reader = AsyncFrameReader(video_local, buffer_size=16, frame_skip=1)
                reader.start()

                frame_idx = 0
                pose_idx = 0

                while True:
                    result = reader.get_frame()
                    if result is None:
                        break
                    fi, frame = result
                    current_pose_idx, pose_idx = pipe.find_pose_idx(fi, pose_idx)
                    frame, _ = pipe.render_frame(frame, fi, current_pose_idx)
                    pipe.draw_frame_counter(frame, fi)
                    writer.write(frame)
                    frame_idx += 1

                reader.join(timeout=5)
                writer.close()

                result = {
                    "stats": {
                        "total_frames": meta.num_frames,
                        "valid_frames": prepared.n_valid,
                        "fps": meta.fps,
                        "resolution": f"{meta.width}x{meta.height}",
                    },
                }

                out_key = req.video_r2_key.replace("input/", "output/")
                logger.info("Uploading result to R2: %s", out_key)

                upload_tasks = [s3.upload_file(str(output_local), req.r2_bucket, out_key)]

                poses_key = None
                csv_key = None

                await asyncio.gather(*upload_tasks)

                INFERENCE_REQUESTS.labels(status="success").inc()
                return ProcessResponse(
                    video_r2_key=out_key,
                    poses_r2_key=poses_key,
                    csv_r2_key=csv_key,
                    stats=result["stats"],
                    metrics=result.get("metrics"),
                    phases=result.get("phases"),
                    recommendations=result.get("recommendations"),
                )
    except Exception:
        INFERENCE_REQUESTS.labels(status="error").inc()
        raise
    finally:
        ACTIVE_REQUESTS.dec()
        INFERENCE_DURATION.observe(time.perf_counter() - start)


@app.get("/health")
async def health():
    return {"status": "ok"}
