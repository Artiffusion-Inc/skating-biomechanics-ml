# Prod Patterns Design: Resilience, Observability & Optimization

**Date:** 2026-05-06  
**Scope:** 7 production-grade patterns for ML pipeline + async GPU dispatch  
**Target:** MVP stage (100–10 000 video jobs/day), Vast.ai Serverless only

---

## 1. Overview

The pipeline processes figure skating videos via RTMPose → H3.6M conversion → biomechanics analysis. All GPU work dispatches to Vast.ai Serverless; there is no local GPU fallback. The 7 patterns address three categories:

| Category | Patterns | Problem Solved |
|----------|----------|----------------|
| **Core (Observability & Stability)** | Correlation ID, Schema Version, Idempotent Job Keys | Traceability, backward compatibility, duplicate prevention |
| **Resilience** | Circuit Breaker, Graceful Frame Degradation | Vast.ai downtime, bad frames killing pipelines |
| **Optimization** | VRAM-Aware Batch Sizing, Warm Pool | GPU cost reduction, cold-start elimination |

---

## 2. Architecture

```
Frontend → Litestar API → arq Worker → [Valkey Queue] → Vast.ai GPU (FastAPI) → R2
              ↑              ↑                ↑
              |              |                |
        Correlation ID   Idempotent         Schema Version
        (structlog)      Job Keys           (S3 metadata)
                         (PG+Valkey)        Warm Pool
                         Circuit            (GPU startup)
                         Breaker
                         (Worker)
                         VRAM-Aware
                         Batch
                         (GPU Server)
                         Graceful Frame
                         Degradation
                         (MogaNet-B
                         extractor)
```

**Key constraint:** No local GPU fallback. Worker must survive Vast.ai unavailability via queue backlog.

---

## 3. Data Flow

### 3.1 Correlation ID

1. Frontend sets `X-Request-ID` (or API generates `uuid4()`)
2. Litestar middleware binds `correlation_id` to `structlog` contextvars
3. Worker `enqueue_job(correlation_id=...)` propagates through arq
4. GPU server reads `X-Request-ID` from HTTP request headers
5. All logs searchable by single ID across API / worker / GPU

### 3.2 Idempotent Job Keys

1. Upload complete → `job_key = hash(video_key + schema_version + params_json)`
2. Worker checks `PG job_dedup` table → hit → return cached `task_id`
3. Miss → `INSERT INTO job_dedup` + `SETEX task:dedup:<key> <task_id> TTL` + enqueue arq
4. TTL = `task_ttl_seconds × 2` (survives task completion)
5. Cleanup: cron deletes rows older than 7 days

### 3.3 Circuit Breaker

1. Worker dispatches Vast.ai → fail → `INCR breaker:<endpoint>:failures`
2. Failures ≥ 3 → `SETEX breaker:<endpoint>:state OPEN 30s`
3. OPEN state → skip dispatch, job remains in arq queue (backlog)
4. After 30s → HALF_OPEN → 1 probe request
5. Probe success × 2 → `DEL breaker:*` → CLOSED
6. Probe fail → OPEN again, backoff 30s → 60s → 120s max

### 3.4 Graceful Frame Degradation

1. MogaNet-B inference frame N → heatmap max < 0.3 or no detection → `pose = None`
2. Post-processor (GapFiller) → cubic interpolation from neighbors
3. Interpolated frames marked `confidence: interpolated`
4. Gap > 10% → warning log, pipeline continues
5. Gap > 30% → fail with `InsufficientDataError`, client gets actionable message
6. All frames fail → fail early (no GPU waste)

### 3.5 Schema Version

1. GPU server injects `"_schema_version": int(os.environ["SCHEMA_VERSION"])` into JSON result
2. S3 upload includes `Metadata={"schema-version": str(version)}`
3. API `/sessions/{id}` reads metadata via `head_object`
4. Version 3 → parse with `rotation_speed`. Version 2 → parse without `rotation_speed`
5. Unknown version → best-effort parse + warning (forward compatibility)

### 3.6 VRAM-Aware Batch Sizing

1. GPU server receives batch → try `ort_session.run(None, {input: batch})`
2. `RuntimeError` OOM → `batch_size = batch_size // 2` → retry
3. Floor `batch_size = 1` (single frame guaranteed)
4. Success → cache `optimal_batch_size` in Valkey per `model_version`
5. Next request starts from cached optimal
6. Prometheus metrics: `batch_size_oom_total`, `optimal_batch_size`

### 3.7 Warm Pool

1. GPU server startup → `on_event("startup")` → create ONNX session
2. Dummy inference on random tensor (warmup CUDA/cuDNN)
3. `/ready` returns 200 only after warmup complete
4. Vast.ai LB routes traffic only to `/ready` = 200 instances
5. `/warm` optional endpoint for extended warmup stages
6. Prometheus: `model_load_duration_seconds`

---

## 4. Component Design

### 4.1 Correlation ID (`structlog` + `X-Request-ID`)

**Litestar middleware:**
```python
from structlog.contextvars import bind_contextvars

@app.before_request
async def bind_correlation_id(request):
    cid = request.headers.get("X-Request-ID", str(uuid4()))
    bind_contextvars(correlation_id=cid)
    request.state.correlation_id = cid
```

**Worker propagation:**
```python
await redis.enqueue_job(
    "process_video",
    correlation_id=request.state.correlation_id,
    ...
)
```

**GPU server middleware:**
```python
@app.middleware("http")
async def correlation_middleware(request, call_next):
    cid = request.headers.get("X-Request-ID", str(uuid4()))
    logger.info("Request start", extra={"correlation_id": cid})
    response = await call_next(request)
    response.headers["X-Request-ID"] = cid
    return response
```

**Key point:** No new dependencies. `structlog` already configured in `backend/app/logging_config.py`.

### 4.2 Idempotent Job Keys (`job_dedup` table + Valkey)

**PG schema:**
```sql
CREATE TABLE job_dedup (
    job_key TEXT PRIMARY KEY,
    task_id TEXT NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX idx_job_dedup_created_at ON job_dedup(created_at);
```

**Worker logic (pseudocode):**
```python
async def process_video_task(ctx, video_key, params):
    job_key = compute_hash(video_key, SCHEMA_VERSION, params)
    
    # Check cache first (fast)
    cached = await redis.get(f"task:dedup:{job_key}")
    if cached:
        return {"task_id": cached, "cached": True}
    
    # Check persistent store
    row = await pg.fetchrow("SELECT task_id FROM job_dedup WHERE job_key=$1", job_key)
    if row:
        await redis.setex(f"task:dedup:{job_key}", task_ttl * 2, row["task_id"])
        return {"task_id": row["task_id"], "cached": True}
    
    # New job
    task_id = generate_uuid()
    await pg.execute("INSERT INTO job_dedup(job_key, task_id) VALUES ($1, $2)", job_key, task_id)
    await redis.setex(f"task:dedup:{job_key}", task_ttl * 2, task_id)
    
    # Proceed with actual processing...
```

**Race condition:** concurrent workers both see cache miss. One INSERT succeeds, other gets `IntegrityError` → retry SELECT → return existing task_id.

**Cleanup cron (weekly):**
```sql
DELETE FROM job_dedup WHERE created_at < NOW() - INTERVAL '7 days';
```

### 4.3 Circuit Breaker (custom lightweight)

**States:**
- `CLOSED` — normal operation, all requests go through
- `OPEN` — fail fast, no dispatch, jobs accumulate in queue
- `HALF_OPEN` — 1 probe request allowed, testing recovery

**Valkey keys:**
- `breaker:<endpoint>:state` — "CLOSED", "OPEN", "HALF_OPEN"
- `breaker:<endpoint>:failures` — integer counter
- `breaker:<endpoint>:last_failure_time` — timestamp for backoff calculation

**Logic:**
```python
async def dispatch_with_breaker(ctx, endpoint, job_data):
    state = await redis.get(f"breaker:{endpoint}:state") or "CLOSED"
    
    if state == "OPEN":
        last_fail = await redis.get(f"breaker:{endpoint}:last_failure_time")
        backoff = min(30 * (2 ** retries), 120)  # max 120s
        if time.time() - float(last_fail) < backoff:
            logger.info("Breaker OPEN, skipping dispatch")
            return  # job stays in queue
        # Transition to HALF_OPEN
        await redis.set(f"breaker:{endpoint}:state", "HALF_OPEN")
        state = "HALF_OPEN"
    
    try:
        result = await dispatch_to_vastai(endpoint, job_data)
        # Success
        await redis.delete(f"breaker:{endpoint}:failures")
        await redis.set(f"breaker:{endpoint}:state", "CLOSED")
        return result
    except Exception:
        failures = await redis.incr(f"breaker:{endpoint}:failures")
        await redis.set(f"breaker:{endpoint}:last_failure_time", str(time.time()))
        
        if failures >= 3 or state == "HALF_OPEN":
            await redis.setex(f"breaker:{endpoint}:state", 30, "OPEN")
        raise
```

**Metrics:** `circuit_breaker_state{endpoint="vastai"}`, `circuit_breaker_failures_total`

### 4.4 Graceful Frame Degradation (MogaNet-B extractor)

**Confidence threshold:** heatmap max < 0.3 → frame rejected.

**Pipeline integration:**
```python
def process_frame(crop, model_session):
    heatmaps = model_session.run(None, {"input": crop})[0]
    max_conf = heatmaps.max()
    
    if max_conf < 0.3:
        return None, 0.0  # marker for gap filling
    
    keypoints = decode_heatmaps(heatmaps)
    return keypoints, max_conf

# Batch processing
poses, confidences = [], []
for frame in frames:
    pose, conf = process_frame(frame, session)
    poses.append(pose)
    confidences.append(conf)

# Gap filling
poses = gap_filler.fill(poses, max_gap=10)  # max 10 consecutive frames
interpolated_ratio = sum(1 for c in confidences if c == 0) / len(confidences)

if interpolated_ratio > 0.30:
    raise InsufficientDataError(f"Too many low-confidence frames: {interpolated_ratio:.1%}")
elif interpolated_ratio > 0.10:
    logger.warning("High interpolation ratio: %.1f%%", interpolated_ratio)
```

**Result annotation:**
```json
{
  "poses": [...],
  "frame_metadata": [
    {"frame": 0, "confidence": 0.95, "source": "inference"},
    {"frame": 1, "confidence": 0.0, "source": "interpolated"}
  ],
  "interpolated_ratio": 0.15
}
```

### 4.5 Schema Version (env var + S3 metadata)

**GPU server env:**
```bash
SCHEMA_VERSION=3
MODEL_VERSION=moganet-b-v1.2
```

**Result JSON (always includes `_schema_version`):**
```json
{
  "_schema_version": 3,
  "_model_version": "moganet-b-v1.2",
  "poses": [...],
  "metrics": {
    "airtime": 0.5,
    "rotation_speed": 180.0  // NEW in v3
  }
}
```

**S3 upload:**
```python
s3_client.put_object(
    Bucket=bucket,
    Key=key,
    Body=json_bytes,
    Metadata={"schema-version": "3", "model-version": "moganet-b-v1.2"}
)
```

**API reader (backward compat):**
```python
async def get_session_result(session_id):
    head = await s3_client.head_object(Bucket=bucket, Key=f"results/{session_id}.json")
    version = int(head["Metadata"].get("schema-version", "1"))
    
    body = await s3_client.get_object(Bucket=bucket, Key=f"results/{session_id}.json")
    data = json.loads(body["Body"].read())
    
    return parse_result(data, version)

def parse_result(data, version):
    if version >= 3:
        return ResultV3.from_dict(data)
    elif version == 2:
        return ResultV2.from_dict(data)
    else:
        logger.warning("Unknown schema version %d, best-effort parse", version)
        return ResultV3.from_dict(data)  # forward compat: ignore unknown fields
```

### 4.6 VRAM-Aware Batch Sizing

**Exponential backoff:**
```python
async def infer_batch(crops, model_session, model_version):
    cached = await redis.get(f"optimal_batch:{model_version}")
    batch_size = int(cached) if cached else 32
    
    while batch_size >= 1:
        try:
            batches = [crops[i:i+batch_size] for i in range(0, len(crops), batch_size)]
            results = []
            for batch in batches:
                tensor = preprocess(batch)
                out = model_session.run(None, {"input": tensor})
                results.extend(out)
            await redis.set(f"optimal_batch:{model_version}", batch_size)
            return results
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                logger.warning("OOM at batch_size=%d, halving", batch_size)
                batch_size //= 2
                continue
            raise
    
    raise RuntimeError("Cannot process even single frame")
```

**Metrics:**
- `batch_size_oom_total{model_version="moganet-b-v1.2"}` — counter
- `optimal_batch_size{model_version="moganet-b-v1.2"}` — gauge
- `inference_batch_size{model_version="moganet-b-v1.2"}` — histogram

### 4.7 Warm Pool

**Startup sequence:**
```python
@app.on_event("startup")
async def warmup():
    global _model_ready
    _model_ready = False
    
    # Stage 1: Load model
    model_path = os.environ["MODEL_PATH"]
    session = ort.InferenceSession(model_path, providers=["CUDAExecutionProvider"])
    
    # Stage 2: Warmup inference
    dummy = np.random.randn(1, 3, 288, 384).astype(np.float32)
    _ = session.run(None, {"input": dummy})
    
    _model_ready = True
    logger.info("GPU warmup complete")

@app.get("/ready")
async def ready():
    if not _model_ready:
        raise HTTPException(status_code=503, detail="Model not ready")
    providers = ort.get_available_providers()
    if "CUDAExecutionProvider" not in providers:
        raise HTTPException(status_code=503, detail="CUDA not available")
    return {"status": "ready", "providers": providers}
```

**Vast.ai LB config:** readiness probe hits `/ready` every 5s, 2 failures → remove from pool.

---

## 5. Error Handling

| Pattern | Failure Mode | Mitigation |
|---------|--------------|------------|
| Correlation ID | Header missing | Generate `uuid4()` at ingress |
| Job Keys | Concurrent INSERT race | `IntegrityError` → retry SELECT |
| Circuit Breaker | State lost on worker restart | State in Valkey, not worker memory |
| Frame Degradation | Gap > 30% | `InsufficientDataError` with frame analysis |
| Schema Version | Unknown version | Best-effort parse + warning |
| VRAM Batch | Process abort on OOM | Container restart (Vast.ai), next request fresh |
| Warm Pool | Warmup > 60s | Split `/ready` (stage 1) and `/warm` (stage 2) |

---

## 6. Testing Strategy

| Pattern | Test Type | What to Verify |
|---------|-----------|--------------|
| Correlation ID | Integration | `X-Request-ID` propagates API → worker → GPU mock |
| Job Keys | Integration + concurrent | Same `job_key` returns same `task_id` under load |
| Circuit Breaker | Unit (mocked Valkey) | 3 failures → OPEN, probe → CLOSED, backoff doubling |
| Frame Degradation | Unit (synthetic data) | 50% low-confidence frames → interpolation, 90% → error |
| Schema Version | Unit | v2 JSON parsed by v3 reader, unknown version → warning |
| VRAM Batch | Unit (mocked OOM) | batch_size=8 OOM → retry 4 → success, optimal cached |
| Warm Pool | Integration | `/ready` 503 before startup, 200 after, LB behavior |

---

## 7. Implementation Phases

### PR1: Resilience (Circuit Breaker + Graceful Degradation)

**Files:**
- `backend/app/worker.py` — add breaker dispatch wrapper
- `backend/app/services/circuit_breaker.py` — new module
- `ml/src/pose_estimation/moganet_batch.py` — add confidence threshold + gap filling
- `ml/src/pose_estimation/_frame_processor.py` — degradation hooks
- `backend/tests/test_circuit_breaker.py` — unit tests
- `ml/tests/pose_estimation/test_moganet_batch.py` — degradation tests

**Goal:** Pipeline survives Vast.ai downtime and bad frames.

### PR2: Core (Correlation ID + Schema Version + Idempotent Keys)

**Files:**
- `backend/app/main.py` — correlation middleware
- `backend/app/worker.py` — propagate correlation_id
- `backend/app/task_manager.py` — add dedup helpers
- `backend/alembic/versions/` — new migration `job_dedup` table
- `ml/gpu_server/server.py` — inject `_schema_version`, S3 metadata
- `backend/app/storage.py` — `head_object` with metadata read
- `backend/app/schemas.py` — versioned result parsers

**Goal:** Full observability and stability.

### PR3: Optimization (VRAM Batch + Warm Pool)

**Files:**
- `ml/gpu_server/server.py` — batch sizing loop, startup warmup
- `ml/gpu_server/` — readiness probe config
- `ml/src/pose_estimation/moganet_batch.py` — optimal batch cache
- `infra/Containerfile.gpu` — Vast.ai LB readiness probe

**Goal:** GPU utilization 100%, zero cold-start.

---

## 8. Metrics & Monitoring

| Metric | Type | Where | Alert |
|--------|------|-------|-------|
| `correlation_id_missing_total` | Counter | API | >0/hour |
| `job_dedup_hit_total` | Counter | Worker | — |
| `circuit_breaker_state` | Gauge | Worker | =1 (OPEN) for >5min |
| `frame_interpolated_ratio` | Histogram | GPU Server | p99 > 0.20 |
| `schema_version_unknown_total` | Counter | API | >0/hour |
| `batch_size_oom_total` | Counter | GPU Server | >10/hour |
| `optimal_batch_size` | Gauge | GPU Server | — |
| `model_load_duration_seconds` | Histogram | GPU Server | >30s |
| `warmup_ready_timestamp` | Gauge | GPU Server | — |

---

## 9. Dependencies

**No new production dependencies.** All patterns use existing stack:
- `structlog` — already in `backend/app/logging_config.py`
- `redis.asyncio` — already used in `task_manager.py`
- `aiobotocore` — already used in `storage.py`
- `onnxruntime` — already used in `moganet_batch.py`

**Optional dev/test:**
- `pytest-asyncio` — for concurrent job key tests
- `fakeredis` — for circuit breaker unit tests without Valkey

---

## 10. Risks & Trade-offs

| Risk | Impact | Mitigation |
|------|--------|------------|
| Job dedup table grows unbounded | PG disk | Weekly cron cleanup, `job_key` is 64-char hash |
| Circuit breaker false positive | Jobs stuck in queue | Threshold 3 failures (not 1), HALF_OPEN probes |
| Frame degradation hides real bugs | Lower accuracy | Interpolation ratio in result, alert at >10% |
| Schema version drift | Forward compat break | Best-effort parse for unknown versions |
| VRAM batch cache stale | Wrong optimal after model update | Cache keyed by `model_version` |
| Warm pool adds startup time | Vast.ai considers unhealthy | `/ready` returns 200 after stage 1 (model load) |

---

*End of design document.*
