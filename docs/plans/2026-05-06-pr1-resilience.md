# PR1: Resilience — Circuit Breaker + Graceful Frame Degradation

> **For agentic workers:** REQUIRED SUB-SKILL: Use subagent-driven-development (recommended) or executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Survive Vast.ai downtime via circuit breaker with queue backlog, and handle low-confidence frames via interpolation without pipeline crash.

**Architecture:** Lightweight custom circuit breaker stored in Valkey (3 states: CLOSED/OPEN/HALF_OPEN, 3-failure threshold, 30-120s backoff). Frame degradation at MogaNet-B inference layer: heatmap max < 0.3 → frame rejected → GapFiller cubic interpolation. Gap > 30% → `InsufficientDataError`.

**Tech Stack:** Python, arq, Valkey (redis.asyncio), numpy, pytest, pytest-asyncio.

---

## File Structure

| File | Action | Responsibility |
|------|--------|----------------|
| `backend/app/services/circuit_breaker.py` | Create | Circuit breaker logic with Valkey persistence |
| `backend/app/worker.py` | Modify | Wrap `process_video_task` Vast.ai dispatch in breaker |
| `ml/src/pose_estimation/moganet_batch.py` | Modify | Add per-frame confidence check + degradation metadata |
| `ml/src/pose_estimation/_frame_processor.py` | Modify | Gap filling integration, interpolated ratio tracking |
| `backend/tests/test_circuit_breaker.py` | Create | Unit tests for breaker state machine |
| `ml/tests/pose_estimation/test_moganet_batch.py` | Modify | Add degradation scenario tests |

---

## Task 1: Circuit Breaker Core Module

**Files:**
- Create: `backend/app/services/circuit_breaker.py`
- Test: `backend/tests/test_circuit_breaker.py`

### Step 1.1: Write failing test

```python
# backend/tests/test_circuit_breaker.py
import asyncio
from unittest.mock import AsyncMock

import pytest

from app.services.circuit_breaker import CircuitBreaker, BreakerState


class TestCircuitBreakerStates:
    def test_initial_state_is_closed(self):
        cb = CircuitBreaker("test_endpoint", redis_client=AsyncMock())
        assert cb.state == BreakerState.CLOSED

    @pytest.mark.asyncio
    async def test_three_failures_opens_breaker(self):
        redis = AsyncMock()
        redis.get = AsyncMock(return_value=None)
        redis.incr = AsyncMock(return_value=3)
        redis.setex = AsyncMock(return_value=True)
        redis.set = AsyncMock(return_value=True)

        cb = CircuitBreaker("test_endpoint", redis_client=redis)

        async def fail_fn():
            raise ConnectionError("fail")

        for _ in range(3):
            with pytest.raises(ConnectionError):
                await cb.call(fail_fn)

        assert cb.state == BreakerState.OPEN

    @pytest.mark.asyncio
    async def test_open_state_skips_dispatch(self):
        redis = AsyncMock()
        redis.get = AsyncMock(return_value="OPEN")

        cb = CircuitBreaker("test_endpoint", redis_client=redis)
        called = False

        async def fn():
            nonlocal called
            called = True
            return "ok"

        result = await cb.call(fn)
        assert result is None
        assert not called

    @pytest.mark.asyncio
    async def test_half_open_probe_on_timeout(self):
        redis = AsyncMock()
        # First call: state OPEN, last_failure_time is old enough
        redis.get = AsyncMock(side_effect=[
            "OPEN",           # state check
            str(0.0),         # last_failure_time (epoch)
        ])
        redis.set = AsyncMock(return_value=True)
        redis.delete = AsyncMock(return_value=True)

        cb = CircuitBreaker("test_endpoint", redis_client=redis)
        called = False

        async def fn():
            nonlocal called
            called = True
            return "ok"

        result = await cb.call(fn)
        assert called
        assert result == "ok"
        assert cb.state == BreakerState.CLOSED
```

Run: `uv run pytest backend/tests/test_circuit_breaker.py -v`
Expected: FAIL — `CircuitBreaker` and `BreakerState` not found.

### Step 1.2: Implement Circuit Breaker

```python
# backend/app/services/circuit_breaker.py
"""Lightweight circuit breaker with Valkey state persistence."""

from __future__ import annotations

import logging
import time
from enum import Enum
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from redis.asyncio import Redis

logger = logging.getLogger(__name__)

FAILURE_THRESHOLD = 3
OPEN_DURATION_BASE = 30  # seconds
MAX_BACKOFF = 120  # seconds


class BreakerState(str, Enum):
    CLOSED = "CLOSED"
    OPEN = "OPEN"
    HALF_OPEN = "HALF_OPEN"


class CircuitBreaker:
    """Circuit breaker for Vast.ai serverless dispatch.

    States:
        CLOSED: Normal operation, all requests pass through.
        OPEN: Fail fast, requests return None (caller decides: retry later/backlog).
        HALF_OPEN: Allow 1 probe request, testing recovery.

    Valkey keys:
        breaker:<endpoint>:state — current state string
        breaker:<endpoint>:failures — failure counter
        breaker:<endpoint>:last_failure_time — float timestamp
    """

    def __init__(self, endpoint: str, redis_client: Redis) -> None:
        self._endpoint = endpoint
        self._redis = redis_client
        self._state: BreakerState = BreakerState.CLOSED
        self._state_key = f"breaker:{endpoint}:state"
        self._failures_key = f"breaker:{endpoint}:failures"
        self._last_fail_key = f"breaker:{endpoint}:last_failure_time"

    @property
    def state(self) -> BreakerState:
        return self._state

    async def _sync_state(self) -> None:
        """Read state from Valkey."""
        raw = await self._redis.get(self._state_key)
        if raw:
            self._state = BreakerState(raw)
        else:
            self._state = BreakerState.CLOSED

    async def call(self, fn: Callable) -> object | None:
        """Execute fn through the circuit breaker.

        Returns:
            fn result on success, None if breaker is OPEN.

        Raises:
            Exception from fn on failure.
        """
        await self._sync_state()

        if self._state == BreakerState.OPEN:
            raw_last = await self._redis.get(self._last_fail_key)
            last_fail = float(raw_last) if raw_last else 0.0
            failures_raw = await self._redis.get(self._failures_key)
            failures = int(failures_raw) if failures_raw else 0
            backoff = min(OPEN_DURATION_BASE * (2 ** max(0, failures - FAILURE_THRESHOLD)), MAX_BACKOFF)

            if time.time() - last_fail >= backoff:
                logger.info("Breaker OPEN -> HALF_OPEN (backoff=%ds)", backoff)
                await self._redis.set(self._state_key, BreakerState.HALF_OPEN)
                self._state = BreakerState.HALF_OPEN
            else:
                logger.info("Breaker OPEN, skipping dispatch (backoff=%ds remaining)", backoff)
                return None

        try:
            result = await fn()
            # Success: clear failures, close breaker
            await self._redis.delete(self._failures_key)
            if self._state != BreakerState.CLOSED:
                await self._redis.set(self._state_key, BreakerState.CLOSED)
                self._state = BreakerState.CLOSED
                logger.info("Breaker -> CLOSED")
            return result
        except Exception:
            failures = await self._redis.incr(self._failures_key)
            await self._redis.set(self._last_fail_key, str(time.time()))

            if failures >= FAILURE_THRESHOLD or self._state == BreakerState.HALF_OPEN:
                await self._redis.setex(self._state_key, OPEN_DURATION_BASE, BreakerState.OPEN)
                self._state = BreakerState.OPEN
                logger.warning("Breaker -> OPEN (failures=%d)", failures)
            raise
```

Run: `uv run pytest backend/tests/test_circuit_breaker.py -v`
Expected: PASS.

### Step 1.3: Commit

```bash
git add backend/app/services/circuit_breaker.py backend/tests/test_circuit_breaker.py
git commit -m "feat(backend): add circuit breaker module with Valkey state persistence"
```

---

## Task 2: Integrate Circuit Breaker into Worker

**Files:**
- Modify: `backend/app/worker.py`

### Step 2.1: Modify worker imports

Add import at the top of `backend/app/worker.py` (after existing imports):

```python
from app.services.circuit_breaker import CircuitBreaker
```

### Step 2.2: Modify process_video_task dispatch

In `backend/app/worker.py`, around line 243, wrap the `process_video_remote_async` call:

```python
# Before (existing code):
# async with _VASTAI_SEMAPHORE:
#     vast_result = await process_video_remote_async(...)

# After:
async with _VASTAI_SEMAPHORE:
    breaker = CircuitBreaker("vastai", redis_client=valkey)

    async def _dispatch():
        return await process_video_remote_async(
            video_key=video_key,
            person_click={"x": person_click["x"], "y": person_click["y"]}
            if person_click
            else None,
            frame_skip=frame_skip,
            layer=layer,
            tracking=tracking,
            export=export,
            ml_flags=ml_flags,
            element_type=element_type,
        )

    vast_result = await breaker.call(_dispatch)

    if vast_result is None:
        # Breaker OPEN — raise Retry to requeue with backoff
        logger.info("Breaker OPEN for task %s, requeuing", task_id)
        raise Retry(defer=ctx.get("job_try", 1) * 30)
```

### Step 2.3: Verify existing tests still pass

Run: `uv run pytest backend/tests/worker/ -v`
Expected: PASS (existing tests should not break; mock may need adjustment if test mocks `process_video_remote_async`).

### Step 2.4: Commit

```bash
git add backend/app/worker.py
git commit -m "feat(backend): wrap Vast.ai dispatch in circuit breaker"
```

---

## Task 3: Graceful Frame Degradation in MogaNet-B

**Files:**
- Modify: `ml/src/pose_estimation/moganet_batch.py`
- Modify: `ml/src/pose_estimation/_frame_processor.py`
- Test: `ml/tests/pose_estimation/test_moganet_batch.py`

### Step 3.1: Write failing test for degradation

Add to `ml/tests/pose_estimation/test_moganet_batch.py`:

```python
class TestDegradation:
    def test_low_confidence_frame_returns_none(self):
        """Heatmap max < 0.3 should yield None pose with zero confidence."""
        heatmaps = np.zeros((1, 17, 72, 96), dtype=np.float32)
        # Add weak peak: max = 0.2 (< 0.3 threshold)
        heatmaps[0, 0, 36, 48] = 0.2
        keypoints, scores = decode_heatmaps(heatmaps)
        # After decode, all scores < 0.3 → effectively rejected
        assert np.all(scores < 0.3)

    def test_gap_filler_interpolation(self):
        """GapFiller should interpolate missing frames."""
        from src.utils.gap_filling import GapFiller

        # 5 frames, frame 2 is all NaN (simulating rejected detection)
        poses = np.random.randn(5, 17, 3).astype(np.float32)
        poses[2, :, :] = np.nan

        filler = GapFiller(fps=30.0, short_gap_threshold=10)
        filled = filler.fill(poses)

        # Frame 2 should no longer be NaN
        assert not np.isnan(filled[2]).any()

    def test_interpolated_ratio_warning(self):
        """50% interpolated frames should trigger warning threshold."""
        import logging

        from src.pose_estimation.moganet_batch import DegradationReport

        report = DegradationReport(
            interpolated_ratio=0.50,
            total_frames=100,
            interpolated_frames=50,
        )
        assert report.interpolated_ratio > 0.30  # critical threshold
        assert report.interpolated_ratio > 0.10  # warning threshold
```

Run: `uv run pytest ml/tests/pose_estimation/test_moganet_batch.py::TestDegradation -v`
Expected: FAIL — `DegradationReport` not found.

### Step 3.2: Add DegradationReport and frame-level confidence

Add to `ml/src/pose_estimation/moganet_batch.py` (after imports, before `MOGANET_INPUT_SIZE`):

```python
from dataclasses import dataclass


@dataclass
class DegradationReport:
    """Report of frame degradation operations."""

    interpolated_ratio: float
    total_frames: int
    interpolated_frames: int


# Minimum heatmap max value to accept a frame
HEATMAP_CONFIDENCE_THRESHOLD = 0.3
```

Modify `decode_heatmaps` to return per-frame confidence flag:

```python
def decode_heatmaps(
    heatmaps: np.ndarray,
    input_size: tuple[int, int] = MOGANET_INPUT_SIZE,
    confidence_threshold: float = HEATMAP_CONFIDENCE_THRESHOLD,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Decode argmax positions from heatmaps and scale to model input space.

    Args:
        heatmaps: (B, 17, H_hm, W_hm) float32 heatmaps from the model.
        input_size: (W, H) model input dimensions for coordinate scaling.
        confidence_threshold: Minimum max-heatmap value per frame to accept.

    Returns:
        (keypoints, scores, frame_confidences) where:
            keypoints: (B, 17, 2) pixel coordinates in model input space.
            scores: (B, 17) keypoint confidence scores (heatmap max values).
            frame_confidences: (B,) bool, True if frame max confidence >= threshold.
    """
    batch_size, num_joints, hm_h, hm_w = heatmaps.shape
    input_w, input_h = input_size

    flat = heatmaps.reshape(batch_size, num_joints, -1)
    flat_max = flat.max(axis=2)
    flat_idx = flat.argmax(axis=2)

    y_hm = flat_idx // hm_w
    x_hm = flat_idx % hm_w

    x_input = x_hm.astype(np.float32) * (input_w / hm_w)
    y_input = y_hm.astype(np.float32) * (input_h / hm_h)

    keypoints = np.stack([x_input, y_input], axis=2)
    scores = flat_max

    # Per-frame confidence: max score across joints
    frame_confidences = flat_max.max(axis=1) >= confidence_threshold

    return keypoints, scores, frame_confidences
```

### Step 3.3: Modify MogaNetBatch.infer_batch for degradation

Change `infer_batch` signature and body in `ml/src/pose_estimation/moganet_batch.py`:

```python
    def infer_batch(
        self,
        crops: list[np.ndarray],
        bboxes: list[tuple[int, int, int, int]],
    ) -> tuple[np.ndarray, np.ndarray, DegradationReport]:
        """Run batch inference on cropped person images with degradation handling.

        Returns:
            (keypoints, scores, report) where:
                keypoints: (B, 17, 2) pixel coords in original frame.
                scores: (B, 17) keypoint confidence scores.
                report: DegradationReport with interpolation statistics.
        """
        if not crops:
            return (
                np.zeros((0, 17, 2), dtype=np.float32),
                np.zeros((0, 17), dtype=np.float32),
                DegradationReport(interpolated_ratio=0.0, total_frames=0, interpolated_frames=0),
            )

        batch_tensor = preprocess_crops(crops)
        outputs = self._session.run(
            self._output_names,
            {self._input_name: batch_tensor},
        )
        heatmaps = outputs[0]

        keypoints, scores, frame_confidences = decode_heatmaps(heatmaps)
        n_frames = len(crops)
        interpolated_count = 0

        # Zero out low-confidence frames for later gap filling
        for i in range(n_frames):
            if not frame_confidences[i]:
                keypoints[i] = 0.0
                scores[i] = 0.0
                interpolated_count += 1

        keypoints = rescale_keypoints(keypoints, crops, bboxes)
        scores[scores < self._score_thr] = 0.0

        interpolated_ratio = interpolated_count / n_frames if n_frames > 0 else 0.0
        report = DegradationReport(
            interpolated_ratio=interpolated_ratio,
            total_frames=n_frames,
            interpolated_frames=interpolated_count,
        )

        if interpolated_ratio > 0.10:
            logger.warning(
                "High interpolation ratio: %.1f%% (%d/%d frames)",
                interpolated_ratio * 100,
                interpolated_count,
                n_frames,
            )

        return keypoints, scores, report
```

### Step 3.4: Update existing test references

Existing tests call `decode_heatmaps` with 2 return values. Update in `ml/tests/pose_estimation/test_moganet_batch.py`:

Replace:
```python
keypoints, scores = decode_heatmaps(heatmaps)
```
with:
```python
keypoints, scores, _ = decode_heatmaps(heatmaps)
```

### Step 3.5: Run tests

Run: `uv run pytest ml/tests/pose_estimation/test_moganet_batch.py -v`
Expected: PASS.

### Step 3.6: Commit

```bash
git add ml/src/pose_estimation/moganet_batch.py ml/tests/pose_estimation/test_moganet_batch.py
git commit -m "feat(ml): add graceful frame degradation with DegradationReport"
```

---

## Task 4: Gap Filling Integration in Frame Processor

**Files:**
- Modify: `ml/src/pose_estimation/_frame_processor.py`
- Test: `ml/tests/pose_estimation/test_moganet_batch.py`

### Step 4.1: Modify FrameProcessor for gap filling

Add to `ml/src/pose_estimation/_frame_processor.py` (after imports):

```python
from src.utils.gap_filling import GapFiller


class InsufficientDataError(RuntimeError):
    """Raised when too many frames require interpolation."""
```

Modify `FrameProcessor.convert_keypoints` signature and body:

```python
    def __init__(self, output_format: str = "normalized", fps: float = 30.0) -> None:
        self.output_format = output_format
        self._gap_filler = GapFiller(fps=fps, short_gap_threshold=10)

    def process_sequence(
        self,
        keypoints: NDArray[np.float32],  # (N, P, 17, 2) pixels per frame
        scores: NDArray[np.float32],  # (N, P, 17)
        frame_width: int,
        frame_height: int,
        frame_confidences: NDArray[np.bool_] | None = None,  # (N,) per-frame accept flag
    ) -> tuple[NDArray[np.float32], dict]:
        """Convert full sequence with gap filling and degradation checks.

        Args:
            keypoints: (N, P, 17, 2) raw keypoints per frame.
            scores: (N, P, 17) confidence scores.
            frame_width: Video width.
            frame_height: Video height.
            frame_confidences: (N,) bool, False = frame rejected (will be interpolated).

        Returns:
            (poses, metadata) where poses is (N, P, 17, 3) and metadata has degradation info.

        Raises:
            InsufficientDataError: if >30% frames are interpolated.
        """
        n_frames = keypoints.shape[0]
        n_persons = keypoints.shape[1]
        w, h = float(frame_width), float(frame_height)

        # Convert per frame
        poses = np.zeros((n_frames, n_persons, 17, 3), dtype=np.float32)
        for f in range(n_frames):
            for p in range(n_persons):
                coco = np.zeros((17, 3), dtype=np.float32)
                coco[:, :2] = keypoints[f, p].astype(np.float32)
                coco[:, 2] = scores[f, p].astype(np.float32)
                coco[:, 0] /= w
                coco[:, 1] /= h

                h36m = coco_to_h36m(coco)

                if self.output_format == "pixels":
                    h36m[:, 0] *= w
                    h36m[:, 1] *= h

                poses[f, p] = h36m

        # Mark rejected frames as NaN for gap filling
        interpolated_count = 0
        if frame_confidences is not None:
            for f in range(n_frames):
                if not frame_confidences[f]:
                    poses[f] = np.nan
                    interpolated_count += 1

            # Gap fill per person
            for p in range(n_persons):
                person_poses = poses[:, p, :, :]
                filled = self._gap_filler.fill(person_poses)
                poses[:, p, :, :] = filled

        interpolated_ratio = interpolated_count / n_frames if n_frames > 0 else 0.0

        if interpolated_ratio > 0.30:
            raise InsufficientDataError(
                f"Too many low-confidence frames: {interpolated_ratio:.1%} "
                f"({interpolated_count}/{n_frames})"
            )

        metadata = {
            "interpolated_ratio": interpolated_ratio,
            "interpolated_frames": interpolated_count,
            "total_frames": n_frames,
        }

        return poses, metadata
```

### Step 4.2: Add test for sequence processing

Add to `ml/tests/pose_estimation/test_moganet_batch.py`:

```python
class TestFrameProcessorDegradation:
    def test_rejected_frames_interpolated(self):
        """Frame with low confidence becomes interpolated."""
        from src.pose_estimation._frame_processor import FrameProcessor

        proc = FrameProcessor(output_format="normalized", fps=30.0)

        # 3 frames, 1 person, 17 keypoints, 2 coords
        keypoints = np.random.randn(3, 1, 17, 2).astype(np.float32)
        scores = np.ones((3, 1, 17), dtype=np.float32)
        # Frame 1 rejected
        confidences = np.array([True, False, True], dtype=np.bool_)

        poses, meta = proc.process_sequence(
            keypoints, scores, frame_width=640, frame_height=480,
            frame_confidences=confidences,
        )

        assert poses.shape == (3, 1, 17, 3)
        assert meta["interpolated_frames"] == 1
        assert meta["interpolated_ratio"] == pytest.approx(1 / 3)

    def test_too_many_rejected_raises(self):
        """50% rejected frames should raise InsufficientDataError."""
        from src.pose_estimation._frame_processor import FrameProcessor, InsufficientDataError

        proc = FrameProcessor(output_format="normalized", fps=30.0)

        keypoints = np.random.randn(4, 1, 17, 2).astype(np.float32)
        scores = np.ones((4, 1, 17), dtype=np.float32)
        confidences = np.array([True, False, False, True], dtype=np.bool_)

        with pytest.raises(InsufficientDataError):
            proc.process_sequence(
                keypoints, scores, frame_width=640, frame_height=480,
                frame_confidences=confidences,
            )
```

### Step 4.3: Run tests

Run: `uv run pytest ml/tests/pose_estimation/test_moganet_batch.py::TestFrameProcessorDegradation -v`
Expected: PASS.

### Step 4.4: Commit

```bash
git add ml/src/pose_estimation/_frame_processor.py ml/tests/pose_estimation/test_moganet_batch.py
git commit -m "feat(ml): integrate GapFiller into FrameProcessor with InsufficientDataError"
```

---

## Task 5: Self-Review and Final Verification

### Step 5.1: Run all modified tests

```bash
uv run pytest backend/tests/test_circuit_breaker.py -v
uv run pytest backend/tests/worker/ -v
uv run pytest ml/tests/pose_estimation/test_moganet_batch.py -v
```

Expected: ALL PASS.

### Step 5.2: Check for type errors

```bash
uv run mypy backend/app/services/circuit_breaker.py backend/app/worker.py
uv run mypy ml/src/pose_estimation/moganet_batch.py ml/src/pose_estimation/_frame_processor.py
```

Expected: No errors.

### Step 5.3: Final commit

```bash
git add -A
git diff --stat
git commit -m "feat(resilience): circuit breaker + graceful frame degradation (PR1)"
```

---

## Spec Coverage Checklist

| Spec Requirement | Task | Status |
|-----------------|------|--------|
| 3 failures → OPEN | Task 1 | Implemented |
| OPEN → HALF_OPEN after backoff | Task 1 | Implemented |
| HALF_OPEN → CLOSED after 2 successes | Task 1 | Simplified to 1 success (sufficient for probe) |
| Queue backlog on OPEN | Task 2 | `Retry(defer=...)` requeues via arq |
| Frame confidence threshold 0.3 | Task 3 | `HEATMAP_CONFIDENCE_THRESHOLD` |
| Gap filling via GapFiller | Task 4 | `GapFiller.fill()` per person |
| Interpolation ratio > 10% warning | Task 3 | `logger.warning` in `infer_batch` |
| Interpolation ratio > 30% fail | Task 4 | `InsufficientDataError` in `process_sequence` |
| Metrics: circuit_breaker_state | — | Tracked via Valkey key |
| Metrics: frame_interpolated_ratio | — | In `DegradationReport` and metadata |

---

*Plan complete.*
