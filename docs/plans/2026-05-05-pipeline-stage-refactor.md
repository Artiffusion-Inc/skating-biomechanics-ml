# Pipeline Stage Refactor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use subagent-driven-development (recommended) or executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Decompose `AnalysisPipeline.analyze()` (197 lines) and `segment_video()` into stage functions. Eliminate duplication between `analyze` and `segment_video` (both call `_extract_and_track` + normalize + smooth). Make each stage independently testable.

**Architecture:** `analyze()` becomes a thin orchestrator calling `_stage_extract`, `_stage_smooth`, `_stage_detect_phases`, `_stage_metrics`, `_stage_recommend`. `segment_video` reuses `_stage_extract` and `_stage_smooth`. Each stage is a pure function (input in → output out, no side effects) except for profiler recording.

**Tech Stack:** Python 3.11, numpy, pytest.

---

## File Structure

### Files to Create
- `ml/src/pipeline/_stage_extract.py` — pure extraction stage
- `ml/src/pipeline/_stage_smooth.py` — normalization + smoothing stage
- `ml/src/pipeline/_stage_phases.py` — phase detection stage
- `ml/src/pipeline/_stage_metrics.py` — biomechanics metrics stage
- `ml/src/pipeline/_stage_recommend.py` — recommendation stage
- `ml/tests/pipeline/test_stage_extract.py`
- `ml/tests/pipeline/test_stage_smooth.py`
- `ml/tests/pipeline/test_stage_phases.py`
- `ml/tests/pipeline/test_stage_metrics.py`

### Files to Modify
- `ml/src/pipeline.py` — reduce `analyze()` to ~40-line orchestrator, reduce `segment_video()` to reuse stages
- `ml/src/__init__.py` — update if needed

### Files to Delete
- None

---

## Task 1: Create `_stage_extract` module

**Files:**
- Create: `ml/src/pipeline/_stage_extract.py`
- Modify: `ml/src/pipeline.py` (remove extraction from `_extract_and_track`)
- Test: `ml/tests/pipeline/test_stage_extract.py`

**Motivation:** `_extract_and_track` combines extraction, gap filling, and spatial reference. Extract the pure data transformation into a reusable stage.

- [ ] **Step 1: Write the failing test**

```python
# ml/tests/pipeline/test_stage_extract.py
import numpy as np
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

from src.pipeline._stage_extract import run_extract_stage


def test_run_extract_stage_no_video():
    with pytest.raises(RuntimeError, match="Failed to open"):
        run_extract_stage(Path("nonexistent.mp4"), person_click=None)


def test_run_extract_stage_returns_tuple():
    """Mock the extractor to avoid needing a real video."""
    with patch("src.pipeline._stage_extract.get_video_meta") as mock_meta:
        mock_meta.return_value = MagicMock(
            num_frames=10, width=640, height=480, fps=30.0
        )
        with patch("src.pipeline._stage_extract.PoseExtractor") as mock_ext:
            mock_extraction = MagicMock()
            mock_extraction.poses = np.random.rand(10, 17, 3).astype(np.float32)
            mock_extraction.first_detection_frame = 0
            mock_extraction.first_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            mock_extraction.valid_mask.return_value = np.ones(10, dtype=bool)
            mock_ext.return_value.extract_video_tracked.return_value = mock_extraction

            with patch("src.pipeline._stage_extract.GapFiller") as mock_filler:
                mock_filler.return_value.fill_gaps.return_value = (
                    mock_extraction.poses, {}
                )
                result = run_extract_stage(
                    Path("fake.mp4"), person_click=None
                )
                assert result.poses is not None
                assert result.frame_offset == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest ml/tests/pipeline/test_stage_extract.py -v`
Expected: FAIL

- [ ] **Step 3: Write minimal implementation**

```python
# ml/src/pipeline/_stage_extract.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from ..pose_estimation.pose_extractor import PoseExtractor
from ..types import PersonClick
from ..utils.gap_filling import GapFiller
from ..utils.video import VideoMeta, get_video_meta


@dataclass
class ExtractStageResult:
    poses: NDArray[np.float32]  # (N, 17, 3) compensated
    frame_offset: int
    meta: VideoMeta
    first_frame: NDArray[np.uint8] | None


def run_extract_stage(
    video_path: Path,
    person_click: PersonClick | None = None,
    device: str = "auto",
    reestimate_camera: bool = False,
) -> ExtractStageResult:
    """Extract poses with tracking, gap filling, and spatial compensation.

    Args:
        video_path: Path to video file.
        person_click: Optional click to select target person.
        device: Device for inference.
        reestimate_camera: Enable per-frame camera re-estimation.

    Returns:
        ExtractStageResult with compensated poses and metadata.
    """
    meta = get_video_meta(video_path)

    # 1. Tracked extraction
    extractor = PoseExtractor(output_format="normalized", device=device)
    extraction = extractor.extract_video_tracked(video_path, person_click=person_click)

    # 2. Skip pre-roll
    frame_offset = extraction.first_detection_frame
    poses = extraction.poses[frame_offset:]
    valid = extraction.valid_mask()[frame_offset:]
    first_frame = extraction.first_frame

    # 3. Gap filling
    filler = GapFiller()
    filled, _ = filler.fill_gaps(poses, valid)
    filled = GapFiller.interpolate_low_confidence(filled, threshold=0.3)

    # 4. Spatial reference
    if reestimate_camera:
        from ..detection.spatial_reference import (
            compensate_poses_per_frame,
            estimate_pose_sequence,
        )
        camera_poses = estimate_pose_sequence(str(video_path), interval=30, fps=meta.fps)
        compensated = compensate_poses_per_frame(
            filled, camera_poses, video_width=meta.width, video_height=meta.height
        )
    else:
        import cv2
        from ..detection.spatial_reference import CameraPose, SpatialReferenceDetector
        spatial_detector = SpatialReferenceDetector()
        if first_frame is not None:
            camera_pose = spatial_detector.estimate_pose(first_frame)
        else:
            cap = cv2.VideoCapture(str(video_path))
            ret, ff = cap.read()
            camera_pose = spatial_detector.estimate_pose(ff) if ret else CameraPose()
            cap.release()
        if camera_pose.confidence > 0.1:
            from ..detection.spatial_reference import compensate_poses_per_frame
            compensated = compensate_poses_per_frame(filled, [(0, camera_pose)])
        else:
            compensated = filled

    return ExtractStageResult(
        poses=compensated,
        frame_offset=frame_offset,
        meta=meta,
        first_frame=first_frame,
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest ml/tests/pipeline/test_stage_extract.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add ml/src/pipeline/_stage_extract.py ml/tests/pipeline/test_stage_extract.py
git commit -m "feat(pipeline): add _stage_extract module for pure extraction stage"
```

---

## Task 2: Create `_stage_smooth` module

**Files:**
- Create: `ml/src/pipeline/_stage_smooth.py`
- Test: `ml/tests/pipeline/test_stage_smooth.py`

**Motivation:** Normalization and smoothing are pure transformations, testable without video I/O. Also handles the phase-aware smoothing logic for jumps.

- [ ] **Step 1: Write the failing test**

```python
# ml/tests/pipeline/test_stage_smooth.py
import numpy as np
import pytest
from src.pipeline._stage_smooth import run_smooth_stage


def test_smooth_no_phases():
    poses = np.random.rand(100, 17, 3).astype(np.float32)
    result = run_smooth_stage(poses, fps=30.0, element_type="three_turn", manual_phases=None)
    assert result.smoothed.shape == (100, 17, 3)
    assert result.phases is None


def test_smooth_with_manual_phases():
    poses = np.random.rand(100, 17, 3).astype(np.float32)
    from src.types import ElementPhase
    phases = ElementPhase(name="waltz_jump", start=10, takeoff=30, peak=40, landing=60, end=80)
    result = run_smooth_stage(poses, fps=30.0, element_type="waltz_jump", manual_phases=phases)
    assert result.smoothed.shape == (100, 17, 3)
    assert result.phases is not None
    assert result.phases.takeoff == 30
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest ml/tests/pipeline/test_stage_smooth.py -v`
Expected: FAIL

- [ ] **Step 3: Write minimal implementation**

```python
# ml/src/pipeline/_stage_smooth.py
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from ..types import ElementPhase


@dataclass
class SmoothStageResult:
    smoothed: NDArray[np.float32]  # (N, 17, 3)
    phases: ElementPhase | None  # pre-detected phases for jump elements
    normalized: NDArray[np.float32]  # (N, 17, 3) before smoothing


def run_smooth_stage(
    poses: NDArray[np.float32],
    fps: float,
    element_type: str | None = None,
    manual_phases: ElementPhase | None = None,
    enable_smoothing: bool = True,
) -> SmoothStageResult:
    """Normalize and smooth poses. Pre-detects phases for jump elements."""
    from ..pose_estimation.normalizer import PoseNormalizer
    from ..utils.smoothing import OneEuroFilterConfig, PoseSmoother, get_skating_optimized_config

    # Normalization
    normalizer = PoseNormalizer(target_spine_length=0.4)
    normalized = normalizer.normalize(poses)

    pre_phases = None
    if not enable_smoothing:
        config = OneEuroFilterConfig(min_cutoff=100.0, beta=0.0, freq=fps)
        smoother = PoseSmoother(config=config, freq=fps)
        smoothed = smoother.smooth(normalized)
    elif manual_phases is not None:
        boundaries = [manual_phases.takeoff, manual_phases.peak, manual_phases.landing]
        boundaries = [b for b in boundaries if b > 0]
        config = get_skating_optimized_config(fps)
        smoother = PoseSmoother(config=config, freq=fps)
        smoothed = smoother.smooth_phase_aware(normalized, boundaries)
        pre_phases = manual_phases
    elif element_type is not None and _is_jump(element_type):
        from ..analysis.phase_detector import PhaseDetector
        phase_result = PhaseDetector().detect_phases(normalized, fps, element_type)
        pre_phases = phase_result.phases
        boundaries = [pre_phases.takeoff, pre_phases.peak, pre_phases.landing]
        boundaries = [b for b in boundaries if b > 0]
        config = get_skating_optimized_config(fps)
        smoother = PoseSmoother(config=config, freq=fps)
        smoothed = smoother.smooth_phase_aware(normalized, boundaries)
    else:
        config = get_skating_optimized_config(fps)
        smoother = PoseSmoother(config=config, freq=fps)
        smoothed = smoother.smooth(normalized)

    return SmoothStageResult(smoothed=smoothed, phases=pre_phases, normalized=normalized)


def _is_jump(element_type: str) -> bool:
    from ..analysis import element_defs
    return element_defs.is_jump(element_type)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest ml/tests/pipeline/test_stage_smooth.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add ml/src/pipeline/_stage_smooth.py ml/tests/pipeline/test_stage_smooth.py
git commit -m "feat(pipeline): add _stage_smooth module for normalization + smoothing"
```

---

## Task 3: Create `_stage_phases` module

**Files:**
- Create: `ml/src/pipeline/_stage_phases.py`
- Test: `ml/tests/pipeline/test_stage_phases.py`

**Motivation:** Phase detection is a pure function: poses in → phases out. Should be independently testable.

- [ ] **Step 1: Write the failing test**

```python
# ml/tests/pipeline/test_stage_phases.py
import numpy as np
import pytest
from src.pipeline._stage_phases import run_phase_stage


def test_detect_phases_for_jump():
    # Simulate a jump: CoM goes up then down
    poses = np.zeros((100, 17, 3), dtype=np.float32)
    poses[:, 0, 1] = np.concatenate([
        np.linspace(0.5, 0.5, 30),   # approach
        np.linspace(0.5, 0.3, 20), # takeoff (CoM up)
        np.linspace(0.3, 0.5, 20), # landing (CoM down)
        np.linspace(0.5, 0.5, 30), # recovery
    ])
    phases = run_phase_stage(poses, fps=30.0, element_type="waltz_jump")
    assert phases is not None
    assert phases.takeoff > 0
    assert phases.landing > phases.takeoff
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest ml/tests/pipeline/test_stage_phases.py -v`
Expected: FAIL

- [ ] **Step 3: Write minimal implementation**

```python
# ml/src/pipeline/_stage_phases.py
from __future__ import annotations

from ..analysis.phase_detector import PhaseDetector
from ..types import ElementPhase
import numpy as np
from numpy.typing import NDArray


def run_phase_stage(
    smoothed: NDArray[np.float32],
    fps: float,
    element_type: str | None = None,
    pre_phases: ElementPhase | None = None,
    manual_phases: ElementPhase | None = None,
) -> ElementPhase:
    """Detect phases from smoothed poses."""
    if manual_phases is not None:
        return manual_phases
    if pre_phases is not None:
        return pre_phases
    if element_type is not None:
        result = PhaseDetector().detect_phases(smoothed, fps, element_type)
        return result.phases
    return ElementPhase(name="unknown", start=0, takeoff=0, peak=0, landing=0, end=0)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest ml/tests/pipeline/test_stage_phases.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add ml/src/pipeline/_stage_phases.py ml/tests/pipeline/test_stage_phases.py
git commit -m "feat(pipeline): add _stage_phases module for phase detection"
```

---

## Task 4: Create `_stage_metrics` module

**Files:**
- Create: `ml/src/pipeline/_stage_metrics.py`
- Test: `ml/tests/pipeline/test_stage_metrics.py`

**Motivation:** Metrics computation is pure: poses + phases → metrics list. Should be testable without pipeline.

- [ ] **Step 1: Write the failing test**

```python
# ml/tests/pipeline/test_stage_metrics.py
import numpy as np
import pytest
from src.pipeline._stage_metrics import run_metrics_stage
from src.types import ElementPhase


def test_metrics_for_jump():
    poses = np.zeros((100, 17, 3), dtype=np.float32)
    poses[:, 0, 1] = np.concatenate([
        np.linspace(0.5, 0.5, 30),
        np.linspace(0.5, 0.3, 20),
        np.linspace(0.3, 0.5, 20),
        np.linspace(0.5, 0.5, 30),
    ])
    phases = ElementPhase(name="waltz_jump", start=10, takeoff=30, peak=40, landing=60, end=80)
    metrics = run_metrics_stage(poses, phases, fps=30.0, element_type="waltz_jump")
    assert len(metrics) > 0
    metric_names = [m.name for m in metrics]
    assert "airtime" in metric_names or "jump_height" in metric_names
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest ml/tests/pipeline/test_stage_metrics.py -v`
Expected: FAIL

- [ ] **Step 3: Write minimal implementation**

```python
# ml/src/pipeline/_stage_metrics.py
from __future__ import annotations

from ..analysis.element_defs import get_element_def
from ..analysis.metrics import BiomechanicsAnalyzer
from ..types import ElementPhase, MetricResult
from numpy.typing import NDArray
import numpy as np


def run_metrics_stage(
    smoothed: NDArray[np.float32],
    phases: ElementPhase,
    fps: float,
    element_type: str | None = None,
) -> list:
    """Compute biomechanics metrics for the detected element."""
    if element_type is None:
        return []
    element_def = get_element_def(element_type)
    if element_def is None:
        return []
    analyzer = BiomechanicsAnalyzer(element_def=element_def)
    return analyzer.analyze(smoothed, phases, fps)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest ml/tests/pipeline/test_stage_metrics.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add ml/src/pipeline/_stage_metrics.py ml/tests/pipeline/test_stage_metrics.py
git commit -m "feat(pipeline): add _stage_metrics module for biomechanics metrics"
```

---

## Task 5: Create `_stage_recommend` module

**Files:**
- Create: `ml/src/pipeline/_stage_recommend.py`
- Test: `ml/tests/pipeline/test_stage_recommend.py`

**Motivation:** Recommendation generation is pure: metrics + element_type → recommendations.

- [ ] **Step 1: Write the failing test**

```python
# ml/tests/pipeline/test_stage_recommend.py
import pytest
from src.pipeline._stage_recommend import run_recommend_stage


def test_recommend_for_jump():
    metrics = []
    recommendations = run_recommend_stage(metrics, element_type="waltz_jump")
    assert isinstance(recommendations, list)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest ml/tests/pipeline/test_stage_recommend.py -v`
Expected: FAIL

- [ ] **Step 3: Write minimal implementation**

```python
# ml/src/pipeline/_stage_recommend.py
from __future__ import annotations

from ..analysis.recommender import Recommender


def run_recommend_stage(
    metrics: list,
    element_type: str | None = None,
) -> list:
    """Generate Russian text recommendations from metrics."""
    if element_type is None:
        return []
    recommender = Recommender()
    return recommender.recommend(metrics, element_type)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest ml/tests/pipeline/test_stage_recommend.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add ml/src/pipeline/_stage_recommend.py ml/tests/pipeline/test_stage_recommend.py
git commit -m "feat(pipeline): add _stage_recommend module for recommendation generation"
```

---

## Task 6: Refactor `AnalysisPipeline.analyze()` to use stages

**Files:**
- Modify: `ml/src/pipeline.py`

**Motivation:** Reduce `analyze()` from 197 lines to ~40-line orchestrator. All heavy logic delegated to stage modules.

- [ ] **Step 1: Replace `analyze()` body**

Replace the method body (lines 189-385) with:

```python
def analyze(
    self,
    video_path: Path,
    element_type: str | None = None,
    manual_phases: ElementPhase | None = None,
    reference_path: Path | None = None,
) -> AnalysisReport:
    from .analysis import element_defs
    from .pipeline._stage_extract import run_extract_stage
    from .pipeline._stage_metrics import run_metrics_stage
    from .pipeline._stage_phases import run_phase_stage
    from .pipeline._stage_recommend import run_recommend_stage
    from .pipeline._stage_smooth import run_smooth_stage

    element_def = None
    if element_type is not None:
        element_def = element_defs.get_element_def(element_type)
        if element_def is None:
            raise ValueError(f"Unknown element type: {element_type}")

    # Stage 1: Extract
    t0 = time.perf_counter()
    extract_result = run_extract_stage(
        video_path,
        person_click=self._person_click,
        device=self._device_config.device,
        reestimate_camera=self._reestimate_camera,
    )
    self._profiler.record("extract_and_track", time.perf_counter() - t0)

    # Stage 2: Smooth
    t0 = time.perf_counter()
    smooth_result = run_smooth_stage(
        extract_result.poses,
        fps=extract_result.meta.fps,
        element_type=element_type,
        manual_phases=manual_phases,
        enable_smoothing=self._enable_smoothing,
    )
    self._profiler.record("smooth", time.perf_counter() - t0)

    if element_type is not None and element_def is not None:
        # Stage 3: Phases
        t0 = time.perf_counter()
        phases = run_phase_stage(
            smooth_result.smoothed,
            fps=extract_result.meta.fps,
            element_type=element_type,
            pre_phases=smooth_result.phases,
            manual_phases=manual_phases,
        )
        self._profiler.record("phase_detection", time.perf_counter() - t0)

        # Stage 4: Metrics
        t0 = time.perf_counter()
        metrics = run_metrics_stage(smooth_result.smoothed, phases, extract_result.meta.fps, element_type)
        self._profiler.record("metrics", time.perf_counter() - t0)

        # Stage 5: DTW alignment
        t0 = time.perf_counter()
        dtw_distance = self._run_dtw(smooth_result.smoothed, phases, element_type)
        self._profiler.record("dtw_alignment", time.perf_counter() - t0)

        # Stage 6: Recommendations
        t0 = time.perf_counter()
        recommendations = run_recommend_stage(metrics, element_type)
        self._profiler.record("recommendations", time.perf_counter() - t0)

        overall_score = self._compute_overall_score(metrics)

        return AnalysisReport(
            element_type=element_type,
            phases=phases,
            metrics=metrics,
            recommendations=recommendations,
            overall_score=overall_score if overall_score is not None else 0.0,
            dtw_distance=dtw_distance if dtw_distance is not None else 0.0,
            blade_summary_left={},
            blade_summary_right={},
            physics={},
            profiling=self._profiler.to_dict(),
        )

    # No element type — poses only
    return AnalysisReport(
        element_type=element_type or "unknown",
        phases=ElementPhase(name="unknown", start=0, takeoff=0, peak=0, landing=0, end=0),
        metrics=[],
        recommendations=[],
        overall_score=0.0,
        dtw_distance=0.0,
        blade_summary_left={},
        blade_summary_right={},
        physics={},
        profiling=self._profiler.to_dict(),
    )
```

- [ ] **Step 2: Extract `_run_dtw` helper**

Move DTW alignment block (lines 319-329) into:

```python
def _run_dtw(self, smoothed, phases, element_type: str) -> float | None:
    if self._reference_store is None:
        return None
    reference = self._reference_store.get_best_match(element_type)
    if reference is None:
        return None
    aligner = self._get_aligner()
    return aligner.compute_distance(
        smoothed[phases.start : phases.end],
        reference.poses[reference.phases.start : reference.phases.end],
    )
```

- [ ] **Step 3: Refactor `segment_video()` to use `_stage_extract` and `_stage_smooth`**

Replace `segment_video()` body (lines 387-423) with:

```python
def segment_video(self, video_path: Path) -> SegmentationResult:
    from .pipeline._stage_extract import run_extract_stage
    from .pipeline._stage_smooth import run_smooth_stage

    # Stage 1-2: Extract
    extract_result = run_extract_stage(
        video_path,
        person_click=self._person_click,
        device=self._device_config.device,
        reestimate_camera=self._reestimate_camera,
    )

    # Stage 3: Smooth
    smooth_result = run_smooth_stage(
        extract_result.poses,
        fps=extract_result.meta.fps,
        enable_smoothing=self._enable_smoothing,
    )

    # Stage 4: Segment
    from .analysis.element_segmenter import ElementSegmenter
    segmenter = ElementSegmenter()
    return segmenter.segment(smooth_result.smoothed, video_path, extract_result.meta)
```

- [ ] **Step 4: Remove `_extract_and_track` method**

Delete `_extract_and_track()` (lines 100-187) since logic moved to `_stage_extract`.

- [ ] **Step 5: Run existing tests**

Run: `uv run pytest ml/tests/ -v -k "pipeline"`
Expected: All pipeline tests pass.

- [ ] **Step 6: Commit**

```bash
git add ml/src/pipeline.py ml/src/pipeline/_stage_*.py ml/tests/pipeline/
git commit -m "refactor(pipeline): decompose analyze() and segment_video() into stage modules"
```

---

## Self-Review

1. **Spec coverage:**
   - `analyze()` decomposition → Tasks 1-6
   - `segment_video()` reuse → Task 6
   - Stage modules → `_stage_extract`, `_stage_smooth`, `_stage_phases`, `_stage_metrics`, `_stage_recommend`
   - All stages pure functions, independently testable

2. **Placeholder scan:** Clean.

3. **Type consistency:**
   - `ExtractStageResult.poses` = same shape as old `compensated_h36m`
   - `SmoothStageResult.smoothed` = same as old `smoothed`
   - `ElementPhase` used consistently across stages
   - `AnalysisReport` construction unchanged

4. **No duplication:** `segment_video` now reuses `_stage_extract` and `_stage_smooth` instead of duplicating `_extract_and_track` + normalize + smooth.

---

## Execution Handoff

**Plan complete and saved to `docs/plans/2026-05-05-pipeline-stage-refactor.md`.**

Two execution options:

**1. Subagent-Driven (recommended)** — Fresh subagent per task + review loop. Commit after every step. All tests green before next Wave.

**2. Inline Execution** — Execute tasks in this session using executing-plans.

**Which approach?**
