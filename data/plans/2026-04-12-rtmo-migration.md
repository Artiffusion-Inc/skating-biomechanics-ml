# RTMO Migration & HALPE26 Removal Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace RTMPose (HALPE26, 26kp) with RTMO (COCO 17kp) and remove all foot keypoint code.

**Architecture:** RTMO outputs COCO 17 keypoints directly — no HALPE26 intermediate step needed. Convert COCO 17 → H3.6M 17 using existing `_coco_to_h36m_single()` in `h36m.py`. Remove `halpe26.py` entirely. Remove `foot_keypoints` field from `TrackedExtraction` and all downstream consumers.

**Tech Stack:** rtmlib (Body class for RTMO), numpy, COCO 17kp format

**Key insight:** RTMO uses `Body` class (not `BodyWithFeet`). rtmlib's `PoseTracker` works identically with both — one-stage (RTMO, no detection needed) vs two-stage (RTMPose). API is the same: `keypoints, scores = tracker(frame)`.

---

### Task 1: Make `coco_to_h36m` public in h36m.py

**Files:**
- Modify: `ml/skating_ml/pose_estimation/h36m.py:106`

- [ ] **Step 1: Rename `_coco_to_h36m_single` to `coco_to_h36m`**

The function at line 106 is the COCO 17 → H3.6M 17 converter. It's currently private (underscore prefix). Make it public since it will become the primary conversion path (replacing `halpe26_to_h36m`).

```python
# Change line 106 from:
def _coco_to_h36m_single(coco_pose: np.ndarray) -> np.ndarray:
# To:
def coco_to_h36m(coco_pose: np.ndarray) -> np.ndarray:
```

- [ ] **Step 2: Update the internal reference in halpe26.py**

In `ml/skating_ml/pose_estimation/halpe26.py`, the `halpe26_to_h36m` function duplicates this logic. This file will be deleted in Task 5, so no need to update it.

- [ ] **Step 3: Run existing tests to verify nothing breaks**

Run: `cd /home/michael/Github/skating-biomechanics-ml/.claude/worktrees/dataset-unification && uv run python -m pytest ml/tests/ -x -q 2>&1 | tail -20`
Expected: All tests pass (no internal callers of the renamed function)

- [ ] **Step 4: Commit**

```bash
git add ml/skating_ml/pose_estimation/h36m.py
git commit -m "refactor(pose): make coco_to_h36m public for RTMO migration"
```

---

### Task 2: Remove `foot_keypoints` from `TrackedExtraction`

**Files:**
- Modify: `ml/skating_ml/types.py:810-835`

- [ ] **Step 1: Remove the `foot_keypoints` field from `TrackedExtraction`**

In `ml/skating_ml/types.py`, remove lines 833-835:
```python
    # DELETE these lines:
    foot_keypoints: np.ndarray | None = (
        None  # (N, 6, 3) [L_Heel, L_BigToe, L_SmallToe, R_Heel, R_BigToe, R_SmallToe]
    )
```

- [ ] **Step 2: Update the class docstring**

Change the docstring (lines 812-825) to remove mention of foot keypoints:
```python
@dataclass
class TrackedExtraction:
    """Pose extraction result with multi-person tracking support.

    Holds a pose sequence for a single tracked person across video frames.
    Missing frames (gaps) are represented as NaN values in the poses array.

    Attributes:
        poses: Pose array of shape (N, 17, 3) with NaN for missing frames.
        frame_indices: Real frame numbers, shape (N,), monotonically increasing.
        first_detection_frame: Index of first frame with a valid detection
            (pre-roll boundary, used by gap filler).
        target_track_id: The locked-on track ID, or None if tracking not used.
        fps: Video frame rate.
        video_meta: Full video metadata.
    """
```

- [ ] **Step 3: Run tests to find all broken references**

Run: `cd /home/michael/Github/skating-biomechanics-ml/.claude/worktrees/dataset-unification && uv run python -m pytest ml/tests/ -x -q 2>&1 | tail -20`
Expected: Some tests fail due to `foot_keypoints` references — this is expected, will be fixed in later tasks.

- [ ] **Step 4: Commit**

```bash
git add ml/skating_ml/types.py
git commit -m "refactor(types): remove foot_keypoints from TrackedExtraction"
```

---

### Task 3: Rewrite extractor to use RTMO (COCO 17)

**Files:**
- Modify: `ml/skating_ml/pose_estimation/rtmlib_extractor.py`

This is the core change. Replace `BodyWithFeet` → `Body` (RTMO), remove HALPE26 intermediate format, remove all foot keypoint handling.

- [ ] **Step 1: Update imports and module docstring**

Replace lines 1-46 with:
```python
"""RTMO-based pose extractor using rtmlib.

Uses the rtmlib PoseTracker with Body model (RTMO) to extract 17-keypoint
COCO poses.  The output is converted to H3.6M 17-keypoint format for the
analysis pipeline.

Architecture:
    Video → rtmlib PoseTracker (Body/RTMO) → COCO 17kp → H3.6M 17kp

Key advantages over RTMPose BodyWithFeet:
    - 2x faster inference (one-stage: detection + pose in single pass)
    - Same COCO 17 accuracy
    - Simpler pipeline (no HALPE26 intermediate format)
    - ONNX Runtime inference (no PyTorch dependency)

References:
    - rtmlib: https://github.com/Tau-J/rtmlib
    - RTMO: https://arxiv.org/abs/2307.00689
"""

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np
from tqdm import tqdm

if TYPE_CHECKING:
    from rtmlib import Body, PoseTracker
else:
    try:
        from rtmlib import Body, PoseTracker
    except ImportError:
        PoseTracker = None  # type: ignore[assignment]
        Body = None  # type: ignore[assignment]

from ..detection.pose_tracker import PoseTracker as CustomPoseTracker
from ..tracking.skeletal_identity import compute_2d_skeletal_ratios
from ..tracking.tracklet_merger import TrackletMerger, build_tracklets
from ..types import PersonClick, TrackedExtraction
from ..utils.video import get_video_meta
from .h36m import _biometric_distance, coco_to_h36m

logger = logging.getLogger(__name__)
```

- [ ] **Step 2: Update class docstring and `__init__`**

Replace the class (lines 51-128) with:
```python
class RTMPoseExtractor:
    """COCO 17kp pose extractor using rtmlib RTMO Body model.

    Provides H3.6M 17-keypoint poses using RTMO (one-stage detection+pose).
    Uses rtmlib's built-in tracking for multi-person handling.

    Args:
        mode: Model preset — ``"lightweight"`` (fast), ``"balanced"``
            (default), ``"performance"`` (accurate).
        tracking_backend: ``"rtmlib"`` uses rtmlib's built-in tracker;
            ``"custom"`` feeds detections into our PoseTracker
            (OC-SORT + biometric Re-ID).
        conf_threshold: Minimum keypoint confidence to accept [0, 1].
        output_format: ``"normalized"`` for [0, 1] coords. ``"pixels"``
            for absolute pixel coords.
        det_frequency: Ignored for RTMO (one-stage, no separate detection).
        frame_skip: Process every Nth frame for pose estimation (1 = every
            frame). Higher values = faster but less accurate. Skipped
            frames are filled with NaN for downstream interpolation.
        device: ``"cpu"`` or ``"cuda"``.
        backend: Inference backend — ``"onnxruntime"`` or ``"opencv"``.
    """

    def __init__(
        self,
        mode: str = "balanced",
        tracking_backend: str = "rtmlib",
        tracking_mode: str = "auto",
        conf_threshold: float = 0.3,
        output_format: str = "normalized",
        det_frequency: int = 1,
        frame_skip: int = 1,
        device: str = "auto",
        backend: str = "onnxruntime",
    ) -> None:
        if PoseTracker is None:
            raise ImportError("rtmlib is not installed. Install with: uv add rtmlib")

        self._mode = mode
        self._tracking_backend = tracking_backend
        self._tracking_mode = tracking_mode
        self._conf_threshold = conf_threshold
        self._output_format = output_format
        self._det_frequency = det_frequency
        self._frame_skip = max(1, frame_skip)
        self._device = device
        self._backend = backend

        # Resolve device via DeviceConfig for consistent GPU-first behavior
        if device == "auto":
            from ..device import DeviceConfig

            self._device = DeviceConfig(device="auto").device

        # Lazy-initialised on first call
        self._tracker: PoseTracker | None = None

    @property
    def tracker(self):
        """Lazy-initialise rtmlib PoseTracker on first access."""
        if self._tracker is None:
            if Body is None:
                raise ImportError("rtmlib Body (RTMO) model not available")
            from rtmlib import Body as RTMBody
            from rtmlib import PoseTracker as RTMPoseTracker

            self._tracker = RTMPoseTracker(
                RTMBody,
                tracking=True,
                tracking_thr=0.3,
                mode=self._mode,
                to_openpose=False,
                backend=self._backend,
                device=self._device,
            )
        return self._tracker
```

Note: `det_frequency` is kept as a parameter for backward compatibility but ignored (RTMO is one-stage).

- [ ] **Step 3: Rewrite `extract_video_tracked` to use COCO 17 directly**

Replace the entire `extract_video_tracked` method body. Key changes:
- Remove `all_feet` allocation (line 166)
- Remove `foot_kps_list` (line 289)
- Replace HALPE26 construction (lines 295-298) with COCO 17 handling
- Replace `halpe26_to_h36m(halpe26)` with `coco_to_h36m(coco_pose)`
- Remove `extract_foot_keypoints(halpe26)` calls
- Remove `foot_keypoints=all_feet` from return value (line 553)
- Remove all `all_feet[...]` assignments throughout

The core per-person loop changes from:
```python
# OLD (HALPE26 26kp):
kp = keypoints[p].astype(np.float32)  # (26, 2) pixels
conf = scores[p].astype(np.float32)  # (26,)
halpe26 = np.zeros((26, 3), dtype=np.float32)
halpe26[:, :2] = kp
halpe26[:, 2] = conf
halpe26[:, 0] /= w
halpe26[:, 1] /= h
h36m = halpe26_to_h36m(halpe26)
foot = extract_foot_keypoints(halpe26)
```

To:
```python
# NEW (COCO 17kp from RTMO):
kp = keypoints[p].astype(np.float32)  # (17, 2) pixels
conf = scores[p].astype(np.float32)  # (17,)
coco = np.zeros((17, 3), dtype=np.float32)
coco[:, :2] = kp
coco[:, 2] = conf
coco[:, 0] /= w
coco[:, 1] /= h
h36m = coco_to_h36m(coco)
```

Also update `frame_track_data` type from `dict[int, dict[int, tuple[np.ndarray, np.ndarray]]]` to `dict[int, dict[int, np.ndarray]]` (no more foot_kps tuple). All places that store/retrieve from `frame_track_data` need updating — remove the foot_kps component.

Update `TrackedExtraction` return to remove `foot_keypoints=all_feet`.

- [ ] **Step 4: Rewrite `preview_persons` to use COCO 17 directly**

Same changes as Step 3 but for the preview method. Replace HALPE26 construction with COCO 17, use `coco_to_h36m`.

- [ ] **Step 5: Update `extract_rtmpose_poses` convenience function**

Update docstring to remove foot_keypoints mention. The function body stays the same.

- [ ] **Step 6: Run tests**

Run: `cd /home/michael/Github/skating-biomechanics-ml/.claude/worktrees/dataset-unification && uv run python -m pytest ml/tests/test_tracked_extraction.py -x -v 2>&1 | tail -30`
Expected: May fail due to foot_keypoints assertions in tests (fixed in Task 7).

- [ ] **Step 7: Commit**

```bash
git add ml/skating_ml/pose_estimation/rtmlib_extractor.py
git commit -m "feat(pose): switch from RTMPose/BodyWithFeet to RTMO/Body (COCO 17kp)"
```

---

### Task 4: Remove foot keypoints from Tracklet and TrackletMerger

**Files:**
- Modify: `ml/skating_ml/tracking/tracklet_merger.py`

- [ ] **Step 1: Remove `foot_keypoints` from Tracklet dataclass**

In `ml/skating_ml/tracking/tracklet_merger.py`, remove line 31:
```python
# DELETE:
    foot_keypoints: dict[int, np.ndarray] = field(default_factory=dict)
```

- [ ] **Step 2: Update `build_tracklets` to remove foot_kps handling**

Replace `build_tracklets` function (lines 64-91):
```python
def build_tracklets(
    frame_track_data: dict[int, dict[int, np.ndarray]],
) -> list[Tracklet]:
    """Build Tracklet objects from frame_track_data.

    Args:
        frame_track_data: {frame_idx: {track_id: pose (17,3)}}

    Returns:
        List of Tracklet objects, one per unique track_id.
    """
    track_data: dict[int, list[tuple[int, np.ndarray]]] = {}
    for frame_idx, tid_map in frame_track_data.items():
        for tid, pose in tid_map.items():
            track_data.setdefault(tid, []).append((frame_idx, pose))

    tracklets: list[Tracklet] = []
    for tid, entries in track_data.items():
        entries.sort(key=lambda x: x[0])
        tracklets.append(
            Tracklet(
                track_id=tid,
                frames=[e[0] for e in entries],
                poses={e[0]: e[1] for e in entries},
            )
        )
    return tracklets
```

- [ ] **Step 3: Update `TrackletMerger.merge` to remove foot_keypoints**

Replace line 178:
```python
# OLD:
    foot_keypoints={**target.foot_keypoints, **match.foot_keypoints},
# NEW: remove the line entirely
```

- [ ] **Step 4: Commit**

```bash
git add ml/skating_ml/tracking/tracklet_merger.py
git commit -m "refactor(tracking): remove foot_keypoints from Tracklet"
```

---

### Task 5: Remove foot keypoints from visualization

**Files:**
- Modify: `ml/skating_ml/visualization/skeleton/drawer.py`
- Modify: `ml/skating_ml/visualization/pipeline.py`

- [ ] **Step 1: Remove `foot_keypoints` parameter from `draw_skeleton`**

In `ml/skating_ml/visualization/skeleton/drawer.py`, line 54, remove the `foot_keypoints` parameter from the function signature.

- [ ] **Step 2: Remove foot keypoint drawing logic**

Remove lines 141-143 (the `if foot_keypoints is not None` block that calls `_draw_foot_keypoints`).

- [ ] **Step 3: Delete `_draw_foot_keypoints` function and foot constants**

Delete the `_draw_foot_keypoints` function (lines 415-460+) and the foot index constants at the top of the file (around lines 407-412: `_FOOT_HEEL_L`, `_FOOT_BIG_TOE_L`, etc.).

- [ ] **Step 4: Remove `foot_kps` from VizPipeline**

In `ml/skating_ml/visualization/pipeline.py`:
- Remove `foot_kps` field from `VizPipeline` dataclass (line 52)
- Remove `foot_kps` from `RenderState` dataclass (line 269)
- Remove all `foot_kp` / `raw_foot_kps` logic in `prepare()` and `render_frame()` methods (lines 353-386, 431)

- [ ] **Step 5: Commit**

```bash
git add ml/skating_ml/visualization/
git commit -m "refactor(viz): remove foot keypoints from skeleton drawing and viz pipeline"
```

---

### Task 6: Clean up web_helpers.py

**Files:**
- Modify: `ml/skating_ml/web_helpers.py`

- [ ] **Step 1: Remove `foot_kps` reference**

In `ml/skating_ml/web_helpers.py`, line 309, remove the `foot_kps=prepared.foot_kps` argument. Search for any other `foot_kps` or `foot_keypoints` references in the file and remove them.

- [ ] **Step 2: Commit**

```bash
git add ml/skating_ml/web_helpers.py
git commit -m "refactor(web): remove foot_keypoints from detect endpoint"
```

---

### Task 7: Delete `halpe26.py` and clean up dead imports

**Files:**
- Delete: `ml/skating_ml/pose_estimation/halpe26.py`
- Modify: `ml/skating_ml/pose_estimation/__init__.py`

- [ ] **Step 1: Delete halpe26.py**

```bash
git rm ml/skating_ml/pose_estimation/halpe26.py
```

- [ ] **Step 2: Update `__init__.py` docstring**

In `ml/skating_ml/pose_estimation/__init__.py`, update the module docstring:
```python
"""Pose estimation module for figure skating analysis.

This module provides H3.6M 17-keypoint pose extraction as the primary format.
Uses RTMO via rtmlib (COCO 17kp) as the sole backend.

Architecture:
    Video -> RTMPoseExtractor (rtmlib Body/RTMO) -> H3.6M 17kp
"""
```

- [ ] **Step 3: Remove geometry.py foot angle functions**

In `ml/skating_ml/utils/geometry.py`, remove the foot angle functions (around lines 233-370):
- `compute_foot_angles` function
- `compute_foot_angle_single` function
- The comment block `# Foot angle functions for blade edge detection (HALPE26 foot keypoints)`

Keep all other geometry functions (angles, distances, etc.).

- [ ] **Step 4: Commit**

```bash
git add ml/skating_ml/pose_estimation/ ml/skating_ml/utils/geometry.py
git commit -m "refactor(pose): delete halpe26.py, remove foot angle functions"
```

---

### Task 8: Update docstrings across pipeline, CLI, and ML docs

**Files:**
- Modify: `ml/skating_ml/pipeline.py`
- Modify: `ml/skating_ml/cli.py`
- Modify: `ml/CLAUDE.md`

- [ ] **Step 1: Update pipeline.py docstrings**

In `ml/skating_ml/pipeline.py`:
- Line 5: Change `RTMPoseExtractor (rtmlib BodyWithFeet)` → `RTMPoseExtractor (rtmlib RTMO Body)`
- Line 9: Change `RTMPoseExtractor.extract_video_tracked()` stays the same
- Line 44: Change `17 keypoints, normalized [0,1], rtmlib backend` stays the same
- Line 370-378: Update `_get_pose_2d_extractor` docstring if it mentions BodyWithFeet

- [ ] **Step 2: Update cli.py docstrings**

In `ml/skating_ml/cli.py`:
- Line 6: Change `RTMPoseExtractor (rtmlib BodyWithFeet)` → `RTMPoseExtractor (rtmlib RTMO Body)`
- Line 413: Change `RTMPose (rtmlib, HALPE26 26kp)` → `RTMO (rtmlib, COCO 17kp)`

- [ ] **Step 3: Update ml/CLAUDE.md**

In `ml/CLAUDE.md`:
- Change `rtmlib_extractor.py # RTMPose via rtmlib (HALPE26, 26kp) — PRIMARY` → `rtmlib_extractor.py # RTMO via rtmlib (COCO 17kp) — PRIMARY`
- Change `halpe26.py # HALPE26 26kp format + foot angles` → remove this line (file deleted)
- Update Pipeline Flow diagram: remove `HALPE26 26kp` mention, change to `RTMO (COCO 17kp)`
- Remove `Key advantages over YOLO26-Pose` section mentioning foot keypoints

- [ ] **Step 4: Commit**

```bash
git add ml/skating_ml/pipeline.py ml/skating_ml/cli.py ml/CLAUDE.md
git commit -m "docs: update docstrings for RTMO migration"
```

---

### Task 9: Update tests

**Files:**
- Modify: `ml/tests/test_tracked_extraction.py`
- Modify: `ml/tests/test_types.py`
- Modify: `ml/tests/tracking/test_tracklet_merger.py`
- Modify: `ml/tests/visualization/test_drawer.py`
- Modify: `ml/tests/visualization/test_pipeline.py`
- Modify: `ml/tests/test_pipeline.py`
- Delete: `ml/tests/test_coco_builder.py` (if it only tests HALPE26 building)

- [ ] **Step 1: Fix test_tracked_extraction.py**

Remove any assertions checking for `foot_keypoints` in `TrackedExtraction`. Remove mock data that includes foot keypoints.

- [ ] **Step 2: Fix test_types.py**

Remove any test that creates `TrackedExtraction` with `foot_keypoints` parameter.

- [ ] **Step 3: Fix test_tracklet_merger.py**

Update `build_tracklets` calls to pass `np.ndarray` instead of `tuple[np.ndarray, np.ndarray]`. Remove foot_keypoints assertions.

- [ ] **Step 4: Fix test_drawer.py**

Remove `foot_keypoints` parameter from `draw_skeleton` test calls.

- [ ] **Step 5: Fix test_pipeline.py (visualization)**

Remove `foot_kps` from VizPipeline construction.

- [ ] **Step 6: Fix test_pipeline.py (main)**

Remove any `foot_keypoints` assertions on `TrackedExtraction` results.

- [ ] **Step 7: Delete test_coco_builder.py if HALPE26-only**

If `test_coco_builder.py` only tests HALPE26 annotation building, delete it:
```bash
git rm ml/tests/test_coco_builder.py
```

- [ ] **Step 8: Run full test suite**

Run: `cd /home/michael/Github/skating-biomechanics-ml/.claude/worktrees/dataset-unification && uv run python -m pytest ml/tests/ -x -q 2>&1 | tail -20`
Expected: All tests pass.

- [ ] **Step 9: Commit**

```bash
git add ml/tests/
git commit -m "test: update tests for RTMO migration, remove foot keypoints"
```

---

### Task 10: Lint and type check

**Files:** None (verification only)

- [ ] **Step 1: Run ruff lint**

Run: `cd /home/michael/Github/skating-biomechanics-ml/.claude/worktrees/dataset-unification && uv run ruff check ml/skating_ml/ 2>&1`
Expected: No errors.

- [ ] **Step 2: Fix any lint errors if found**

- [ ] **Step 3: Run basedpyright**

Run: `cd /home/michael/Github/skating-biomechanics-ml/.claude/worktrees/dataset-unification && uv run basedpyright ml/skating_ml/ --level error 2>&1 | tail -20`
Expected: No errors.

- [ ] **Step 4: Commit any lint/type fixes**

```bash
git add -A
git commit -m "fix(lint): address lint and type errors from RTMO migration"
```

---

## Summary of Changes

| File | Action | What changes |
|------|--------|-------------|
| `ml/skating_ml/pose_estimation/rtmlib_extractor.py` | **Rewrite** | BodyWithFeet → Body (RTMO), HALPE26 → COCO 17, remove foot kp |
| `ml/skating_ml/pose_estimation/halpe26.py` | **Delete** | Entire file (HALPE26Key, halpe26_to_h36m, extract_foot_keypoints) |
| `ml/skating_ml/pose_estimation/h36m.py` | **Modify** | Make `coco_to_h36m` public |
| `ml/skating_ml/pose_estimation/__init__.py` | **Modify** | Update docstring |
| `ml/skating_ml/types.py` | **Modify** | Remove `foot_keypoints` from TrackedExtraction |
| `ml/skating_ml/tracking/tracklet_merger.py` | **Modify** | Remove foot_keypoints from Tracklet |
| `ml/skating_ml/visualization/skeleton/drawer.py` | **Modify** | Remove foot kp drawing |
| `ml/skating_ml/visualization/pipeline.py` | **Modify** | Remove foot_kps |
| `ml/skating_ml/web_helpers.py` | **Modify** | Remove foot_kps |
| `ml/skating_ml/utils/geometry.py` | **Modify** | Remove foot angle functions |
| `ml/skating_ml/pipeline.py` | **Modify** | Update docstrings |
| `ml/skating_ml/cli.py` | **Modify** | Update docstrings |
| `ml/CLAUDE.md` | **Modify** | Update docs |
| `ml/tests/*` | **Modify/Delete** | Update tests, delete test_coco_builder.py |
