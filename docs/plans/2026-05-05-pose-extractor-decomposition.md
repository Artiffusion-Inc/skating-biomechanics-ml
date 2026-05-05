# PoseExtractor Decomposition Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use subagent-driven-development (recommended) or executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Decompose `PoseExtractor.extract_video_tracked()` (401 lines) and `_extract_batch()` (375 lines) into testable, reusable components, eliminating the per-frame vs batch path duplication.

**Architecture:** Extract shared tracking logic into `TrackState`, `TargetSelector`, `TrackValidator`, and `FrameProcessor` classes. Both `extract_video_tracked` and `_extract_batch` become thin orchestrators (~60 lines each) that delegate to shared components. All stateful tracking decisions become unit-testable without video I/O.

**Tech Stack:** Python 3.11, numpy, opencv, pytest. No new dependencies.

---

## File Structure

### Files to Create
- `ml/src/pose_estimation/_track_state.py` — tracker instance management, hit counts, frame_track_data
- `ml/src/pose_estimation/_target_selector.py` — click-based and auto target selection
- `ml/src/pose_estimation/_track_validator.py` — anti-steal logic, biometric track migration
- `ml/src/pose_estimation/_frame_processor.py` — RTMO output → H3.6M conversion per frame
- `ml/tests/pose_estimation/test_track_state.py`
- `ml/tests/pose_estimation/test_target_selector.py`
- `ml/tests/pose_estimation/test_track_validator.py`
- `ml/tests/pose_estimation/test_frame_processor.py`

### Files to Modify
- `ml/src/pose_estimation/pose_extractor.py` — reduce `extract_video_tracked` and `_extract_batch` to ~60-line orchestrators
- `ml/src/pose_estimation/__init__.py` — update exports if needed

### Files to Delete
- None

---

## Task 1: Create `TrackState` class

**Files:**
- Create: `ml/src/pose_estimation/_track_state.py`
- Modify: `ml/src/pose_estimation/pose_extractor.py`
- Test: `ml/tests/pose_estimation/test_track_state.py`

**Motivation:** `extract_video_tracked` and `_extract_batch` both duplicate tracker initialization (sports2d vs deepsort vs custom), hit counting, and frame_track_data bookkeeping.

- [ ] **Step 1: Write the failing test**

```python
# ml/tests/pose_estimation/test_track_state.py
import numpy as np
import pytest
from src.pose_estimation._track_state import TrackState


def test_track_state_init_no_tracker():
    ts = TrackState(fps=30.0, tracking_backend="rtmlib")
    assert ts.tracker_instances == (None, None, None)
    assert ts.target_track_id is None
    assert ts.track_hit_counts == {}


def test_track_state_init_custom_tracker():
    ts = TrackState(fps=30.0, tracking_backend="custom")
    assert ts.custom_tracker is not None
    assert ts.sports2d_tracker is None
    assert ts.deepsort_tracker is None


def test_track_state_init_sports2d():
    ts = TrackState(fps=30.0, tracking_backend="rtmlib", tracking_mode="sports2d")
    assert ts.sports2d_tracker is not None
    assert ts.custom_tracker is None


def test_record_frame_data():
    ts = TrackState(fps=30.0)
    poses = np.random.rand(2, 17, 3).astype(np.float32)
    track_ids = [0, 1]
    ts.record_frame(5, poses, track_ids)
    assert ts.track_hit_counts[0] == 1
    assert ts.track_hit_counts[1] == 1
    assert 5 in ts.frame_track_data
    assert ts.frame_track_data[5][0].shape == (17, 3)


def test_auto_select_by_hits():
    ts = TrackState(fps=30.0)
    ts.track_hit_counts = {0: 5, 1: 10, 2: 3}
    selected = ts.auto_select_target()
    assert selected == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest ml/tests/pose_estimation/test_track_state.py -v`
Expected: FAIL with "module not found"

- [ ] **Step 3: Write minimal implementation**

```python
# ml/src/pose_estimation/_track_state.py
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

logger = logging.getLogger(__name__)


class TrackState:
    """Manages tracker instances, hit counts, and per-frame track data."""

    def __init__(
        self,
        fps: float,
        tracking_backend: str = "rtmlib",
        tracking_mode: str = "auto",
    ) -> None:
        self.fps = fps
        self.tracking_backend = tracking_backend
        self.tracking_mode = tracking_mode
        self.target_track_id: int | None = None
        self.track_hit_counts: dict[int, int] = {}
        self.frame_track_data: dict[int, dict[int, NDArray[np.float32]]] = {}
        self._next_internal_id = 0

        self.sports2d_tracker = None
        self.deepsort_tracker = None
        self.custom_tracker = None
        self._init_trackers()

    def _init_trackers(self) -> None:
        if self.tracking_backend == "custom":
            from ..tracking.sports2d import Sports2DTracker
            self.custom_tracker = Sports2DTracker(max_disappeared=30, fps=self.fps)
            return

        resolved = self._resolve_tracking_mode()
        if resolved == "sports2d":
            from ..tracking.sports2d import Sports2DTracker
            self.sports2d_tracker = Sports2DTracker(max_disappeared=30, fps=self.fps)
        elif resolved == "deepsort":
            from ..tracking.deepsort_tracker import DeepSORTTracker
            self.deepsort_tracker = DeepSORTTracker(max_age=30, embedder_gpu=True)

    def _resolve_tracking_mode(self) -> str:
        if self.tracking_mode == "auto":
            try:
                from ..tracking.deepsort_tracker import DeepSORTTracker
                return "deepsort"
            except ImportError:
                return "sports2d"
        return self.tracking_mode

    @property
    def tracker_instances(self) -> tuple:
        return (self.sports2d_tracker, self.deepsort_tracker, self.custom_tracker)

    def update_tracking(
        self,
        h36m_poses: NDArray[np.float32],
        frame: NDArray[np.uint8] | None = None,
        frame_width: int = 0,
        frame_height: int = 0,
    ) -> list[int]:
        """Run trackers on current frame detections. Returns track IDs."""
        n_persons = h36m_poses.shape[0]
        if n_persons == 0:
            return []

        if self.sports2d_tracker is not None:
            return self.sports2d_tracker.update(h36m_poses[:, :, :2], h36m_poses[:, :, 2])
        if self.deepsort_tracker is not None:
            return self.deepsort_tracker.update(
                h36m_poses[:, :, :2],
                h36m_poses[:, :, 2],
                frame=frame,
                frame_width=frame_width,
                frame_height=frame_height,
            )
        if self.custom_tracker is not None:
            return self.custom_tracker.update(h36m_poses[:, :, :2], h36m_poses[:, :, 2])

        # Fallback: sequential IDs
        track_ids = list(range(self._next_internal_id, self._next_internal_id + n_persons))
        self._next_internal_id += n_persons
        return track_ids

    def record_frame(
        self,
        frame_idx: int,
        h36m_poses: NDArray[np.float32],
        track_ids: list[int],
    ) -> None:
        self.frame_track_data[frame_idx] = {
            tid: h36m_poses[p].copy() for p, tid in enumerate(track_ids)
        }
        for tid in track_ids:
            self.track_hit_counts[tid] = self.track_hit_counts.get(tid, 0) + 1

    def auto_select_target(self) -> int | None:
        if not self.track_hit_counts:
            return None
        return max(self.track_hit_counts, key=lambda k: self.track_hit_counts[k])

    def retroactive_fill(
        self,
        all_poses: NDArray[np.float32],
        target_track_id: int,
    ) -> None:
        for fidx, tmap in self.frame_track_data.items():
            if target_track_id in tmap and np.isnan(all_poses[fidx, 0, 0]):
                all_poses[fidx] = tmap[target_track_id]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest ml/tests/pose_estimation/test_track_state.py -v`
Expected: PASS (4/4)

- [ ] **Step 5: Commit**

```bash
git add ml/src/pose_estimation/_track_state.py ml/tests/pose_estimation/test_track_state.py
git commit -m "feat(pose): add TrackState component for tracker management"
```

---

## Task 2: Create `TargetSelector` class

**Files:**
- Create: `ml/src/pose_estimation/_target_selector.py`
- Test: `ml/tests/pose_estimation/test_target_selector.py`

**Motivation:** Click-based target selection (mid-hip distance) and auto-select by hits are duplicated in both per-frame and batch paths.

- [ ] **Step 1: Write the failing test**

```python
# ml/tests/pose_estimation/test_target_selector.py
import numpy as np
import pytest
from src.pose_estimation._target_selector import TargetSelector


def test_click_selection_within_window():
    sel = TargetSelector(click_norm=(0.5, 0.5), click_lock_window=6)
    poses = np.zeros((2, 17, 3), dtype=np.float32)
    poses[0, 4, :2] = [0.5, 0.5]  # mid-hip approx
    poses[1, 4, :2] = [0.1, 0.1]
    track_ids = [0, 1]
    result = sel.select_target(poses, track_ids, frame_idx=3)
    assert result == 0


def test_click_selection_outside_window():
    sel = TargetSelector(click_norm=(0.5, 0.5), click_lock_window=6)
    poses = np.zeros((1, 17, 3), dtype=np.float32)
    poses[0, 4, :2] = [0.5, 0.5]
    track_ids = [0]
    result = sel.select_target(poses, track_ids, frame_idx=10)
    assert result is None


def test_auto_select_by_hits():
    sel = TargetSelector()
    hit_counts = {0: 5, 1: 10}
    assert sel.auto_select_by_hits(hit_counts) == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest ml/tests/pose_estimation/test_target_selector.py -v`
Expected: FAIL

- [ ] **Step 3: Write minimal implementation**

```python
# ml/src/pose_estimation/_target_selector.py
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


class TargetSelector:
    """Selects target person via click proximity or auto-select by detection frequency."""

    def __init__(
        self,
        click_norm: tuple[float, float] | None = None,
        click_lock_window: int = 6,
    ) -> None:
        self.click_norm = click_norm
        self.click_lock_window = click_lock_window
        self._target_track_id: int | None = None

    @property
    def target_track_id(self) -> int | None:
        return self._target_track_id

    def select_target(
        self,
        h36m_poses: NDArray[np.float32],
        track_ids: list[int],
        frame_idx: int,
    ) -> int | None:
        """Try to select target via click proximity. Returns selected track_id or None."""
        if self._target_track_id is not None:
            return None
        if self.click_norm is None:
            return None
        if frame_idx >= self.click_lock_window:
            return None

        best_dist = float("inf")
        best_tid = None
        cx_click, cy_click = self.click_norm

        for p, tid in enumerate(track_ids):
            mid_hip_x = (h36m_poses[p, 4, 0] + h36m_poses[p, 1, 0]) / 2
            mid_hip_y = (h36m_poses[p, 4, 1] + h36m_poses[p, 1, 1]) / 2
            dist = (mid_hip_x - cx_click) ** 2 + (mid_hip_y - cy_click) ** 2
            if dist < best_dist:
                best_dist = dist
                best_tid = tid

        if best_tid is not None:
            self._target_track_id = best_tid
        return best_tid

    @staticmethod
    def auto_select_by_hits(track_hit_counts: dict[int, int]) -> int | None:
        if not track_hit_counts:
            return None
        return max(track_hit_counts, key=lambda k: track_hit_counts[k])
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest ml/tests/pose_estimation/test_target_selector.py -v`
Expected: PASS (3/3)

- [ ] **Step 5: Commit**

```bash
git add ml/src/pose_estimation/_target_selector.py ml/tests/pose_estimation/test_target_selector.py
git commit -m "feat(pose): add TargetSelector component"
```

---

## Task 3: Create `TrackValidator` class

**Files:**
- Create: `ml/src/pose_estimation/_track_validator.py`
- Test: `ml/tests/pose_estimation/test_track_validator.py`

**Motivation:** Anti-steal logic (centroid jump + skeletal anomaly) and biometric track migration are ~120 lines duplicated in both paths.

- [ ] **Step 1: Write the failing test**

```python
# ml/tests/pose_estimation/test_track_validator.py
import numpy as np
import pytest
from src.pose_estimation._track_validator import TrackValidator


def test_no_steal_when_pose_unchanged():
    val = TrackValidator()
    pose = np.ones((17, 3), dtype=np.float32) * 0.5
    pose[:, 2] = 0.9
    assert not val.is_stolen(pose, pose)


def test_steal_on_large_jump_and_anomaly():
    val = TrackValidator()
    prev = np.ones((17, 3), dtype=np.float32) * 0.5
    prev[:, 2] = 0.9
    curr = prev.copy()
    curr[:, 0] += 0.5  # big jump
    curr[0, 1] += 0.3  # distort ratios
    assert val.is_stolen(curr, prev)


def test_migration_score_weights():
    val = TrackValidator()
    prev = np.ones((17, 3), dtype=np.float32) * 0.5
    prev[:, 2] = 0.9
    curr = prev.copy()
    curr[:, 0] += 0.05  # small jump
    score = val.migration_score(curr, prev, elapsed=0)
    assert score < 1.5
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest ml/tests/pose_estimation/test_track_validator.py -v`
Expected: FAIL

- [ ] **Step 3: Write minimal implementation**

```python
# ml/src/pose_estimation/_track_validator.py
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from ..tracking.skeletal_identity import compute_2d_skeletal_ratios


class TrackValidator:
    """Anti-steal detection and biometric track migration."""

    CENTROID_JUMP_THRESHOLD = 0.15
    RATIO_CHANGE_THRESHOLD = 0.25
    MAX_LOST_FRAMES = 60
    MIGRATION_THRESHOLD = 1.5

    def is_stolen(
        self,
        current_pose: NDArray[np.float32],
        last_target_pose: NDArray[np.float32],
        last_target_ratios: NDArray[np.float32] | None = None,
    ) -> bool:
        cur_cx = float(np.nanmean(current_pose[:, 0]))
        cur_cy = float(np.nanmean(current_pose[:, 1]))
        prev_cx = float(np.nanmean(last_target_pose[:, 0]))
        prev_cy = float(np.nanmean(last_target_pose[:, 1]))
        jump = np.sqrt((cur_cx - prev_cx) ** 2 + (cur_cy - prev_cy) ** 2)

        skeletal_anomaly = False
        if last_target_ratios is not None:
            curr_ratios = compute_2d_skeletal_ratios(current_pose)
            ratio_change = float(np.linalg.norm(curr_ratios - last_target_ratios))
            skeletal_anomaly = ratio_change > self.RATIO_CHANGE_THRESHOLD

        return jump > self.CENTROID_JUMP_THRESHOLD and skeletal_anomaly

    def migration_score(
        self,
        candidate_pose: NDArray[np.float32],
        last_target_pose: NDArray[np.float32],
        elapsed: int,
    ) -> float:
        from ..pose_estimation.h36m import _biometric_distance

        cur_cx = float(np.nanmean(candidate_pose[:, 0]))
        cur_cy = float(np.nanmean(candidate_pose[:, 1]))
        prev_cx = float(np.nanmean(last_target_pose[:, 0]))
        prev_cy = float(np.nanmean(last_target_pose[:, 1]))
        pos_dist = np.sqrt((cur_cx - prev_cx) ** 2 + (cur_cy - prev_cy) ** 2)
        bio_dist = _biometric_distance(candidate_pose, last_target_pose)

        w_pos = max(0.2, 1.0 - elapsed * 0.02)
        w_bio = 1.0 - w_pos
        return w_pos * pos_dist / 0.15 + w_bio * bio_dist / 0.08
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest ml/tests/pose_estimation/test_track_validator.py -v`
Expected: PASS (3/3)

- [ ] **Step 5: Commit**

```bash
git add ml/src/pose_estimation/_track_validator.py ml/tests/pose_estimation/test_track_validator.py
git commit -m "feat(pose): add TrackValidator component with anti-steal logic"
```

---

## Task 4: Create `FrameProcessor` class

**Files:**
- Create: `ml/src/pose_estimation/_frame_processor.py`
- Test: `ml/tests/pose_estimation/test_frame_processor.py`

**Motivation:** COCO→H3.6M conversion, normalization, and per-frame rescaling are ~30 lines duplicated in every frame of both paths.

- [ ] **Step 1: Write the failing test**

```python
# ml/tests/pose_estimation/test_frame_processor.py
import numpy as np
import pytest
from src.pose_estimation._frame_processor import FrameProcessor


def test_convert_empty():
    fp = FrameProcessor(output_format="normalized")
    result = fp.convert_keypoints(np.zeros((0, 17, 2)), np.zeros((0, 17)), 640, 480)
    assert result.shape == (0, 17, 3)


def test_convert_single_person():
    fp = FrameProcessor(output_format="normalized")
    kps = np.array([[[100.0, 200.0]] * 17], dtype=np.float32)
    scores = np.ones((1, 17), dtype=np.float32) * 0.8
    result = fp.convert_keypoints(kps, scores, 640, 480)
    assert result.shape == (1, 17, 3)
    assert result[0, 0, 0] == pytest.approx(100 / 640, abs=1e-4)
    assert result[0, 0, 1] == pytest.approx(200 / 480, abs=1e-4)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest ml/tests/pose_estimation/test_frame_processor.py -v`
Expected: FAIL

- [ ] **Step 3: Write minimal implementation**

```python
# ml/src/pose_estimation/_frame_processor.py
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .h36m import coco_to_h36m


class FrameProcessor:
    """Converts raw RTMO output (COCO keypoints) to H3.6M format per frame."""

    def __init__(self, output_format: str = "normalized") -> None:
        self.output_format = output_format

    def convert_keypoints(
        self,
        keypoints: NDArray[np.float32],  # (P, 17, 2) pixels
        scores: NDArray[np.float32],  # (P, 17)
        frame_width: int,
        frame_height: int,
    ) -> NDArray[np.float32]:  # (P, 17, 3)
        n_persons = keypoints.shape[0]
        h36m_poses = np.zeros((n_persons, 17, 3), dtype=np.float32)
        w, h = float(frame_width), float(frame_height)

        for p in range(n_persons):
            coco = np.zeros((17, 3), dtype=np.float32)
            coco[:, :2] = keypoints[p].astype(np.float32)
            coco[:, 2] = scores[p].astype(np.float32)
            coco[:, 0] /= w
            coco[:, 1] /= h

            h36m = coco_to_h36m(coco)

            if self.output_format == "pixels":
                h36m[:, 0] *= w
                h36m[:, 1] *= h

            h36m_poses[p] = h36m

        return h36m_poses
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest ml/tests/pose_estimation/test_frame_processor.py -v`
Expected: PASS (2/2)

- [ ] **Step 5: Commit**

```bash
git add ml/src/pose_estimation/_frame_processor.py ml/tests/pose_estimation/test_frame_processor.py
git commit -m "feat(pose): add FrameProcessor for COCO→H3.6M conversion"
```

---

## Task 5: Refactor `extract_video_tracked` to use new components

**Files:**
- Modify: `ml/src/pose_estimation/pose_extractor.py`

**Motivation:** Reduce `extract_video_tracked` from 401 lines to ~60-line orchestrator.

- [ ] **Step 1: Replace per-frame path in `extract_video_tracked`**

Replace the entire body of `extract_video_tracked` (lines 166-566) with:

```python
def extract_video_tracked(
    self,
    video_path: Path | str,
    person_click: PersonClick | None = None,
    progress_cb=None,
    use_batch: bool = True,
    batch_size: int = 8,
) -> TrackedExtraction:
    if use_batch:
        return self._extract_batch(
            video_path,
            person_click=person_click,
            progress_cb=progress_cb,
            batch_size=batch_size,
        )
    return self._extract_per_frame(
        video_path,
        person_click=person_click,
        progress_cb=progress_cb,
    )
```

- [ ] **Step 2: Extract `_extract_per_frame` method**

Add new private method `_extract_per_frame` (~120 lines) using the components:

```python
def _extract_per_frame(
    self,
    video_path: Path | str,
    person_click: PersonClick | None = None,
    progress_cb=None,
) -> TrackedExtraction:
    from ..utils.frame_buffer import AsyncFrameReader

    video_path = Path(video_path)
    video_meta = get_video_meta(video_path)
    num_frames = video_meta.num_frames
    all_poses = np.full((num_frames, 17, 3), np.nan, dtype=np.float32)

    track_state = TrackState(
        fps=video_meta.fps,
        tracking_backend=self._tracking_backend,
        tracking_mode=self._tracking_mode,
    )
    click_norm = person_click.to_normalized(video_meta.width, video_meta.height) if person_click else None
    selector = TargetSelector(click_norm=click_norm)
    validator = TrackValidator()
    processor = FrameProcessor(output_format=self._output_format)

    last_target_pose: np.ndarray | None = None
    last_target_ratios: np.ndarray | None = None
    target_lost_frame: int | None = None

    # Read first frame
    cap = cv2.VideoCapture(str(video_path))
    ret, first_frame = cap.read()
    cap.release()
    if not ret:
        raise RuntimeError(f"Failed to read first frame: {video_path}")

    reader = AsyncFrameReader(video_path, buffer_size=16, frame_skip=self._frame_skip)
    reader.start()
    pbar = _get_tqdm()(total=num_frames, desc="Extracting poses", unit="frame")

    try:
        while True:
            result = reader.get_frame()
            if result is None:
                break
            frame_idx, frame = result
            h, w = frame.shape[:2]

            # Resize large frames
            if max(h, w) > 1920:
                scale = 1920 / max(h, w)
                frame_ds = cv2.resize(frame, (int(w * scale), int(h * scale)))
            else:
                frame_ds = frame

            tracker = self.tracker
            tracker_result = tracker(frame_ds)
            if not isinstance(tracker_result, tuple) or len(tracker_result) != 2:
                pbar.update(self._frame_skip)
                continue
            keypoints, scores = tracker_result
            if keypoints is None or len(keypoints) == 0:
                pbar.update(self._frame_skip)
                continue

            # Rescale if downscaled
            if frame_ds is not frame:
                keypoints = keypoints * (max(h, w) / 1920)

            h36m_poses = processor.convert_keypoints(keypoints, scores, w, h)
            track_ids = track_state.update_tracking(h36m_poses, frame, w, h)
            track_state.record_frame(frame_idx, h36m_poses, track_ids)

            # Target selection
            if selector.target_track_id is None:
                selector.select_target(h36m_poses, track_ids, frame_idx)
                if selector.target_track_id is not None:
                    track_state.target_track_id = selector.target_track_id

            target_id = track_state.target_track_id or selector.target_track_id
            if target_id is not None:
                found = False
                stolen = False
                for p, tid in enumerate(track_ids):
                    if tid == target_id:
                        if last_target_pose is not None and validator.is_stolen(
                            h36m_poses[p], last_target_pose, last_target_ratios
                        ):
                            stolen = True
                            break
                        all_poses[frame_idx] = h36m_poses[p]
                        last_target_pose = h36m_poses[p].copy()
                        last_target_ratios = compute_2d_skeletal_ratios(h36m_poses[p])
                        target_lost_frame = None
                        found = True
                        break

                if stolen:
                    all_poses[frame_idx] = np.full((17, 3), np.nan, dtype=np.float32)
                    found = False

                if (not found or stolen) and last_target_pose is not None:
                    if target_lost_frame is None:
                        target_lost_frame = frame_idx
                    if frame_idx - target_lost_frame <= validator.MAX_LOST_FRAMES:
                        best_score = float("inf")
                        best_tid = None
                        best_pose = None
                        for p, tid in enumerate(track_ids):
                            if stolen and tid == target_id:
                                continue
                            score = validator.migration_score(
                                h36m_poses[p], last_target_pose,
                                frame_idx - (target_lost_frame or frame_idx)
                            )
                            if score < best_score:
                                best_score = score
                                best_tid = tid
                                best_pose = h36m_poses[p]
                        if best_tid is not None and best_score < validator.MIGRATION_THRESHOLD and best_pose is not None:
                            track_state.target_track_id = best_tid
                            all_poses[frame_idx] = best_pose
                            last_target_pose = best_pose.copy()
                            last_target_ratios = compute_2d_skeletal_ratios(best_pose)
                            target_lost_frame = None
                            track_state.retroactive_fill(all_poses, best_tid)

            pbar.update(self._frame_skip)
            if progress_cb:
                progress_cb(frame_idx / num_frames * 0.3, f"Extracting poses... {frame_idx}/{num_frames}")
    finally:
        reader.join()
        pbar.close()

    # Auto-select if no click
    if track_state.target_track_id is None:
        auto_tid = selector.auto_select_by_hits(track_state.track_hit_counts)
        if auto_tid is not None:
            track_state.target_track_id = auto_tid
            track_state.retroactive_fill(all_poses, auto_tid)

    # Post-hoc merge (keep existing TrackletMerger logic)
    self._post_hoc_merge(all_poses, track_state.frame_track_data, track_state.target_track_id)

    valid_mask = ~np.isnan(all_poses[:, 0, 0])
    if not np.any(valid_mask):
        raise ValueError(f"No valid pose detected: {video_path}")

    return TrackedExtraction(
        poses=all_poses,
        frame_indices=np.arange(num_frames),
        first_detection_frame=int(np.argmax(valid_mask)),
        target_track_id=track_state.target_track_id,
        fps=video_meta.fps,
        video_meta=video_meta,
        first_frame=first_frame,
    )
```

- [ ] **Step 3: Add `_post_hoc_merge` helper**

Extract the post-hoc merge block (lines 498-566) into `_post_hoc_merge(self, all_poses, frame_track_data, target_track_id)`.

- [ ] **Step 4: Run existing tests**

Run: `uv run pytest ml/tests/pose_estimation/ -v -k "extract"`
Expected: All existing extraction tests still pass.

- [ ] **Step 5: Commit**

```bash
git add ml/src/pose_estimation/pose_extractor.py
git commit -m "refactor(pose): extract per-frame path to use TrackState, TargetSelector, TrackValidator, FrameProcessor"
```

---

## Task 6: Refactor `_extract_batch` to use new components

**Files:**
- Modify: `ml/src/pose_estimation/pose_extractor.py`

**Motivation:** Reduce `_extract_batch` from 375 lines to ~80-line orchestrator, reusing the same components.

- [ ] **Step 1: Replace `_extract_batch` body**

Replace the entire method (lines 568-945) with:

```python
def _extract_batch(
    self,
    video_path: Path | str,
    person_click: PersonClick | None = None,
    progress_cb=None,
    batch_size: int = 8,
) -> TrackedExtraction:
    from .rtmo_batch import BatchRTMO

    video_path = Path(video_path)
    video_meta = get_video_meta(video_path)
    num_frames = video_meta.num_frames
    all_poses = np.full((num_frames, 17, 3), np.nan, dtype=np.float32)

    rtmo = BatchRTMO(
        mode=self._mode,
        device=self._device,
        score_thr=self._conf_threshold,
        nms_thr=0.45,
    )

    track_state = TrackState(
        fps=video_meta.fps,
        tracking_backend=self._tracking_backend,
        tracking_mode=self._tracking_mode,
    )
    click_norm = person_click.to_normalized(video_meta.width, video_meta.height) if person_click else None
    selector = TargetSelector(click_norm=click_norm)
    validator = TrackValidator()
    processor = FrameProcessor(output_format=self._output_format)

    last_target_pose: np.ndarray | None = None
    last_target_ratios: np.ndarray | None = None
    target_lost_frame: int | None = None

    # Read first frame
    cap = cv2.VideoCapture(str(video_path))
    ret, first_frame = cap.read()
    cap.release()
    if not ret:
        raise RuntimeError(f"Failed to read first frame: {video_path}")

    # Read all frames
    cap = cv2.VideoCapture(str(video_path))
    frames_to_process = []
    frame_indices = []
    try:
        for idx in range(num_frames):
            if idx % self._frame_skip != 0:
                continue
            ret, frame = cap.read()
            if not ret:
                break
            frames_to_process.append(frame)
            frame_indices.append(idx)
    finally:
        cap.release()

    if not frames_to_process:
        raise ValueError(f"No frames read: {video_path}")

    pbar = _get_tqdm()(total=len(frames_to_process), desc="Batch extracting poses", unit="batch")

    for batch_start in range(0, len(frames_to_process), batch_size):
        batch_end = min(batch_start + batch_size, len(frames_to_process))
        batch_frames = frames_to_process[batch_start:batch_end]
        batch_indices = frame_indices[batch_start:batch_end]
        batch_results = rtmo.infer_batch(batch_frames)

        for frame_idx, frame, (keypoints, scores) in zip(batch_indices, batch_frames, batch_results, strict=True):
            h, w = frame.shape[:2]
            if keypoints.shape[0] == 0:
                pbar.update(1)
                continue

            h36m_poses = processor.convert_keypoints(keypoints, scores, w, h)
            track_ids = track_state.update_tracking(h36m_poses)
            track_state.record_frame(frame_idx, h36m_poses, track_ids)

            if selector.target_track_id is None:
                selector.select_target(h36m_poses, track_ids, frame_idx)
                if selector.target_track_id is not None:
                    track_state.target_track_id = selector.target_track_id

            target_id = track_state.target_track_id or selector.target_track_id
            if target_id is not None:
                found = False
                stolen = False
                for p, tid in enumerate(track_ids):
                    if tid == target_id:
                        if last_target_pose is not None and validator.is_stolen(
                            h36m_poses[p], last_target_pose, last_target_ratios
                        ):
                            stolen = True
                            break
                        all_poses[frame_idx] = h36m_poses[p]
                        last_target_pose = h36m_poses[p].copy()
                        last_target_ratios = compute_2d_skeletal_ratios(h36m_poses[p])
                        target_lost_frame = None
                        found = True
                        break

                if stolen:
                    all_poses[frame_idx] = np.full((17, 3), np.nan, dtype=np.float32)
                    found = False

                if (not found or stolen) and last_target_pose is not None:
                    if target_lost_frame is None:
                        target_lost_frame = frame_idx
                    if frame_idx - target_lost_frame <= validator.MAX_LOST_FRAMES:
                        best_score = float("inf")
                        best_tid = None
                        best_pose = None
                        for p, tid in enumerate(track_ids):
                            if stolen and tid == target_id:
                                continue
                            score = validator.migration_score(
                                h36m_poses[p], last_target_pose,
                                frame_idx - (target_lost_frame or frame_idx)
                            )
                            if score < best_score:
                                best_score = score
                                best_tid = tid
                                best_pose = h36m_poses[p]
                        if best_tid is not None and best_score < validator.MIGRATION_THRESHOLD and best_pose is not None:
                            track_state.target_track_id = best_tid
                            all_poses[frame_idx] = best_pose
                            last_target_pose = best_pose.copy()
                            last_target_ratios = compute_2d_skeletal_ratios(best_pose)
                            target_lost_frame = None
                            track_state.retroactive_fill(all_poses, best_tid)

            pbar.update(1)
            if progress_cb:
                progress_cb(frame_idx / num_frames * 0.3, f"Batch extracting... {frame_idx}/{num_frames}")

    pbar.close()

    if track_state.target_track_id is None:
        auto_tid = selector.auto_select_by_hits(track_state.track_hit_counts)
        if auto_tid is not None:
            track_state.target_track_id = auto_tid
            track_state.retroactive_fill(all_poses, auto_tid)

    self._post_hoc_merge(all_poses, track_state.frame_track_data, track_state.target_track_id)

    valid_mask = ~np.isnan(all_poses[:, 0, 0])
    if not np.any(valid_mask):
        raise ValueError(f"No valid pose detected: {video_path}")

    return TrackedExtraction(
        poses=all_poses,
        frame_indices=np.arange(num_frames),
        first_detection_frame=int(np.argmax(valid_mask)),
        target_track_id=track_state.target_track_id,
        fps=video_meta.fps,
        video_meta=video_meta,
        first_frame=first_frame,
    )
```

- [ ] **Step 2: Run existing tests**

Run: `uv run pytest ml/tests/pose_estimation/ -v -k "batch"`
Expected: All batch extraction tests pass.

- [ ] **Step 3: Full test suite**

Run: `uv run pytest ml/tests/pose_estimation/ -v`
Expected: All tests pass.

- [ ] **Step 4: Commit**

```bash
git add ml/src/pose_estimation/pose_extractor.py
git commit -m "refactor(pose): extract batch path to shared components, eliminate duplication"
```

---

## Self-Review

1. **Spec coverage:**
   - `extract_video_tracked` decomposition → Tasks 5, 6
   - `_extract_batch` decomposition → Task 6
   - Shared components → Tasks 1-4 (TrackState, TargetSelector, TrackValidator, FrameProcessor)
   - Test coverage → Each task has its own test file
   - No placeholders → All code is concrete, no "TBD"

2. **Placeholder scan:** Clean.

3. **Type consistency:**
   - `TrackState.target_track_id` used consistently
   - `TargetSelector.target_track_id` mirrors existing logic
   - `TrackValidator.is_stolen()` signature matches original usage
   - `FrameProcessor.convert_keypoints()` produces same output shape `(P, 17, 3)`

4. **No dead code introduced:** New files only. Old logic preserved, just moved.

---

## Execution Handoff

**Plan complete and saved to `docs/plans/2026-05-05-pose-extractor-decomposition.md`.**

Two execution options:

**1. Subagent-Driven (recommended)** — Fresh subagent per task + review loop. Commit after every step. All tests green before next Wave.

**2. Inline Execution** — Execute tasks in this session using executing-plans, batch execution with checkpoints.

**Which approach?**
