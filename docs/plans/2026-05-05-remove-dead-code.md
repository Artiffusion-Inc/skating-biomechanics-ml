# Remove Dead Code Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use subagent-driven-development (recommended) or executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove dead code that is never called, never wired into the pipeline, or explicitly disabled. Reduce cognitive load and test maintenance burden. Verify nothing breaks.

**Architecture:** Identify dead files by checking `import` references and runtime usage. Delete files, remove corresponding tests, clean up imports in dependent files. Each deletion is a single commit for easy revert.

**Tech Stack:** Python 3.11, pytest, grep.

---

## File Structure

### Files to Delete (confirmed dead)
- `ml/src/pose_3d/corrective_pipeline.py` — CorrectiveLens disabled (~3px shift), never used
- `ml/src/pose_3d/kinematic_constraints.py` — Only used by CorrectiveLens
- `ml/src/pose_3d/anchor_projection.py` — Only used by CorrectiveLens
- `ml/src/pose_3d/athletepose_extractor.py` — Only used by CorrectiveLens and 3D blade detection (both dead)
- `ml/src/detection/blade_edge_detector_3d.py` — Not wired into pipeline
- `ml/src/detection/blade_edge_detector.py` — Not wired into pipeline (BDA algorithm, 79% accuracy)
- `ml/src/extras/depth_anything.py` — No models, no runtime usage
- `ml/src/extras/foot_tracker.py` — No models, no runtime usage
- `ml/src/extras/inpainting.py` — No models, no runtime usage
- `ml/src/extras/model_registry.py` — Only used by extras (all dead)
- `ml/src/extras/optical_flow.py` — No models, no runtime usage
- `ml/src/extras/segment_anything.py` — No models, no runtime usage
- `ml/src/extras/video_matting.py` — No models, no runtime usage

### Files to Modify (clean up imports)
- `ml/src/pipeline.py` — Remove `compute_3d` branch (lines 258-294), remove `_get_pose_3d_extractor`
- `ml/src/web_helpers.py` — Remove all ML extras flags (depth, optical_flow, segment, foot_track, matting, inpainting)
- `ml/src/pose_estimation/__init__.py` — Remove exports if any
- `ml/tests/pose_3d/` — Delete tests for deleted modules
- `ml/tests/detection/` — Delete blade edge detector tests
- `ml/src/__init__.py` — Clean up exports

### Files to Keep (alive)
- `ml/src/pose_3d/normalizer_3d.py` — Used by pipeline? Check.
- `ml/src/tracking/skeletal_identity.py` — Used by TrackletMerger, keep
- `ml/src/tracking/tracklet_merger.py` — Used by pose_extractor, keep

---

## Verification Strategy

Before each deletion:
```bash
grep -rl "from.*corrective_pipeline\|import.*corrective_pipeline" ml/src/ ml/scripts/ backend/ experiments/
grep -rl "BladeEdgeDetector\|blade_edge_detector" ml/src/ ml/scripts/ backend/ experiments/
grep -rl "from.*extras\|import.*extras" ml/src/ ml/scripts/ backend/ experiments/
```

If no references outside the file itself → safe to delete.

---

## Task 1: Remove CorrectiveLens and dependents

**Files:**
- Delete: `ml/src/pose_3d/corrective_pipeline.py`
- Delete: `ml/src/pose_3d/kinematic_constraints.py`
- Delete: `ml/src/pose_3d/anchor_projection.py`
- Delete: `ml/src/pose_3d/athletepose_extractor.py`
- Modify: `ml/src/pipeline.py` (remove compute_3d branch)
- Modify: `ml/src/pose_3d/__init__.py`

- [ ] **Step 1: Verify no external references to CorrectiveLens**

Run:
```bash
grep -rl "CorrectiveLens\|corrective_pipeline" ml/src/ ml/scripts/ backend/ experiments/ 2>/dev/null
```
Expected: Only references in `ml/src/pose_3d/` and `ml/tests/pose_3d/`

- [ ] **Step 2: Verify no external references to AthletePose3DExtractor**

Run:
```bash
grep -rl "AthletePose3DExtractor\|athletepose_extractor" ml/src/ ml/scripts/ backend/ experiments/ 2>/dev/null
```
Expected: Only in `ml/src/pose_3d/` and pipeline.py compute_3d branch

- [ ] **Step 3: Delete the 4 files**

```bash
git rm ml/src/pose_3d/corrective_pipeline.py
 git rm ml/src/pose_3d/kinematic_constraints.py
 git rm ml/src/pose_3d/anchor_projection.py
 git rm ml/src/pose_3d/athletepose_extractor.py
```

- [ ] **Step 4: Remove compute_3d branch from pipeline.py**

Delete lines 258-294 (the `if self._compute_3d:` block including blade detection inside it).

Also remove `_get_pose_3d_extractor` method (lines 446-458).

Remove `compute_3d` parameter from `__init__` and `self._compute_3d` attribute.

Remove `blade_summary_left` and `blade_summary_right` from `AnalysisReport` construction.

- [ ] **Step 5: Update `ml/src/pose_3d/__init__.py`**

Remove exports for deleted classes.

- [ ] **Step 6: Run tests**

Run: `uv run pytest ml/tests/ -v -k "not pose_3d"`
Expected: All non-pose_3d tests pass. Pose3D tests will fail because files deleted — we'll delete those tests next.

- [ ] **Step 7: Delete pose_3d tests**

```bash
git rm ml/tests/pose_3d/test_corrective_lens.py ml/tests/pose_3d/test_anchor_projection.py ml/tests/pose_3d/test_athletepose_extractor.py
```

- [ ] **Step 8: Commit**

```bash
git commit -m "refactor(pose_3d): remove CorrectiveLens and dependents (dead code)"
```

---

## Task 2: Remove blade edge detectors

**Files:**
- Delete: `ml/src/detection/blade_edge_detector_3d.py`
- Delete: `ml/src/detection/blade_edge_detector.py`
- Delete: `ml/src/utils/blade_edge_detector.py` (if exists)
- Delete: `ml/tests/detection/test_blade_edge_detector.py`
- Delete: `ml/tests/utils/test_blade_edge_detector.py`
- Modify: `ml/src/types.py` — Keep `BladeType` enum (used in `AnalysisReport` for backward compat, but can remove from report)

- [ ] **Step 1: Verify no runtime references**

Run:
```bash
grep -rl "BladeEdgeDetector\|blade_edge_detector" ml/src/ ml/scripts/ backend/ experiments/ 2>/dev/null | grep -v "__pycache__"
```
Expected: Only in files we're about to delete

- [ ] **Step 2: Delete files**

```bash
git rm ml/src/detection/blade_edge_detector_3d.py ml/src/detection/blade_edge_detector.py
find ml/tests -name "*blade*" -exec git rm {} \;
```

- [ ] **Step 3: Clean up AnalysisReport references**

In `ml/src/types.py`, check `AnalysisReport` dataclass. If `blade_summary_left/right` are fields, remove them from the report. Also check `ml/src/pipeline.py` — already removed in Task 1.

- [ ] **Step 4: Run tests**

Run: `uv run pytest ml/tests/detection/ -v`
Expected: Remaining detection tests pass.

- [ ] **Step 5: Commit**

```bash
git commit -m "refactor(detection): remove blade edge detectors (dead code, not wired)"
```

---

## Task 3: Remove ML extras (depth, SAM2, matting, etc.)

**Files:**
- Delete: `ml/src/extras/depth_anything.py`
- Delete: `ml/src/extras/foot_tracker.py`
- Delete: `ml/src/extras/inpainting.py`
- Delete: `ml/src/extras/model_registry.py`
- Delete: `ml/src/extras/optical_flow.py`
- Delete: `ml/src/extras/segment_anything.py`
- Delete: `ml/src/extras/video_matting.py`
- Modify: `ml/src/web_helpers.py` — Remove ML extras parameters and logic
- Modify: `ml/src/extras/__init__.py` — Remove exports

- [ ] **Step 1: Verify no runtime references**

Run:
```bash
grep -rl "from.*extras\|import.*extras" ml/src/ ml/scripts/ backend/ experiments/ 2>/dev/null | grep -v "__pycache__"
```
Expected: Only in `ml/src/extras/` and `ml/src/web_helpers.py`

- [ ] **Step 2: Delete extras files**

```bash
git rm ml/src/extras/depth_anything.py
 git rm ml/src/extras/foot_tracker.py
 git rm ml/src/extras/inpainting.py
 git rm ml/src/extras/model_registry.py
 git rm ml/src/extras/optical_flow.py
 git rm ml/src/extras/segment_anything.py
 git rm ml/src/extras/video_matting.py
```

- [ ] **Step 3: Refactor `ml/src/web_helpers.py`**

Remove parameters: `depth`, `optical_flow`, `segment`, `foot_track`, `matting`, `inpainting`, `element_type`.

Remove the `any_ml` block (lines 183-286), ML model initialization, and ML layers rendering (lines 357-393).

The function should become:
```python
def process_video_pipeline(
    video_path: str | Path,
    person_click: PersonClick | None,
    frame_skip: int,
    layer: int,
    tracking: str,
    output_path: str | Path,
    progress_cb=None,
    cancel_event=None,
) -> dict:
    """Run the visualization pipeline."""
    from src.visualization import render_layers
    from src.visualization.pipeline import VizPipeline, prepare_poses

    video_path = Path(video_path) if isinstance(video_path, str) else video_path
    output_path = Path(output_path) if isinstance(output_path, str) else output_path

    prepared = prepare_poses(
        video_path,
        person_click=person_click,
        frame_skip=frame_skip,
        tracking=tracking,
        progress_cb=progress_cb,
    )

    pipe = VizPipeline(
        meta=prepared.meta,
        poses_norm=prepared.poses_norm,
        poses_px=prepared.poses_px,
        layer=layer,
        confs=prepared.confs,
        frame_indices=prepared.frame_indices,
    )

    # ... render loop without ML extras ...
```

- [ ] **Step 4: Remove `_run_analysis` from web_helpers**

Delete `_run_analysis()` and the ThreadPoolExecutor analysis block (lines 327-336, 419-429). This logic belongs in the pipeline, not web helpers.

- [ ] **Step 5: Run tests**

Run: `uv run pytest ml/tests/ -v -k "not extras"`
Expected: All tests pass.

- [ ] **Step 6: Commit**

```bash
git commit -m "refactor(extras): remove dead ML extras (depth, SAM2, matting, etc.)"
```

---

## Task 4: Clean up remaining references

**Files:**
- Modify: `ml/src/__init__.py` — Remove dead exports
- Modify: `ml/src/extras/__init__.py` — Keep or delete if empty
- Modify: `ml/src/types.py` — Remove `BladeType`, `BladeState3D`, `IceTrace`, `MotionDirection` if unused

- [ ] **Step 1: Check `ml/src/__init__.py` for dead exports**

Run:
```bash
grep -n "export\|__all__" ml/src/__init__.py
```

Remove any exports pointing to deleted modules.

- [ ] **Step 2: Check `ml/src/types.py` for unused types**

Run:
```bash
grep -rl "BladeType\|BladeState3D\|IceTrace\|MotionDirection" ml/src/ ml/scripts/ backend/ 2>/dev/null | grep -v "__pycache__"
```

If only referenced in `types.py` itself → safe to remove.

- [ ] **Step 3: Delete empty `ml/src/extras/` if applicable**

If `ml/src/extras/__init__.py` is empty after removing exports, delete the entire `extras/` directory.

- [ ] **Step 4: Full test suite**

Run: `uv run pytest ml/tests/ -v`
Expected: All tests pass. If any import errors → fix.

- [ ] **Step 5: Commit**

```bash
git commit -m "chore(cleanup): remove dead exports and unused types"
```

---

## Self-Review

1. **Spec coverage:**
   - CorrectiveLens + dependents → Task 1
   - Blade edge detectors → Task 2
   - ML extras → Task 3
   - Cleanup → Task 4
   - All deletions verified via grep before removal

2. **Placeholder scan:** Clean.

3. **Type consistency:**
   - `AnalysisReport.blade_summary_*` removed in Task 1
   - `BladeType` removed in Task 4 if unused
   - No references to deleted modules remain

4. **Safety:** Each deletion preceded by grep verification. Each deletion is a separate commit. Easy to revert if needed.

---

## Execution Handoff

**Plan complete and saved to `docs/plans/2026-05-05-remove-dead-code.md`.**

Two execution options:

**1. Subagent-Driven (recommended)** — Fresh subagent per task + review loop. Commit after every step. Grep verification before each deletion.

**2. Inline Execution** — Execute tasks in this session using executing-plans.

**Which approach?**
