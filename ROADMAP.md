# Figure Skating Biomechanics ML - Roadmap

**Status:** MVP ~85% complete | Last updated: 2026-03-27

> **This is the SINGLE SOURCE OF TRUTH for project status.** All implementation decisions and priority changes must be reflected here first.

---

## Vision

AI-тренер по фигурному катанию который анализирует видео и даёт рекомендации на русском языке.

**Target Users:** Figure skaters and coaches looking for technical feedback
**Input:** Video recordings (mp4, webm, etc.)
**Output:** Biomechanics analysis + Russian recommendations

---

## Architecture Overview

```
┌─────────────┐    ┌──────────────┐    ┌─────────────┐    ┌──────────────┐
│   Video     │ -> │  Detection   │ -> │  Pose 2D    │ -> │ Normalized   │
│   Input     │    │  (YOLOv11n)   │    │ (BlazePose)  │    │   Poses       │
└─────────────┘    └──────────────┘    └─────────────┘    └──────────────┘
                                                                  v
┌──────────────────────────────────────────────────────────────────────┐
│                        Analysis Pipeline                            │
│  ┌──────────────┐  ┌─────────────┐  ┌─────────────┐  ┌───────────┐ │
│  │ Phase Detect │->│  Metrics    │->│  DTW Align  │->│ Recommend │ │
│  │   (TODO)     │  │  (Done)      │  │  (Fixing)   │  │ (Done)    │ │
│  └──────────────┘  └─────────────┘  └─────────────┘  └───────────┘ │
└──────────────────────────────────────────────────────────────────────┘
                                                                  v
                                                      ┌─────────────────────┐
                                                      │ Russian Text Report │
                                                      └─────────────────────┘
```

---

## Implementation Phases

### Phase 0: Foundation ✅ 100%
**Status:** Complete

- [x] Project structure
- [x] Type definitions (BKey, FrameKeypoints, etc.)
- [x] Video utilities (extract_frames, get_video_meta)
- [x] Geometry utilities (angles, distances)
- [x] Quality tooling (ruff, mypy, pytest)

**Files:** `types.py`, `utils/`, `pyproject.toml`

---

### Phase 1: Person Detection ✅ 100%
**Status:** Complete

- [x] YOLOv11n integration
- [x] PersonDetector class
- [x] Single-frame detection
- [x] Full-video detection
- [x] BoundingBox type

**Files:** `detection/person_detector.py`
**Tests:** `tests/detection/test_person_detector.py`

---

### Phase 2: 2D Pose Estimation ✅ 100%
**Status:** Complete

- [x] BlazePose integration (33 keypoints)
- [x] BlazePoseExtractor class
- [x] Pixel coordinates extraction
- [x] Confidence values
- [x] Alternative: YOLO-Pose (17 keypoints)

**Files:** `pose_2d/blazepose_extractor.py`, `pose_2d/pose_extractor.py`
**Tests:** `tests/pose_2d/test_blazepose_extractor.py`

---

### Phase 3: Pose Normalization ✅ 100%
**Status:** Complete

- [x] Root-centering (mid-hip → origin)
- [x] Scale normalization (spine length → 0.4)
- [x] PoseNormalizer class
- [x] Coordinate type system (PixelPose vs NormalizedPose)
- [x] Runtime validation (assert_pose_format)

**Files:** `pose_2d/normalizer.py`, `types.py` (coordinate types)
**Tests:** `tests/pose_2d/test_normalizer.py`

---

### Phase 4: Temporal Smoothing ✅ 100%
**Status:** Complete

- [x] One-Euro Filter implementation
- [x] PoseSmoother class
- [x] Skating-optimized config
- [x] 29% jitter reduction achieved
- [x] Normalized-space smoothing

**Files:** `utils/smoothing.py`
**Tests:** `tests/utils/test_smoothing.py`

---

### Phase 5: Biomechanics Metrics ✅ 100%
**Status:** Complete

- [x] Airtime calculation
- [x] Jump height (hip trajectory)
- [x] Knee angles (hip-knee-ankle)
- [x] Arm position
- [x] Edge detection (inside/outside/flat)
- [x] Rotation speed
- [x] BiomechanicsAnalyzer class

**Files:** `analysis/metrics.py`
**Tests:** `tests/analysis/test_metrics.py`

---

### Phase 6: Phase Detection ⚠️ 50%
**Status:** MANUAL ONLY - auto-detection NOT working

- [x] ElementPhase data structure
- [x] Manual phase specification via CLI
- [ ] **Auto takeoff detection** (height threshold)
- [ ] **Auto peak detection** (min hip y)
- [ ] **Auto landing detection** (impact detection)
- [ ] Phase transition smoothing

**Issue:** PhaseDetector always returns takeoff=0, landing=end
**Priority:** HIGH - blocks fully automated analysis

**Files:** `analysis/phase_detector.py`
**Tests:** `tests/analysis/test_phase_detector.py`

---

### Phase 7: DTW Motion Alignment ⚠️ 70%
**Status:** Code exists, tests failing

- [x] DTW implementation (dtw-python)
- [x] Sakoe-Chiba window
- [x] MotionAligner class
- [ ] **Fix tests** - expects 17 keypoints, BlazePose has 33
- [ ] Multi-segment alignment
- [ ] Alignment quality metrics

**Issue:** Test suite uses old 17-keypoint format, need update for 33-keypoint BlazePose
**Priority:** MEDIUM - manual analysis works without this

**Files:** `alignment/aligner.py`, `alignment/motion_dtw.py`
**Tests:** `tests/alignment/test_aligner.py` (7 failing)

---

### Phase 8: Rule-Based Recommender ✅ 100%
**Status:** Complete

- [x] Rule engine for each element
- [x] MetricResult validation
- [x] Russian text generation
- [x] jump_rules.py (waltz_jump, toe_loop, flip, salchow, loop, lutz, axel)
- [x] three_turn_rules.py
- [x] Recommender class

**Files:** `analysis/recommender.py`, `analysis/rules/`
**Tests:** `tests/analysis/test_recommender.py`

---

### Phase 9: Reference System ✅ 100%
**Status:** Complete

- [x] ReferenceData type
- [x] Save/Load .npz files
- [x] ReferenceBuilder CLI
- [x] Element definitions (ideal ranges)
- [x] Reference directory structure

**Files:** `references/element_defs.py`, `references/reference_builder.py`, `references/reference_store.py`

**Usage:**
```bash
uv run python -m skating_biomechanics_ml.cli build-ref expert.mp4 \
    --element waltz_jump --takeoff 1.0 --peak 1.2 --landing 1.4
```

---

### Phase 10: Automatic Segmentation ✅ 90%
**Status:** Mostly working

- [x] ElementSegmenter class
- [x] Motion-based segmentation
- [x] Element type classification
- [x] JSON export
- [x] Segment visualization
- [ ] Refine segment boundaries (includes preparation/recovery)

**Issue:** Segments include too much context (preparation, recovery)
**Priority:** LOW - core functionality works

**Files:** `segmentation/element_segmenter.py`, `scripts/visualize_segmentation.py`

---

### Phase 11: Visualization ✅ 100%
**Status:** Complete

- [x] draw_skeleton() - 33-keypoint overlay
- [x] draw_velocity_vectors() - speed visualization
- [x] draw_trails() - motion history
- [x] draw_edge_indicators() - inside/outside/flat
- [x] draw_debug_hud() - telemetry overlay
- [x] Layered HUD system (0-3)
- [x] Cyrillic text support (Pillow)
- [x] Frame-perfect synchronization

**Files:** `utils/visualization.py`, `scripts/visualize_with_skeleton.py`
**Tests:** `tests/utils/test_visualization.py` (19 passing)

---

### Phase 12: CLI & Pipeline ✅ 100%
**Status:** Complete

- [x] argparse CLI (analyze, build-ref, segment)
- [x] AnalysisPipeline orchestrator
- [x] Russian output
- [x] Help text and examples

**Commands:**
```bash
# Analyze video
uv run python -m skating_biomechanics_ml.cli analyze video.mp4 --element waltz_jump

# Build reference
uv run python -m skating_biomechanics_ml.cli build-ref expert.mp4 --element three_turn

# Segment video
uv run python -m skating_biomechanics_ml.cli segment video.mp4

# Visualize with debug overlay
uv run python scripts/visualize_with_skeleton.py video.mp4 --layer 3
```

**Files:** `cli.py`, `pipeline.py`

---

## Current Blockers

### HIGH Priority
1. **Auto phase detection** - Manual specification required
   - Impact: Cannot fully automate analysis
   - Solution: Implement height-based takeoff, peak, landing detection

### MEDIUM Priority
2. **DTW alignment tests** - Test suite outdated
   - Impact: Cannot verify alignment correctness
   - Solution: Update tests for 33-keypoint format

### LOW Priority
3. **Segment boundaries** - Too broad, includes preparation
   - Impact: Segments not precise
   - Solution: Trim to element core motion

---

## Next Steps (Priority Order)

1. **Fix Auto Phase Detection** ⚠️ HIGH
   - Implement height threshold for takeoff
   - Find minimum hip y for peak
   - Detect landing impact
   - Estimated: 2-3 hours

2. **Fix DTW Tests** ⚠️ MEDIUM
   - Update test data for 33 keypoints
   - Fix shape mismatches
   - Estimated: 1 hour

3. **Commit Current Work** ⚠️ HIGH
   - 36 untracked files
   - All phases need commit
   - Estimated: 30 minutes

4. **Improve Segmentation** 📝 LOW
   - Trim segment boundaries
   - Remove preparation/recovery
   - Estimated: 2-3 hours

---

## Version History

| Version | Date | Status | Notes |
|---------|------|--------|-------|
| 0.1 | 2026-03-27 | MVP 85% | Core pipeline working, visualization complete |

---

## Reference

- Original architecture: `research/RESEARCH.md`
- Visualization research: `research/VISUALIZATION_RESEARCH_PROMPT.md`
- API documentation: See individual module docstrings
