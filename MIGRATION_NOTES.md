# 3D-Only Migration Notes

**Date:** 2026-03-29
**Branch:** feature/3d-only-migration
**Baseline:** 255 tests passing, 62% coverage

---

## Migration Goal

Transition from **BlazePose 2D (33 keypoints)** to **H3.6M 3D (17 keypoints)** as the primary pose format.

---

## Baseline Metrics (2D)

- **Test Pass Rate:** 255/255 (100%)
- **Code Coverage:** 62%
- **Pose Format:** (N, 33, 2) normalized coordinates
- **Keypoints:** 33 (BlazePose format)

---

## Breaking Changes

### Lost Keypoints (16 total)
- **Eyes:** LEFT_EYE_INNER, LEFT_EYE, LEFT_EYE_OUTER, RIGHT_EYE_INNER, RIGHT_EYE, RIGHT_EYE_OUTER
- **Ears:** LEFT_EAR, RIGHT_EAR
- **Mouth:** MOUTH_LEFT, MOUTH_RIGHT
- **Hands (fingers):** LEFT_PINKY, LEFT_INDEX, LEFT_THUMB, RIGHT_PINKY, RIGHT_INDEX, RIGHT_THUMB
- **Feet (detail):** LEFT_HEEL, LEFT_FOOT_INDEX, RIGHT_HEEL, RIGHT_FOOT_INDEX

### Affected Features
- **Blade detection:** Loses heel/toe keypoints, must use 3D physics
- **Arm position:** Loses hand keypoints, uses wrist-to-shoulder distance
- **Face tracking:** Loses eye/ear/mouth keypoints

---

## Rollback Plan

If migration fails:
1. Keep `master` branch stable
2. Use `src/compat.py` feature flags for gradual rollout
3. Revert individual commits if needed
4. Delete feature branch

**Trigger for rollback:**
- Test pass rate < 80%
- Critical metrics error > 20%
- Performance degradation > 2x

---

## Progress Tracking

- [x] Phase 0: Preparation & Safety
- [x] Phase 1: Core Type System (BKey → H36Key) ✅ COMPLETE
- [x] Phase 2: Pipeline Core (2D extraction → 3D) ✅ COMPLETE
  - 263 tests passing
  - Coverage: 59%
  - Files created: pose_3d/normalizer_3d.py
  - Files updated: normalizer.py, smoothing.py, conftest.py, test files
- [x] Phase 3: Metrics & Analysis Migration ✅ COMPLETE
  - Files updated: metrics.py, geometry.py, element_defs.py
  - All BKey references → H36Key
  - Re-implemented edge detection for 3D (body lean proxy)
  - Re-implemented CoM calculation for H3.6M (HEAD instead of NOSE, LFOOT/RFOOT instead of ankles)
- [x] Phase 4: Visualization Migration ✅ COMPLETE
  - Files updated: visualization.py
  - Skeleton edges updated to 17kp H3.6M format
  - draw_edge_indicators deprecated for H3.6M (no heel/toe points)
  - All visualization functions updated for H36Key
- [x] Phase 5: Integration & Cleanup ✅ COMPLETE
  - Updated pipeline.py to use BladeEdgeDetector3D with 3D poses
  - Updated phase_detector.py to make blade detection optional (CoM-based primary)
  - Updated scripts/visualize_with_skeleton.py to require --blade-3d for edge detection
  - Legacy 2D blade_edge_detector.py kept for reference but no longer used by default

## Remaining BKey References (Intentional)

- `src/blade_edge_detector.py` - Legacy 2D version (deprecated, use blade_edge_detector_3d.py instead)

## Migration Complete!

The core analysis pipeline now uses H3.6M 17kp 3D format throughout:
- Type system: H36Key enum with backward compatibility aliases
- Metrics: All biomechanics calculations work with 17kp
- Visualization: Skeleton, velocity, trails all updated
- Pipeline: 3D-first architecture with optional 2D fallback
- Test suite: 263 tests passing, 59% coverage

---

## Phase 6: Cleanup & Deprecation ✅ COMPLETE

### Files Cleaned:
1. **src/compat.py** - REMOVED (unused migration compatibility layer)
2. **src/pipeline.py** - Removed unused imports (pose_extractor, reference_builder)
3. **src/phase_detector.py** - Removed unused imports (geometry, video, smoothing, subtitles, visualization)

### Deprecation Notices Added:
1. **src/blade_edge_detector.py** - Added deprecation warning
   - Docstring now indicates it's deprecated for H3.6M format
   - __init__ emits DeprecationWarning when instantiated
   - Users directed to use BladeEdgeDetector3D instead

### Test Status:
- 263 tests passing
- 15 warnings (expected: 10 deprecation warnings from 2D blade detector tests)
- All other warnings are pre-existing (compute_jump_height, CUDA)

### Legacy Code Status:
- **src/blade_edge_detector.py** - DEPRECATED but kept for backward compatibility
- **tests/utils/test_blade_edge_detector.py** - Tests pass with deprecation warnings
- **src/blazepose_extractor.py** - DEPRECATED (raises NotImplementedError, use H36MExtractor instead)
- **src/pose_estimation/h36m_extractor.py** - Contains deprecated blazepose_to_h36m() function (raises NotImplementedError)
