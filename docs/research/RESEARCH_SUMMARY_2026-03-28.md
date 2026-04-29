---
title: "Research Summary: Figure Skating ML System Enhancements"
date: "2026-03-28"
status: active
citekey: "leuthold2025physicsinformed"
---

# Research Summary: Figure Skating ML System Enhancements

**Date:** 2026-03-28  
**Sources:** Exa Web Search + Gemini Deep Research (41 papers)  
**Status:** ACTIVE

---

## Executive Summary

Two parallel research efforts identified key improvements:
1. **Exa MCP Web Search** — fast literature review (5 themes)
2. **Gemini Deep Research** — comprehensive technical report (41 cited papers)

**Critical Finding:** Flight Time method for jump height has **60% error** for low jumps — must use CoM parabolic trajectory instead.

---

## Key Findings by Theme

### Theme 1: Blade Edge Detection

| Source | Accuracy | Method |
|--------|----------|--------|
| Chen 2025 | 79.09% | MediaPipe BDA |
| Tanaka 2023 | 83% | 3D pose + IMU |
| Gemini rec. | 75-80% | Two-stream hybrid (kinematic + CNN patches) |

**Decision:** BDA Algorithm implemented in `utils/blade_edge_detector.py` (19 tests passing). Two-stream CNN future work.

### Theme 2: 2D vs 3D Pose Estimation for Occlusion

| Model | MPJPE | Params | VRAM Fit |
|-------|-------|--------|----------|
| MotionBERT | 39.2mm | 42.5M | No |
| MotionAGFormer-L | 38.4mm | 25.0M | Maybe |
| **Pose3DM-L** | **37.9mm** | **7.43M** | **Yes** |
| **Pose3DM-S** | **42.1mm** | **0.50M** | **Real-time** |
| BlazePose + Physics | -10% MPJPE | <5M | Mobile |

**Key paper:** Leuthold et al. (2025) — Kalman + bone constraints, -10.2% MPJPE, post-processing only (NumPy + scipy).

**Recommendation:** Cascade — BlazePose → Physics optimizer → optional Pose3DM-S.

### Theme 3: Multi-Person Tracking

**Problem:** Black clothing eliminates color Re-ID; white ice = poor contrast.

**Solution:** OC-SORT + anatomical pose biometrics (shoulder/torso, femur/tibia, arm_span/height ratios). No extra NN, works with identical clothing.

| Tracker | Occlusion (>5s) | Same Clothing | Compute |
|---------|-----------------|---------------|---------|
| SORT/DeepSORT | Low | Low | Low |
| ByteTrack | Medium | Low | Low |
| **OC-SORT + Pose Bio** | **High** | **Max** | **Low** |

### Theme 4: Physical Parameter Estimation

**Critical:** Flight time overestimates jump height by 18% (medium) to 60% (low). Skaters land with bent knees.

**Solution:** `CoM(t) = (1/M) × Σ(mᵢ × pᵢ)` — parabolic trajectory during flight, physically accurate.

| Parameter | Recommendation |
|-----------|---------------|
| Height | User input (one-time) |
| Weight | User input + Dempster tables |
| Bone lengths | Physics-informed optimizer |
| Segment masses | Biomechanics tables from total weight |

**Sensitivity:** Moment of inertia I depends on r² (height critical). Angular velocity ω highly sensitive to height error.

### Theme 5: Hierarchical Element Classification

**New datasets:** FSBench (783 videos, 76h+), FSAnno (fine-grained), YourSkatingCoach (BIOES-tagging), MMFS (11,672 clips, 256 classes).

**BIOES-Tagging:** Begin-Inside-Outside-End-Single. Enables precise takeoff/landing frame detection.

**Architecture:** GCN (SkelFormer/SAFSAR) for basic elements → rule-based combinations for complex elements.

---

## Hardware Constraints (RTX 3050 Ti, 4GB VRAM)

| Module | Current | Recommended | VRAM | Time |
|--------|---------|-------------|------|------|
| Detection (YOLOv11n) | ~100MB | Keep | ~100MB | ~10ms |
| Pose (BlazePose) | ~150MB | Keep | ~150MB | ~20ms |
| Physics Optimizer | None | **Add** | ~0MB | ~5ms |
| 3D Pose (Pose3DM-S) | None | Optional | ~200MB | ~30ms |
| OC-SORT | None | **Add** | ~20MB | ~5ms |
| **TOTAL** | **~250MB** | **~520MB** | **4GB** | **~50-90ms** |

All enhancements fit within budget.

---

## Priority Roadmap

### Phase 1: Quick Wins (1-2 days)
1. **Physics-Informed Post-Processor** — Leuthold 2025 Kalman + bone constraints. Expected: -10% MPJPE.
2. **Replace Flight Time with CoM** — parabolic trajectory. Eliminates 60% error for low jumps.

### Phase 2: Tracking (1-2 days)
3. **OC-SORT + Pose Biometrics** — anatomical ratio Re-ID. Prevents skeleton switching.
4. **Integrate Blade Detection** — add to AnalysisPipeline, update rules.

### Phase 3: Advanced (3-5 days)
5. **Pose3DM-S for Occlusion** — evaluate vs BlazePose + Physics.
6. **GCN Element Classifier** — collect YouTube dataset, train SAFSAR-style model.

### Phase 4: Future
7. Two-stream blade detection (MobileNetV3 patches)
8. Audio-visual fusion (skate sound classification)

---

## Open Research Questions

1. **Blade visual ambiguity** — multimodal (audio+visual) solve?
2. **Mass from video** — impossible without reference force?
3. **Motion blur at 4-5 rev/s** — angular momentum constraints help?
4. **Rocker vs Counter** — requires arc history + blade state.

---

## Key References

1. **Leuthold et al. (2025)** — Physics Informed Human Posture Estimation. arXiv:2512.06783. Kalman + bone constraints, -10% MPJPE.
2. **Pose3DM (2025)** — Bidirectional Mamba-Enhanced 3D HPE. 0.5M params, linear complexity.
3. **Chen et al. (2025)** — Automated Blade Type Discrimination. BDA Algorithm, 79.09%.
4. **FSBench (CVPR 2025)** — Figure Skating Benchmark. 783 videos, FSBench/FSAnno.
5. **YourSkatingCoach (2024)** — Fine-Grained Element Analysis. BIOES-tagging.
6. **Zhou et al. (2024)** — AQA Survey. arXiv:2412.11149.

**Full 41-paper analysis preserved in git history.** This file is a compressed decision record.
