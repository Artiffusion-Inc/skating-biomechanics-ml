# CLAUDE.md

> **PROJECT ROADMAP:** @ROADMAP.md — SINGLE SOURCE OF TRUTH for implementation status
> **RESEARCH:** @research/RESEARCH_SUMMARY_2026-03-28.md — Exa + Gemini findings (41 papers)

---

## Project Overview

ML-based AI coach for figure skating. Analyzes video, compares attempts to professional references, provides biomechanical feedback in Russian.

**Vision:** AI-тренер по фигурному катанию — анализ видео и рекомендации на русском.

## Tech Stack

| Component | Technology |
|-----------|-----------|
| **Language** | Python 3.11+ (`uv`) |
| **2D Pose** | RTMPose via rtmlib (HALPE26, 26kp with feet) **default** |
| **2D Pose (alt)** | YOLO26-Pose (H3.6M, 17kp) |
| **3D Lifting** | MotionAGFormer-S / Biomechanics3DEstimator |
| **3D Correction** | CorrectiveLens (kinematic constraints + anchor projection) |
| **Tracking** | PoseTracker (OC-SORT + anatomical biometrics) |
| **Alignment** | DTW (dtw-python) with Sakoe-Chiba window |
| **Analysis** | CoM trajectory, physics engine (Dempster tables) |
| **GPU** | CUDA via onnxruntime-gpu (7.1x speedup) |
| **Testing** | pytest + pytest-cov (279+ tests) |

## Architecture

```
Video → RTMPose (rtmlib, CUDA) → HALPE26 (26kp)
  → H3.6M (17kp) conversion → GapFiller → Smoothing
  → [Optional] CorrectiveLens (3D lift → kinematic constraints → project back to 2D)
  → Phase Detection → Biomechanics Metrics → DTW (vs reference)
  → Rule-based Recommender → Russian Text Report
```

**Key decisions:**
- **rtmlib > YOLO-Pose** for 2D: better tracking, foot keypoints, ONNX (fast on CPU, CUDA on GPU)
- **HALPE26 (26kp)** as intermediate format, converted to H3.6M (17kp) for downstream
- **CorrectiveLens**: uses 3D lifting as corrective layer for 2D skeleton (Kinovea-style angles)
- **PoseTracker**: anatomical biometric Re-ID instead of color (solves black clothing on ice)
- **CoM trajectory** instead of flight time (eliminates 60% error for low jumps)

---

## Git Workflow

Commit types: `feat`, `fix`, `docs`, `refactor`, `test`, `chore`, `perf`

```
feat(pose): add RTMPoseExtractor with HALPE26 foot keypoints
fix(tracking): give each track its own Kalman state
perf(viz): add frame_skip, render-scale for 3x speedup
```

Pre-commit: `uv run pytest tests/ -v -m "not slow"` + `uv run ruff check .`

## GPU Requirements

**GPU-only. CPU inference is forbidden.** Always use `device='cuda'`.
Before running: `bash scripts/setup_cuda_compat.sh` (required after `uv sync`).
System has CUDA 13.2, onnxruntime-gpu needs CUDA 12 compat libs in `.venv/cuda-compat/`.

---

## Project Structure

```
src/
├── types.py                          # H36Key, BladeType, PersonClick, TrackedExtraction
├── pipeline.py                       # AnalysisPipeline orchestrator
├── cli.py                            # argparse CLI (analyze, build-ref, segment, compare)
├── pose_estimation/
│   ├── rtmlib_extractor.py           # RTMPose via rtmlib (HALPE26, tracking, CUDA)
│   ├── h36m_extractor.py             # YOLO26-Pose (H3.6M, tracked extraction)
│   ├── halpe26.py                    # HALPE26 constants + H3.6M mapping + foot angles
│   └── normalizer.py                 # Root-centering + scale normalization
├── pose_3d/
│   ├── corrective_pipeline.py        # CorrectiveLens: 3D→2D corrective overlay
│   ├── kinematic_constraints.py      # Bone length + joint angle limits (3D)
│   ├── anchor_projection.py          # 3D→2D projection + confidence blending
│   ├── athletepose_extractor.py      # MotionAGFormer / TCPFormer wrapper
│   └── biomechanics_estimator.py     # Simple 3D estimation (no model)
├── detection/
│   ├── pose_tracker.py               # OC-SORT + anatomical biometric Re-ID
│   ├── spatial_reference.py          # Per-frame camera pose estimation
│   └── blade_edge_detector_3d.py     # 3D blade edge detection (BDA)
├── analysis/
│   ├── physics_engine.py             # CoM, parabolic trajectory, Dempster tables
│   ├── phase_detector.py             # CoM-based auto takeoff/peak/landing
│   ├── metrics.py                    # BiomechanicsAnalyzer
│   └── recommender.py                # Rule-based Russian recommendations
├── visualization/
│   ├── comparison.py                 # ComparisonRenderer (side-by-side, overlay)
│   ├── layers/                       # skeleton, velocity, trail, blade, joint_angle, timer, vertical_axis
│   ├── hud/                          # HUD elements, layout, panel
│   └── skeleton/                     # Skeleton drawing (2D/3D, joints)
├── utils/
│   ├── gap_filling.py                # GapFiller (linear interp + velocity extrapolation)
│   ├── geometry.py                   # Angles, distances, foot angles
│   └── smoothing.py                  # One-Euro Filter, PoseSmoother
└── references/
    ├── reference_builder.py          # Build reference from expert video
    └── reference_store.py            # Save/load .npz

scripts/
├── visualize_with_skeleton.py        # Main viz script (layered HUD, --3d, --pose-backend)
├── setup_cuda_compat.sh              # CUDA 12 compat for onnxruntime on CUDA 13.x
├── check_all.py                      # Quality checks
└── download_models.py                # Download model weights

tests/
├── pose_3d/                          # 37 tests (corrective pipeline)
├── detection/                        # Tracker tests
├── analysis/                         # Metrics, physics, recommender
└── alignment/                        # DTW aligner
```

---

## CLI Usage

```bash
# Analyze video (full pipeline)
uv run python -m src.cli analyze video.mp4 --element waltz_jump --pose-backend rtmlib

# Build reference from expert video
uv run python -m src.cli build-ref expert.mp4 --element waltz_jump

# Compare two videos (training mode)
uv run python -m src.cli compare attempt.mp4 reference.mp4 --overlays skeleton,angles,timer

# Visualize with 3D-corrected skeleton
uv run python scripts/visualize_with_skeleton.py video.mp4 --layer 2 --3d --output out.mp4

# Interactive person selection
uv run python -m src.cli analyze video.mp4 --element three_turn --select-person
```

### Visualization Options

```bash
--pose-backend rtmlib|yolo   # Pose estimation backend
--3d                         # Enable 3D-corrected 2D overlay (CorrectiveLens)
--layer 0-3                  # HUD layer (0=skeleton, 3=full coaching HUD)
--render-scale 0.5           # Downscale rendering for speed
--frame-skip 8               # Process every Nth frame only
--select-person              # Interactive person selection
--overlays skeleton,angles   # Comparison overlays
```

---

## Environment

- **OS**: Artix Linux (Ryzen 7 5800H / RTX 3050 Ti 4GB VRAM)
- **CUDA**: 13.2 system, onnxruntime-gpu uses CUDA 12 compat libs
- **GPU Setup**: `bash scripts/setup_cuda_compat.sh` after `uv sync`

## Supported Elements

| Element | Type | Key Metrics |
|---------|------|-------------|
| `three_turn` | Step | trunk_lean, edge_change, knee_angle |
| `waltz_jump` | Jump | airtime, max_height, landing_knee |
| `toe_loop` | Jump | airtime, rotation_speed, toe_pick |
| `flip` | Jump | airtime, pick_quality |
| `salchow` | Jump | airtime, rotation_speed |
| `loop` | Jump | airtime, height |
| `lutz` | Jump | toe_pick_quality, rotation |
| `axel` | Jump | height, rotation |

---

## Key Concepts

### Coordinate Convention

- `poses_norm` — Normalized [0,1]
- `poses_px` — Pixel coordinates
- Validate with `assert_pose_format()` from `types.py`

### HALPE26 → H3.6M Mapping

`halpe26_to_h36m()` converts 26kp (COCO 17 + 6 foot + 3 face) to 17kp H3.6M format. Foot keypoints (heel, big_toe, small_toe) preserved separately for blade edge detection.

### CorrectiveLens (3D→2D)

```
RTMPose 2D → MotionAGFormer 3D lift → kinematic constraints → anchor-based projection → blend with raw 2D
```

- Bone length enforcement (iterative Jacobian)
- Joint angle limits (knees 0-180°, elbows 0-160°, hips 30-180°)
- Per-frame scale from torso ratio (no camera calibration needed)
- Confidence-based blending (trust corrected at low confidence)

### CUDA Compatibility

System has CUDA 13.2, onnxruntime-gpu needs CUDA 12. Solution: standalone CUDA 12 libs in `.venv/cuda-compat/` with patched RUNPATH. Script `setup_cuda_compat.sh` automates this.

---

## Performance

| Config | 364 frames (14.5s video) | 1800 frames (60s video) |
|--------|--------------------------|------------------------|
| CPU (rtmlib, frame_skip=8) | ~50s | ~247s |
| **GPU (rtmlib, frame_skip=8)** | **~12s** | **~59s** |
| GPU + render-scale 0.5 | ~10s | ~49s |
| GPU + render-scale 0.33 | ~8s | ~40s |

---

## Known Issues

1. **Distant skaters**: rtmpib may miss very small figures (<10% frame width). Use `--person-click X Y` or `--select-person`.
2. **CUDA compat**: Must run `setup_cuda_compat.sh` after `uv sync` on this system.
3. **Segment boundaries**: Phase 10 includes preparation/recovery in segments.

---

## References

- @ROADMAP.md — project status (SINGLE SOURCE OF TRUTH)
- @research/RESEARCH_SUMMARY_2026-03-28.md — research findings (41 papers)
- @research/RESEARCH.md — research memory bank (index)
- @MIGRATION_NOTES.md — BlazePose 33kp → H3.6M 17kp migration details
