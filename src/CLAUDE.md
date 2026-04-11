# src/CLAUDE.md — ML Pipeline

## Project Structure

```
src/
├── types.py                          # H36Key, BladeType, PersonClick, TrackedExtraction
├── pipeline.py                       # AnalysisPipeline orchestrator
├── cli.py                            # argparse CLI (analyze, build-ref, segment, compare)
├── pose_estimation/
│   ├── rtmlib_extractor.py           # RTMPose via rtmlib (HALPE26, tracking, CUDA)
│   ├── h36m.py                       # H3.6M constants, skeleton edges, conversion functions
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
├── storage.py                        # S3-compatible upload/download (Cloudflare R2)
├── vastai/
│   ├── __init__.py                   # Package marker
│   └── client.py                     # Vast.ai Serverless: route → process → download
├── worker.py                         # arq worker (remote GPU dispatch + local fallback)
├── config.py                         # pydantic-settings (Valkey, R2, Vast.ai)
├── web_helpers.py                    # process_video_pipeline(), ModelRegistry
└── references/
    ├── reference_builder.py          # Build reference from expert video
    └── reference_store.py            # Save/load .npz

tests/
├── pose_3d/                          # Corrective pipeline tests
├── detection/                        # Tracker tests
├── analysis/                         # Metrics, physics, recommender
└── alignment/                        # DTW aligner
```

## CLI Usage

```bash
uv run python -m src.cli analyze video.mp4 --element waltz_jump
uv run python -m src.cli build-ref expert.mp4 --element waltz_jump
uv run python -m src.cli compare attempt.mp4 reference.mp4 --overlays skeleton,angles,timer
uv run python scripts/visualize_with_skeleton.py video.mp4 --layer 2 --3d --output out.mp4
```

### Visualization Options

```
--3d                         # Enable 3D-corrected 2D overlay (CorrectiveLens)
--layer 0-3                  # HUD layer (0=skeleton, 3=full coaching HUD)
--render-scale 0.5           # Downscale rendering for speed
--frame-skip 8               # Process every Nth frame only
--select-person              # Interactive person selection
--overlays skeleton,angles   # Comparison overlays
```

## GPU Requirements

**GPU-only. CPU inference is forbidden.** Always use `device='cuda'`.
Before running: `bash scripts/setup_cuda_compat.sh` (required after `uv sync`).
System has CUDA 13.2, onnxruntime-gpu needs CUDA 12 compat libs in `.venv/cuda-compat/`.

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

## Key Concepts

- `poses_norm` — Normalized [0,1], `poses_px` — Pixel coordinates. Validate with `assert_pose_format()`.
- `halpe26_to_h36m()`: 26kp (COCO 17 + 6 foot + 3 face) → 17kp H3.6M. Foot keypoints preserved separately.
- **RTMPose (rtmlib)** is the sole pose estimation backend. HALPE26 26kp extracted, converted to H3.6M 17kp for downstream analysis.
- **CorrectiveLens**: 2D → MotionAGFormer 3D lift → kinematic constraints → anchor projection → blend.
- **CUDA compat**: standalone CUDA 12 libs in `.venv/cuda-compat/` with patched RUNPATH.

## Remote GPU Processing (Vast.ai Serverless)

Worker dispatches to Vast.ai Serverless GPU when `VASTAI_API_KEY` is set, falls back to local GPU.

```
Frontend → FastAPI → Valkey queue → arq worker
  → [VASTAI_API_KEY set?]
    → YES: upload to R2 → Vast.ai route → GPU worker → download from R2
    → NO:  local GPU (process_video_pipeline)
```

**Image**: `ghcr.io/xpos587/skating-ml-gpu:latest` — multi-stage, 4.9GB, no torch/timm/triton.

## Tracking Debugging Workflow

When tracking quality degrades (skeleton jumps to wrong person), follow this data-driven analysis approach. **Do NOT guess — extract data and find the exact divergence frame.**

### Step 1: Isolate the layer

The tracking pipeline has 3 layers that can independently cause track switches:
1. **Sports2DTracker** — per-frame centroid association (Kalman-predicted distance matrix)
2. **Anti-steal logic** — in `rtmlib_extractor.py`, guards against centroid jumps
3. **Tracklet merger** — post-hoc NaN gap filling with biometric re-association

### Step 2: Analyze centroid trajectories

In the CSV, look for:
- **Sports2D misassignment**: track ID on wrong detection index without anti-steal trigger
- **Anti-steal false positive**: `target_track_id` switched even though Sports2D assigned correctly
- **Tracklet merger error**: wrong track merged into the gap

### Step 3: Fix the root cause

Key lessons learned:
- **Anti-steal must use AND (not OR)** for combining position + biometric signals.
- **Kalman dt=1.0 (frame-based), not dt=1/fps**.
- **Figure skating movements are NOT anomalies** — leg swings, rotations are normal.

### Anti-steal thresholds

- Centroid jump: `> 0.15` (normalized coordinates)
- Skeletal anomaly: `> 0.25` (bone ratio change)
- **Logic: AND** — both must exceed threshold simultaneously

## Known Issues

1. **Distant skaters**: rtmpose may miss very small figures (<10% frame width). Use `--person-click X Y` or `--select-person`.
2. **CUDA compat**: Must run `setup_cuda_compat.sh` after `uv sync` on this system.
3. **Segment boundaries**: Phase 10 includes preparation/recovery in segments.
4. **Foot keypoints on skates**: RTMPose foot keypoints (HALPE26 indices 17-22) are unreliable on ice skates.
5. **Multi-bbox per person**: RTMPose sometimes produces multiple bounding boxes for the same person. NMS deduplication may be needed.

## Before Committing

1. **Tests**: `lefthook run test`
2. **Type check**: `lefthook run typecheck`
3. **Lint**: `lefthook run format`
4. All checks must pass. Lefthook pre-commit hooks run automatically.
