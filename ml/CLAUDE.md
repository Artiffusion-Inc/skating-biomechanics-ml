# ml/CLAUDE.md — ML Pipeline & arq Worker

## Architectural Constraint

**ml use backend for infra, never for ML.** `ml/` = pure ML lib. No DB, queue, web framework knowledge. arq worker (orchestrate ML dispatch) live in `backend/app/`, import `ml/` for types (`H36Key`, `VastResult`) and GPU dispatch.

## Project Structure

```
ml/
├── src/                       # Python package (src.*)
│   ├── __init__.py                   # Exports DeviceConfig
│   ├── types.py                      # Core types: H36Key, FrameKeypoints, PersonClick, etc.
│   ├── device.py                     # DeviceConfig — GPU/CPU auto-detection
│   ├── pipeline.py                   # AnalysisPipeline orchestrator
│   ├── web_helpers.py                # Preview rendering for detect endpoint
│   ├── pose_estimation/              # 2D pose extraction
│   │   ├── pose_extractor.py       # PersonDetector + MogaNetBatch (COCO 17kp) — PRIMARY
│   │   ├── h36m.py                   # H3.6M 17kp format handling
│   │   ├── halpe26.py                # DELETED - no longer used (MogaNet-B uses COCO 17kp)
│   │   ├── normalizer.py             # Root-centering + scale normalization
│   │   └── person_selector.py        # Interactive person selection
│   ├── pose_3d/                      # 3D pose lifting
│   │   ├── corrective_pipeline.py    # CorrectiveLens: 3D lift → constraints → project
│   │   ├── kinematic_constraints.py  # Bone length + joint angle enforcement
│   │   ├── anchor_projection.py      # 3D→2D projection with confidence blending
│   │   ├── athletepose_extractor.py  # MotionAGFormer / TCPFormer integration
│   │   └── normalizer_3d.py          # 3D pose normalization
│   ├── analysis/                     # Biomechanics analysis
│   │   ├── metrics.py                # Airtime, height, knee angles, rotation, etc.
│   │   ├── phase_detector.py         # CoM-based takeoff/peak/landing detection
│   │   ├── physics_engine.py         # CoM, moment of inertia, angular momentum
│   │   ├── recommender.py            # Rule-based Russian text recommendations
│   │   ├── element_defs.py           # Figure skating element definitions
│   │   ├── element_segmenter.py      # Automatic element segmentation
│   │   └── rules/                    # Per-element recommendation rules
│   ├── detection/                    # Person detection & tracking
│   │   ├── person_detector.py        # YOLO-based person detection
│   │   ├── pose_tracker.py           # Kalman filter + biometric Re-ID
│   │   ├── spatial_reference.py      # Per-frame camera pose estimation
│   │   └── blade_edge_detector_3d.py # BDA algorithm (not wired into pipeline)
│   ├── tracking/                     # Multi-person tracking
│   │   ├── sports2d.py               # Sports2D centroid association
│   │   ├── tracklet_merger.py        # NaN gap filling with biometric re-association
│   │   ├── deepsort_tracker.py       # DeepSORT integration
│   │   └── skeletal_identity.py      # Anatomical ratio Re-ID
│   ├── alignment/                    # Motion comparison
│   │   ├── aligner.py                # MotionAligner class
│   │   └── motion_dtw.py             # DTW with Sakoe-Chiba window
│   ├── visualization/                # Rendering & HUD
│   │   ├── comparison.py             # Side-by-side / overlay comparison
│   │   ├── export_3d.py              # 3D skeleton export (glTF)
│   │   ├── export_3d_animated.py     # Animated 3D export
│   │   ├── hud/                      # HUD layer system (0-3)
│   │   ├── skeleton/                 # Skeleton drawer
│   │   └── layers/                   # Individual overlay layers
│   ├── utils/                        # Shared utilities
│   │   ├── video.py                  # extract_frames, get_video_meta
│   │   ├── geometry.py               # Angles, distances
│   │   ├── smoothing.py              # One-Euro filter
│   │   ├── gap_filling.py            # 3-tier gap filling
│   │   └── subtitles.py              # VTT subtitle generation
│   ├── references/                   # Reference motion database
│   │   ├── reference_builder.py      # Build .npz references from video
│   │   └── reference_store.py        # Save/load reference files
│   ├── datasets/                     # Dataset handling
│   │   ├── coco_builder.py           # COCO format builder
│   │   └── projector.py              # 3D projection utilities
│   └── extras/                       # Optional ML models (not in pipeline)
│       ├── model_registry.py         # Model download/management
│       ├── depth_anything.py         # Depth estimation
│       ├── optical_flow.py           # Optical flow
│       ├── segment_anything.py       # SAM segmentation
│       ├── inpainting.py             # Video inpainting
│       └── foot_tracker.py           # Foot tracking
├── gpu_server/                       # Vast.ai GPU server
│   ├── server.py                     # FastAPI server for GPU worker
│   └── Containerfile                 # Multi-stage build (4.9GB)
├── scripts/                          # Standalone scripts
│   ├── cli.py                        # Backend API CLI — upload, enqueue, poll, JSON stdout
│   ├── visualize_with_skeleton.py    # Main viz script (--layer, --3d, --select-person)
│   ├── setup_cuda_compat.sh          # CUDA 12 compat libs for CUDA 13.x
│   ├── download_ml_models.py         # Download model weights
│   ├── normalize_video.py            # H.264, 1280px, 30fps
│   ├── compare_videos.py             # Side-by-side analysis comparison
│   ├── build_references.py           # Build reference database
│   └── deploy.sh                     # Deploy to Vast.ai
├── tests/                            # 93+ tests
│   ├── conftest.py                   # Shared fixtures
│   ├── analysis/                     # Metrics, phase detector, recommender
│   ├── detection/                    # Person detector, pose tracker
│   ├── tracking/                     # DeepSORT, skeletal identity
│   ├── visualization/                # HUD, skeleton, layers
│   ├── alignment/                    # DTW, aligner
│   ├── pose_3d/                      # Corrective lens, anchor projection
│   └── utils/                        # Geometry, smoothing
└── pyproject.toml                    # ML deps (pure library, no backend deps)
```

## Key Types (`src.types`)

| Type | Purpose |
|------|---------|
| `H36Key` (IntEnum) | H3.6M 17-keypoint indices. Primary format |
| `FrameKeypoints` | Single frame: (17, 2) or (17, 3) array |
| `PersonClick` | User person selection (x, y, frame) |
| `TrackedExtraction` | Per-person tracked pose sequence |
| `ElementPhase` | takeoff/flight/landing phase markers |
| `BladeType` (Enum) | INSIDE/OUTSIDE/FLAT/TOE_PICK/UNKNOWN |

## Pipeline Flow

```
Video → PoseExtractor (PersonDetector + MogaNetBatch, COCO 17kp)
     → GapFiller → Smoothing (One-Euro)
     → [Optional] CorrectiveLens (3D lift → constraints → project back)
     → PhaseDetector (CoM-based)
     → BiomechanicsAnalyzer (airtime, height, angles, rotation)
     → DTW alignment vs reference
     → Recommender → Russian text report
```

## GPU Requirements

**GPU only. CPU inference forbidden.** Always use `device='cuda'`.

```bash
# Required after every uv sync
bash ml/scripts/setup_cuda_compat.sh
```

System has CUDA 13.2, onnxruntime-gpu needs CUDA 12 compat libs in `.venv/cuda-compat/`.

## Device Configuration

```python
from src.device import DeviceConfig

cfg = DeviceConfig.default()        # Auto-detect (CUDA preferred)
cfg = DeviceConfig(device="cpu")    # Explicit CPU
cfg.onnx_providers                  # ["CUDAExecutionProvider", "CPUExecutionProvider"]

# Environment override: SKATING_DEVICE=cpu
```

## Numba JIT Optimizations

Compute-heavy fn use Numba JIT:

- **Geometry:** `_angle_3pt_rad_numba()`, `angle_3pt_batch()` - 5M+ ops/sec
- **Smoothing:** `_one_euro_filter_sequence_numba()`, `smooth_trajectory_2d_numba()` - 44M+ frames/sec
- **Metrics:** `_compute_knee_angle_series_numba()`, `_compute_trunk_lean_series_numba()` - 50K+ ops/sec

**Expected speedup:** 10-100x repeated ops (after JIT compile).

**Usage:** Auto. No API change. First call slow (compile), later fast.

**Benchmark:** `uv run python ml/scripts/benchmark_numba.py`

**Note:** TensorRT experimental. ONNX default for serverless compat.

## Worker Jobs (`backend/app.worker`)

Two async arq jobs:

| Job | Trigger | What it does |
|-----|---------|-------------|
| `process_video_task` | `POST /process` | Full ML pipeline → save to DB |
| `detect_video_task` | `POST /detect` | PersonDetector + MogaNetBatch → render preview → store Valkey |

Both jobs download video from R2, process on GPU, store results. When `VASTAI_API_KEY` set, dispatch to Vast.ai Serverless GPU.

## CorrectiveLens (Disabled by Default)

3D lift as corrective layer for 2D skeleton. ~3px max shift. Not worth compute for most. Enable with `--3d` flag in viz scripts.

## Tracking Debugging

Skeleton jump to wrong person → follow data-driven approach in @CLAUDE.md (Tracking Debugging Workflow).

## CLI for Remote Processing (`scripts/cli.py`)

Use `scripts/cli.py` to dispatch video processing to Vast.ai Serverless GPU via the backend API. No local GPU needed.

```bash
cd ml

# Authenticate (stores JWT in ~/.config/skating-cli/credentials.json, mode 600)
uv run python scripts/cli.py login

# Analyze a video — upload → enqueue → poll → JSON stdout
uv run python scripts/cli.py analyze /path/to/video.mov --element waltz_jump

# Or via go-task
go-task cli-analyze VIDEO=/path/to/video.mov ELEMENT=waltz_jump
```

**Auth flow:** `login` → save refresh token → each request auto-refreshes access token on expiry.

**Output:** Only JSON to stdout. Progress to stderr.

**Backend API endpoints used:** `POST /auth/login`, `POST /auth/refresh`, `GET /users/me`, `POST /uploads/init`, `PUT` presigned chunks, `POST /uploads/complete`, `POST /process/queue`, `GET /process/{task_id}/status`.

## Before Committing

1. **Tests**: `uv run python -m pytest ml/tests/ --no-cov`
2. **Lint**: `uv run ruff check ml/src/`
3. **Type check**: `uv run basedpyright ml/src/ --level error`
