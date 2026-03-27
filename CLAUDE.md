# CLAUDE.md

> **⚠️ PROJECT ROADMAP:** See @ROADMAP.md for the SINGLE SOURCE OF TRUTH on implementation status, phases, and blockers.
>
> This file provides project context and development conventions.

## Project Overview

ML-based personal AI coach for figure skating using computer vision. Analyzes skating technique from video and provides specific recommendations in Russian.

## Tech Stack (MVP)

| Component           | Technology                               |
| ------------------- | ---------------------------------------- |
| **Language**        | Python 3.11+                             |
| **Package Manager** | `uv`                                     |
| **Detection**       | YOLOv11n (Ultralytics)                   |
| **2D Pose**         | MediaPipe BlazePose (33 keypoints)       |
| **Normalization**   | Root-centering + scale normalization     |
| **Alignment**       | DTW (dtw-python) with Sakoe-Chiba window |
| **Analysis**        | Custom biomechanics metrics              |
| **Recommendations** | Rule-based engine (Russian output)       |
| **Testing**         | Pytest + pytest-cov                      |

## MVP Architecture (2D-only)

```
Video Input → YOLOv11n (detect) → BlazePose (2D keypoints) → Normalization
    ↓
Phase Detection → Biomechanics Metrics → DTW (vs reference)
    ↓
Rule-based Recommender → Text Report (Russian)
```

**Key Decision**: MVP uses 2D normalized poses instead of 3D lifting. This simplifies the pipeline while providing sufficient information for basic biomechanics analysis. 3D lifting can be added later as an enhancement.

## Project Structure

```
src/skating_biomechanics_ml/
├── types.py              # Shared data types (BKey, FrameKeypoints, etc.)
├── pipeline.py           # Main AnalysisPipeline orchestrator
├── cli.py                # argparse CLI (analyze, build-ref commands)
├── detection/
│   └── person_detector.py    # YOLOv11n wrapper
├── pose_2d/
│   ├── pose_extractor.py     # BlazePose wrapper
│   └── normalizer.py          # Root-centering, scale normalization
├── analysis/
│   ├── metrics.py             # BiomechanicsAnalyzer (airtime, angles, etc.)
│   ├── phase_detector.py      # Auto-detect takeoff/peak/landing
│   ├── recommender.py         # Rule-based recommendation engine
│   └── rules/
│       ├── jump_rules.py      # Rules for waltz_jump, toe_loop, flip
│       └── three_turn_rules.py # Rules for three_turn
├── alignment/
│   └── aligner.py             # DTW motion alignment
├── references/
│   ├── element_defs.py        # Element definitions & ideal metrics
│   ├── reference_builder.py   # Build reference from expert video
│   └── reference_store.py     # Store/load .npz reference files
└── utils/
    ├── video.py               # cv2 video utilities
    ├── geometry.py            # Angles, distances, smoothing
    ├── smoothing.py           # One-Euro Filter for pose smoothing
    ├── visualization.py       # Skeleton, kinematics, HUD drawing
    └── subtitles.py           # VTT subtitle parser for coach commentary

scripts/
├── check_all.py               # Run all quality checks
├── build_references.py        # CLI to build references from video
├── download_models.py         # Download YOLOv11n weights
└── visualize_with_skeleton.py # Enhanced debug visualization with layered HUD

tests/
├── conftest.py            # Shared fixtures
├── test_types.py          # Type tests
├── test_pipeline.py       # Integration tests
├── detection/             # PersonDetector tests
├── pose_2d/               # BlazePose tests
├── analysis/              # Metrics, phase detector, recommender tests
└── alignment/             # DTW aligner tests

data/
└── references/            # Expert reference .npz files (not in git)
    ├── three_turn/
    ├── waltz_jump/
    ├── toe_loop/
    └── flip/
```

## Development Workflow

### Quality Checks

```bash
# Run all checks
uv run python scripts/check_all.py

# Individual checks
uv run ruff check .          # Lint
uv run ruff format .         # Format
uv run mypy src/             # Type check
uv run vulture src/ tests/   # Dead code
uv run pytest tests/ -v -m "not slow"  # Tests (exclude slow ML tests)
```

### CLI Usage

```bash
# Analyze a skating video
uv run python -m skating_biomechanics_ml.cli analyze video.mp4 --element three_turn

# Build reference from expert video
uv run python -m skating_biomechanics_ml.cli build-ref expert.mp4 --element waltz_jump \\
    --takeoff 1.0 --peak 1.2 --landing 1.4

# With reference directory
uv run python -m skating_biomechanics_ml.cli analyze video.mp4 --element waltz_jump \\
    --reference-dir data/references --output report.txt
```

### Supported Elements

| Element      | Type | Key Metrics                                           |
| ------------ | ---- | ----------------------------------------------------- |
| `three_turn` | Step | trunk_lean, edge_change_smoothness, knee_angle        |
| `waltz_jump` | Jump | airtime, max_height, landing_knee_angle, arm_position |
| `toe_loop`   | Jump | airtime, rotation_speed, toe_pick_timing              |
| `flip`       | Jump | airtime, pick_quality, air_position                   |

## Key Concepts

### Data Types

- **FrameKeypoints**: `(N, 33, 3)` — x, y, confidence from BlazePose (pixel coords)
- **NormalizedPose**: `(N, 33, 2)` — x, y in [0,1] normalized coordinates
- **PixelPose**: `(N, 33, 2)` — x, y in pixel coordinates
- **ElementPhase**: start, takeoff, peak, landing, end frame indices
- **MetricResult**: name, value, unit, is_good, reference_range

### Coordinate System Convention (CRITICAL)

**Always clarify coordinate system in variable names and function signatures:**

```python
# Naming convention
poses_norm = ...  # Normalized [0,1]
poses_px = ...    # Pixel coordinates

# Use validation from types.py
from skating_biomechanics_ml.types import assert_pose_format
assert_pose_format(poses, "normalized", context="my_function")

# Convert between formats
from skating_biomechanics_ml.types import normalize_pixel_poses, pixelize_normalized_poses
poses_norm = normalize_pixel_poses(poses_px, width=1920, height=1080)
poses_px = pixelize_normalized_poses(poses_norm, width=1920, height=1080)
```

**Visualization functions expect NORMALIZED coordinates:**

- `draw_velocity_vectors()` → normalized [0,1]
- `draw_trails()` → normalized [0,1]
- `draw_edge_indicators()` → normalized [0,1]
- `draw_skeleton()` → both (handles conversion internally)

**Common bugs to avoid:**

1. Passing pixel coords to functions expecting normalized → wrong calculations
2. Passing normalized coords to functions expecting pixels → skeleton misaligned
3. Smoothing in wrong coordinate space → inconsistent results

### Normalization

1. **Root-centering**: mid-hip → origin (0, 0)
2. **Scale normalization**: spine length → 0.4 (typical adult athlete)

### Biomechanics Metrics

- **Airtime**: `(landing - takeoff) / fps` seconds
- **Jump height**: `hip_y[landing] - min(hip_y[takeoff:landing])`
- **Knee angle**: Angle at hip-knee-ankle joint
- **Arm position**: Distance from wrist to shoulder (0 = close, 1 = extended)
- **Edge indicator**: +1 (inside edge), -1 (outside edge), 0 (flat)

### Visualization System

Enhanced debug visualization with layered HUD architecture:

**Layers:**

- **Layer 0 (Raw)**: Skeleton only
- **Layer 1 (Kinematics)**: + velocity vectors + motion trails
- **Layer 2 (Technical)**: + edge indicators + joint angles
- **Layer 3 (Coaching)**: + subtitles + full HUD

**Usage:**

```bash
# Generate debug visualization
uv run python scripts/visualize_with_skeleton.py video.mp4 --layer 3 --output video_debug.mp4

# With pre-computed poses
uv run python scripts/visualize_with_skeleton.py video.mp4 --layer 1 --poses poses.npz
```

**Color scheme (skeleton):**

- Left side (arm/leg): Blue
- Right side (arm/leg): Red
- Center (torso/head): Green
- Joints: White

**Smoothing:**

- One-Euro Filter applied in normalized coordinate space
- Reduces jitter by ~30% while preserving fast motions

## Environment

- **OS**: Artix Linux (Ryzen 7 5800H / RTX 3050 Ti)
- **Python**: 3.11+ via `uv`
- **VRAM**: <200MB for full pipeline (YOLOv11n ~100MB + BlazePose CPU)

## Implementation Status

✅ **Overall: MVP ~85% complete**

**See [`ROADMAP.md`](ROADMAP.md)** for detailed phase-by-phase status:

**Complete (100%):**

- Phase 0: Foundation (types, utils)
- Phase 1: Person Detection (YOLOv11n)
- Phase 2: Pose Estimation (BlazePose 33kp)
- Phase 3: Normalization (root-centering + scale)
- Phase 4: Smoothing (One-Euro Filter, 29% jitter reduction)
- Phase 5: Metrics (airtime, height, angles, edge)
- Phase 8: Recommender (rule-based, Russian output)
- Phase 9: Reference System (save/load .npz)
- Phase 11: Visualization (layered HUD, skeleton, kinematics)
- Phase 12: CLI & Pipeline (analyze, build-ref, segment)

**Partial (50-90%):**

- Phase 6: Phase Detection ⚠️ 50% - MANUAL ONLY, auto-detection NOT working
- Phase 7: DTW Alignment ⚠️ 70% - code exists, tests failing (17 vs 33 keypoints)
- Phase 10: Segmentation ✅ 90% - working, but boundaries too broad

## Recent Improvements (2026-03)

### Coordinate System Architecture

- Added explicit `PixelPose` and `NormalizedPose` type aliases
- Runtime validation with `assert_pose_format()` catches coordinate bugs early
- Helper functions: `normalize_pixel_poses()`, `pixelize_normalized_poses()`
- Documented convention in CLAUDE.md to prevent future confusion

### Enhanced Visualization

- Frame-perfect synchronization between poses and video frames
- One-Euro Filter smoothing in normalized coordinate space (~30% jitter reduction)
- Support for VTT subtitles with Russian/Cyrillic text (via Pillow)
- Layered HUD system for focused debugging

## Known Issues & Workarounds

### BlazePose Frame Skipping

BlazePose may skip frames where person detection confidence is low. This causes:

- Fewer extracted poses than video frames
- Potential synchronization issues

**Solution:** The visualization script tracks frame indices and only draws skeleton when pose data exists for that frame.

### Coordinate System Confusion

Historically, mixing pixel and normalized coordinates caused visualization bugs.

**Solution:** Always use variable name suffixes (`_px`, `_norm`) and validate with `assert_pose_format()`.

## Next Steps (Future Enhancements)

1. **3D Lifting**: Add Pose3DM-L for true 3D biomechanics
2. **RAG System**: Integrate Qwen3/GPT-4o for natural language recommendations
3. **More Elements**: Add salchow, loop, lutz, axel
4. **Real-time Processing**: Optimize for live feedback
5. **Mobile Deployment**: Port to smartphone for on-ice analysis

## References

- Original architecture research: [`research/RESEARCH.md`](research/RESEARCH.md)
- BlazePose keypoints: <https://google.github.io/mediapipe/solutions/pose.html>
- DTW in Python: <https://dynamictimewarping.github.io/>
