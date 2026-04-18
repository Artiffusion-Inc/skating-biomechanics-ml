# Pipeline Profiling Results

**Date:** 2026-04-18
**Tool:** `PipelineProfiler` (`time.perf_counter`) — see `ml/skating_ml/utils/profiling.py`
**Script:** `ml/scripts/profile_pipeline.py`
**Branch:** `refactor/pipeline-profiling`

## Test Setup

| Parameter | Value |
|-----------|-------|
| Video | `athletepose3d/train_set/S1/Axel_10_cam_1.mp4` |
| Element type | waltz_jump |
| Device | CPU (CUDA unavailable — "CUDA unknown error") |
| Pipeline | `AnalysisPipeline.analyze()` |
| 3D lift | Failed (no model file), exception caught |
| DTW | Skipped (no reference loaded) |
| Physics | Skipped (depends on 3D) |

**Note:** Run on CPU, not GPU. Real production uses GPU (CUDA). CPU numbers are an upper bound for GPU performance — GPU inference is ~7x faster per frame (measured previously: 5.6s vs 39.4s for 364 frames).

## Stage Timings

```
Stage                              Time (s)        %    Calls
-------------------------------------------------------------------
video_meta                           0.0052     0.0%        1
extract_and_track                 1128.1576    99.8%        1
normalize                            0.0015     0.0%        1
smooth                               0.4504     0.0%        1
3d_lift_and_blade                    0.0001     0.0%        1
phase_detection                      1.1046     0.1%        1
metrics                              0.6012     0.1%        1
dtw_alignment                        0.0000     0.0%        1
physics                              0.0000     0.0%        1
recommendations                      0.0025     0.0%        1
-------------------------------------------------------------------
TOTAL                             1130.3255   100.0%
```

## Time Distribution (ASCII)

```
extract_and_track  ████████████████████████████████████████████ 99.8%
phase_detection    █                                                0.1%
metrics            █                                                0.1%
smooth             █                                                0.0%
recommendations    █                                                0.0%
video_meta         █                                                0.0%
normalize          █                                                0.0%
3d_lift_and_blade  █                                                0.0%
dtw_alignment      █                                                0.0%
physics            █                                                0.0%
```

## Analysis

### Bottleneck: `extract_and_track` = 99.8% of total time

RTMPose inference (via rtmlib/ONNX Runtime) dominates pipeline execution at 1128.2s on CPU. This includes:
- Frame extraction from video
- RTMO model loading (~17 min cold start — first-time model download at ~25-50 kB/s)
- Per-frame 2D pose estimation
- Person tracking (Kalman filter + biometric Re-ID)
- Gap filling

### CPU-only stages combined: < 0.2% of total

| Category | Time (s) | % of total |
|----------|----------|------------|
| normalize | 0.0015 | 0.0001% |
| smooth (One-Euro filter) | 0.4504 | 0.0398% |
| phase_detection | 1.1046 | 0.0977% |
| metrics | 0.6012 | 0.0532% |
| recommendations | 0.0025 | 0.0002% |
| **Total CPU-only** | **2.160** | **0.191%** |

### Numba JIT targets — impact assessment

The functions optimized with Numba JIT in PR #29 (`_angle_3pt_rad`, `smooth_trajectory_2d`, `_compute_knee_angle_series`, `_compute_trunk_lean_series`) are called within stages that total < 0.2% of pipeline time. Even a 100x speedup on these functions would save at most ~2s out of 1130s.

### 3D lifting

Not measured — model file not found, exception silently caught. When enabled, this adds MotionAGFormer/TCPFormer inference. Previous measurements showed CorrectiveLens adds ~3px max shift at unknown time cost. Worth profiling separately with GPU.

### DTW / Physics

Both effectively 0s — DTW skipped (no reference), physics skipped (depends on 3D poses).

## Recommendations for Optimization

1. **RTMPose inference is the only meaningful bottleneck.** All other optimization is noise.
2. **GPU acceleration already provides 7x speedup** (measured: 5.6s vs 39.4s for 364 frames). Ensure CUDA works in production.
3. **Model download caching.** Cold start downloads 78.8MB RTMO model at ~25-50 kB/s (~17 min). Cache model locally or pre-download.
4. **Frame skip.** Already available (`frame_skip=8` in visualization). For analysis pipeline, consider adaptive frame skip during non-critical phases (preparation, recovery).
5. **Do NOT optimize Numba targets further.** The 0.2% CPU-only stages are not the bottleneck. Numba JIT compilation overhead may exceed its benefit for single-run analysis.

## Reproduction

```bash
cd ml && .venv/bin/python scripts/profile_pipeline.py \
    /path/to/video.mp4 --element waltz_jump --json /tmp/profiling_results.json
```

Raw data: `/tmp/profiling_results.json`
