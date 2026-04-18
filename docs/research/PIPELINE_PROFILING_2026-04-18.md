# Pipeline Profiling Results

**Date:** 2026-04-18
**Tool:** `PipelineProfiler` (`time.perf_counter`) — see `ml/skating_ml/utils/profiling.py`
**Script:** `ml/scripts/profile_pipeline.py`
**Branch:** `refactor/pipeline-profiling`

## Test Setup

| Parameter | Value |
|-----------|-------|
| Video | `athletepose3d/train_set/S1/Axel_10_cam_1.mp4` (186 frames, 60fps, 3.1s) |
| Element type | waltz_jump |
| Device | CPU (CUDA unavailable — "CUDA unknown error") |
| Pipeline | `AnalysisPipeline.analyze()` |
| RTMO model | `rtmo-m` (balanced, 640x640) |
| 3D lift | Failed (no model file), exception caught |
| DTW | Skipped (no reference loaded) |
| Physics | Skipped (depends on 3D) |

**Note:** Run on CPU, not GPU. Real production uses GPU (CUDA).

## Run 1: Cold Start (model download)

First run — RTMO model (78.8MB) downloaded at ~25-50 kB/s.

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

~17 min of the 1128s was model download (not captured separately by profiler).

## Run 2: Warm (cached model) + Deep Profile

Model already cached in `~/.cache/rtmlib/hub/checkpoints/`. Deep `--deep` flag enabled.

```
Stage                              Time (s)        %    Calls
-------------------------------------------------------------------
video_meta                           0.0036     0.0%        1
extractor_init                       0.0000     0.0%        1
rtmo_inference_loop                 69.7847    98.8%        1
gap_filling                          0.0002     0.0%        1
spatial_reference                    0.0669     0.1%        1
extract_and_track                   69.8522    98.9%        1
normalize                            0.0012     0.0%        1
smooth                               0.2577     0.4%        1
3d_lift_and_blade                    0.0001     0.0%        1
phase_detection                      0.4912     0.7%        1
metrics                              0.0316     0.0%        1
dtw_alignment                        0.0000     0.0%        1
physics                              0.0000     0.0%        1
recommendations                      0.0014     0.0%        1
-------------------------------------------------------------------
TOTAL                               70.6401   100.0%
```

### Deep: `extract_and_track` Breakdown

| Sub-stage | Time (s) | % of extract_and_track |
|-----------|----------|------------------------|
| RTMO model init (ONNX session) | 0.000 | 0.0% (cached) |
| rtmo_inference_loop (186 frames) | 69.785 | 99.9% |
| gap_filling | 0.000 | 0.0% |
| spatial_reference | 0.067 | 0.1% |
| **Total** | **69.852** | **100.0%** |

**Per-frame RTMO inference: 375.2ms/frame (CPU)**

### Run 3: Ultra-deep (ONNX hook, DeepSORT hook)

Monkey-patched `onnxruntime.InferenceSession.run()` and `DeepSORTTracker.update()` to isolate inference vs tracking vs overhead.

| Component | Time (s) | % of pipeline | Per-frame |
|-----------|----------|---------------|-----------|
| **ONNX RTMO inference** | 50.5 | 74.9% | 271.5ms |
| **DeepSORT tracking** | 12.1 | 17.9% | 71.4ms |
| Other (cv2 decode, resize, coco2h36m, rtmlib IoU) | 4.9 | 7.2% | 26.3ms |
| **Total** | **67.5** | **100%** | **362.7ms** |

ONNX inference details: 186 calls, min=171ms, max=549ms, std=70ms. First few frames slower (ONNX session warmup / thread pool init).

**DeepSORT = 17.9% of pipeline** — second largest consumer. Uses PyTorch MobileNet embedder for appearance-based Re-ID. Runs 169/186 frames (skipped when no detections).

DeepSORT internals: `generate_embeds()` = 100% of DeepSORT time (68.8ms/call). Kalman predict + Hungarian matching = negligible. The bottleneck is PyTorch MobileNet forward pass.

### Run 4: ONNX Runtime Op-Level Profiling (RTMO model)

Profiled RTMO-M (89.3MB) directly via `ort.SessionOptions(enable_profiling=True)`, 5 inference runs. 6314 trace events, 45 unique ONNX operators.

| ONNX Op | Time (ms) | % | Calls | Avg (ms) |
|---------|-----------|---|-------|----------|
| **Conv** | 1818.9 | 62.4% | 800 | 2.274 |
| **QuickGelu** | 320.2 | 11.0% | 704 | 0.455 |
| **ReorderInput** | 278.1 | 9.5% | 640 | 0.435 |
| **ReorderOutput** | 203.4 | 7.0% | 840 | 0.242 |
| Concat | 81.5 | 2.8% | 400 | 0.204 |
| Add | 58.4 | 2.0% | 416 | 0.140 |
| MatMul | 43.8 | 1.5% | 96 | 0.456 |
| Slice | 28.3 | 1.0% | 272 | 0.104 |
| Split | 26.7 | 0.9% | 32 | 0.836 |
| MaxPool | 14.7 | 0.5% | 24 | 0.611 |
| Rest (36 ops) | 141.8 | 4.9% | — | — |
| **Total** | **2914.8** | **100%** | **6240** | **0.467** |

Key observations:
- **Conv layers = 62.4%** — dominant. CPU matrix multiplication. Benefits massively from GPU.
- **QuickGelu = 11.0%** — GELU activation variant. Could be fused with Conv in GPU kernels.
- **ReorderInput + ReorderOutput = 16.5%** — data layout conversion (NHWC↔NCHW). Zero cost on GPU (just memory view), significant CPU overhead.
- **ReorderInput (9.5%) and ReorderOutput (7.0%) together = 281.5ms/5runs = 56.3ms/frame** — this is pure CPU memory copy overhead that disappears on GPU.

## Full Pipeline Breakdown (Warm Run)

```
extract_and_track  ████████████████████████████████████████████ 98.9%
phase_detection    ▏                                                0.7%
smooth             ▏                                                0.4%
spatial_reference  ▏                                                0.1%
metrics            ▏                                                0.0%
recommendations    ▏                                                0.0%
video_meta         ▏                                                0.0%
normalize          ▏                                                0.0%
gap_filling        ▏                                                0.0%
3d_lift_and_blade  ▏                                                0.0%
dtw_alignment      ▏                                                0.0%
physics            ▏                                                0.0%
```

## Analysis

### Bottleneck: RTMO inference + DeepSORT = 92.8% of total time

362.7ms/frame on CPU for 186 frames = 67.5s.

Inside the per-frame loop (`extract_video_tracked`), every frame runs:
1. `cv2.VideoCapture.read()` — 4.6ms (frame decode)
2. Optional resize (if > 1920px)
3. `tracker(frame_ds)` — RTMO ONNX inference (271.5ms) + rtmlib IoU tracking
4. `DeepSORTTracker.update()` — PyTorch MobileNet embedder (71.4ms)
5. COCO→H3.6M conversion + normalization + biometric anti-steal

**Two neural networks run per frame on CPU:**
- RTMO (ONNX, 271.5ms) — pose estimation
- DeepSORT embedder (PyTorch, 71.4ms) — appearance Re-ID

### Everything else combined: 1.2%

| Category | Time (s) | % of total |
|----------|----------|------------|
| phase_detection | 0.491 | 0.70% |
| smooth (One-Euro filter) | 0.258 | 0.36% |
| spatial_reference | 0.067 | 0.09% |
| metrics | 0.032 | 0.04% |
| video_meta | 0.004 | 0.01% |
| normalize | 0.001 | 0.00% |
| recommendations | 0.001 | 0.00% |
| gap_filling | 0.000 | 0.00% |
| **Total non-inference** | **0.854** | **1.21%** |

### Numba JIT targets — confirmed noise

PR #29 optimized `_angle_3pt_rad`, `smooth_trajectory_2d`, `_compute_knee_angle_series`, `_compute_trunk_lean_series` with Numba JIT. These run inside `smooth` (0.258s) and `metrics` (0.032s) — totaling 0.29s out of 70.6s. Even complete elimination would save 0.4%.

### Cold start impact

First run took 1130s vs warm 70.6s. Difference = 1059s, almost entirely RTMO model download (~78.8MB at ~25-50 kB/s). Model is cached after first run to `~/.cache/rtmlib/hub/checkpoints/`.

## Recommendations

1. **Two bottlenecks: RTMO (74.9%) + DeepSORT (17.9%).** Combined 92.8% of pipeline.
2. **GPU acceleration is the primary lever.** On GPU: Conv moves to CUDA cores (10-50x faster), ReorderInput/Output becomes zero-cost (memory view), QuickGelu fuses with Conv. Previously measured 7x speedup for RTMO on GPU. DeepSORT embedder also GPU-accelerated via PyTorch CUDA.
3. **Disable DeepSORT for single-person videos.** DeepSORT embedder runs even when only 1 person is detected. For single-skater analysis (most use cases), appearance Re-ID is unnecessary — rtmlib's built-in IoU tracking suffices. Use `tracking_backend="rtmlib"` or `tracking_mode="rtmlib"` to skip DeepSORT entirely. Expected savings: 71.4ms/frame (19.7%).
4. **Frame skip for analysis.** Already available (`frame_skip` parameter). Skipping every other frame halves both RTMO and DeepSORT time.
5. **Lighter RTMO variant.** `rtmo-s` (small) vs current `rtmo-m` (medium). Trade accuracy for speed.
6. **Batch inference.** Currently one frame at a time. ONNX Runtime supports batching — processing N frames per batch could improve GPU utilization.
7. **ReorderInput/ReorderOutput = 16.5% of ONNX time on CPU.** This is NHWC↔NCHW layout conversion. Zero cost on GPU. If CPU-only deployment needed, consider converting RTMO to NHWC-native ONNX graph.
8. **Do NOT optimize Numba targets.** 0.4% of pipeline time. Numba JIT cold-start compilation may exceed saved time for single-run analysis.

## Reproduction

```bash
# Basic profile
cd ml && .venv/bin/python scripts/profile_pipeline.py \
    /path/to/video.mp4 --element waltz_jump --json /tmp/profiling.json

# Deep profile (model init vs inference)
cd ml && .venv/bin/python scripts/profile_pipeline.py \
    /path/to/video.mp4 --element waltz_jump --deep --json /tmp/profiling_deep.json
```

Raw data: `/tmp/profiling_results.json`, `/tmp/profiling_deep.json`
