# MogaNet-B Inference Benchmark

> **Date:** 2026-05-05
> **GPU:** NVIDIA GeForce GTX 1070 Ti (Pascal, 8 GB VRAM, CUDA 6.1)
> **Host:** Vast.ai (South Korea, $0.0387/hr)
> **Model:** `moganet_b_ap2d_384x288.pth` (47.4M params, fine-tuned on AthletePose3D)

---

## Raw Inference Latency

Measured on 200 iterations, batch=1, input=(3, 288, 384), CUDA sync after each forward.

| Metric | Value |
|--------|-------|
| Mean | **48.13 ms** |
| Median | **47.86 ms** |
| Std | 0.93 ms |
| P95 | 49.16 ms |
| P99 | 52.22 ms |
| FPS | **20.8** |

**torch.compile:** Not supported on Pascal (CUDA capability 6.1 < 7.0, Triton requires ≥7.0).

### Comparison with RTX 5090 (from `data/plans/2026-04-18-kd-moganet-yolo26-plan.md`)

| GPU | Latency | Relative |
|-----|---------|----------|
| RTX 5090 | ~33.8 ms | 1.0× |
| GTX 1070 Ti | **48.1 ms** | **1.42×** |

> DLPerf ratio (5090 vs 1070 Ti) is ~5×, but MogaNet-B (47M params) is small enough that Pascal FP32 keeps up. The gap is much smaller than the raw TFLOPS difference.

---

## Video Processing Speed (YOLOv8n + MogaNet-B)

Full pipeline: `frame → YOLOv8 detect → crop each person → MogaNet-B → skeleton overlay`

| Metric | Value |
|--------|-------|
| Video | Парное.MOV, 1080×1920 @ 59.5 fps, 376 frames |
| Total time | **86.5 s** |
| FPS | **4.0** |
| Max persons/frame | 8 |
| Min FPS (8 persons) | 1.4 |
| Avg persons/frame | ~3.5 |

### Per-frame breakdown (estimated)

| Stage | Time |
|-------|------|
| YOLOv8n detect (1×1080p) | ~10–15 ms |
| MogaNet-B per person | ~48 ms |
| Overhead (crop, draw, I/O) | ~10 ms |
| **1 person total** | **~70 ms → 14 FPS** |
| **8 persons total** | **~400 ms → 2.5 FPS** |

---

## Files

| File | Purpose |
|------|---------|
| `benchmark_moganet.py` | Pure inference benchmark (latency, FPS, percentiles) |
| `process_video_moganet.py` | Naive full-frame MogaNet (no detector) |
| `process_video_yolo_moganet.py` | YOLOv8 + MogaNet top-down pipeline |

---

## Notes

1. **Person detector required.** MogaNet-B is top-down; without a detector it hallucinates keypoints when multiple people are present.
2. **No batch inference.** Each crop runs through MogaNet separately. Batch processing all crops in one forward would improve throughput significantly.
3. **YOLO false positives.** YOLOv8n occasionally detects >2 people on ice (reflections, shadows). Confidence filtering (`conf > 0.5`) and NMS would help.
4. **Pascal CUDA limitation.** No FP16 Tensor Cores, no torch.compile. FP32 only.

---

## Reproduce

```bash
# On a CUDA-capable machine with PyTorch
git clone <repo>
cd experiments/moganet-benchmark

# Benchmark
python3 benchmark_moganet.py

# Video with detector
python3 process_video_yolo_moganet.py
```

Weights: `data/models/athletepose3d/moganet_b_ap2d_384x288.pth` (544 MB)
