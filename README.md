# Skating Biomechanics ML

AI-тренер по фигурному катанию — анализ видео, сравнение с эталонами, биомеханическая обратная связь на русском.

## Quick Start

```bash
uv sync
bash scripts/setup_cuda_compat.sh   # CUDA GPU setup (RTX 3050 Ti)

# Анализ видео
uv run python -m src.cli analyze video.mp4 --element waltz_jump --pose-backend rtmlib

# Сравнение двух видео (тренировочный режим)
uv run python -m src.cli compare attempt.mp4 reference.mp4 --overlays skeleton,angles,timer

# Визуализация с 3D-коррекцией скелета
uv run python scripts/visualize_with_skeleton.py video.mp4 --layer 2 --3d --output out.mp4
```

## Architecture

```
Video → RTMPose (rtmlib, CUDA) → HALPE26 (26kp) → H3.6M (17kp)
  → GapFiller → Smoothing → [Optional] CorrectiveLens (3D→2D correction)
  → Phase Detection → Biomechanics Metrics → DTW → Recommender → Russian Report
```

| Component | Technology |
|-----------|-----------|
| **2D Pose** | RTMPose via rtmlib (HALPE26, 26kp, CUDA) |
| **3D Lifting** | MotionAGFormer-S / Biomechanics3DEstimator |
| **3D Correction** | CorrectiveLens (kinematic constraints + anchor projection) |
| **Tracking** | OC-SORT + anatomical biometric Re-ID |
| **Physics** | CoM trajectory, Dempster anthropometric tables |
| **GPU** | CUDA via onnxruntime-gpu (7.1x speedup) |

## Project Structure

```
src/
├── pose_estimation/     # RTMPose (rtmlib), YOLO26-Pose
├── pose_3d/             # CorrectiveLens, MotionAGFormer, TCPFormer
├── detection/           # PoseTracker, spatial reference, blade detection
├── analysis/            # Physics engine, metrics, recommender
├── visualization/       # Layered HUD, comparison, skeleton
├── alignment/           # DTW motion alignment
└── utils/               # GapFiller, geometry, smoothing
```

## Research

See [`research/RESEARCH.md`](research/RESEARCH.md) — index of all research materials, memory bank.

## Quality

```bash
uv run pytest tests/ -v -m "not slow"   # 272+ tests
uv run ruff check .                      # Lint
uv run ruff format .                     # Format
```

## License

MIT
