# experiments/CLAUDE.md — ML Experiments

## Purpose

Exploratory ML experiments. Not part of the production pipeline (`ml/src/`).
These are scripts and notebooks for testing hypotheses before integrating into the main system.

## Directory Convention

```
experiments/
├── README.md              # Master report: hypotheses, results table, conclusions
├── CLAUDE.md              # This file
├── checkpoints/           # Model checkpoints (not in git)
├── exp_<short_name>.py    # Experiment scripts
└── YYYY-MM-DD-<topic>.md  # Standalone experiment reports
```

## Experiment Template

Every experiment script must document at the top:

```python
"""
Experiment: <short name>
Hypothesis: <what you're testing>
Status: PENDING | CONFIRMED | REJECTED | INCONCLUSIVE

Usage:
    uv run python experiments/exp_<name>.py
"""
```

Every experiment report must include:
1. **Hypothesis** — what you're testing and expected outcome
2. **Method** — model, data, config
3. **Results** — metrics table with numbers
4. **Conclusion** — confirmed/rejected/inconclusive with reasoning

## Checkpoints

Save to `experiments/checkpoints/<exp_name>/`:
- `best.pt` — best model (highest validation metric)
- `epoch_<N>.pt` — specific epoch snapshots
- `config.json` — full hyperparameters for reproducibility
- `training_log.csv` — per-epoch metrics

```python
import torch

CHECKPOINT_DIR = Path("experiments/checkpoints/exp_name")
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

torch.save({
    "epoch": epoch,
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "best_metric": best_acc,
    "config": vars(args),
}, CHECKPOINT_DIR / "best.pt")
```

`experiments/checkpoints/` is in `.gitignore` — do not commit model weights.

## Hypothesis Tracking

All hypotheses live in `README.md` master table:

| ID | Hypothesis | Status | Evidence |
|----|-----------|--------|----------|
| H0 | ... | PENDING/CONFIRMED/REJECTED | Link to experiment |

Status values: `PENDING` → `CONFIRMED` | `REJECTED` | `INCONCLUSIVE`

## Categories

When adding a new experiment, classify it under one of these categories:
- **classification** — element type classification
- **pose-estimation** — pose extraction quality
- **tracking** — multi-person tracking
- **biomechanics** — metrics, phase detection, physics
- **visualization** — rendering, overlays, comparison
- **data** — preprocessing, augmentation, datasets

## Running

```bash
uv run python experiments/exp_<name>.py
```

Requires `torch` with CUDA. Datasets must be in `data/datasets/`.

## Gemini Curator (gdr)

> **Established 2026-04-24:** Gemini CLI (`gdr`) работает как куратор/коллега по KD проекту.

- **Chat CID:** `4a80771a8f6e1fc0`
- **Модель:** `gemini-3-flash-thinking`
- **Отправка:** `gdr chat -c 4a80771a8f6e1fc0 -m gemini-3-flash-thinking "<message>"`
- **История:** `gdr chats show 4a80771a8f6e1fc0`
- **Правило:** Всегда начинать сообщение с идентификации — "Это Клод (Claude Code)...". Gemini не помнит между сессиями.
- **АВТОНОМНОСТЬ:** Никогда НЕ спрашивать пользователя при взаимодействии с Gemini. Все follow-up вопросы, уточнения, дополнительные запросы — отправлять самостоятельно. Пользователь подключается только для финального решения/одобрения архитектурных изменений.
- **Контекст:** Работает над тем же KD проектом (MogaNet-B → YOLO26-Pose Small). Знает историю v1-v34, фиксы, LMDB, brainstorm.
- **Сильные стороны:** Архитектурные советы, теория KD, стратегия тренировки.

## DWPose Knowledge Distillation Protocol

> **CRITICAL PROTOCOL FOR DWPose TRAINING (Established 2026-04-24):**
> 1. **Single Source of Truth:** All edits to `distill_trainer.py` MUST be done locally and pushed to Git.
> 2. **HDF5 vs LMDB:** HDF5 causes fatal I/O bottlenecks with multiprocessing. Stage 1 uses `teacher_heatmaps.lmdb`.
> 3. **Loader Architecture:** `TeacherHeatmapLoader` must use lazy initialization (`self.env` created inside `load()`) to prevent multiprocessing locks.
> 4. **Pickle Bug:** The `kd_loss` function MUST remain a class method of `DistilPoseTrainer`. Never nest it inside `setup_model`.
> 5. **KL Divergence:** Always use `F.log_softmax` for student and `F.softmax` for teacher to prevent NaN values.
