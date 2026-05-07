# Handoff: TAS Coarse Segmentation + RF Classifier

## Goal

Replace rule-based `element_segmenter.py` with ML-based temporal action segmentation:
- **Coarse TAS**: BiGRU per-frame classifier → 4 classes (None/Jump/Spin/Step)
- **Fine classifier**: Random Forest on segment biomechanical features → ~30 fine element types
- **Dataset**: MCFS (271 videos, 1.7M frame-level labels, 130 fine classes)
- **Metric**: OverlapF1@50 (IoU ≥ 0.5 for segment match)

## Current Progress

All 7 plan tasks implemented and committed on branch `feature/tas-coarse-segmentation`:

| Task | File | Status |
|------|------|--------|
| Data loader | `ml/src/tas/dataset.py` | Done — MCFS OP25→COCO17→H3.6M, coarse labels |
| BiGRU model | `ml/src/tas/model.py` | Done — pack_padded, 3 passing tests |
| OverlapF1 metric | `ml/src/tas/metrics.py` | Done — segment matching, 9 passing tests |
| Training script | `experiments/train_tas.py` | Done — 5-fold CV, Adam, CE(-1 ignore) |
| RF classifier | `ml/src/tas/classifier.py` | Done — 5 features, sklearn RF wrapper |
| Inference | `ml/src/tas/inference.py` | Done — TASElementSegmenter, config auto-load |
| Integration | `ml/src/analysis/element_segmenter.py` | Done — `method="tas_ml"` with rule fallback |

**Tests**: 15 tests, all green when run individually (`pytest ml/tests/tas/test_*.py` one-by-one).

## What Worked

- `BiGRUTAS` architecture: `(B,T,17,2) → Linear(34→128) → BiGRU(128,2 layers) → Dense(256→128→4)`
- `pack_padded_sequence` handles variable-length sequences correctly
- `OverlapF1` matching with IoU threshold 0.5 — follows MCFS paper (AAAI 2021)
- RF on 5 biomech features: duration, hip_y_range, motion_energy, rotation_speed, num_frames
- Checkpoint config embedding (`config` dict in .pt) lets inference auto-reconstruct model arch
- Direct file loading in tests bypasses `types.py` → stdlib shadowing issue

## What Didn't Work

- **torch missing in venv**: System Python had torch, worktree venv didn't. Fixed with `uv add torch scikit-learn --directory ml`.
- **`types.py` shadowing stdlib**: `ml/src/types.py` conflicts with Python stdlib `types`. When PYTHONPATH includes `ml/src`, `import types` gets ML types → `ImportError: cannot import name 'MappingProxyType'`.
  - **Workaround**: Tests use `importlib.util.spec_from_file_location` + manual `sys.modules` injection. Don't import through package init chain.
  - **Real fix**: Rename `types.py` → `skating_types.py` (breaking change, deferred).
- **pytest parallel collection**: Running `pytest ml/tests/tas/` fails because test_dataset.py injects fake `ml.src.*` packages into `sys.modules`, conflicting with other tests. Run individually.
- **mayupei repo**: Their approach = 2-stage LSTM-CNN, but only 3 classes (Jump/Spin/Background). Step mapped to background. They don't solve fine classification at all. F1@50 = 0.893, frame acc = 91.7%.

## Next Steps

1. **Train BiGRU** — `experiments/train_tas.py` ready. Needs GPU (user mentioned V100). Run: `uv run python experiments/train_tas.py`
2. **Train RF classifier** — `experiments/train_rf_classifier.py` ready. Can run on CPU.
3. **Evaluate** — `experiments/evaluate_tas.py --checkpoint <path>`
4. **Compare with rule-based** — run both methods on same videos, compare OverlapF1
5. **Consider unified model** — Research shows 130-class end-to-end possible but boundary detection degrades. Could try hierarchical head (4 coarse + 30 fine) or keep 2-stage.
6. **Rename `types.py`** — If approved, rename to eliminate shadowing once and for all.

## Reference

- Plan: `docs/plans/2026-05-05-tas-coarse-segmentation.md` (in main repo, symlinked)
- Research: `docs/research/RESEARCH_TEMPORAL_ACTION_SEGMENTATION_2026-04-12.md`
- External repo analyzed: https://github.com/mayupei/figure-skating-action-segmentation
