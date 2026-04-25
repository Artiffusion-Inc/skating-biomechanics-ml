# v35 KD Code Review: Consolidated Report

**Date:** 2026-04-25
**Scope:** MogaNet-B teacher -> YOLO26-Pose student KD pipeline (4-agent review)
**Verdict:** 8 CRITICAL bugs found. Pipeline produces no meaningful KD signal in current state.

---

## Summary

| Severity | Count | Status |
|----------|-------|--------|
| CRITICAL | 8 | Must fix before v35 launch |
| HIGH | 7 | Fix before meaningful training |
| MEDIUM | 3 | Fix after CRITICALs |
| LOW | 3 | Cosmetic / perf |
| PASS | 4 | Verified correct |

**Key findings:**
1. Teacher heatmap generation applied `torch.sigmoid()` incorrectly (D2) -- currently running generation is INVALID
2. Student keypoint decode is a no-op -- `.view()` result discarded (C3/T1)
3. Student decode formula missing `*2.0` and `-0.5` vs actual Pose26 (D5)
4. KD loss is ~0.005% of total loss due to batch scaling mismatch (T4)
5. Inverse affine transform ignores the 1.4x padding used in crop (C1/D7)
6. Feature adapters created lazily, never registered in optimizer (F1/T3)
7. Spatial alignment between teacher crop features and student full-image features is fundamentally broken (F2)
8. No IoU matching -- student multi-anchor output silently uses anchor 0 (C2)

---

## Deduplicated Findings (Ordered by Priority)

### CRITICAL

| ID | Finding | Agents | File:Line | Fix |
|----|---------|--------|-----------|-----|
| **D2** | **Sigmoid applied to teacher heatmaps** -- MogaNet DeconvHead outputs raw logits; JointsMSELoss uses raw values. `torch.sigmoid()` flattens Gaussian peaks (1.0 -> 0.73), destroying spatial precision. **Heatmap generation currently running with sigmoid -- must re-do.** | A4 | `generate_teacher_heatmaps.py:623,655` | Remove `torch.sigmoid()` from both batch flush paths. Re-run heatmap generation. |
| **C3/T1** | **`.view()` is a no-op** -- `kpts.view(B, -1, K, 3)` on line 617 returns new tensor but result is discarded. Only indices [0,1,2] of the 51-element raw vector are ever used. | A1,A3 | `distill_trainer.py:617` | Change to `kpts = kpts.view(B, -1, K, 3)` |
| **D5** | **Student decode formula wrong** -- Actual Pose26 `kpts_decode`: `(raw * 2.0 + (anchor - 0.5)) * stride`. Code uses: `(raw + anchor) * stride`. Missing `*2.0` factor and `-0.5` offset. Student coordinates are in wrong scale/offset vs teacher. | A4 | `distill_trainer.py:624-626` | Fix decode to match Pose26: `(raw * 2.0 + (anchor - 0.5)) * stride` |
| **T4** | **KD loss negligible (0.005% of total)** -- PoseLoss26 returns `loss * batch_size` (~100). coord_loss and feat_loss are mean-reduced (~0.01). KD contributes nothing. | A3 | `distill_trainer.py` (kd_loss method) | Either (a) multiply KD losses by batch_size, or (b) divide GT loss by batch_size before summing. |
| **C1/D7** | **Inverse affine ignores 1.4x padding** -- `crop_and_resize()` expands bbox by 20% per side (`bw * 1.4`). `_inverse_affine_transform()` uses original `bw`, `bh` without padding. Teacher coords mapped to wrong region. | A1,A4 | `distill_trainer.py:655-667` vs `generate_teacher_heatmaps.py:459-461` | Store actual padded bbox dimensions (or the crop parameters) during heatmap generation in the HDF5. Use `bw*1.4`, `bh*1.4` in inverse transform. |
| **C2** | **No IoU matching -- hardcoded anchor 0** -- Student predicts K keypoints at every spatial anchor (8400+ anchors for YOLO26s). Code takes anchor index 0 (top-left, lowest confidence). No matching against GT/person bbox. | A1 | `distill_trainer.py` (kd_loss) | Implement IoU matching: compute student detection bbox at each anchor, IoU with GT bbox, select highest-IoU anchor. |
| **F1/T3** | **Feature adapters not in optimizer** -- `setup_model()` registers placeholder adapters (256->256). `_get_or_create_adapter()` lazily replaces with correct dims (64->128 etc.) but new module is NOT registered as model submodule. Optimizer still holds references to discarded placeholders. | A2,A3 | `distill_trainer.py:523-533,573-576` | Create adapters with correct dims in `setup_model()` (requires knowing teacher channel dims at init time), OR replace model submodule: `setattr(model, f'kd_adapter_{idx}', new_adapter)` and rebuild optimizer. |
| **F2** | **Spatial content misalignment** -- Teacher features encode tight person crop (288x384, portrait, ~1 person). Student features encode full image (384x384, square, multiple people). Pixel correspondence is meaningless. Interpolating crop features to match full-image feature maps destroys spatial alignment. | A2 | `distill_trainer.py` (feature KD section) | **Architectural decision needed.** Options: (a) crop student features to GT bbox region before comparison, (b) channel-only distillation (spatial average pooling), (c) drop feature KD entirely until coordinate KD works. **Recommendation: option (c).** |

### HIGH

| ID | Finding | Agents | File:Line | Fix |
|----|---------|--------|-----------|-----|
| **T2** | **No `build_optimizer` override** -- When `--freeze-backbone` is used, backbone unfrozen at epoch 11 but NOT added to optimizer param groups. Frozen params stay frozen permanently. | A3 | `distill_trainer.py` | Override `build_optimizer` in the Ultralytics trainer subclass to create param groups with differential LR. |
| **T5** | **No differential LR** -- All params get `lr=0.002`. Standard practice: backbone at 0.1x head LR. Without this, pretrained backbone features destabilize. | A3 | `distill_trainer.py` | Implement in `build_optimizer`: `param_groups = [{'params': head_params, 'lr': lr}, {'params': backbone_params, 'lr': lr * 0.1}]` |
| **C4** | **Decode formula mismatch with PoseLoss26** -- distill_trainer uses inference-style decode `(raw + anchor) * stride`. PoseLoss26 uses `raw + anchor` (no stride), with GT scaled to anchor space. Scale mismatch between student coords (pixel) and teacher coords (normalized). | A1 | `distill_trainer.py:624-626` | Coordinate KD should work in a consistent space. Either: (a) decode student to pixel space AND teacher to pixel space, or (b) convert both to normalized [0,1]. |
| **C5** | **Edge clamping not compensated** -- `crop_and_resize` clamps to image bounds when bbox extends beyond edges. Inverse transform doesn't know actual crop was smaller than padded bbox. | A1 | `generate_teacher_heatmaps.py:469-473` | Store actual crop origin and dimensions in HDF5 alongside bboxes. |
| **F3** | **Aspect ratio distortion** -- Teacher features 4:3 (72x96) interpolated to student 1:1 (48x48). Stretches features horizontally. | A2 | `distill_trainer.py` (feature interpolation) | Use `mode='bicubic'` with `align_corners=False` and accept minor distortion, or crop student features to matching aspect ratio. |
| **F4** | **No explicit adapter init** -- Default Kaiming uniform. For C_in != C_out, identity init (eye_) fails. Need Xavier or truncated SVD projection. | A2 | `distill_trainer.py:458-460` | Initialize `FeatureAdapter.projection.weight` with Xavier normal and `bias` with zeros. Or SVD projection from teacher to student channel space. |
| **T8** | **Unit tests reference old v34 API** -- Tests import from old module paths and call removed methods. | A3 | `experiments/yolo26-pose-kd/scripts/test_*.py` | Update test fixtures to match v35 API. |

### MEDIUM

| ID | Finding | Agents | Fix |
|----|---------|--------|-----|
| **C6** | No confidence threshold on teacher coords | A1 | Skip keypoints with confidence < 0.3 in KD loss |
| **C7** | Occluded keypoints (vis=1) included in KD | A1 | Filter by visibility flag from YOLO labels |
| **T9** | KD warmup (5 epochs) > LR warmup (3 epochs) -- 2 epochs at full LR with zero KD | A3 | Align KD warmup <= LR warmup, or set KD warmup to 3 epochs |

### LOW

| ID | Finding | Agents | Fix |
|----|---------|--------|-----|
| **F5** | L2 normalization before MSE is non-standard | A2 | Use cosine similarity loss instead, or remove L2 norm |
| **F6** | feat_alpha=0.01 very conservative | A2 | Increase to 0.1 after feature alignment is fixed |
| **F8** | Docstring says layers [4,6,8] but default is [3,9,31] | A2 | Fix docstring in `extract_backbone_features()` |
| **D8** | TeacherFeatureLoader linear scan fallback O(N*M) | A4 | Build reverse index on first access |
| **F7** | No HDF5 compression | A2 | Use `compression='gzip'` for features |

### VERIFIED CORRECT

| ID | Finding | Agent | Notes |
|----|---------|-------|-------|
| D1 | ImageNet normalization constants | A4 | Match official MogaNet-B config |
| D3 | Soft argmax implementation | A4 | Correct sub-pixel refinement |
| D4 | HDF5 index naming consistency | A4 | Consistent per pipeline |
| D6 | MogaNet-B confirmed heatmap model | A4 | Not SimCC |

---

## Recommended Fix Order

Fixes must be applied in dependency order. Some fixes unlock others.

### Phase 0: Stop the Bleeding (Immediate)

**0.1. D2 -- Remove sigmoid from heatmap generation** [BLOCKS EVERYTHING]
- The currently running heatmap generation produces invalid data
- Fix `generate_teacher_heatmaps.py` lines 623 and 655
- Re-run heatmap generation after fix
- Then re-extract coordinates (`extract_teacher_coords.py`)

### Phase 1: Fix Coordinate KD (Core Pipeline)

**1.1. C3/T1 + D5 -- Fix student decode** [BLOCKS coordinate KD]
- These two bugs are in the same function `_decode_student_kpts()`
- C3: `kpts = kpts.view(B, -1, K, 3)` (assign, don't discard)
- D5: `(raw * 2.0 + (anchor - 0.5)) * stride` (match Pose26)
- Can be fixed in one edit

**1.2. C1/D7 -- Fix inverse affine padding** [BLOCKS coordinate KD]
- Store padded bbox dims in HDF5 during heatmap generation
- Update `_inverse_affine_transform()` to use `bw*1.4`, `bh*1.4`

**1.3. C4 -- Consistent coordinate space** [BLOCKS coordinate KD]
- Decide: pixel space or normalized [0,1] for KD comparison
- Ensure student decode and teacher inverse affine produce same space

**1.4. C5 -- Edge clamping compensation** [DEPENDS on 1.2]
- Store actual crop params in HDF5
- Use in inverse transform

**1.5. T4 -- Fix batch scaling** [BLOCKS KD effectiveness]
- Divide GT loss by batch_size, OR multiply KD losses by batch_size
- This is what makes KD actually do something

**1.6. C2 -- IoU matching** [COMPLEX, can defer]
- Student predicts at 8400+ anchors; need to select the right one
- Workaround: for single-person crops, anchor nearest to image center is likely correct
- Full fix: compute IoU between student detection bbox and GT bbox

### Phase 2: Fix Training Pipeline

**2.1. T2 + T5 -- build_optimizer with differential LR** [BLOCKS progressive unfreeze]
- Override `build_optimizer` in Ultralytics trainer
- Create param groups: head at full LR, backbone at 0.1x

**2.2. T6 + T7 -- Progressive unfreeze defaults** [POLICY]
- Consider making `--freeze-backbone` the default for KD training
- Add LR warm-up for backbone at unfreeze epoch

### Phase 3: Fix Feature KD (or Drop It)

**Decision point: Keep or drop feature distillation?**

Feature KD has 3 CRITICAL issues (F1, F2, F3) that are architectural, not bug-fixable without significant rework:
- Spatial misalignment is fundamental (crop vs full-image features)
- Adapters not in optimizer is fixable but meaningless if spatial alignment is wrong
- Aspect ratio distortion compounds spatial misalignment

**Recommendation: DROP feature distillation for v35.** Focus on getting coordinate KD correct first. Feature KD can be re-added in v36 after coordinate KD is validated.

If keeping feature KD:
- **3.1. F1** -- Fix adapter registration (use correct dims at setup_model time)
- **3.2. F2** -- Implement bbox-region cropping of student features
- **3.3. F3** -- Handle aspect ratio in interpolation
- **3.4. F4** -- Add proper adapter initialization

### Phase 4: Polish

- C6, C7: confidence/visibility filtering
- T8: update unit tests
- T9: align warmup schedules
- F5-F8: minor improvements

---

## Dependency Graph

```
D2 (remove sigmoid) ──────────────────────────────────────────────┐
  └─> Re-run heatmap gen ──> Re-extract coords ──> ─────────────┐│
                                                                ││
C3/T1 (fix .view) ──────────────────────────────────────────────┐││
  + D5 (fix decode formula)                                     │││
  └─> Coordinate KD produces valid student coords              │││
                                                                │││
C1/D7 (fix padding) ──> C5 (edge clamping) ────────────────────┐│││
  └─> Inverse affine produces correct teacher coords           ││││
                                                               ││││
C4 (consistent space) ────────────────────────────────────────┐││││
  └─> Student and teacher coords comparable                   │││││
                                                              │││││
T4 (batch scaling) ───────────────────────────────────────────┐│││││
  └─> KD loss has non-zero gradient                           ││││││
                                                              ││││││
C2 (IoU matching) ───────────────────────────────────────────┐││││││
  └─> Correct anchor selected                                │││││││
                                                              │││││││
  ═════════════════════════════════════════════════════════════╧╧╧╧╧╧╧
                        COORDINATE KD WORKS
                                    │
T2+T5 (optimizer) ──> Progressive unfreeze works ─────────────┤
                                    │
F1+F2+F3+F4 (feature KD) ──> OR DROP FEATURE KD ─────────────┘
```

---

## Action Items Before v35 Launch

| # | Item | Estimated Effort | Blocks |
|---|------|-----------------|--------|
| 1 | Fix D2 (remove sigmoid) + re-run heatmap gen + re-extract coords | 2-3h (generation time) | All KD |
| 2 | Fix C3/T1 + D5 (student decode) | 15 min | Coordinate KD |
| 3 | Fix C1/D7 + C5 (inverse affine + edge clamping) | 30 min | Coordinate KD |
| 4 | Fix C4 (consistent coordinate space) | 20 min | Coordinate KD |
| 5 | Fix T4 (batch scaling) | 15 min | KD effectiveness |
| 6 | Implement C2 (IoU matching) or workaround | 1-2h | Coordinate accuracy |
| 7 | Fix T2+T5 (build_optimizer + differential LR) | 30 min | Progressive unfreeze |
| 8 | Decision: keep or drop feature KD | 15 min | Architecture |
| 9 | Update unit tests (T8) | 30 min | CI |

**Minimum viable v35:** Items 1-5 + item 8 (drop feature KD) = ~4h of work + 2-3h generation time.

---

## Architecture Recommendation

**Drop feature distillation for v35.** Rationale:

1. Feature KD has fundamental spatial misalignment (teacher crop vs student full-image) that requires architectural changes, not bug fixes
2. Coordinate KD alone is the standard approach for pose distillation (DWPose, PoseConv, etc.)
3. Getting coordinate KD right first provides a measurable baseline
4. Feature KD can be re-added in v36 with proper bbox-region alignment

**Simplified v35 architecture:**
```
Teacher (MogaNet-B) ──> offline heatmaps ──> soft argmax ──> coords (crop space)
                                                                │
                                              inverse affine + padding fix
                                                                │
                                                         coords (full image)
                                                                │
Student (YOLO26-Pose) ──> fixed decode ──> coords (pixel space)
                                │
                         normalize to [0,1]
                                │
                    MSE(student_kpts, teacher_kpts) * confidence * batch_size
                                │
                         coord_loss (scaled by coord_alpha)
                                +
                         gt_loss (PoseLoss26 / batch_size)
```
