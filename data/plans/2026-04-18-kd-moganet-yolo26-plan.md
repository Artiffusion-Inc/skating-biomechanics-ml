# Implementation Plan: KD MogaNet-B → YOLO26-Pose (DistilPose-Style)

**Design spec:** `data/specs/2026-04-18-kd-moganet-yolo26-design.md`
**Created:** 2026-04-18
**Last updated:** 2026-04-22 (implementation progress update)
**Branch:** `docs/kd-moganet-yolo26-spec`

---

## Change Log

**2026-04-22 (v5 - critical fixes applied):**
- ✅ **Feature caching implemented** — generate_teacher_features.py creates ~4GB HDF5 with backbone features [4,6,8]
- ✅ **Full DWPose Two-Stage KD** — feature + logit distillation, α=0.00005, β=0.1, weight decay schedule
- ⚠️ **CRITICAL BUG FOUND:** Teacher heatmaps generated with `torch.clamp()` instead of `torch.sigmoid()`
  - MogaNet-B DeconvHead outputs raw logits (no activation) → [-1.19, +1.19] range
  - Clamp destroyed peak structure: peaks 0.094 instead of ~0.7, many zeros introduced
  - **FIX APPLIED:** Changed `torch.clamp(0,1)` → `torch.sigmoid()` in generate_teacher_heatmaps.py
  - **Regenerating 58GB heatmaps** (~45-60 min on RTX 5090)
- 🔄 **Next:** After heatmaps complete → verify peaks ~0.7 → start full DWPose training (210 epochs, ~2-3 days)

**2026-04-22 (final v3 - research + implementation complete):**
- ✅ **Teacher heatmaps GENERATED** — (264,874, 17, 72, 96), 62GB, MSRA Gaussian, batch=128 (1.3h)
- ✅ **Critical fixes applied** — MSRA norm, Adam(lr=0.001), kd_weight=0.001, data clamp
- ✅ **Research completed** — 4 parallel agents analyzed DistilPose, YOLO-KD, best practices
- ✅ **Key findings:**
  - DistilPose: model-level integration, online teacher, multi-loss
  - YOLO-KD: MGD best for detection (+2.8% mAP), DWPose for pose
  - Best practice: two-stage scheduling (0.1→1.0), temperature annealing (3→1)
- ⚠️ **Issue discovered:** `yolo train` does NOT support KD losses (baseline only!)
- ✅ **Solution:** Use `distill_trainer.py` directly (already working, 13/13 tests pass)

**2026-04-22 (final v2 - critical fixes applied):**
- ✅ Task 12: distill_trainer.py — 13/13 tests pass, 6 bugs fixed (softplus→sigmoid, teacher norm, mosaic=0, padding, lr0, student conf)
- ✅ Task 11: simulate_heatmap.py — MSRA encoding, 12/12 tests pass
- ✅ MogaNet-B verified — (1,17,72,96) output, 33.8ms/crop on RTX 5090
- ✅ ultralytics 8.4.41 + YOLO26n/s-pose.pt — sigma head verified
- ✅ AP3D-FS — 35,705 train / 21,368 val
- ✅ COCO 10% — 5,659 train images
- ✅ FineFS v2 complete — 229,169 train / 57,943 val (287K total) via 24× multiprocessing
- ✅ Training configs — calibration.yaml, stage3_distill.yaml, stage3_baseline.yaml
- ✅ **CRITICAL FIXES APPLIED (2026-04-22 16:00 UTC):**
  - ❌ **Heatmap norm:** Per-channel [0,1] → **MSRA Gaussian (raw + clamp)** — destroys spatial structure!
  - ❌ **LR:** optimizer=auto(lr=0.01) → **Adam(lr=0.001)** — too aggressive for pose KD
  - ❌ **kd_weight:** 1.0 → **0.001** — DistilPose uses 0.0005
  - ✅ **score_weight:** 0.01 (already correct)
  - ✅ **Data clamp:** Added to AP3D-FS converter (FineFS already had it)

**Sources for fixes:**
- DistilPose: `configs/.../DistilPose_S_coco_256x192.py` — Adam(lr=1e-3), kd_weight=0.0005
- MMPose: `mmpose/codecs/utils/gaussian_heatmap.py:167-169` — MSRA encoding peak=1.0
- Ultralytics: `default.yaml` — "SGD=1e-2, Adam/AdamW=1e-3"

**2026-04-21:**
- ✅ Pre-flight checks completed (FineFS quality, Sigma head, HDF5 benchmark, FSAnno availability)
- ✅ **DISCOVERY:** YOLO26 has built-in sigma head (no architecture changes needed)
- ✅ **DISCOVERY:** AP3D contains 50% figure skating data (not just athletic proxy)
- ❌ FSAnno SKIPPED (61.5% YouTube videos unavailable)
- 📊 Data budget updated: ~280K train (vs 632K original — actual counts after conversion)
- 💰 Budget: RTX 5090 rented at $0.593/hr ($0.371 GPU + $0.223 disk 800GB)

---

## Task 0: Training Time & Risk Estimation

**Reference:** Ultralytics docs — "baseline: ~2.8s compute per 1000 images on RTX 4090". **WARNING:** This is an internal cost-estimator coefficient, NOT wall-clock time. Cross-check with Ultralytics cost table shows 72× discrepancy between formula and real cost-derived time. No reliable per-epoch benchmark available.

**Approach: Measure, don't estimate.**

### Step 0: Calibration Run (first thing on rented GPU)

Before any training, run 5 epochs on actual data to get real wall-clock time:

```bash
yolo train model=yolo26n-pose.pt data=skating_full.yaml epochs=5 batch=32 imgsz=640 device=0
# Record: total_wall_time / 5 = seconds_per_epoch
```

Then extrapolate:
```
total_hours = (seconds_per_epoch / 3600) × planned_epochs × KD_overhead
```

**Budget assumption (pessimistic, until calibrated):**
- Use previous estimate: **$46 on RTX 4090, $23 on RTX 5090** (with 1.5× contingency)
- If calibration shows faster → free up budget for gated stages or more data
- If calibration shows slower → cut gated stages or reduce epochs

### Risk Mitigation Strategies

**1. Offline teacher heatmaps (biggest cost saving)**
Pre-compute MogaNet-B heatmaps ONCE for all training images, store as **single HDF5 file** (not individual .npy — 632K files = I/O bottleneck). Eliminates 1.5× teacher overhead during KD, eliminates VRAM concern.

**CRITICAL: MogaNet-B is top-down.** Requires bounding boxes as input — use GT boxes from YOLO labels. Algorithm:
```python
for img, label in dataset:
    bbox = label['bbox']          # GT bbox from YOLO .txt
    crop = crop_image(img, bbox, padding=0.2)
    hm = moganet_topdown_inference(crop)
    save to HDF5
```
**Pre-flight validated (2026-04-21):** HDF5 throughput = 4330 heatmaps/sec (43× target). Single file sufficient.

**2. Skip Stage 2 ablation — POST-MORTEM already answered**
Previous experiment: freeze=20 was best (0.517 vs 0.406 at freeze=0). Start KD from pretrained + freeze=10-20. Fallback to ablation only if KD fails.

**3. Gate Stage 2.5 — skip if teacher already good on skating**
Run MogaNet-B eval on 100 skating val images first. If AP > 0.85: skip teacher adaptation.

**4. Gate Stage 3.5 — skip if gap is small**
After Stage 3 converges: check teacher-student gap. gap < 0.08 → DONE, skip TDE.

**5. Progressive sizing — n first, then s/m only if needed**
Train YOLO26n (100 epochs). If AP >= 0.85 → DONE.

**6. Data validation locally — zero GPU cost**
**COMPLETED (2026-04-21):**
- ✅ FineFS quality: 94.1% visible, H3.6M-compatible, zero artifacts
- ✅ Sigma head: exists in YOLO26, pretrained weights available
- ✅ HDF5 performance: 4330 heatmaps/sec (43× target)
- ✅ FSAnno: 61.5% YouTube unavailable → SKIPPED
- ✅ AP3D-fs: 50% IS figure skating → KEEP (350K samples)

**7. Heatmap resolution — VERIFIED**
MogaNet-B output shape confirmed: (17, 72, 96) for 384×288 input (stride=4).

### Budget (updated 2026-04-21)

**Data budget increased:**
- FineFS: 272K train (validated excellent)
- AP3D-fs: 350K train (figure skating only, not tnf/rm)
- COCO: 10K train (10% dynamic)
- **Total: 632K train** (vs 369K in original plan = 1.7× increase)

| GPU | ETA (280K images) | Cost | Notes |
|-----|-------------------|------|-------|
| RTX 5090 ($0.593/hr) | ~95h | **$56** | GPU $0.371 + Disk $0.223 (800GB) |

**Note:** Actual cost = $0.593/hr (not $0.305 — that's GPU-only). Disk 800GB adds $0.223/hr.

**After calibration:** Replace ETA with `(seconds_per_epoch / 3600) × planned_epochs`. Update cost accordingly.

**Actions:**
- [x] Pre-flight checks COMPLETED (all validations passed)
- [ ] Rent GPU, run 5-epoch calibration → record `seconds_per_epoch`
- [ ] Recalculate ALL estimates using real measurement
- [ ] Pre-compute teacher heatmaps after calibration

**Validation:** Pre-flight checks complete, calibration run completes without errors.

---

## Task 1: Project Structure & Dependencies

Create experiment directory and install dependencies.

**Actions:**
- [ ] Create `experiments/yolo26-pose-kd/` with subdirs: `scripts/`, `configs/`, `checkpoints/`, `results/`
- [ ] Add `checkpoints/` and `results/` to `.gitignore`
- [ ] Create `experiments/yolo26-pose-kd/requirements.txt` (ultralytics, mmpose, torch, torchvision)
- [ ] Verify `ultralytics` supports YOLO26-Pose (`pip show ultralytics` or `uv add`)

**Files:**
- `experiments/yolo26-pose-kd/.gitignore`
- `experiments/yolo26-pose-kd/requirements.txt`

**Validation:** `ls experiments/yolo26-pose-kd/` shows expected structure.

---

## Task 2: Explore FineFS Data Format

Before writing converters, understand FineFS data.

**Actions:**
- [ ] Extract `skeleton.zip` from `/home/michael/Downloads/FineFS/data/`
- [ ] Read 1 NPZ file — check shape, keypoint format, coordinate system
- [ ] Read 1 annotation JSON — check structure, timing, element labels
- [ ] Determine: 3D or 2D? Normalized or pixel? Which 17kp mapping?
- [ ] **Document FineFS 17kp → COCO 17kp mapping** — usually same order, but verify (some datasets swap ankle/knee indices)
- [ ] Document format in a comment at top of converter script

**Output:** Format spec for FineFS (shape, dtype, keypoint order, coordinate range).

**Blocker:** Tasks 3 and 4 depend on this.

---

## Task 3: FineFS → YOLO Converter

Convert FineFS dataset to YOLO pose format.

**Context:** FineFS has 1,167 videos, NPZ shape (4,350, 17, 3) per video = ~5M raw frames. Files: `skeleton.zip` (868MB), `video.zip` (40GB), `annotation.zip` (1.1MB) — all at `/home/michael/Downloads/FineFS/data/`. Videos already extracted (1167 MP4 files).

**Sampling strategy:** 2fps from video (assume 30fps source → every 15th frame) → ~290 frames/video → 338K raw. After filter (>= 5 visible keypoints, ~80% pass) → **~270K frames**. This is the primary dataset (75% of training data).

**Why 2fps:** Adjacent frames in skating video are nearly identical (30fps). 2fps captures diverse poses without redundancy. Research shows diminishing returns after ~100K images — our 270K is above sweet spot but necessary for domain coverage across 1167 different skaters/elements.

**Actions:**
- [ ] Create `experiments/yolo26-pose-kd/scripts/convert_finefs.py`
- [ ] Read annotation JSON — check structure, timing, element labels, visibility flags
- [ ] Map FineFS 17kp to COCO 17kp (if different order)
- [ ] 3D→2D projection (take x,y, discard z)
- [ ] Extract frames from videos at 2fps (OpenCV)
- [ ] Generate bounding boxes from keypoints (PCK-based padding, factor=0.2)
- [ ] Filter frames with < 5 visible keypoints
- [ ] Split: 80% train / 20% val (video-level split, not frame-level — no leakage)
- [ ] Output: YOLO format (images/ + labels/*.txt per image)
- [ ] Spot-check: visual overlay of keypoints on 10 random frames
- [ ] Record actual frame count after sampling

**Input:** `/home/michael/Downloads/FineFS/data/skeleton/` + `video/` + `annotation/`
**Output:** `experiments/yolo26-pose-kd/data/finefs/train/` (~216K) and `val/` (~54K)

**Validation:** Count total frames, check label distribution, visual spot-check.

---

## ~~Task 4: FSAnno → YOLO Converter~~

**Status:** **SKIPPED** — YouTube videos unavailable (61.5% deleted/private)

**Reason:** Pre-flight check (2026-04-21) tested 13 random YouTube URLs from video_sources.json. Only 5/13 (38.5%) were available. Without source videos, FSAnno annotations cannot be used for video-based training.

**Data available:** FSAnno annotations and metadata are intact but unusable without videos.

**Impact:** MINIMAL. FineFS + AP3D-fs provide sufficient skating data (620K+ frames).

---

## ~~Task 5: FSC + MCFS → EXCLUDED~~

**Status:** EXCLUDED — no original video frames available.

**Reason:** FSC (4168 sequences, 150 frames each, PKL) and MCFS (2668 segments, 141 frames each, PKL) contain only pose data without source video. YOLO training requires images for person detection + pose regression — skeleton-only data on black background is unsuitable. Generating synthetic images from poses would introduce domain gap.

**Data available (for reference only):**
- FSC: `data/datasets/figure-skating-classification/train_data.pkl` — shape (150, 17, 3), 4161 train + 1007 test sequences
- MCFS: `data/datasets/mcfs/segments.pkl` — shape (T, 17, 2), 2668 segments

**Impact:** Minor. FineFS (270K) + FSAnno (59K) = 329K skating frames already above sweet spot (~100K). FSC/MCFS would add redundancy, not diversity.
- [ ] Record actual frame count

---

## Task 4.5: AP3D-FS → YOLO Converter

**NEW TASK:** Convert AthletePose3D figure skating portion to YOLO format.

**Context:** AP3D contains 350,674 figure skating samples (49.9% of dataset). Pre-flight check (2026-04-21) revealed AP3D-fs is direct domain match — not proxy data.

**AP3D structure:**
- Location: `data/datasets/athletepose3d/pose_2d/`
- Format: COCO-style annotations (`annotations/train_set.json`)
- Content: 350K fs action samples (Axel, Flip, Loop, Lutz, Salchow, Toeloop, combinations)
- Resolution: Various (multi-camera setup)
- Keypoint format: COCO 17kp (H3.6M compatible)

**Action:** SKIP conversion — AP3D already in COCO format! Just need path configuration.

**Quality:** Human-verified 3D annotations projected to 2D. Tier 1 (high quality).

---

## Task 5: ~~FSC + MCFS~~ → EXCLUDED

**Status:** EXCLUDED — source videos unavailable. (Same as FSAnno)

---

## Task 6: Combine Datasets → data.yaml

Merge all datasets into single Ultralytics-compatible dataset.

**Expected data budget (UPDATED 2026-04-21):**

| Dataset | Train | Val | Role |
|---------|-------|-----|------|
| FineFS (2fps) | ~272K | ~68K | Skating domain (tier 1 — validated excellent) |
| AP3D-fs (figure skating) | ~350K | — | Skating domain (tier 1 — direct fs data) |
| ~~AP3D-tnf/rm~~ | ~~353K~~ | — | ~~DROPPED (not skating-relevant)~~ |
| ~~FSAnno~~ | ~~17K~~ | ~~4K~~ | ~~SKIPPED (YouTube unavailable)~~ |
| COCO (10% dynamic) | ~10K | — | Catastrophic forgetting prevention |
| **TOTAL** | **~632K** | **~68K** | **2.2× original estimate!** |

**Key changes from plan:**
- ✅ FineFS: Validated — 94.1% visible, H3.6M-compatible, zero artifacts
- ✅ AP3D-fs: KEEP — 50% IS figure skating, not proxy data
- ❌ AP3D-tnf/rm: DROP — track&field/running not skating-relevant
- ❌ FSAnno: SKIP — 61.5% YouTube videos unavailable
- ✅ COCO: 10% dynamic (not 15% fixed) — monitor val AP, adjust mix

**Critical:** COCO dynamic mix — monitor COCO val AP every 5 epochs. If AP drops >5% relative → increase COCO mix to 15-20%.

**Actions:**
- [ ] Create `experiments/yolo26-pose-kd/configs/data.yaml`
- [ ] Paths for train: finefs_train + ap3d_train (fs only) + coco_10pct
- [ ] Paths for val: finefs_val (skating-only, primary quality metric)
- [ ] `kpt_shape: [17, 3]` (COCO 17kp + visibility)
- [ ] `names: ['person']`
- [ ] Verify: `yolo val model=yolo26n-pose.pt data=data.yaml` runs without errors
- [ ] Count actual train/val images

**Output:** `data.yaml` with all dataset paths.

**Validation:** Ultralytics can load the dataset without errors.

---

## Task 7: Vast.ai Environment Setup

Prepare remote training environment on Vast.ai.

**GPU Selection:**
- Primary: RTX 4090 24GB (DLPerf≈55, ~$0.14-0.28/hr, best $/perf ratio)
- Fallback: A100 40GB ($0.26-0.52/hr) if VRAM insufficient for YOLO26m + teacher

**Rental Strategy — Unverified + Smoke Test:**
Unverified ≠ broken, just "not yet evaluated by platform". Verification is fully automated (reliability >= 90%, CUDA >= 12, 500+ Mbps). Real risks: provisioning failures (Docker/SSH), thermal throttling under load. Mitigated by 15-min smoke test. If machine survives 15 min at 100% GPU load → likely stable. Savings: 50-80% vs verified.

**Actions:**
- [ ] Search Vast.ai for RTX 4090 24GB, 200GB+ disk, on-demand
- [ ] Filter: reliability >= 95% (even unverified), CUDA >= 12
- [ ] Rent instance (unverified OK — cheaper, see strategy above)
- [ ] **Smoke test (15 min):** run `gpu_burn` or training on 100 images at 100% load
  - Check `nvidia-smi` — temp >85°C or clock drops → destroy instance, rent another
  - Check network — download test file, if <10 MB/s → destroy
  - If smoke test passes → proceed with setup
- [ ] Install: Python 3.11+, PyTorch with CUDA, ultralytics, mmpose
- [ ] Upload: all YOLO format datasets (rsync, compress)
- [ ] Upload: MogaNet-B weights (`moganet_b_ap2d_384x288.pth`)
- [ ] Upload: pretrained YOLO26 weights (`yolo26n/s/m-pose.pt`)
- [ ] Verify: MogaNet-B inference works on 1 test image
- [ ] Verify: YOLO26 validation works on skating val set
- [ ] Set up persistent tmux/screen session
- [ ] Set up checkpointing: best + every 10 epochs (required for interruptible recovery)

**Checkpointing Strategy (for unverified instances):**
- Save best model (by skating val AP)
- Save every 10 epochs to disk
- Optionally sync checkpoints to R2 (external storage)
- If instance dies: resume from last checkpoint on new instance

**Output:** Working remote environment with all data and models.

---

## Task 8: Stage 1 — Baseline Validation

Measure pretrained baselines on skating val. CRITICAL before any training.

**Actions:**
- [ ] Run `yolo val model=yolo26n-pose.pt data=skating.yaml` → record AP, AP50, per-joint AP
- [ ] Run `yolo val model=yolo26s-pose.pt data=skating.yaml` → same metrics
- [ ] Run `yolo val model=yolo26m-pose.pt data=skating.yaml` → same metrics
- [ ] Run MogaNet-B eval on skating val → AP, AP50
- [ ] Run YOLO26n on AP3D val (cross-domain check)
- [ ] Compile baseline comparison table
- [ ] Save to `results/baseline_comparison.csv`

**Output:** Baseline table with real numbers.

**Decision point:** If pretrained AP < 0.5 on skating val, data quality may be insufficient.

---

## Task 9: Stage 2 — Fine-tune Ablation [SKIP BY DEFAULT]

**Status:** SKIP — POST-MORTEM already answered. Previous experiment showed freeze=20 was best (0.517 vs 0.406 at freeze=0). Start KD from pretrained + freeze=10-20.

**Fallback trigger:** Only run if Stage 3 KD fails (student AP < 0.70 after convergence).

**Actions (only if triggered):**
- [ ] Create fine-tune training script
- [ ] Grid: freeze=[10,20], lr=[0.0005,0.001], epochs=50
- [ ] All configs: `val=true, save_period=10, patience=20`
- [ ] Use fine-tuned weights for Stage 3 retry

**Output:** `results/stage2_finetune_results.csv` (only if triggered)

---

## Task 10: Stage 2.5 — Teacher Domain Adaptation [GATED]

**Gate:** Skip if MogaNet-B AP > 0.85 on 100 skating val images (likely — AP=0.962 on AP3D).

**Actions:**
- [ ] Run MogaNet-B eval on 100 skating val images → record AP
- [ ] If AP > 0.85: **SKIP**, use original weights. Proceed to Task 11.
- [ ] If AP <= 0.85: fine-tune deconv head only (frozen backbone)
  - [ ] Config: lr=0.0001, epochs=10-15, data=skating train
  - [ ] Validate AP before and after
  - [ ] Save: `moganet_b_skating_adapted.pth`

**Output:** Decision (skip or adapted weights).

---

## Task 11: Simulated Heatmap Module + Built-in Sigma

Implement MSRA encoding for DistilPose-style KD using YOLO26's **built-in sigma head**.

**Source:** DistilPose `distilpose/models/losses/dist_loss.py` — `Reg2HMLoss` class.

**DISCOVERY (2026-04-21):** YOLO26-Pose **already has sigma head** — `cv4_sigma` with 34 channels (17kp × 2). Pre-trained weights available. NO architecture modification needed!

**How it works:** Student uses built-in `kpts_sigma` from YOLO26 forward output. From `(x, y, sigma_x, sigma_y)`, a 2D Gaussian heatmap is generated via MSRA unbiased encoding. This heatmap is compared via MSE against teacher's heatmap.

```python
# YOLO26 already provides sigma during training:
pred = model.forward(x)  # Returns kpts + kpts_sigma
kpts = pred["kpts"]     # (B, 51, anchors) → 17kp × 3
sigma = pred["kpts_sigma"]  # (B, 34, anchors) → 17kp × 2

# Fully differentiable (vectorized, no loops)
mu_x = kpts_x * W  # normalized [0,1] → pixel coords
mu_y = kpts_y * H
sigma_x = sigma[:, 0::2]  # Extract sigma_x from sigma output
sigma_y = sigma[:, 1::2]  # Extract sigma_y
g = exp(-0.5 * ((x - mu_x)^2 / (sigma_x^2 + eps) + (y - mu_y)^2 / (sigma_y^2 + eps)))
```

**Actions:**
- [ ] Load MogaNet-B weights, run `model.forward()` on test crop → record exact `heatmap.shape` (confirmed: (17, 72, 96) for 384×288 input)
- [ ] Create `experiments/yolo26-pose-kd/scripts/simulate_heatmap.py`
- [ ] Function `keypoints_to_heatmap(kpts, sigma, visibility, hm_shape=(17, 72, 96))`
  - Input: `kpts [B,K,2]` normalized [0,1], `sigma [B,K,2]` from YOLO26 cv4_sigma, `visibility [B,K]`
  - Output: `heatmap [B,K,H,W]` — 2D Gaussian per keypoint
  - **hm_shape = (17, 72, 96) confirmed from MogaNet-B**
- [ ] MSRA unbiased encoding: vectorized grid computation
- [ ] Visibility masking: zero out invisible keypoints (visibility=0)
- [ ] Unit tests: known (x,y,sigma) → expected heatmap peak location and width
- [ ] Test differentiability: `heatmap.sum().backward()` must work

**Implementation note:** YOLO26 sigma head is already trained. Fine-tune only sigma head on skating data:
```python
for name, param in model.model.named_parameters():
    if 'sigma' in name:
        param.requires_grad = True  # Unfreeze sigma
```

**Output:** `simulate_heatmap.py` with verified heatmap shape and differentiability test.

---

## Task 12: DistilPoseTrainer Implementation (Offline Heatmaps + Built-in Sigma)

Implement custom Ultralytics trainer with DistilPose-style KD loss using **pre-computed offline teacher heatmaps** and **YOLO26's built-in sigma head**.

**Key changes from original plan:**
1. Teacher model NOT loaded during training (heatmaps pre-computed)
2. YOLO26 already has sigma head — use it, don't add new one
3. HDF5 single file sufficient (4330 heatmaps/sec = 43× target)
4. MSE not KL divergence (no temperature parameter)

**Source:** DistilPose `distilpose/models/losses/dist_loss.py` + pre-flight findings.

**DistilPose KD Loss (verified):**
```
L_total = L_gt + 1.0 * L_reg2hm + 0.01 * L_score
```
| Component | Loss | Weight | What it does |
|-----------|------|--------|-------------|
| L_gt | Standard YOLO pose loss | 1.0 | GT keypoints regression |
| L_reg2hm | MSE | 1.0 | Simulated heatmap vs teacher heatmap |
| L_score | L1 | 0.01 | Student confidence vs teacher value at pred location |

**Implementation approach (Option A - monkey-patch):**

```python
class DistilPoseTrainer(PoseTrainer):
    def setup_model(self):
        super().setup_model()
        # Unfreeze sigma head only
        for name, param in self.model.model.named_parameters():
            if 'sigma' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False  # Freeze backbone
    
    def preprocess_batch(self, batch):
        # Attach teacher heatmaps from HDF5 to batch
        batch["teacher_hm"] = self.load_teacher_heatmaps(batch["im_file"])
        return batch
    
    def compute_loss(self, batch):
        # GT loss
        gt_loss = super().compute_loss(batch)
        
        # Extract student outputs
        preds = self.model(batch["img"])
        kpts = preds["kpts"]  # (B, 51, anchors)
        sigma = preds["kpts_sigma"]  # (B, 34, anchors)
        
        # Generate simulated heatmap from kpts + sigma
        sim_hm = keypoints_to_heatmap(kpts, sigma, batch["keypoints_visible"])
        
        # Load teacher heatmaps (pre-computed)
        teacher_hm = batch["teacher_hm"]  # (B, 17, 72, 96)
        
        # KD loss: MSE (not KL!)
        kd_loss = F.mse_loss(sim_hm, teacher_hm)
        
        # Optional ScoreLoss
        student_conf = kpts[..., 2]  # visibility channel
        teacher_conf_at_pred = extract_teacher_value(teacher_hm, kpts[..., :2])
        score_loss = F.l1_loss(student_conf, teacher_conf_at_pred)
        
        return gt_loss + 1.0 * kd_loss + 0.01 * score_loss
```

**Actions:**
- [ ] Create `experiments/yolo26-pose-kd/scripts/distill_trainer.py`
- [ ] Subclass PoseTrainer (not BaseTrainer)
- [ ] Override `setup_model()` to unfreeze sigma head
- [ ] Override `preprocess_batch()` to load teacher heatmaps from HDF5
- [ ] Override `compute_loss()` with monkey-patch approach above
- [ ] HDF5 config: `h5py.File(path, 'r', libver='latest', swmr=True)` for concurrent read
- [ ] Warm-up: reg2hm_loss=0 for first 5 epochs (let student learn coordinates)
- [ ] No annealing needed (DistilPose uses fixed weight=1.0)
- [ ] Test: run 1 epoch on 10 images, verify loss decreases

**HDF5 structure (verified):**
```python
# teacher_heatmaps.h5
# /heatmaps: shape (N_images, 17, 72, 96), dtype float16
# /indices: image path → row index (JSON sidecar)
# Performance: 4330 heatmaps/sec (43× target)
```

**Output:** `distill_trainer.py` with DistilPoseTrainer class.

---

## Task 13: Stage 3 — DistilPose Response KD (Progressive Sizing)

Run knowledge distillation training with offline teacher heatmaps.

**Step 0: Pre-compute teacher heatmaps (first thing on rented GPU)**
- [ ] Load MogaNet-B, verify heatmap output shape (Task 11)
- [ ] For each training image:
  - Read GT bbox from YOLO label (.txt)
  - Crop image around bbox with 0.2 padding
  - Resize crop to MogaNet input size (384×288)
  - Run MogaNet top-down inference → get heatmap
  - Store in HDF5: `teacher_heatmaps.h5` (float16 to save disk)
- [ ] Create index JSON: image_path → HDF5 row number
- [ ] Verify: load 10 random heatmaps, visualize, check peaks match GT keypoints
- [ ] Estimated time: ~1.5h on RTX 5090, ~3h on RTX 4090

**Step 1: Train YOLO26n (primary)**
- [ ] Create `configs/stage3_distill.yaml` with KD params
- [ ] Run: `model.train(data=skating_full.yaml, trainer=DistilPoseTrainer, freeze=10, epochs=100)`
- [ ] Monitor: GT loss, KD loss, total loss, val AP per epoch
- [ ] Early stop on val AP (patience=20)
- [ ] Save best model by skating val AP

**Step 2: Check success criteria**
- [ ] If YOLO26n AP >= 0.85 on skating val → **DONE**, skip s/m
- [ ] If YOLO26n AP < 0.85 → train YOLO26s with same config (Step 3)
- [ ] If YOLO26s AP < 0.85 → train YOLO26m (Step 4, last resort)

**Output:** Trained YOLO26 model(s) with distilled knowledge.

**Decision point:**
- If gap (teacher - student) < 0.08 AP → skip Stage 3.5
- If gap >= 0.08 AP → proceed to Stage 3.5

---

## Task 14: Stage 3.5 — Optional TDE + Feature KD [GATED]

**Gate:** Only if Stage 3 gap >= 0.08 AP.

**Actions:**
- [ ] Research DistilPose TDE implementation from github.com/yshMars/DistilPose
- [ ] Implement Token-distilling Encoder (attention + tokenization)
- [ ] Add feature KD loss: `beta * MSE(student_feat, TDE(teacher_feat))`
- [ ] Re-run training with combined loss
- [ ] Compare: Stage 3 only vs Stage 3 + 3.5

**Output:** Potentially improved student model.

---

## Task 15: Stage 4 — Student Size Selection [PROGRESSIVE]

Already handled in Task 13 (progressive sizing). This task is for final evaluation.

**Actions:**
- [ ] Compile final comparison table:
  - Model | AP (skating val) | AP (AP3D val) | Speed (FPS) | Params | Gap vs teacher
- [ ] Select smallest model meeting all success criteria
- [ ] Export selected model to ONNX for production use
- [ ] If no model meets criteria → increase data or add Stage 3.5

**Output:** Final model + comparison table.

---

## Task 16: Results Documentation

Document all results.

**Actions:**
- [ ] Write `experiments/yolo26-pose-kd/results/README.md` with:
  - Baseline comparison table
  - Fine-tune ablation results
  - KD training curves summary
  - Final model performance
  - Lessons learned
- [ ] Update project ROADMAP.md with KD experiment status
- [ ] Commit all scripts and results (not checkpoints)

---

## Dependency Graph

```
Task 0 (time estimation) ──→ recalculate after Task 3-4 (actual N_images)

Task 1 (structure)
  ├── Task 2 (explore FineFS) ──→ Task 3 (FineFS converter)
  └── Task 4 (FSAnno converter)

Task 3 + Task 4 ──→ Task 6 (data.yaml) ──→ Task 7 (Vast.ai setup)
                                                         └──→ Task 8 (baseline)
                                                                └──→ Task 10 (teacher adap, GATED)
                                                                       └──→ Task 13 step 0 (pre-compute heatmaps)

Task 11 (heatmap module) ──→ Task 12 (DistilPoseTrainer)

Task 10 + Task 12 ──→ Task 13 (KD training + progressive sizing)
  └──→ Task 14 (TDE, GATED)
       └──→ Task 15 (final evaluation)
            └──→ Task 16 (results)

Task 9 (fine-tune ablation) — SKIP by default, only if KD fails
```

## Parallelization Opportunities

| Group | Tasks | Can run in parallel? |
|-------|-------|---------------------|
| Structure + data explore | 0, 1, 2 | Yes |
| Data converters | 3, 4 | Yes (after Task 2) |
| KD modules | 11, 12 | Yes (sequential, 11→12) |
| Vast.ai setup | 7 | After Task 6 (needs data ready) |
| Baseline → teacher eval → heatmaps | 8, 10, 13-step0 | Sequential (8→10→13) |
| Task 9 (ablation) | 9 | SKIP by default |
