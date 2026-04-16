# YOLO26-Pose Migration & COCO 17kp Standardization — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use subagent-driven-development (recommended) or executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace RTMO/rtmlib with fine-tuned YOLO26-Pose AND standardize on COCO 17kp (drop H3.6M).

**Architecture:** Fine-tune YOLO26-Pose on AthletePose3D (COCO 17kp GT, all 12 cameras, subject-level split S1 train / S2 val, k-means frame selection, imgsz=1280). Export ONNX + TensorRT. ONNX runtime (no ultralytics at inference). Standardize 2D pipeline from H3.6M→COCO indices. pose_3d/ (CorrectiveLens) excluded — disabled, will be rebuilt natively on COCO 17kp in future.

**Tech Stack:** ultralytics (training), ONNX Runtime (runtime), TensorRT (optional Vast.ai), Vast.ai 2x RTX 4090 (training + GPU inference)

---

## Context

RTMO/rtmlib works (19.1px mean error vs GT). YOLO26-Pose: better speed, TensorRT, BoTSORT, easy fine-tuning. COCO 17kp is universal — all models (YOLO26, RTMO, ViTPose, CIGPose) and datasets (AthletePose3D, FSC, MCFS) use it natively. Dropping H3.6M eliminates `coco_to_h36m()` conversion with zero quality impact.

## Data Research Summary

| Question | Decision | Evidence |
|----------|----------|----------|
| **Camera sampling** | All 12 cameras + split by subject (S1 train, S2 val) | Wang et al. 2020: viewpoint diversity > correlation risk. Only S1,S2 have skating data (S3-S5 are athletics). Subject split prevents memorization. |
| **Frame sampling** | k-means ~25 frames per video (fallback: uniform 5 FPS) | DeepLabCut: k-means selects visually diverse poses, avoids redundant adjacent frames at 60 FPS. 5 FPS = 37 frames ≈ reasonable approximation. |
| **NaN handling** | vis=0, (x,y)=mean of visible keypoints. Skip frames with <3 visible. | Ultralytics: vis=0 keypoints excluded from pose loss. But (x,y) used for bbox — must set to reasonable values, not (0,0). |
| **Other sports** | Skating only (1,398 videos) | Sports survey 2025: domain-specific data critical. Boxing→other doesn't transfer. COCO pretrained already provides general pose knowledge. Fine-tuning should specialize. |
| **COCO-Pose mixing** | 15% COCO-Pose images in train set | Gandhi et al. 2025: freeze=10 fine-tuning with mixed data yields +10% mAP target, <0.1% COCO forgetting. Adds body/pose/background diversity, prevents overfitting on 1 subject. |
| **SkatingVerse pseudo-labels** | **From round 1** (not deferred) | Federated learning (Ni et al. 2025): single-subject data creates body-proportion bias. Jaus et al. 2025: "label quality has minimal impact for feature transfer." SkatingVerse adds hundreds of diverse skaters + real rink environments. Drolet-Roy et al. 2026: 12K diverse sports images = +17.3 AP. |
| **Subject diversity** | Priority over volume | 226K frames from 1 person = overkill volume, underkill diversity. Research shows subject diversity is #1 factor for generalization. |
| **Augmentation on ice** | Image-level only, conservative | GCN experiments (exp_augmentation.py): mirror harmful for classification (-4.1pp), joint noise/SkeletonMix no gain. But image-level mosaic/mixup/fliplr are fundamentally different (don't change element identity). copy_paste keypoints removed (analogue of SkeletonMix). |

**Dataset stats:**
- S1 train: 753 skating clips × 12 cameras = ~9,036 clips × ~25 frames = **~226K GT frames**
- SkatingVerse: 28K videos × 2 FPS × ~3s avg = **~168K pseudo-labeled frames** (filtered by confidence ≥0.5, ≥8 visible keypoints) — **DEFERRED**
- COCO mix: 15% of total = **~59K COCO-Pose frames**
- S2 val: 755 skating clips × 12 cameras × ~25 frames = **~227K GT frames**
- **First round train: ~285K frames** (226K GT + 59K COCO)
- **With SkatingVerse: ~453K frames** (226K GT + 168K pseudo + 59K COCO)
- **Total val: ~227K frames** (S2 GT only — hold-out subject)

**IMPORTANT:** Only S1 and S2 contain skating data. S3-S5 are athletics only (Running, Discus, Javelin, Shot_put, Spin_discus). S1 valid+test splits exist (1522 more clips with GT) but same person — not used to avoid body-proportion overfitting.

**Available but NOT used:**
- S1 valid_set + test_set (1522 clips, GT) — same person as S1 train, risk of overfitting
- S3-S5 (athletics only) — different kinematics, pose estimation transfers but not worth dilution
- FSC (5,168 seq) + MCFS (2,668 seq) — poses only, no images. Useful for benchmark.
- FineFS (1,167) — quality scores only, no poses.

## Key Design Decisions

| Decision | Choice | Why |
|----------|--------|-----|
| Keypoint format | COCO 17kp (drop H3.6M) | Universal standard, all models native, no approximations |
| Runtime backend | ONNX (not ultralytics) | Lightweight, portable |
| Training resolution | 1280 (not 640) | Preserves wrist/ankle detail from 1920px source |
| Camera strategy | All 12 cameras, S1 train / S2 val | Viewpoint diversity > correlation risk. Only S1,S2 have skating (S3-S5 athletics). |
| Frame selection | k-means ~25 frames per video | DeepLabCut-validated: visually diverse poses, no redundancy |
| NaN handling | vis=0 + mean visible coords, skip <3 visible | vis=0 excluded from pose loss, coords for bbox only |
| Domain | S1 GT + SkatingVerse pseudo + 15% COCO | Subject diversity > volume. 226K from 1 person < 168K from hundreds of skaters. |
| Tracking | BoTSORT + SkeletalIdentity | BoTSORT general, SkeletalIdentity for black clothing |
| NMS | None (e2e) | YOLO26 e2e already handles NMS |
| Freeze | 10 layers | Small domain, prevent catastrophic forgetting |
| LR | 0.0005, lrf=0.88 | Official recipe uses gentle decay (88% not 5%) |
| Augmentation | mosaic=0.9, mixup=0.05, fliplr=0.5 | Image-level only. GCN experiments showed mirror harmful for classification, but fliplr+flip_idx is safe for pose estimation (symmetric task). copy_paste=0 (SkeletonMix analogue didn't help). No rotation (vertical sport). |
| Pseudo-label source | Pretrained YOLO26-Pose | From round 1, not after fine-tuning. "Label quality minimal impact for feature transfer" (Jaus et al. 2025). |

## COCO 17kp Index Reference

```
COCO 17kp:
 0: NOSE          5: L_SHOULDER   10: R_WRIST   15: L_ANKLE
 1: L_EYE         6: R_SHOULDER   11: L_HIP     16: R_ANKLE
 2: R_EYE         7: L_ELBOW      12: R_HIP
 3: L_EAR         8: R_ELBOW      13: L_KNEE
 4: R_EAR         9: L_WRIST      14: R_KNEE

Virtual joints (computed helpers, NOT in array):
  pelvis  = midpoint(11, 12)    thorax = midpoint(5, 6)
  spine   = midpoint(pelvis, thorax)    head   = midpoint(1, 2)
```

## File Structure

| File | Action | Responsibility |
|------|--------|---------------|
| `ml/scripts/prepare_yolo_dataset.py` | Create | AthletePose3D + COCO → YOLO labels |
| `ml/scripts/pseudo_label_skatingverse.py` | Create | SkatingVerse pseudo-label generation (pretrained model) |
| `ml/scripts/benchmark_pretrained.py` | Create | Pretrained YOLO26 vs RTMO benchmark |
| `ml/scripts/train_yolo26_pose.py` | Create | Fine-tuning + ONNX + TensorRT export |
| `ml/skating_ml/pose_estimation/pose_extractor.py` | Rewrite | ONNX YOLO26-Pose backend |
| `ml/skating_ml/pose_estimation/coco17.py` | Create | COCO 17kp constants (replaces h36m.py) |
| `ml/skating_ml/pose_estimation/h36m.py` | Keep | Still used by pose_3d/ (disabled pipeline) |
| `ml/skating_ml/types.py` | Modify | COCOKey enum (replaces H36Key for 2D pipeline) |
| `ml/skating_ml/worker.py` | Modify | Update PoseExtractor constructor call (remove deprecated params) |
| `ml/skating_ml/pipeline.py` | Modify | Update PoseExtractor constructor call |
| `ml/skating_ml/visualization/pipeline.py` | Modify | Update PoseExtractor constructor call |
| ~20 files (2D pipeline) | Modify | H36Key.X → COCOKey.X: 13 simple renames + 7 structural (virtual joints) |
| ~14 test files (ml/tests/) | Modify | H36Key → COCOKey in test fixtures and assertions |
| pose_3d/*.py | Exclude | Disabled pipeline, will be rebuilt natively on COCO 17kp |
| `ml/pyproject.toml` | Modify | ultralytics (training), remove rtmlib |

**Out of scope (future work):**
- pose_3d/ CorrectiveLens rebuild — will be completely redone with COCO-native 3D lifter
- 2D→3D pose reconstruction — important near-term priority, but separate from this migration

## GPU Instance & Data Setup

**Hardware:** Vast.ai 2x RTX 4090 (offer 33657001, contract 34994328, NL, 1 TB RAM, 300 GB disk, Driver 580.95)
**Disk:** 300 GB container storage (`--disk 300`) — 166 GB datasets + training runs + checkpoints
**Cost:** ~$0.64/hr → **~$7.7 за 12 часов**
**Strategy:** Параллельные эксперименты на 2 GPU (не DDP). GPU 0 = HP search, GPU 1 = full train.

### Dataset Requirements

| Dataset | Format | Size | Source | Status | Needed for |
|---------|--------|------|--------|--------|------------|
| **AthletePose3D** data.zip | Видео (mp4) | ~71 GB | Google Drive (gdown) | **CRITICAL** | Task 0, 1, 2 (GT videos) |
| **AthletePose3D** pose_2d.zip | GT keypoints (_coco.npy) | ~30 GB | Google Drive (gdown) | **CRITICAL** | Task 0, 1 (GT labels) |
| **COCO** train2017 | Картинки + annotations | ~19 GB | cocodataset.org (S3) | **CRITICAL** | Task 1 (15% mix) |
| **SkatingVerse** | Видео (mp4) | ~46 GB | ModelScope.cn | **DEFERRED** | Task 2.5 (pseudo-labels) |
| MCFS | Только poses (no images) | 103 MB | GitHub | **NOT NEEDED** | GCN classifier only |
| FSC | Только poses (no images) | 340 MB | HuggingFace | **NOT NEEDED** | GCN classifier only |

**Первый раунд (Task 0-2): 120 GB** (AthletePose3D 101 + COCO 19)
**С SkatingVerse (Task 2.5): 166 GB**

### Download Limits

| Source | Speed | Limit | Workaround |
|--------|-------|-------|------------|
| Google Drive (gdown) | ~10 MB/s | **Обрыв через 1 час** | `gdown --continue` + retry loop |
| COCO (Amazon S3) | ~10-50 MB/s | Resume supported | `wget -c` |
| ModelScope.cn | 2-8 MB/s из EU | Signed URL expires | Re-fetch URL, или R2 proxy |

### Dataset Downloads (run on Vast.ai instance)

```bash
# === Setup ===
apt update && apt install -y python3-pip ffmpeg git aria2
pip3 install gdown uv

# === Helper: gdown → aria2c multi-thread download ===
# gdown парсит Google Drive confirmation page, aria2c качает в 8 потоков.
# Fallback на gdown --continue если confirm=t не сработает (quota/cookie issue).
gdown_aria2c() {
    local FILE_ID="$1" OUTPUT="$2"
    DIRECT_URL=$(python3 -c "
from gdown.download import get_url_from_gdrive_confirmation
url = 'https://drive.google.com/uc?id=${FILE_ID}&export=download'
result = get_url_from_gdrive_confirmation(url, '${FILE_ID}')
print(result)
" 2>/dev/null)

    if [ -n "$DIRECT_URL" ] && echo "$DIRECT_URL" | grep -q "google"; then
        echo "Got direct URL, downloading with aria2c (8 streams)..."
        aria2c -x 8 -s 8 -k 1M -c --max-tries=0 --retry-wait=3 "$DIRECT_URL" -o "$OUTPUT"
    else
        echo "Failed to get direct URL, falling back to gdown..."
        while ! gdown --continue "https://drive.google.com/uc?id=${FILE_ID}" -O "$OUTPUT"; do
            echo "Download interrupted, retrying in 10s..."
            sleep 10
        done
    fi
}

# === 1. AthletePose3D (101 GB) — Google Drive ===
# Source: https://github.com/calvinyeungck/AthletePose3D
# Folder: https://drive.google.com/drive/folders/10YnMJAluiscnLkrdiluIeehNetdry5Ft
# NOTE: Individual file IDs below were extracted from this folder.
#        If they don't work, use: gdown --folder "10YnMJAluiscnLkrdiluIeehNetdry5Ft"

mkdir -p /root/data/datasets/athletepose3d && cd /root/data/datasets/athletepose3d

# data.zip (~71 GB) — видео, ~30-50 мин с aria2c
gdown_aria2c "1xnQDxvTjS9D9eYJMWsizbsfHSvxTnCxp" "data.zip"

# pose_2d.zip (~30 GB) — GT keypoints в COCO 17kp (_coco.npy), ~15-25 мин с aria2c
gdown_aria2c "13ISVY8G_NxrwLWFdxyTlOUhSyxVJsd93" "pose_2d.zip"

unzip data.zip && rm data.zip
unzip pose_2d.zip && rm pose_2d.zip

# pose_3d.zip — SKIP (only for future 3D lifter, not needed now)

# === 2. COCO train2017 (19 GB) — Amazon S3 (verified HTTP 200) ===
mkdir -p /root/data/datasets/coco && cd /root/data/datasets/coco

wget https://images.cocodataset.org/annotations/annotations_trainval2017.zip
unzip annotations_trainval2017.zip && rm annotations_trainval2017.zip

aria2c -x 8 -s 8 -k 1M -c \
  http://images.cocodataset.org/zips/train2017.zip
unzip train2017.zip && rm train2017.zip

# === 3. SkatingVerse (44.6 GB) — DEFERRED to Task 2.5 ===
# НЕ скачивать сейчас. Скачать позже, если Task 2 mAP50 < 80%.
# Source: https://modelscope.cn/datasets/awei2003/1st_SkatingVerse_Dataset
# NOTE: zip лежит на OSS attachments, НЕ в git repo.
#        URL /api/v1/.../repo?FilePath=... НЕ работает для attachments!
#        Правильный способ — через OSS tree API → presigned URL:
#
# Step 1: Получить presigned URL
# curl -sL "https://www.modelscope.cn/api/v1/datasets/awei2003/1st_SkatingVerse_Dataset/oss/tree/?MaxLimit=100&Revision=master&Recursive=True&FilterDir=True" | python3 -c "import json,sys; data=json.load(sys.stdin); [print(f['Name'], f['Url']) for f in data.get('Data',{}).get('Files',[])]"
#
# Step 2: Скачать через aria2c (16 потоков, OSS до 1000 conn, ~15-30 мин)
# aria2c -x 16 -s 16 -k 1M -c --max-tries=0 --retry-wait=3 "<PRESIGNED_URL>" -o skatingverse.zip
#
# Альтернатива: ModelScope SDK
# pip install modelscope
# python3 -c "from modelscope.hub.snapshot_download import dataset_snapshot_download; dataset_snapshot_download('awei2003/1st_SkatingVerse_Dataset')"
```

### 12-Hour Session Timeline

```
0:00-1:30  Скачивание (параллельно, aria2c multi-thread):
           GPU idle
           AthletePose3D: ~45-75 мин (8 потоков, Google Drive)
           COCO: ~10-15 мин (8 потоков, S3)

1:30-2:00  Dataset prep (k-means frame selection + COCO mix) — 30 мин

2:00-5:00  GPU 0: HP search batch 1 (25K subset, 640px, yolo26n-pose, 5 configs)
           GPU 1: HP search batch 2 (25K subset, 640px, yolo26n-pose, 5 configs)
           → ~10 экспериментов за 3 часа

5:00-11:00 GPU 0: Full train v1 (285K, 1280px, yolo26s-pose, best HPs)
           GPU 1: Full train v2 (285K, 1280px, yolo26s-pose, alt HPs)
           → 2 финальных модели (~10 epochs каждая, patience=20)

11:00-12:00 GPU 0: Task 2b per-joint PCK evaluation
           GPU 1: ONNX export + speed benchmark
```

### Code Deployment to Vast.ai

**Auth via gh cli (private repo):**

```bash
# 1. Generate PAT locally: https://github.com/settings/tokens
#    Scopes: repo (full)

# 2. On Vast.ai:
apt install -y git gh
gh auth login --with-token <<< "ghp_YOUR_TOKEN_HERE"

# 3. Clone repo
git clone -b feature/yolo26-migration git@github.com:USER/skating-biomechanics-ml.git
cd skating-biomechanics-ml
pip3 install uv
uv sync --project ml/
```

**Option B: rsync from local** (if repo not ready yet)

```bash
rsync -avz --exclude='.venv' --exclude='node_modules' --exclude='__pycache__' \
  /home/michael/Github/skating-biomechanics-ml/ vastai:/root/skating-biomechanics-ml/
```

### Environment Setup on Vast.ai

```bash
# CUDA 13.0, driver 570.x — PyTorch cu124 forward-compatible
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip3 install ultralytics onnxruntime-gpu scikit-learn opencv-python-headless aria2

# Verify
python3 -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0)}, GPU count: {torch.cuda.device_count()}')"
```

### tmux Session Layout (pre-installed on Vast.ai)

**IMPORTANT:** All tmux sessions created with `-d` (detached). Survive SSH disconnects.
Downloads and training continue even if SSH drops.

**Logging:** All output piped to log files via `tee`. Python runs unbuffered (`PYTHONUNBUFFERED=1`).
Log files survive tmux restart (pane scrollback is lost, but files persist).

```bash
# Create detached session (survives SSH disconnect)
tmux new-session -d -s skating

# Pane 0: downloads (background, survives SSH drop)
tmux split-window -h -t 0

# Pane 1: GPU 0 experiments
tmux split-window -h -t 0

# Pane 2: GPU 1 experiments
tmux split-window -h -t 0

# Reattach after SSH reconnect:
tmux attach -t skating

# Navigate between panes: Ctrl+B then arrow keys
# Detach without killing: Ctrl+B, D
```

```bash
# download_datasets.sh — all output to log file (unbuffered)
#!/bin/bash
exec > >(tee -a /root/logs/download.log) 2>&1  # unbuffered, survives SSH drop
set -e
mkdir -p /root/logs
echo "=== Download started $(date) ==="
# ... (download commands) ...
echo "=== Download finished $(date) ==="
```

```bash
# Training with persistent logs (unbuffered Python + tee)
# Pane 1 (GPU 0):
CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 \
  uv run python ml/scripts/train_yolo26_pose.py \
  --name hp_lr001 --no-val --save-period 10 \
  2>&1 | tee /root/logs/hp_lr001.log

# Pane 2 (GPU 1):
CUDA_VISIBLE_DEVICES=1 PYTHONUNBUFFERED=1 \
  uv run python ml/scripts/train_yolo26_pose.py \
  --name hp_lr002 --no-val --save-period 10 \
  2>&1 | tee /root/logs/hp_lr002.log
```

```bash
# After SSH reconnect, check logs:
tail -20 /root/logs/download.log
tail -5 /root/logs/hp_lr001.log
grep -E "mAP50|best" /root/logs/hp_lr001.log
```

```bash
# download_datasets.sh — put all download commands here, runs in background
#!/bin/bash
set -e
mkdir -p /root/data/datasets/athletepose3d && cd /root/data/datasets/athletepose3d
gdown_aria2c "1xnQDxvTjS9D9eYJMWsizbsfHSvxTnCxp" "data.zip"
gdown_aria2c "13ISVY8G_NxrwLWFdxyTlOUhSyxVJsd93" "pose_2d.zip"
unzip data.zip && rm data.zip
unzip pose_2d.zip && rm pose_2d.zip

mkdir -p /root/data/datasets/coco && cd /root/data/datasets/coco
wget https://images.cocodataset.org/annotations/annotations_trainval2017.zip
unzip annotations_trainval2017.zip && rm annotations_trainval2017.zip
aria2c -x 8 -s 8 -k 1M -c http://images.cocodataset.org/zips/train2017.zip
unzip train2017.zip && rm train2017.zip

echo "=== ALL DATASETS READY ==="
```

### Training Performance Notes

**Data loading (285K images, imgsz=1280):**
- `cache='ram'` — НЕВОЗМОЖЕН (нужно ~1.15 TB RAM)
- `cache='disk'` — ~787 GB .npy файлов, можно на 1.6 TB disk
- `cache=False` (default) — OK, dataloader не bottleneck при workers=8 на NVMe
- `workers=8` (default) — хорошо для 2x GPU, auto-capped `min(cpu_count/num_gpus, 8)`
- `amp=True` (default) — FP16, auto-checked на 4090

**Validation optimization:**
- Ultralytics НЕ имеет `val_period` параметра
- Для HP search: `val=False` + `save_period=10` — валидировать каждые 10 epochs offline
- Для full train: `val=True` (default) — early stopping + best model tracking

```python
# HP search: skip validation during training, validate checkpoints after
model.train(data="data.yaml", val=False, save_period=10, ...)

# Full train: normal validation with early stopping
model.train(data="data.yaml", val=True, patience=20, ...)
```

### Artifacts & Experiment Tracking

#### Directory Structure (on Vast.ai)

```
~/skating-biomechanics-ml/
├── runs/pose/                        # Ultralytics output (auto-created)
│   ├── hp_lr001/                     # Each experiment = one dir
│   │   ├── weights/best.pt
│   │   ├── weights/last.pt
│   │   ├── results.csv               # Ultralytics epoch metrics
│   │   ├── args.yaml                 # Full training config (reproducible)
│   │   └── confusion_matrix.png
│   └── full_1280_freeze10/
│       └── ...
├── experiments/yolo26-pose/          # OUR experiment tracking (git-tracked structure)
│   ├── README.md                     # Master report — live updated
│   ├── CLAUDE.md                     # Conventions
│   ├── configs/                      # YAML configs for each run
│   │   ├── benchmark_pretrained.yaml
│   │   ├── hp_lr001_640.yaml
│   │   └── full_1280_freeze10.yaml
│   ├── results/                      # Parsed results (not raw CSVs)
│   │   ├── hp_search_results.csv     # HP search summary table
│   │   └── full_train_results.csv
│   └── notes/                        # Session notes, observations
│       └── session_2026-04-15.md
└── /root/logs/                       # tmux session logs
    ├── download.log
    ├── hp_lr001.log
    └── full_1280.log
```

#### Experiment Configs (YAML, not hardcoded)

Every run uses a YAML config file. Training script reads it:

```yaml
# experiments/yolo26-pose/configs/hp_lr001_640.yaml
name: hp_lr001_640
phase: hp_search          # hp_search | full_train | benchmark
model: yolo26s-pose.pt
data: data/datasets/athletepose3d/yolo26_ap3d_s1.yaml
hypothesis: "LR 0.001 is too high — expect oscillation, mAP < 0.85"

hyperparameters:
  lr0: 0.001
  lrf: 0.01
  freeze: 5
  epochs: 50
  imgsz: 640
  batch: 32
  val: false
  save_period: 10
  augment: true
  mosaic: 1.0
  mixup: 0.0
  fliplr: 0.5
  copy_paste: 0.0
  device: 0               # GPU index (always 0, CUDA_VISIBLE_DEVICES handles mapping)

data_config:
  dataset: ap3d_s1
  cameras: all_12
  frame_sampling: kmeans_25
  coco_mix_pct: 0.15
  total_frames: 25000     # HP search subset
```

Training script usage:
```bash
CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 \
  uv run python ml/scripts/train_yolo26_pose.py \
  --config experiments/yolo26-pose/configs/hp_lr001_640.yaml \
  2>&1 | tee /root/logs/hp_lr001_640.log
```

#### Master Report (`README.md`)

Live-updated during session. Template:

```markdown
# YOLO26-Pose Fine-Tuning — Training Session

**Date:** 2026-04-15
**Hardware:** 2x RTX 4090 (Vast.ai offer 33657001, contract 34994328)
**Session duration:** 12 hours target
**Goal:** mAP50 > 0.90 on AthletePose3D S2 (held-out subject)

---

## Session Log

| Time | Event | Notes |
|------|-------|-------|
| 00:00 | Instance started | apt, uv, PyTorch setup |
| 00:30 | Data download started | COCO + AthletePose3D |
| 01:30 | All data ready | 285K frames |
| 02:00 | Benchmark pretrained | Baseline mAP50=?.?? |
| 02:30 | HP search started | GPU0: lr sweep, GPU1: freeze sweep |
| 05:00 | HP search done | Best config: lr=???, freeze=??? |
| 05:30 | Full train started | Both GPUs, imgsz=1280 |
| 11:00 | Training done | Best mAP50=?.?? |
| 11:30 | ONNX export + benchmark | vs RTMO comparison |

## Benchmark: Pretrained YOLO26-Pose vs RTMO

| Model | mAP50 | mAP50-95 | Mean PKE (px) | Median PKE | P90 PKE |
|-------|-------|----------|---------------|------------|---------|
| YOLO26s-pose (pretrained) | — | — | — | — | — |
| RTMO (current) | — | — | 19.1 | 11.8 | 40.2 |

## HP Search Results

| Config | LR | Freeze | Epochs | imgsz | mAP50 | mAP50-95 | Time (min) | Status |
|--------|-----|--------|--------|-------|-------|----------|------------|--------|
| hp_lr001_640 | 0.001 | 5 | 50 | 640 | — | — | — | pending |
| hp_lr0005_640 | 0.0005 | 5 | 50 | 640 | — | — | — | pending |
| hp_lr0001_640 | 0.0001 | 5 | 50 | 640 | — | — | — | pending |
| hp_lr0005_f10_640 | 0.0005 | 10 | 50 | 640 | — | — | — | pending |
| hp_lr0005_f20_640 | 0.0005 | 20 | 50 | 640 | — | — | — | pending |
| hp_lr0005_f0_640 | 0.0005 | 0 | 50 | 640 | — | — | — | pending |
| hp_lr0005_mos0_640 | 0.0005 | 5 | 50 | 640 | — | — | — | pending |
| hp_lr0005_fliplr0_640 | 0.0005 | 5 | 50 | 640 | — | — | — | pending |
| hp_lr0005_mixup01_640 | 0.0005 | 5 | 50 | 640 | — | — | — | pending |
| hp_lr0005_batch64_640 | 0.0005 | 5 | 50 | 640 | — | — | — | pending |

## Full Training Results

| Config | LR | Freeze | Epochs | imgsz | mAP50 | mAP50-95 | Mean PKE | Time (min) |
|--------|-----|--------|--------|-------|-------|----------|----------|------------|
| full_1280_best | — | — | 100 | 1280 | — | — | — | — |
| full_1280_lr_half | — | — | 100 | 1280 | — | — | — | — |

## Final Model Comparison

| Model | mAP50 | mAP50-95 | Mean PKE | Median PKE | P90 PKE | Speed (ms/frame) |
|-------|-------|----------|----------|------------|---------|------------------|
| RTMO (current) | — | — | 19.1 | 11.8 | 40.2 | — |
| YOLO26s-pose (pretrained) | — | — | — | — | — | — |
| YOLO26s-pose (fine-tuned) | — | — | — | — | — | — |
| YOLO26s-pose (ONNX) | — | — | — | — | — | — |

## Observations

<!-- Fill during session: what worked, what didn't, surprises -->
```

#### Auto-Append Script

After each run completes, append to `results/hp_search_results.csv`:

```python
"""Append run result to tracking CSV. Called after each experiment completes."""
import csv
from pathlib import Path
import yaml

def append_result(run_dir: Path, csv_path: Path):
    """Parse Ultralytics results.csv + args.yaml, append one row."""
    args = yaml.safe_load((run_dir / "args.yaml").read_text())
    rows = list(csv.DictReader((run_dir / "results.csv").open()))
    best = max(rows, key=lambda r: float(r.get("metrics/mAP50(B)", 0)))

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = csv_path.exists()
    fieldnames = [
        "name", "lr0", "freeze", "epochs", "imgsz", "batch",
        "map50", "map50_95", "best_epoch", "status"
    ]

    with csv_path.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            w.writeheader()
        w.writerow({
            "name": run_dir.name,
            "lr0": args.get("lr0", ""),
            "freeze": args.get("freeze", ""),
            "epochs": args.get("epochs", ""),
            "imgsz": args.get("imgsz", ""),
            "batch": args.get("batch", ""),
            "map50": best.get("metrics/mAP50(B)", ""),
            "map50_95": best.get("metrics/mAP50-95(B)", ""),
            "best_epoch": best.get("epoch", ""),
            "status": "done",
        })
```

#### Copy Artifacts Back

```bash
# After session — sync everything
rsync -avz --progress vastai:~/skating-biomechanics-ml/experiments/yolo26-pose/ \
    ./experiments/yolo26-pose/

rsync -avz --progress --include='*/' --include='best.pt' --include='best.onnx' \
    --include='args.yaml' --include='results.csv' --exclude='*' \
    vastai:~/skating-biomechanics-ml/runs/pose/ ./runs/pose/
```

#### What Gets Git-Committed

```
experiments/yolo26-pose/
├── README.md              # YES — final report with numbers
├── CLAUDE.md              # YES — conventions
├── configs/               # YES — reproducible YAML configs
│   ├── hp_lr001_640.yaml
│   └── full_1280_freeze10.yaml
├── results/               # YES — summary CSVs (small)
│   └── hp_search_results.csv
└── notes/                 # YES — session observations
    └── session_2026-04-15.md
```

What stays on Vast.ai only (in .gitignore):
- `runs/pose/*/weights/*.pt` — model files (100-400MB each)
- `runs/pose/*/weights/*.onnx` — ONNX exports
- `runs/pose/*/confusion_matrix.png` — images
- `/root/logs/*.log` — raw tmux logs

---

### Task 0: Benchmark Pretrained YOLO26-Pose vs RTMO

**Why first:** Data-driven decision — measure gap between pretrained YOLO26 and RTMO. 30 min.
**Held-out test:** Use S2 (held-out subject, all 12 cameras). Training uses S1 only.

**Files:**
- Create: `ml/scripts/benchmark_pretrained.py`

- [ ] **Step 1: Write benchmark script**

Script that loads pretrained `yolo26s-pose.pt`, runs on AthletePose3D S2 skating videos (held-out subject), compares against `_coco.npy` GT. Reports mean/median/P90 per-joint. Also runs RTMO for direct comparison.

**IMPORTANT:** Only S1 and S2 contain skating data (Axel, Flip, Loop, Lutz, Salchow, Toeloop, Comb). S3-S5 are athletics only.

```python
"""Benchmark pretrained YOLO26s-pose vs RTMO against AthletePose3D GT.
Downloads yolo26s-pose.pt automatically (first run only)."""

import sys
from pathlib import Path
import cv2
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# S2 is held-out subject for benchmark (S1 = train)
BASE = Path("data/datasets/athletepose3d/videos/train_set/S2")
SKATING = {"axel", "comb", "flip", "loop", "lutz", "salchow", "toeloop"}
KP_NAMES = ["nose","leye","reye","lear","rear","lsho","rsho","lelb","relb","lwri","rwri",
            "lhip","rhip","lkne","rkne","lank","rank"]

def find_skating_videos(base, max_videos=5):
    """Find skating video+GT pairs in subject directory."""
    pairs = []
    for gt in sorted(base.glob("*_coco.npy")):
        vid = gt.with_name(gt.name.replace("_coco.npy", ".mp4"))
        if vid.exists() and any(s in gt.stem.lower() for s in SKATING):
            pairs.append((vid, gt))
            if len(pairs) >= max_videos:
                break
    return pairs

def run_yolo26(pairs):
    from ultralytics import YOLO
    model = YOLO("yolo26s-pose.pt")
    all_preds = []
    for vid, gt in pairs:
        cap = cv2.VideoCapture(str(vid))
        preds = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            r = model.predict(source=frame, conf=0.3, verbose=False, classes=[0])
            if r and r[0].keypoints is not None and len(r[0].keypoints) > 0:
                preds.append(r[0].keypoints.xy[0].cpu().numpy())
            else:
                preds.append(None)
        cap.release()
        all_preds.append(preds)
    return all_preds

def run_rtmo(pairs):
    from skating_ml.pose_estimation.pose_extractor import PoseExtractor
    ext = PoseExtractor(mode="lightweight", output_format="pixels")
    all_preds = []
    for vid, gt in pairs:
        extraction = ext.extract_video_tracked(str(vid))
        if extraction and extraction.poses is not None:
            # poses: (N, 17, 3) — take first 2 columns (x, y) in pixel space
            # Need to convert back from normalized to pixels for comparison
            cap = cv2.VideoCapture(str(vid))
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()
            preds = []
            for pose in extraction.poses:
                if np.isnan(pose).all():
                    preds.append(None)
                else:
                    preds.append(pose[:, :2] * np.array([w, h]))
            all_preds.append(preds)
        else:
            all_preds.append([None] * len(np.load(gt)))
    ext.close()
    return all_preds

def compute_metrics(all_preds, all_gt):
    joint_errors, per_joint = [], {n: [] for n in KP_NAMES}
    for preds, gt in zip(all_preds, all_gt):
        for fi, p in enumerate(preds):
            if p is None or fi >= len(gt): continue
            for j in range(17):
                if np.isnan(gt[fi, j, 0]): continue
                d = np.linalg.norm(gt[fi, j, :2] - p[j])  # _coco.npy is (N,17,3), compare x,y only
                joint_errors.append(d)
                per_joint[KP_NAMES[j]].append(d)
    if not joint_errors: return {}
    e = np.array(joint_errors)
    return {"mean": float(np.mean(e)), "median": float(np.median(e)),
            "p90": float(np.percentile(e, 90)), "n_frames": len(e),
            "per_joint": {n: float(np.mean(v)) for n, v in per_joint.items() if v}}

def main():
    pairs = find_skating_videos(BASE, max_videos=5)
    if not pairs:
        print(f"No skating videos found in {BASE}"); return
    print(f"Benchmark: {len(pairs)} S2 skating videos")
    for vid, gt in pairs:
        print(f"  {vid.name}")

    all_gt = [np.load(gt).astype(np.float32) for _, gt in pairs]
    for name, fn in [("RTMO", run_rtmo), ("YOLO26s (pretrained)", run_yolo26)]:
        print(f"\nRunning {name}...")
        m = compute_metrics(fn(pairs), all_gt)
        if not m: print("  No valid comparisons\n"); continue
        print(f"  Mean: {m['mean']:.1f}  Median: {m['median']:.1f}  P90: {m['p90']:.1f}  ({m['n_frames']} frames)")
        for jn, err in sorted(m["per_joint"].items(), key=lambda x: x[1]):
            print(f"    {jn:6s}: {err:5.1f}")

if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run benchmark**

Run: `uv run python ml/scripts/benchmark_pretrained.py`

Expected: RTMO ~19.1px baseline. YOLO26 pretrained — see if within 5px.

- [ ] **Step 3: Commit**

```bash
git add ml/scripts/benchmark_pretrained.py
git commit -m "feat(ml): add pretrained YOLO26-Pose vs RTMO benchmark"
```

---

### Task 1: Dataset Preparation — GT Frames + Pseudo-Labels + COCO Mix

**Files:**
- Create: `ml/scripts/prepare_yolo_dataset.py`
- Create: `ml/scripts/pseudo_label_skatingverse.py`

- [ ] **Step 1: Write AthletePose3D + COCO dataset preparation script**

Key features (research-backed, see Data Research Summary above):
- **All 12 cameras** per sequence (viewpoint diversity > correlation risk)
- **Subject-level split**: S1 train, S2 val (only S1,S2 have skating — S3-S5 are athletics)
- **k-means frame selection**: ~25 visually diverse frames per video (avoids 60 FPS redundancy, 64px resize)
- **NaN handling**: vis=0, (x,y)=mean of visible keypoints, skip frames with <3 visible
- **Skating only**: Axel, Flip, Loop, Lutz, Salchow, Toeloop, Comb (no athletics)
- **COCO-Pose mixing**: 15% COCO train2017 pose images via symlinks (body/pose diversity, prevents overfitting on 1 subject)
- **Dependency**: scikit-learn (MiniBatchKMeans)

```python
"""Prepare YOLO26-Pose training dataset from AthletePose3D + COCO-Pose.

Research-backed decisions:
- All 12 cameras: viewpoint diversity > correlation (Wang et al. 2020)
- Subject-level split: S1 train, S2 val (only S1,S2 have skating — S3-S5 athletics)
- k-means frame selection: ~25 visually diverse frames, 64px resize (DeepLabCut standard)
- NaN: vis=0 + mean visible coords, skip <3 visible (Ultralytics: vis=0 excluded from loss)
- Skating only: Axel, Flip, Loop, Lutz, Salchow, Toeloop, Comb (no athletics)
- COCO-Pose mixing: 15% via symlinks (Gandhi et al. 2025: <0.1% forgetting, no disk copy)
- Subject diversity > volume: S1 valid+test NOT used (same person, risk of body-proportion bias)
"""

import argparse
from pathlib import Path
import cv2
import json
import numpy as np
from sklearn.cluster import MiniBatchKMeans

COCO_KP_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]
FLIP_IDX = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]
SKATING_JUMPS = {"axel", "salchow", "loop", "lutz", "flip", "toeloop", "comb"}

# Subject-level split: only S1,S2 have skating data (S3-S5 athletics only)
TRAIN_SUBJECTS = {"S1"}
VAL_SUBJECTS = {"S2"}

# COCO-Pose mixing: add 15% COCO images to prevent overfitting on 5 subjects
COCO_MIX_RATIO = 0.15


def is_figure_skating(path: Path) -> bool:
    return any(j in path.stem.lower() for j in SKATING_JUMPS)


def get_subject(path: Path) -> str:
    """Extract subject ID (S1, S2, etc.) from path like .../S1/Axel_10_cam_1.mp4"""
    for part in path.parts:
        if part.startswith("S") and part[1:].isdigit():
            return part
    return ""


def select_frames_kmeans(video_path: Path, n_frames: int = 25) -> list[int]:
    """Select visually diverse frames using k-means clustering (DeepLabCut approach).
    
    At 60 FPS, adjacent frames differ <2%. K-means selects representative poses
    from different motion phases (approach, takeoff, flight, landing).
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return []
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= n_frames:
        cap.release()
        return list(range(total))
    
    # Sample every 2nd frame for k-means input (speed), resize to 64px wide (DeepLabCut standard)
    step = 2
    frames = []
    for fi in range(0, total, step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ret, frame = cap.read()
        if ret:
            small = cv2.resize(frame, (64, int(64 * frame.shape[0] / frame.shape[1])))
            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).flatten().astype(np.float32)
            frames.append(gray)
    cap.release()
    
    if len(frames) <= n_frames:
        return list(range(0, total, step))
    
    frames_arr = np.array(frames)
    kmeans = MiniBatchKMeans(n_clusters=n_frames, random_state=42, batch_size=min(len(frames), 256))
    labels = kmeans.fit_predict(frames_arr)
    
    # Pick frame closest to each cluster center
    selected = []
    for c in range(n_frames):
        mask = labels == c
        if mask.any():
            indices = np.where(mask)[0] * step
            selected.append(int(indices[len(indices) // 2]))  # middle of cluster
    return sorted(set(selected))


def coco_npy_to_yolo_label(kp_xy: np.ndarray, w: int, h: int) -> str | None:
    """Convert COCO keypoints to YOLO pose label.
    
    NaN handling: vis=0, (x,y)=mean of visible keypoints.
    Frames with <3 visible keypoints are skipped (can't compute bbox).
    """
    nan_mask = np.isnan(kp_xy[:, 0])
    n_visible = (~nan_mask).sum()
    if n_visible < 3:
        return None
    
    vis = np.where(nan_mask, 0, 2).astype(np.float64)
    valid = ~nan_mask
    
    # For NaN keypoints: set coords to mean of visible (for bbox computation only)
    kp = kp_xy.copy()
    if nan_mask.any():
        mean_xy = kp_xy[valid].mean(axis=0)
        kp[nan_mask] = mean_xy
    
    # Bbox from visible keypoints with 10% padding
    xc, yc = kp[valid, 0], kp[valid, 1]
    xmin, xmax, ymin, ymax = xc.min(), xc.max(), yc.min(), yc.max()
    pad = max((xmax - xmin) * 0.1, 5), max((ymax - ymin) * 0.1, 5)
    xmin, xmax = max(0, xmin - pad[0]), min(w, xmax + pad[0])
    ymin, ymax = max(0, ymin - pad[1]), min(h, ymax + pad[1])
    
    # Normalize keypoints to [0, 1]
    kp_norm = kp.copy()
    kp_norm[:, 0] /= w
    kp_norm[:, 1] /= h
    
    parts = [0, (xmin + xmax) / 2 / w, (ymin + ymax) / 2 / h, (xmax - xmin) / w, (ymax - ymin) / h]
    for i in range(17):
        parts.extend([kp_norm[i, 0], kp_norm[i, 1], vis[i]])
    return " ".join(f"{v:.6f}" for v in parts)


def extract_video(video, gt, out_img, out_lbl, frame_indices, idx=0):
    gt_poses = np.load(gt).astype(np.float64)
    n_gt = len(gt_poses)
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        return 0
    iw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    ih = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    saved = 0
    for fi in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ret, frame = cap.read()
        if not ret:
            continue
        gi = min(fi, n_gt - 1)
        label = coco_npy_to_yolo_label(gt_poses[gi], iw, ih)
        if label:
            stem = f"vid{idx:05d}_f{saved:06d}"
            cv2.imwrite(str(out_img / f"{stem}.jpg"), frame)
            (out_lbl / f"{stem}.txt").write_text(label + "\n")
            saved += 1
    cap.release()
    return saved


def create_data_yaml(root):
    kpt_names_yaml = "  0:\n" + "\n".join(f"    - {n}" for n in COCO_KP_NAMES)
    (root / "data.yaml").write_text(f"""\
path: {root.resolve()}
train: images/train
val: images/val
kpt_shape: [17, 3]
flip_idx: {FLIP_IDX}
nc: 1
names:
  0: person
kpt_names:
{kpt_names_yaml}
""")


def add_coco_pose_mix(out_img_train: Path, out_lbl_train: Path, n_skating: int):
    """Add COCO train2017 person-with-keypoints images to training set.

    Gandhi et al. 2025: mixing COCO with domain data during fine-tuning (freeze=10)
    yields +10% mAP on target with <0.1% COCO forgetting. Adds body/pose/background
    diversity, prevents overfitting on 2 AthletePose3D subjects.

    Downloads COCO annotations, filters for person-with-keypoints, symlinks images+labels.
    Uses symlinks instead of copying (59K images × ~200KB = ~12GB copy avoided).
    Falls back to shutil.copy2 if symlinks fail (e.g., Vast.ai network filesystem).
    """
    import os
    import shutil
    coco_root = Path("data/datasets/coco")
    ann_file = coco_root / "annotations" / "person_keypoints_train2017.json"
    img_dir = coco_root / "train2017"

    if not ann_file.exists():
        print("  COCO annotations not found. Downloading...")
        coco_root.mkdir(parents=True, exist_ok=True)
        import urllib.request
        urllib.request.urlretrieve(
            "https://images.cocodataset.org/annotations/annotations_trainval2017.zip",
            str(coco_root / "annotations_trainval2017.zip"),
        )
        import zipfile
        with zipfile.ZipFile(str(coco_root / "annotations_trainval2017.zip")) as z:
            z.extractall(str(coco_root))
        print("  COCO annotations downloaded.")

    if not img_dir.exists():
        print("  WARNING: COCO train2017 images not found.")
        print(f"  Download from https://cocodataset.org and extract to {img_dir}")
        print("  Skipping COCO-Pose mixing. Run again after downloading.")
        return 0

    n_coco_target = int(n_skating * COCO_MIX_RATIO / (1 - COCO_MIX_RATIO))
    print(f"  COCO mix target: {n_coco_target} images ({COCO_MIX_RATIO*100:.0f}% of train)")

    with open(ann_file) as f:
        coco = json.load(f)

    # Build index: image_id → image info
    img_info = {img["id"]: img for img in coco["images"]}

    # Build index: image_id → best annotation (most keypoints)
    best_anns = {}
    for ann in coco["annotations"]:
        if ann.get("num_keypoints", 0) >= 10 and ann["category_id"] == 1:
            iid = ann["image_id"]
            if iid not in best_anns or ann["num_keypoints"] > best_anns[iid]["num_keypoints"]:
                best_anns[iid] = ann

    # Sample uniformly
    import random
    random.seed(42)
    candidate_ids = [iid for iid in best_anns if iid in img_info]
    sampled_ids = random.sample(candidate_ids, min(n_coco_target, len(candidate_ids)))

    saved = 0
    for img_id in sampled_ids:
        info = img_info[img_id]
        best_ann = best_anns[img_id]
        kps = np.array(best_ann["keypoints"]).reshape(-1, 3)  # (17, 3): x, y, vis
        if kps.shape[0] != 17:
            continue

        # Bbox from annotation
        x, y, w, h = best_ann["bbox"]
        xc, yc = x + w / 2, y + h / 2
        iw, ih = info["width"], info["height"]

        # Build YOLO label
        kp_norm = kps.copy()
        kp_norm[:, 0] /= iw
        kp_norm[:, 1] /= ih
        kp_norm[:, 2] = np.where(kps[:, 2] > 0, 2.0, 0.0)

        parts = [0, xc / iw, yc / ih, w / iw, h / ih]
        for i in range(17):
            parts.extend([float(kp_norm[i, 0]), float(kp_norm[i, 1]), float(kp_norm[i, 2])])

        stem = f"coco_{img_id:012d}"
        src_img = img_dir / info["file_name"]
        dst_img = out_img_train / f"{stem}.jpg"
        if src_img.exists() and not dst_img.exists():
            try:
                os.symlink(str(src_img.resolve()), str(dst_img))
            except OSError:
                # Fallback for filesystems that don't support symlinks (e.g., Vast.ai)
                shutil.copy2(str(src_img), str(dst_img))
            (out_lbl_train / f"{stem}.txt").write_text(" ".join(f"{v:.6f}" for v in parts) + "\n")
            saved += 1

    print(f"  COCO-Pose mix: {saved} images added")
    return saved


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", type=Path, default=Path("data/datasets/athletepose3d"))
    p.add_argument("--output", type=Path, default=Path("data/yolo-skating"))
    p.add_argument("--frames-per-video", type=int, default=25)
    p.add_argument("--max-sequences", type=int, default=0)
    a = p.parse_args()
    src, out = a.source, a.output
    for d in ["images/train", "images/val", "labels/train", "labels/val"]:
        (out / d).mkdir(parents=True, exist_ok=True)
    
    # Find all skating video+GT pairs
    all_pairs = [
        (f.with_name(f.name.replace("_coco.npy", ".mp4")), f)
        for f in src.rglob("*_coco.npy")
        if f.with_name(f.name.replace("_coco.npy", ".mp4")).exists() and is_figure_skating(f)
    ]
    print(f"Found {len(all_pairs)} skating video-GT pairs")
    
    # Filter by max sequences if specified
    if a.max_sequences > 0:
        # Get unique sequences
        seqs = list(set(get_subject(v) + "/" + v.stem.split("_cam_")[0] for v, _ in all_pairs))
        import random
        random.seed(42)
        random.shuffle(seqs)
        keep = set(seqs[: a.max_sequences])
        all_pairs = [(v, g) for v, g in all_pairs
                      if get_subject(v) + "/" + v.stem.split("_cam_")[0] in keep]
    
    # Subject-level split
    train_pairs = [(v, g) for v, g in all_pairs if get_subject(v) in TRAIN_SUBJECTS]
    val_pairs = [(v, g) for v, g in all_pairs if get_subject(v) in VAL_SUBJECTS]
    print(f"Train: {len(train_pairs)} clips ({len(set(get_subject(v) for v, _ in train_pairs))} subjects)")
    print(f"Val:   {len(val_pairs)} clips ({len(set(get_subject(v) for v, _ in val_pairs))} subjects)")
    
    total = 0
    train_total = 0
    for split_name, pairs in [("train", train_pairs), ("val", val_pairs)]:
        split_total = 0
        for i, (v, g) in enumerate(pairs):
            # k-means frame selection
            frame_indices = select_frames_kmeans(v, n_frames=a.frames_per_video)
            if not frame_indices:
                continue
            n = extract_video(
                v, g, out / "images" / split_name, out / "labels" / split_name,
                frame_indices, i,
            )
            split_total += n
            total += n
            if split_name == "train":
                train_total += n
            if (i + 1) % 500 == 0:
                print(f"  [{split_name}] {i + 1}/{len(pairs)}: {split_total} frames")
        print(f"  {split_name}: {split_total} frames from {len(pairs)} clips")

    # COCO-Pose mixing: add 15% COCO images to train set
    print(f"\nAdding COCO-Pose mix ({COCO_MIX_RATIO*100:.0f}%)...")
    n_coco = add_coco_pose_mix(out / "images/train", out / "labels/train", train_total)
    total += n_coco

    print(f"\nDone: {total} frames total ({train_total} skating + {n_coco} COCO)")
    create_data_yaml(out)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Smoke-test**

Run: `uv run python ml/scripts/prepare_yolo_dataset.py --max-sequences 10 --output data/yolo-skating-test`

- [ ] **Step 3: Validate**

Run: `uv run python -c "
from pathlib import Path; import numpy as np
for f in sorted(Path('data/yolo-skating-test/labels/train').glob('*.txt'))[:5]:
    v = list(map(float, f.read_text().strip().split()))
    assert len(v)==56; kp = np.array(v[5:]).reshape(17,3)
    assert kp[:,2].sum() >= 3  # at least 3 visible keypoints
    print(f'{f.name}: {int(kp[:,2].sum())} visible, bbox={v[1]:.3f},{v[2]:.3f},{v[3]:.3f},{v[4]:.3f}')
"`

- [ ] **Step 4: Full dataset**

Run: `uv run python ml/scripts/prepare_yolo_dataset.py --output data/yolo-skating`

Expected: ~226K skating train (S1) + ~59K COCO train (~285K train total before pseudo-labels) + ~227K val frames (S2) (~512K total). ~15GB disk (symlinks for COCO).

- [ ] **Step 6: Generate SkatingVerse pseudo-labels (DEFERRED — Task 2.5)**

**Skip this step in first round.** SkatingVerse (46GB) not downloaded yet. Run only after Task 2 evaluation shows mAP50 < 80%. Download from ModelScope.cn at that point.

Use pretrained YOLO26-Pose to generate pseudo-labels on SkatingVerse videos. This adds **hundreds of diverse skaters + real rink environments** — critical for generalization (federated learning shows single-subject data creates body-proportion bias).

```bash
# On VPS or Vast.ai (GPU needed)
uv run python ml/scripts/pseudo_label_skatingverse.py \
    --video-dir data/datasets/skatingverse \
    --output data/yolo-skating-pseudo \
    --max-videos 10000
```

Expected: ~30K-50K pseudo-labeled frames (2 FPS × 3s avg × 10K videos, filtered by confidence ≥0.5, ≥8 visible keypoints).

- [ ] **Step 7: Create merged data.yaml**

Point to all three data sources in training config:

```python
import yaml
from pathlib import Path

data = {
    "path": str(Path("data/yolo-skating").resolve()),
    "train": [
        str(Path("data/yolo-skating/images/train").resolve()),       # S1 GT (226K)
        str(Path("data/yolo-skating-pseudo/images/train").resolve()), # SkatingVerse pseudo
    ],
    "val": str(Path("data/yolo-skating/images/val").resolve()),      # S2 GT (227K)
    "kpt_shape": [17, 3],
    "flip_idx": [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15],
    "nc": 1,
    "names": {"0": "person"},
}
Path("data/yolo-skating/data.yaml").write_text(yaml.dump(data, default_flow_style=False))
```

Final dataset: **~226K GT + ~40K pseudo + ~59K COCO = ~325K train frames, ~227K val frames**.

- [ ] **Step 8: Commit**

```bash
git add ml/scripts/prepare_yolo_dataset.py ml/scripts/pseudo_label_skatingverse.py
git commit -m "feat(data): YOLO dataset scripts (S1 GT + SkatingVerse pseudo + COCO 15%, S2 val)"
```

**First round: Step 6-7 SKIPPED.** Train on S1 GT + COCO only (285K). Steps 6-7 deferred to Task 2.5 if needed.

---

### Task 2: Fine-Tune YOLO26-Pose (First Round)

**Files:**
- Create: `ml/scripts/train_yolo26_pose.py`

- [ ] **Step 1: Write training script**

```python
"""Fine-tune YOLO26-Pose on figure skating data + COCO-Pose mix.

Augmentation notes (from exp_augmentation.py GCN experiments):
- Image-level mosaic/mixup/fliplr: SAFE for pose estimation (symmetric task,
  flip_idx handles keypoint remapping). Fundamentally different from GCN mirror
  which destroys element identity (flip→lutz).
- copy_paste=0: Removed. Analogue of SkeletonMix which showed no gain in GCN
  experiments. Keypoint-level paste can create physically implausible poses.
- No rotation (degrees=0): Skating is vertically oriented.
- No erasing: Classification-only augmentation, not valid for pose estimation.
- translate=0.275: Matches official YOLO26s recipe.

Usage:
  # HP search (no validation, save every 10 epochs):
  uv run python ml/scripts/train_yolo26_pose.py --model yolo26n-pose --data data/yolo-skating-test/data.yaml --epochs 50 --batch 16 --imgsz 640 --no-val --save-period 10 --name hp_lr001

  # Full train (with validation + early stopping):
  uv run python ml/scripts/train_yolo26_pose.py --model yolo26s-pose --imgsz 1280 --batch -1
  # Vast.ai 2x RTX 4090, batch=-1 for auto-batch
"""
import argparse
from pathlib import Path
from ultralytics import YOLO

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="yolo26s-pose")
    p.add_argument("--data", default="data/yolo-skating/data.yaml")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch", type=int, default=-1)  # auto-batch for RTX 3090
    p.add_argument("--imgsz", type=int, default=1280)
    p.add_argument("--device", default=0)
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--freeze", type=int, default=10)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--val", default=True, action=argparse.BooleanOptionalAction)
    p.add_argument("--save-period", type=int, default=-1)
    a = p.parse_args()
    model = YOLO(f"{a.model}.pt")
    model.train(data=a.data, epochs=a.epochs, imgsz=a.imgsz, batch=a.batch,
        device=a.device, patience=a.patience, freeze=a.freeze, workers=a.workers,
        val=a.val, save_period=a.save_period,
        lr0=0.0005, lrf=0.88, hsv_h=0.015, hsv_s=0.35, hsv_v=0.2,
        degrees=0.0, translate=0.275, scale=0.9, shear=0.0, perspective=0.0,
        fliplr=0.5, flipud=0.0, mosaic=0.9, mixup=0.05,
        close_mosaic=10,
        project="runs/pose", name="skating-ft", exist_ok=True)
    metrics = model.val()
    print(f"mAP50-95: {metrics.pose.map:.4f}  mAP50: {metrics.pose.map50:.4f}")
    best = Path("runs/pose/skating-ft/weights/best.pt")
    if best.exists():
        exp = YOLO(str(best))
        print("Exporting ONNX..."); exp.export(format="onnx", imgsz=a.imgsz, half=True, simplify=True, dynamic=True)
        onnx_path = best.with_suffix(".onnx")
        if onnx_path.exists():
            import time, cv2
            import onnxruntime as ort
            sess = ort.InferenceSession(str(onnx_path), providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
            inp = sess.get_inputs()[0]
            dummy = np.zeros((1, 3, a.imgsz, a.imgsz), dtype=np.float16)
            # Warmup
            for _ in range(10): sess.run(None, {inp.name: dummy})
            # Benchmark
            t0 = time.perf_counter()
            for _ in range(100): sess.run(None, {inp.name: dummy})
            fps = 100 / (time.perf_counter() - t0)
            print(f"ONNX FPS: {fps:.0f} (imgsz={a.imgsz}, FP16)")
        try:
            print("Exporting TensorRT..."); exp.export(format="engine", imgsz=a.imgsz, half=True)
        except Exception as e:
            print(f"TensorRT export skipped (non-critical): {e}")

if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Smoke-test**

Run: `uv run python ml/scripts/train_yolo26_pose.py --model yolo26n-pose --data data/yolo-skating-test/data.yaml --epochs 3 --batch 4 --imgsz 640`

- [ ] **Step 3: Train on Vast.ai**

```bash
uv run python ml/scripts/train_yolo26_pose.py --model yolo26s-pose --imgsz 1280 --batch -1 --epochs 100
```

- [ ] **Step 4: Copy models back**

```bash
scp vastai:runs/pose/skating-ft/weights/best.pt data/models/yolo26s-pose-skating.pt
scp vastai:runs/pose/skating-ft/weights/best.onnx data/models/yolo26s-pose-skating.onnx
```

- [ ] **Step 5: Commit**

```bash
git add ml/scripts/train_yolo26_pose.py
git commit -m "feat(ml): YOLO26-Pose fine-tuning script (imgsz=1280, ONNX+TensorRT export)"
```

---

### Task 2b: Training Monitoring & Per-Joint Evaluation

**Why:** Ultralytics doesn't support per-joint AP out of the box (issue #7417 closed "not planned"). For figure skating, ankle errors are biomechanically critical (edge angle proxy). Need custom evaluation.

**Files:**
- Create: `ml/scripts/evaluate_per_joint.py`

- [ ] **Step 1: Write per-joint PCK evaluation script**

```python
"""Per-joint PCK evaluation for figure skating pose models.
Computes PCK@h for each keypoint on held-out validation set.

PCK = Percentage of Correct Keypoints: fraction of predictions within
threshold h * bbox_diagonal of ground truth.

Reports TWO thresholds per joint:
- PCK@0.5: standard (comparable with literature)
- PCK@strict: 0.05 for biomechanically critical joints (ankles, knees)

**Why PCK@0.05 for ankles/knees:** To measure joint angles within 3° tolerance (required for knee flexion and ankle edge angle calculation), spatial detection error must fall within 0.05 × bbox_diagonal (clinical biomechanics research). A 15px ankle deviation alters Lutz vs Flip edge classification — PCK@0.5 is too lenient for biomechanical validity.
"""
import argparse
import yaml
from pathlib import Path
import numpy as np
from ultralytics import YOLO

COCO_KP_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]

# Stricter thresholds for biomechanically critical joints
STRICT_THRESHOLD = 0.05  # for knees (13,14) and ankles (15,16)
STANDARD_THRESHOLD = 0.5  # standard PCK (all joints)
CRITICAL_JOINTS = {13, 14, 15, 16}  # knees, ankles


def evaluate_per_joint(model_path, data_yaml):
    # Load data config as dict (YOLODataset requires dict, not path string)
    with open(data_yaml) as f:
        data_dict = yaml.safe_load(f)

    model = YOLO(str(model_path))
    results = model.val(data=data_yaml, verbose=False)
    print(f"Pose mAP50-95: {results.pose.map:.4f}  mAP50: {results.pose.map50:.4f}")

    # Get validation dataset
    from ultralytics.data import YOLODataset
    ds = YOLODataset(data_dict, task="pose", imgsz=1280)

    # Compute per-joint PCK at TWO thresholds
    pck_standard = {i: [] for i in range(17)}  # PCK@0.5 for all joints
    pck_strict = {i: [] for i in CRITICAL_JOINTS}  # PCK@0.05 for critical joints
    total_joints = {i: 0 for i in range(17)}

    for idx in range(len(ds)):
        item = ds[idx]
        img = item["img"]
        gt_kpts = item["keypoints"].numpy()  # (M, 17, 3): x, y, vis

        # Run inference
        pred = model.predict(source=img, verbose=False, imgsz=1280)
        if not pred or pred[0].keypoints is None:
            continue

        pred_kpts = pred[0].keypoints.data.cpu().numpy()  # (N, 17, 3)

        # Match predictions to GT (simplified: assume single person per image)
        if len(pred_kpts) == 0 or len(gt_kpts) == 0:
            continue

        pred_kp = pred_kpts[0]  # (17, 3)
        gt_kp = gt_kpts[0]      # (17, 3)

        # Bbox diagonal for PCK normalization
        gt_xy = gt_kp[:, :2]
        gt_visible = gt_kp[:, 2] > 0
        if gt_visible.sum() < 3:
            continue
        bbox_diag = np.linalg.norm(gt_xy[gt_visible].max(axis=0) - gt_xy[gt_visible].min(axis=0))

        for j in range(17):
            if not gt_visible[j]:
                continue
            dist = np.linalg.norm(pred_kp[j, :2] - gt_kp[j, :2])
            # Standard PCK@0.5 (all joints)
            pck_standard[j].append(float(dist < STANDARD_THRESHOLD * bbox_diag))
            # Strict PCK@0.05 (critical joints only)
            if j in CRITICAL_JOINTS:
                pck_strict[j].append(float(dist < STRICT_THRESHOLD * bbox_diag))
            total_joints[j] += 1

    # Print results
    header = f"{'Joint':<15} {'PCK@0.5':>10}"
    if any(total_joints[j] > 0 for j in CRITICAL_JOINTS):
        header += f" {'PCK@0.05':>10}"
    header += f" {'N':>8}"
    print(header)
    print("-" * len(header))

    for j in range(17):
        name = COCO_KP_NAMES[j]
        if total_joints[j] == 0:
            print(f"{name:<15} {'N/A':>10} {'—':>10} {0:>8}")
            continue
        line = f"{name:<15} {np.mean(pck_standard[j]):>10.3f}"
        if j in CRITICAL_JOINTS and pck_strict[j]:
            line += f" {np.mean(pck_strict[j]):>10.3f}"
        elif j in CRITICAL_JOINTS:
            line += f" {'N/A':>10}"
        line += f" {total_joints[j]:>8}"
        print(line)

    # Summary
    all_standard = [v for vals in pck_standard.values() for v in vals]
    print(f"\nOverall PCK@0.5: {np.mean(all_standard):.3f}")
    if pck_strict:
        all_strict = [v for vals in pck_strict.values() for v in vals]
        print(f"Critical PCK@0.05 (knees+ankles): {np.mean(all_strict):.3f}")
    return {COCO_KP_NAMES[j]: float(np.mean(v)) if v else 0.0
            for j, v in pck_standard.items()}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="runs/pose/skating-ft/weights/best.pt")
    p.add_argument("--data", default="data/yolo-skating/data.yaml")
    a = p.parse_args()
    evaluate_per_joint(a.model, a.data)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run after training**

Run on Vast.ai after Task 2:
```bash
uv run python ml/scripts/evaluate_per_joint.py --model runs/pose/skating-ft/weights/best.pt
```

Expected output:
```
Joint           PCK@0.5     PCK@0.05        N
--------------------------------------------------
left_ankle      0.XXX       0.XXX        XXXX
right_ankle     0.XXX       0.XXX        XXXX
left_knee       0.XXX       0.XXX        XXXX
right_knee      0.XXX       0.XXX        XXXX
left_hip        0.XXX       —            XXXX
...
Overall PCK@0.5: 0.XXX
Critical PCK@0.05 (knees+ankles): 0.XXX
```

**Decision gates:**
- Ankle PCK@0.05 < 0.7 → re-pseudo-label with fine-tuned model (round 2)
- Overall PCK@0.5 < 0.8 → investigate data quality or increase epochs
- Ankle PCK significantly lower than shoulder/hip → check annotation quality in AthletePose3D

- [ ] **Step 3: Commit**

```bash
git add ml/scripts/evaluate_per_joint.py
git commit -m "feat(ml): per-joint PCK evaluation for figure skating pose models"
```

#### Training Monitoring Guide

**What to watch during training** (view with `tensorboard --logdir runs/pose`):

| Metric | Healthy | Concerning | Action |
|--------|---------|-----------|--------|
| `pose_loss` | Smooth decay | Oscillates or flat | Reduce `lr0` |
| `val/pose_loss` | Decays with train | Increases | Overfitting — more augmentation or fewer epochs |
| `mAP50(P)` | Rises fast, plateaus | Flat from epoch 1 | Check data quality |
| `mAP50-95(P)` | Slower rise, keeps improving | Flat | Train longer |
| `val/cls_loss` | Stable | Increasing | Overfitting signal (issue #18570) |
| Val-Train mAP gap | 0 to -2% | > -5% | Overfitting |

**Overfitting thresholds for this dataset** (1 GT subject + pseudo-labels):
- Train mAP >> Val mAP by > 5% → add more augmentation or reduce epochs
- `patience=20` should fire before overfitting becomes severe

**Iterative refinement tracking table:**

| Round | Source | mAP50(P) | mAP50-95(P) | Ankle PCK@0.05 | Delta |
|-------|--------|----------|-------------|---------------|-------|
| 0 (pretrained) | COCO pretrained | — | — | — |
| 1 (fine-tune) | S1 GT + SkatingVerse pseudo + COCO | | | |
| 2 (refined) | Re-pseudo-label with round 1 model | | | |

**Stop refining when:** mAP50(P) improvement < 0.5% for 2 consecutive rounds.

---

### Task 2.5: Optional — Iterative Pseudo-Label Refinement

**When:** After Task 2 training. Run ONLY if mAP50 on S2 val < 80%.

**Why:** Fine-tuned model produces better pseudo-labels than pretrained. Re-pseudo-labeling SkatingVerse with the fine-tuned model and retraining can improve quality (self-training theory: Shin 2012, Litrico 2023).

**Files:**
- Create: `ml/scripts/self_train_pseudo_labels.py`

- [ ] **Step 1: Write iterative pseudo-label refinement script**

```python
"""Iterative pseudo-label refinement: re-label SkatingVerse with fine-tuned model.

Workflow:
1. Load fine-tuned YOLO26-Pose (from Task 2)
2. Run inference on SkatingVerse videos
3. Filter by confidence >= 0.5, at least 8 visible keypoints
4. Convert to YOLO format labels (overwrite previous pseudo-labels)
5. Retrain (Task 2 script with updated data.yaml)

Based on: Litrico 2023 (pseudo-label refinement), Shin 2012 (self-training theory).
"""
import argparse
from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO

CONF_THRESHOLD = 0.5  # minimum keypoint confidence
MIN_VISIBLE_KP = 8    # minimum visible keypoints per person (10 too restrictive during flight)


def generate_pseudo_labels(model_path, video_dir, output_dir, max_videos=0):
    """Run inference on unlabeled videos, save confident predictions as labels."""
    model = YOLO(model_path)
    out_img = output_dir / "images" / "train"
    out_lbl = output_dir / "labels" / "train"
    out_img.mkdir(parents=True, exist_ok=True)
    out_lbl.mkdir(parents=True, exist_ok=True)

    videos = sorted(Path(video_dir).rglob("*.mp4"))
    if max_videos > 0:
        videos = videos[:max_videos]

    total_frames = 0
    for vi, vid in enumerate(videos):
        cap = cv2.VideoCapture(str(vid))
        if not cap.isOpened():
            continue
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        frame_interval = max(1, int(fps / 2))  # 2 FPS sampling
        fi = 0
        vid_frames = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if fi % frame_interval != 0:
                fi += 1
                continue
            results = model.predict(source=frame, conf=0.3, verbose=False, classes=[0])
            if results and results[0].keypoints is not None:
                # Correct ultralytics API: keypoints.data shape (N_people, 17, 3)
                for det_idx in range(len(results[0].keypoints.data)):
                    kps = results[0].keypoints.data[det_idx].cpu().numpy()  # (17, 3)
                    confs = kps[:, 2]
                    n_visible = (confs >= CONF_THRESHOLD).sum()
                    if n_visible < MIN_VISIBLE_KP:
                        continue
                    # Build YOLO label
                    xy = kps[:, :2]
                    xmin, ymin = xy.min(axis=0)
                    xmax, ymax = xy.max(axis=0)
                    h, w = frame.shape[:2]
                    xc, yc = (xmin + xmax) / 2 / w, (ymin + ymax) / 2 / h
                    bw, bh = (xmax - xmin) / w, (ymax - ymin) / h
                    vis = np.where(confs >= CONF_THRESHOLD, 2.0, 0.0)
                    kp_norm = xy / [w, h]
                    parts = [0, xc, yc, bw, bh]
                    for i in range(17):
                        parts.extend([float(kp_norm[i, 0]), float(kp_norm[i, 1]), float(vis[i])])
                    stem = f"sv_{vi:05d}_f{vid_frames:06d}"
                    cv2.imwrite(str(out_img / f"{stem}.jpg"), frame)
                    (out_lbl / f"{stem}.txt").write_text(
                        " ".join(f"{v:.6f}" for v in parts) + "\n"
                    )
                    vid_frames += 1
            fi += 1
        cap.release()
        total_frames += vid_frames
        if (vi + 1) % 100 == 0:
            print(f"  [{vi + 1}/{len(videos)}] {total_frames} pseudo-labeled frames")

    print(f"Done: {total_frames} pseudo-labeled frames from {len(videos)} videos")
    return total_frames


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="data/models/yolo26s-pose-skating.pt")
    p.add_argument("--video-dir", default="data/datasets/skatingverse")
    p.add_argument("--output", default="data/yolo-skating-pseudo")
    p.add_argument("--max-videos", type=int, default=10000)  # start with 10K, scale to 28K
    p.add_argument("--conf-threshold", type=float, default=CONF_THRESHOLD)
    a = p.parse_args()
    generate_pseudo_labels(a.model, a.video_dir, a.output, a.max_videos)
    print(f"\nPseudo-labels saved to {a.output}")
    print("Merge with existing data and retrain using train_yolo26_pose.py")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Generate pseudo-labels (subset)**

Run on Vast.ai (GPU needed):
```bash
uv run python ml/scripts/self_train_pseudo_labels.py --max-videos 5000
```

Expected: ~30K-50K pseudo-labeled frames (2 FPS × 3s avg video × 10K videos, filtered by confidence).

- [ ] **Step 3: Merge and retrain**

Create merged data.yaml pointing to both original + pseudo-labeled data:
```python
import os
merged = Path("data/yolo-skating-merged")
merged.mkdir(parents=True, exist_ok=True)

# Write merged data.yaml — point directly to original data paths
# (avoids fragile nested symlinks that ultralytics may not traverse)
merged_data_yaml = f"""
path: {merged.resolve()}
train:
  - {Path('data/yolo-skating').resolve()}/images/train
  - {Path('data/yolo-skating-pseudo').resolve()}/images/train
val:
  - {Path('data/yolo-skating').resolve()}/images/val
kpt_shape: [17, 3]
flip_idx: [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]
nc: 1
names:
  0: person
"""
(merged / "data.yaml").write_text(merged_data_yaml)
```

Retrain from fine-tuned checkpoint (50 epochs, lower LR):
```bash
uv run python ml/scripts/train_yolo26_pose.py \
    --model data/models/yolo26s-pose-skating \
    --data data/yolo-skating-merged/data.yaml \
    --epochs 50
```

- [ ] **Step 4: Evaluate improvement**

Compare mAP50 on S2 val (held-out skating subject) before vs after self-training. If improvement > 2pp mAP50, scale to full 28K videos.

- [ ] **Step 5: Commit**

```bash
git add ml/scripts/self_train_pseudo_labels.py
git commit -m "feat(ml): self-training loop for SkatingVerse pseudo-labels"
```

---

### Task 3: COCO 17kp Standardization — Replace H36Key (2D Pipeline Only)

**This is the index remap task.** Replace H36Key with COCOKey across 2D pipeline files (~20 files). The mapping is mechanical — same physical joints, different indices. No backward compat aliases — clean migration.

**Scope:** Only 2D pipeline files. pose_3d/ is excluded (disabled, will be rebuilt natively on COCO 17kp). h36m.py is kept for pose_3d/ until rebuild.

**Files:**
- Create: `ml/skating_ml/pose_estimation/coco17.py`
- Modify: `ml/skating_ml/types.py` (COCOKey enum)
- Keep: `ml/skating_ml/pose_estimation/h36m.py` (still used by pose_3d/)
- Modify: ~30 files in 2D pipeline (H36Key.X → COCOKey.X)

**Excluded (pose_3d/ — disabled, future rebuild):**
- `pose_3d/corrective_pipeline.py`
- `pose_3d/kinematic_constraints.py`
- `pose_3d/anchor_projection.py`
- `pose_3d/normalizer_3d.py`
- `pose_3d/athletepose_extractor.py`
- `visualization/export_3d.py` — 3D glTF export (uses H36M_SKELETON, disabled pipeline)
- `visualization/export_3d_animated.py` — 3D animated export (uses H36M_SKELETON, disabled pipeline)

**Additional file to migrate (not in list above):**
- `visualization/skeleton/joints.py` — 2D joint rendering, references H36Key

- [ ] **Step 1: Create coco17.py with COCOKey enum and constants**

```python
"""COCO 17 keypoint constants, skeleton edges, and helper functions.

Replaces h36m.py — COCO 17kp is the universal standard for all modern
pose models and datasets (YOLO26, RTMO, ViTPose, CIGPose, AthletePose3D, etc.).
"""

import numpy as np


class COCOKey:
    """COCO 17 keypoint indices."""

    NOSE = 0
    LEFT_EYE = 1
    RIGHT_EYE = 2
    LEFT_EAR = 3
    RIGHT_EAR = 4
    LEFT_SHOULDER = 5
    RIGHT_SHOULDER = 6
    LEFT_ELBOW = 7
    RIGHT_ELBOW = 8
    LEFT_WRIST = 9
    RIGHT_WRIST = 10
    LEFT_HIP = 11
    RIGHT_HIP = 12
    LEFT_KNEE = 13
    RIGHT_KNEE = 14
    LEFT_ANKLE = 15
    RIGHT_ANKLE = 16


# COCO skeleton edges
COCO_SKELETON_EDGES = [
    # Face
    (COCOKey.NOSE, COCOKey.LEFT_EYE),
    (COCOKey.NOSE, COCOKey.RIGHT_EYE),
    (COCOKey.LEFT_EYE, COCOKey.LEFT_EAR),
    (COCOKey.RIGHT_EYE, COCOKey.RIGHT_EAR),
    # Torso
    (COCOKey.LEFT_SHOULDER, COCOKey.RIGHT_SHOULDER),
    (COCOKey.LEFT_SHOULDER, COCOKey.LEFT_HIP),
    (COCOKey.RIGHT_SHOULDER, COCOKey.RIGHT_HIP),
    (COCOKey.LEFT_HIP, COCOKey.RIGHT_HIP),
    # Arms
    (COCOKey.LEFT_SHOULDER, COCOKey.LEFT_ELBOW),
    (COCOKey.LEFT_ELBOW, COCOKey.LEFT_WRIST),
    (COCOKey.RIGHT_SHOULDER, COCOKey.RIGHT_ELBOW),
    (COCOKey.RIGHT_ELBOW, COCOKey.RIGHT_WRIST),
    # Legs
    (COCOKey.LEFT_HIP, COCOKey.LEFT_KNEE),
    (COCOKey.LEFT_KNEE, COCOKey.LEFT_ANKLE),
    (COCOKey.RIGHT_HIP, COCOKey.RIGHT_KNEE),
    (COCOKey.RIGHT_KNEE, COCOKey.RIGHT_ANKLE),
]


COCO_KEYPOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]


# Virtual joint computation helpers (not stored in arrays, used on-the-fly)
def compute_pelvis(pose: np.ndarray) -> np.ndarray:
    """Midpoint of left and right hip."""
    return (pose[COCOKey.LEFT_HIP] + pose[COCOKey.RIGHT_HIP]) / 2

def compute_thorax(pose: np.ndarray) -> np.ndarray:
    """Midpoint of left and right shoulder."""
    return (pose[COCOKey.LEFT_SHOULDER] + pose[COCOKey.RIGHT_SHOULDER]) / 2

def compute_spine(pose: np.ndarray) -> np.ndarray:
    """Midpoint of pelvis and thorax."""
    return (compute_pelvis(pose) + compute_thorax(pose)) / 2

def compute_head(pose: np.ndarray) -> np.ndarray:
    """Midpoint of left and right eye (better for CoM than nose tip)."""
    return (pose[COCOKey.LEFT_EYE] + pose[COCOKey.RIGHT_EYE]) / 2

def compute_neck(pose: np.ndarray) -> np.ndarray:
    """Neck estimate: 30% from thorax toward nose (anatomical approximation).
    NOTE: COCO has no neck keypoint. H3.6M NECK was between THORAX and HEAD.
    This is an approximation — affects torso skeleton chain visualization only."""
    thorax = compute_thorax(pose)
    nose = pose[COCOKey.NOSE]
    return thorax + 0.3 * (nose - thorax)

def compute_mid_hip(pose: np.ndarray) -> np.ndarray:
    """Alias for compute_pelvis."""
    return compute_pelvis(pose)

def compute_mid_shoulder(pose: np.ndarray) -> np.ndarray:
    """Alias for compute_thorax."""
    return compute_thorax(pose)


# Virtual torso edges (computed joints, not stored in array)
# Used for skeleton rendering and biomechanics chain
# Indices 17-20 are virtual (beyond COCO 17kp array bounds)
VIRTUAL_TORSO_EDGES = [
    # These reference computed joints via their function names
    # Not stored in COCO_SKELETON_EDGES because indices > 16 don't exist in arrays
    # Callers must compute these joints explicitly and append to pose array for rendering
]


# No backward compatibility — all call sites updated in Step 2
```

- [ ] **Step 2: Remap all H36Key call sites**

The remapping is:
```
H36Key.HIP_CENTER  → compute_mid_hip(pose)   (was index 0, now computed)
H36Key.RHIP        → COCOKey.R_HIP         (was 1, now 12)
H36Key.RKNEE       → COCOKey.R_KNEE        (was 2, now 14)
H36Key.RFOOT       → COCOKey.R_ANKLE       (was 3, now 16)
H36Key.LHIP        → COCOKey.L_HIP         (was 4, now 11)
H36Key.LKNEE       → COCOKey.L_KNEE        (was 5, now 13)
H36Key.LFOOT       → COCOKey.L_ANKLE       (was 6, now 15)
H36Key.SPINE       → compute_spine(pose)    (was 7, now computed — lossy: midpoint vs calibrated)
H36Key.THORAX      → compute_thorax(pose)   (was 8, now computed)
H36Key.NECK        → compute_neck(pose)    (was 9, now computed — NOT NOSE)
H36Key.HEAD        → compute_head(pose)     (was 10, now computed — eye midpoint)
H36Key.LSHOULDER  → COCOKey.L_SHOULDER    (was 11, now 5)
H36Key.LELBOW     → COCOKey.L_ELBOW       (was 12, now 7)
H36Key.LWRIST     → COCOKey.L_WRIST        (was 13, now 9)
H36Key.RSHOULDER  → COCOKey.R_SHOULDER    (was 14, now 6)
H36Key.RELBOW     → COCOKey.R_ELBOW       (was 15, now 8)
H36Key.RWRIST     → COCOKey.R_WRIST        (was 16, now 10)
```

For each of the 37 files, apply these transformations:

**Direct index remaps** (most files — just change the constant):
- `H36Key.LHIP` → `COCOKey.L_HIP`
- `H36Key.RHIP` → `COCOKey.R_HIP`
- `H36Key.LKNEE` → `COCOKey.L_KNEE`
- etc. for all 12 direct-mapped joints

**Virtual joint remaps** (need function call instead of index):
- `pose[H36Key.HIP_CENTER]` → `compute_mid_hip(pose)` (was index 0, now computed)
- `pose[H36Key.SPINE]` → `compute_spine(pose)` (was index 7, now computed — lossy: midpoint vs calibrated)
- `pose[H36Key.THORAX]` → `compute_thorax(pose)` (was index 8, now computed)
- `pose[H36Key.NECK]` → `compute_neck(pose)` (was index 9, now computed — NOT NOSE, anatomically different)
- `pose[H36Key.HEAD]` → `compute_head(pose)` (was index 10, now computed — eye midpoint, better than nose for CoM)

**Files with virtual joints** (need logic change, not just remap):
- `ml/skating_ml/analysis/physics_engine.py` — uses HEAD, SPINE, THORAX
- `ml/skating_ml/visualization/layers/vertical_axis_layer.py` — uses HEAD
- `ml/skating_ml/detection/pose_tracker.py` — uses mid_hip, mid_shoulder (already computed inline)
- `ml/skating_ml/tracking/skeletal_identity.py` — uses bone ratios (direct indices work after remap)
- `ml/skating_ml/pose_estimation/normalizer.py` — root-centering (uses mid_hip)
- `ml/skating_ml/pose_3d/kinematic_constraints.py` — bone length constraints
- `ml/skating_ml/pose_3d/anchor_projection.py` — torso anchor

**Skeleton edges remap** (types.py + visualization):
- Replace `H36M_SKELETON_EDGES` with `COCO_SKELETON_EDGES` from coco17.py

- [ ] **Step 3: Update types.py**

Replace `H36Key` import and enum with `COCOKey` from `coco17.py`. Update `H36M_SKELETON_EDGES` reference.

- [ ] **Step 4: Keep h36m.py (pose_3d/ still needs it)**

No action — h36m.py stays until pose_3d/ is rebuilt natively on COCO 17kp.

- [ ] **Step 5: Run tests, fix failures**

Run: `uv run python -m pytest ml/tests/ -x --no-cov -q 2>&1 | head -80`

**Expected: ~14 test files fail** (394 H36Key references across tests/conftest.py). Each failure shows exactly which index remap was missed.

Key test files to fix (exclude pose_3d/ tests):
- `tests/conftest.py` — shared fixtures (pose data shape, mock extractors)
- `tests/test_types.py` — H36Key enum tests → COCOKey
- `tests/analysis/test_metrics.py` — uses H36Key indices for angle/metric calculations
- `tests/analysis/test_phase_detector.py` — uses H36Key for CoM computation
- `tests/detection/test_pose_tracker.py` — uses H36Key for biometric tracking
- `tests/tracking/test_skeletal_identity.py` — bone ratio calculations
- `tests/utils/test_geometry.py` — angle/distance functions
- `tests/visualization/test_vertical_axis_layer.py` — uses HEAD virtual joint
- `tests/visualization/test_joint_angle_layer.py` — joint angle display

The remap is mechanical — same mapping as production code (see Step 2 table). Virtual joints (pose[H36Key.X]) → compute functions (compute_X(pose)).

- [ ] **Step 6: All tests pass**

Run: `uv run python -m pytest ml/tests/ --no-cov -q`

- [ ] **Step 7: Commit**

```bash
git add ml/skating_ml/pose_estimation/coco17.py ml/skating_ml/pose_estimation/__init__.py \
       ml/skating_ml/types.py \
       ml/tests/conftest.py ml/tests/test_types.py \
       <all ~20 modified 2D pipeline files> \
       <all ~14 modified test files>
git commit -m "refactor(pose): standardize 2D pipeline on COCO 17kp (pose_3d/ excluded, tests updated)"
```

---

### Task 4: Rewrite PoseExtractor — ONNX Runtime Backend

**Files:**
- Modify: `ml/skating_ml/pose_estimation/pose_extractor.py`
- Modify: `ml/skating_ml/worker.py` (update constructor call)
- Modify: `ml/skating_ml/pipeline.py` (update constructor call)
- Modify: `ml/skating_ml/visualization/pipeline.py` (update constructor call)

- [ ] **Step 1: Read current pose_extractor.py**

Read: `ml/skating_ml/pose_estimation/pose_extractor.py` — understand `extract_video_tracked()`, `preview_persons()`, tracking integration.

Also read callers to understand required constructor params:
- `ml/skating_ml/worker.py` — uses `mode`, `tracking_backend`, `tracking_mode`, `output_format`
- `ml/skating_ml/pipeline.py` — uses `output_format`
- `ml/skating_ml/visualization/pipeline.py` — uses `output_format`, `det_frequency`, `frame_skip`, `tracking_mode`

- [ ] **Step 2: Replace rtmlib with ONNX inference**

Key changes:
1. Import `onnxruntime` instead of `rtmlib`
2. Load ONNX model in `__init__`
3. `_preprocess(frame)` — letterbox resize, BGR→RGB, normalize 0-1, HWC→NCHW
4. `_postprocess(outputs)` — extract keypoints, filter by confidence
5. `predict_frame(frame)` — preprocess → ONNX inference → postprocess → (P, 17, 2) keypoints + (P, 17) scores
6. `extract_video_tracked()` — replace `self.tracker(frame)` with `self.predict_frame(frame)`, feed into Sports2D tracker
7. **No coco_to_h36m()** — pipeline now uses COCO 17kp natively
8. Keep all tracking (Sports2D, SkeletalIdentity, TrackletMerger) unchanged
9. **`mode` parameter**: Deprecate (no-op with warning). ONNX has one model — no lightweight/balanced/performance variants. Keep in API signature for backward compat.
10. **`tracking_backend` parameter**: Deprecate (no-op with warning). ONNX replaces all rtmlib backends.
11. **Preserve these parameters** (used by worker.py, pipeline.py, viz/pipeline.py):
    - `output_format: str = "normalized"` — convert pixel coords to [0,1] in postprocess
    - `tracking_mode: str = "auto"` — pass through to Sports2D/DeepSORT selection
    - `det_frequency: int = 1` — run detection every Nth frame, interpolate rest
    - `frame_skip: int = 1` — skip frames for speed
12. **`preview_persons()`**: Reuse `_predict_frame()` on sampled frames → feed detections to Sports2D → return person summaries (bbox, track hits, mid-hip). Same logic as current, just replace rtmlib inference with ONNX `_predict_frame()`.

**ONNX preprocessing (from ultralytics source):**
```python
def _preprocess(self, frame: np.ndarray) -> np.ndarray:
    """BGR frame → NCHW float32 tensor for ONNX inference."""
    # Letterbox resize to imgsz (maintain aspect ratio, pad with 114)
    h, w = frame.shape[:2]
    ratio = min(self.imgsz / w, self.imgsz / h)
    new_w, new_h = int(w * ratio), int(h * ratio)
    pad_w, pad_h = self.imgsz - new_w, self.imgsz - new_h
    top, left = pad_h // 2, pad_w // 2
    img = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    img = cv2.copyMakeBorder(img, top, pad_h - top, left, pad_w - left,
                              cv2.BORDER_CONSTANT, value=(114, 114, 114))
    self._letterbox_info = (ratio, top, left)  # save for postprocessing
    # BGR → RGB, normalize 0-1, HWC → NCHW
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return np.transpose(img, (2, 0, 1))[np.newaxis]  # (1, 3, H, W)
```

**ONNX postprocessing:**
```python
def _postprocess(self, outputs: list[np.ndarray], conf_threshold: float = 0.3):
    """End2end ONNX output → filtered keypoints + scores + bboxes.

    YOLO26-Pose end2end ONNX output: (1, 300, 57) — same shape regardless of nc.
    Layout per detection: [x1, y1, x2, y2, max_score, class_idx, kpt0_x, kpt0_y, kpt0_v, ...]
    57 = 4 bbox + 1 max_class_score + 1 class_index + 51 keypoints (17 × 3).
    NO objectness channel (removed since YOLOv8). nc class scores collapsed to max+idx by topk.
    """
    pred = outputs[0][0]  # (300, 57)

    # Filter by max class confidence (column 4)
    scores = pred[:, 4]  # (300,) max class probability
    mask = scores > conf_threshold
    pred = pred[mask]

    if len(pred) == 0:
        return np.empty((0, 17, 2)), np.empty((0, 17)), np.empty((0, 4))

    bboxes = pred[:, :4]   # (P, 4) x1, y1, x2, y2 (normalized to imgsz)
    kpts = pred[:, 6:]    # (P, 51) — 17 keypoints × 3 (x, y, visibility)
    kpts = kpts.reshape(-1, 17, 3)

    kp_scores = kpts[:, :, 2]  # (P, 17) keypoint visibility/confidence
    xy = kpts[:, :, :2]        # (P, 17, 2) keypoint coords (normalized to imgsz)

    # Undo letterbox: coords are normalized to imgsz, convert to original pixel space
    ratio, top, left = self._letterbox_info
    xy[:, :, 0] = (xy[:, :, 0] * self.imgsz - left) / ratio
    xy[:, :, 1] = (xy[:, :, 1] * self.imgsz - top) / ratio
    return xy, kp_scores, bboxes
```

**ONNX model loading:**
```python
def __init__(self, model_path="data/models/yolo26s-pose-skating.onnx",
             imgsz=1280, conf_threshold=0.3, device="cuda"):
    import onnxruntime as ort
    from pathlib import Path

    self.imgsz = imgsz
    self.conf_threshold = conf_threshold

    # Try fine-tuned model first, fall back to pretrained
    model_path = Path(model_path)
    if model_path.exists():
        self.model_path = model_path
    else:
        self.model_path = Path("yolo26s-pose.onnx")  # ultralytics auto-downloads

    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if device == "cuda" else ["CPUExecutionProvider"]
    self.session = ort.InferenceSession(str(self.model_path), providers=providers)
    self.input_name = self.session.get_inputs()[0].name

    # Validate output shape — end2e ONNX must be (1, 300, 57)
    dummy = np.zeros((1, 3, imgsz, imgsz), dtype=np.float32)
    out = self.session.run(None, {self.input_name: dummy})
    out_shape = out[0].shape  # expected: (1, 300, 57)

    if out_shape[1] != 300 or out_shape[2] != 57:
        import warnings
        warnings.warn(
            f"Unexpected ONNX output shape: {out_shape}. "
            f"Expected (1, 300, 57) for end2e YOLO26-Pose. "
            f"If using non-end2e export, re-export with end2e=True."
        )
```

**Tracking integration:**
```python
# In extract_video_tracked(), per frame:
keypoints, scores = self.predict_frame(frame)  # (P, 17, 2), (P, 17)
# Feed into Sports2D tracker (same interface as before)
if keypoints.shape[0] > 0:
    track_ids = self.sports2d_tracker.update(keypoints, scores)
```

**Model download:** The fine-tuned ONNX model must exist at `data/models/yolo26s-pose-skating.onnx`.
If not found, fall back to pretrained: `yolo26s-pose.onnx` (auto-downloaded by ultralytics on export).

- [ ] **Step 3: Update __init__.py**

```python
from .pose_extractor import PoseExtractor
__all__ = ["PoseExtractor"]
```

- [ ] **Step 4: Run tests, fix failures**

- [ ] **Step 5: Commit**

```bash
git add ml/skating_ml/pose_estimation/pose_extractor.py ml/skating_ml/pose_estimation/__init__.py
git commit -m "refactor(pose): replace rtmlib/RTMO with ONNX YOLO26-Pose"
```

---

### Task 5: Update Dependencies + Cleanup

**Files:**
- Modify: `ml/pyproject.toml`

- [ ] **Step 1: Update pyproject.toml**

Remove rtmlib, add ultralytics (training dep) and scikit-learn (dataset prep):
```toml
# Remove: "rtmlib>=0.0.7",
# Add: "ultralytics>=8.3.0,<9.0",  # pin major version for API stability
#       "scikit-learn>=1.5.0",       # MiniBatchKMeans for k-means frame selection
```

- [ ] **Step 2: uv sync**

Run: `uv sync`

- [ ] **Step 3: Grep for rtmlib**

Run: `grep -rn "rtmlib\|H36Key\|h36m" ml/ --include="*.py"`

Expected: No rtmlib. H36Key should only appear in coco17.py as alias.

- [ ] **Step 4: Commit**

```bash
git add ml/pyproject.toml ml/uv.lock
git commit -m "chore(deps): replace rtmlib with ultralytics (training) + onnxruntime (runtime), add scikit-learn"
```

---

### Task 6: Integration Verification

- [ ] **Step 1: Full test suite**

Run: `uv run python -m pytest ml/tests/ --no-cov -v`

- [ ] **Step 2: Real video test**

```bash
uv run python ml/scripts/visualize_with_skeleton.py data/uploads/<test.mp4> --layer 1
```

- [ ] **Step 3: Speed benchmark**

```python
import time, cv2
from skating_ml.pose_estimation import PoseExtractor
ext = PoseExtractor()
cap = cv2.VideoCapture("data/uploads/test.mp4")
start = time.perf_counter()
frames = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break
    ext.predict_frame(frame); frames += 1
print(f"{frames} frames in {time.perf_counter()-start:.1f}s = {frames/(time.perf_counter()-start):.0f} FPS")
ext.close()
```

Expected: >= 100 FPS on Vast.ai RTX 3090.

- [ ] **Step 4: Commit**

```bash
git add -A
git commit -m "test(ml): verify YOLO26-Pose + COCO migration integration"
```

---

## Verification Checklist

- [ ] `uv run python -m pytest ml/tests/ --no-cov` — all pass
- [ ] `uv run ruff check ml/skating_ml/` — no errors
- [ ] `grep -rn "rtmlib" ml/` — no matches
- [ ] Skeleton overlay on real skating video looks correct
- [ ] Speed >= RTMO baseline (141 FPS)
- [ ] No `coco_to_h36m()` call in 2D pipeline (format bridge for future 3D lifter — OK)
- [ ] No `H36Key` references in 2D pipeline (pose_3d/ still uses H36Key — OK)
- [ ] Dataset has subject-level split (S1 train, S2 val — only S1,S2 have skating)
- [ ] NaN keypoints properly handled (vis=0, mean visible coords)
- [ ] k-means frame selection produces diverse poses (not redundant adjacent frames)
- [ ] COCO-Pose images present in train set (~15% of total)
- [ ] SkatingVerse pseudo-labels present in train set (confidence >= 0.5, >=8 visible keypoints)
- [ ] No copy_paste in training config (SkeletonMix analogue, didn't help in GCN experiments)

## Rollback Plan

1. If YOLO26 worse than RTMO: use pretrained `yolo26s-pose.onnx` (no fine-tune)
2. If COCO migration breaks things: git revert to pre-Task-3 commit
3. Both are independent — can roll back one without the other

---

## Future Work: 3D Pose Reconstruction Rebuild

**Problem:** Current MotionAGFormer-S has unusable dependencies (mmpose, mmcv, torch 2.x ecosystem conflict). Disabled in pipeline. 3D reconstruction needed for biomechanical analysis (joint angles, CoM trajectory, physics).

**Approach:** Replace with PoseMamba-L (AAAI 2025) — Mamba-based 3D lifter, no mmpose dependency.

| | MotionAGFormer-S (current) | PoseMamba-L (target) |
|---|---|---|
| Params | 4.8M | 6.7M |
| Frames | 81 (2.7s @ 30fps) | 243 (8.1s @ 30fps) |
| MPJPE (detected 2D) | ~42mm | 38.1mm |
| Complexity | O(N²) quadratic | O(N) linear |
| Dependencies | mmpose, mmcv, torch 2.x | Clean (Mamba SSM) |
| Code | github.com/WongKinYiu/MotionAGFormer | github.com/nankingjing/PoseMamba |

**Why PoseMamba-L:**
- 243-frame window = entire jump sequence in one pass
- Linear complexity = faster than MotionAGFormer-L at same accuracy
- No mmpose dependency hell
- Skeleton-aware bidirectional scanning (spatial + temporal)
- Clean PyTorch, easy to export ONNX

**Format bridge:**
- 2D pipeline outputs COCO 17kp (native from YOLO26-Pose)
- `coco_to_h36m()` index reorder before 3D lifting (~10 lines, pure permutation)
- All 3D lifters use H3.6M 17kp internally — no model change needed
- 3D output: H3.6M 17kp → convert back to COCO if needed

**Sports fine-tuning (Suzuki et al., 2024):**
- Unsupervised fine-tuning via multi-view pseudo-labels
- Same Nagoya University group as MotionAGFormer/AthletePose3D
- Can apply to AthletePose3D skating subset directly
- Code: github.com/SZucchini/unsupervised-fine-tuning-pose3d-for-sports

**Scope:** Separate plan. This migration covers 2D pipeline only (pose_3d/ excluded).
