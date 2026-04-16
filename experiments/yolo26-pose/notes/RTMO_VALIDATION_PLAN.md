# RTMO Validation Plan (2026-04-15)

**Goal:** Compare RTMO baseline with YOLO26 on AthletePose3D dataset

---

## Research Findings (tvly + Tavily)

### RTMO Model Sources

| Source | Model | Format | Status |
|--------|-------|--------|--------|
| **Xenova/RTMO-s** | RTMO-s | ONNX (quantized) | ✅ Ready for download |
| **Xenova/RTMO-m** | RTMO-m | ONNX (quantized) | ✅ Ready for download |
| pesi/rtmo | RTMO-s/m/l | ONNX | ⚠️ Requires rtmlib |
| open-mmlab/mmpose | RTMO | PyTorch .pth | ❌ Requires conversion |

### Key Discovery: Xenova Models

**Xenova/RTMO-s** and **Xenova/RTMO-m** are quantized ONNX models optimized for Transformers.js, but they work with ONNX Runtime!

**Advantages:**
- ✅ Direct ONNX download (no conversion needed)
- ✅ Works with onnxruntime-gpu
- ✅ COCO 17-keypoint format (same as YOLO26)
- ✅ No mmcv/mmpose/rtmlib dependencies

**Download URLs:**
```bash
# RTMO-s (small, faster)
https://huggingface.co/Xenova/RTMO-s/resolve/main/onnx/model.onnx

# RTMO-m (medium, more accurate)
https://huggingface.co/Xenova/RTMO-m/resolve/main/onnx/model.onnx
```

---

## Implementation Plan

### Step 1: Download Models Locally (5 min)

```bash
# Create model directory
mkdir -p data/models/rtmo

# Download RTMO-s (small)
wget -O data/models/rtmo/rtmo-s.onnx \
  https://huggingface.co/Xenova/RTMO-s/resolve/main/onnx/model.onnx

# Download RTMO-m (medium)
wget -O data/models/rtmo/rtmo-m.onnx \
  https://huggingface.co/Xenova/RTMO-m/resolve/main/onnx/model.onnx
```

### Step 2: Create Validation Script (10 min)

File: `ml/scripts/validate_rtmo_xenova.py`

```python
#!/usr/bin/env python3
"""Validate Xenova RTMO on AthletePose3D val set."""

import sys
from pathlib import Path
import cv2
import numpy as np
import onnxruntime as ort

# Dataset paths
VAL_IMAGES = Path("/root/data/datasets/yolo26_ap3d/images/val")
VAL_LABELS = Path("/root/data/datasets/yolo26_ap3d/labels/val")

# RTMO model
RTMO_MODEL = "/root/data/models/rtmo/rtmo-s.onnx"

print(f"🔍 RTMO Validation on AthletePose3D val set")
print(f"Model: {RTMO_MODEL}")
print()

# Initialize ONNX Runtime
print("⏳ Loading RTMO model...")
providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
session = ort.InferenceSession(RTMO_MODEL, providers=providers)
print("✅ RTMO loaded")
print()

# Get input/output names
input_name = session.get_inputs()[0].name
output_names = [o.name for o in session.get_outputs()]

print(f"Input: {input_name}")
print(f"Outputs: {output_names}")
print()

# Validation metrics
total_images = 0
correct_keypoints = 0
total_keypoints = 0
keypoint_distances = []

# Process images
image_files = sorted(list(VAL_IMAGES.glob("*.jpg")))[:100]  # First 100
print(f"Processing {len(image_files)} images...")

for img_idx, img_path in enumerate(image_files, 1):
    if img_idx % 20 == 0:
        print(f"  Progress: {img_idx}/{len(image_files)}")

    # Load image
    image = cv2.imread(str(img_path))
    if image is None:
        continue

    h, w = image.shape[:2]
    img_size = max(h, w)

    # Get corresponding label file
    label_path = VAL_LABELS / (img_path.stem + ".txt")
    if not label_path.exists():
        continue

    # Load GT keypoints
    with open(label_path) as f:
        label_data = f.read().strip().split()

    if len(label_data) < 17 * 3:
        continue

    # Parse GT keypoints (YOLO format)
    gt_kpts = []
    for i in range(17):
        x = float(label_data[i * 3 + 0]) * w
        y = float(label_data[i * 3 + 1]) * h
        vis = int(label_data[i * 3 + 2])
        if vis > 0:
            gt_kpts.append([x, y])

    if len(gt_kpts) < 5:
        continue

    # Prepare input for RTMO
    # TODO: Need to check actual input format from model
    # Usually: (1, 3, 640, 640) normalized

    # Run RTMO inference
    # outputs = session.run(output_names, {input_name: input_data})

    # TODO: Parse outputs and calculate metrics

    total_images += 1

print()
print("=" * 60)
print("📊 RTMO VALIDATION RESULTS")
print("=" * 60)
print(f"Total images: {total_images}")
# ... metrics output
```

### Step 3: Upload to Vast.ai (2 min)

```bash
# Upload models
scp data/models/rtmo/*.onnx vastai:/root/data/models/rtmo/

# Upload script
scp ml/scripts/validate_rtmo_xenova.py vastai:/root/
```

### Step 4: Run Validation (5 min)

```bash
ssh vastai
python /root/validate_rtmo_xenova.py
```

---

## Alternative: Use rtmlib with Pre-Downloaded Models

If Xenova models don't work, try rtmlib approach:

```bash
# Install rtmlib on local machine
pip install rtmlib

# Download models via rtmlib
python -c "from rtmlib import RTMO; RTMO('rtmo-s')"

# Find model cache location
# Usually: ~/.cache/rtmlib/ or ~/.cache/huggingface/

# Upload cache to Vast.ai
scp -r ~/.cache/rtmlib vastai:/root/.cache/
```

---

## Expected Results

| Model | Expected mAP50-95(P) | RTMO Paper (COCO) |
|-------|---------------------|-------------------|
| RTMO-s | ~0.60-0.65 | 0.677 |
| RTMO-m | ~0.65-0.70 | 0.715 |

**Our baselines on AthletePose3D:**
- YOLO26n pretrained: **0.604**
- YOLO26n fine-tuned: **0.406**

**If RTMO-s > 0.604:** RTMO is better baseline
**If RTMO-s ≈ 0.604:** Comparable to YOLO26n
**If RTMO-s < 0.604:** YOLO26n is better

---

## Timeline

- [ ] Step 1: Download models (5 min)
- [ ] Step 2: Create validation script (10 min)
- [ ] Step 3: Upload to Vast.ai (2 min)
- [ ] Step 4: Run validation (5 min)
- [ ] Step 5: Compare with YOLO26 baselines

**Total:** ~30 min

---

## References

- Xenova/RTMO-s: https://huggingface.co/Xenova/RTMO-s
- Xenova/RTMO-m: https://huggingface.co/Xenova/RTMO-m
- RTMO Paper: arxiv.org/html/2312.07526v2 (Table 1: RTMO-s = 0.677 mAP50-95)
- Research via tvly (Tavily CLI): `tvly search "RTMO ONNX download"`
