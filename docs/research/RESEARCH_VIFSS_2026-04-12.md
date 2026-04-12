# VIFSS: View-Invariant Figure Skating-Specific Pose Representation

**Paper:** arXiv:2508.10281 (submitted 2025-08-14)
**Authors:** Ryota Tanaka, Tomohiro Suzuki, Keisuke Fujii (Nagoya University / RIKEN AIP)
**Code:** https://github.com/ryota-skating/VIFSS (Apache-2.0, Python, PyTorch 2.4.1)
**Venue:** Extended from ACM MMSports 2024 workshop paper (Tanaka et al., 2024)

---

## 1. Problem Statement

Temporal Action Segmentation (TAS) for figure skating jumps from broadcast video. Two core challenges:

1. **Insufficient annotated data** -- figure skating TAS annotations are sparse (only ~9% of frames are labeled)
2. **View sensitivity** -- 2D poses are projections that vary with camera angle; direct 3D pose regression is bottlenecked by estimation quality (especially for unseen rotations like quads)

The paper proposes learning **latent pose embeddings** rather than using raw 2D/3D coordinates, via a two-stage framework: contrastive pre-training (view-invariant) + fine-tuning (domain-specific).

---

## 2. Architecture: JointFormer as Pose Encoder

### 2.1 JointFormer (Lutz et al., ICPR 2022)

The pose encoder is **JointFormer**, originally designed for single-frame 2D-to-3D lifting. VIFSS repurposes it as a pose embedding extractor by changing the output dimension from 3D coordinates to a latent embedding vector.

**Architecture details:**

```
Input: 2D pose [B, J, 2]  (J = 17 joints, H3.6M format)
  |
  v
Embedding layer (graph-based): SemGraphConv(2 -> d_model)
  |  Adjacency matrix built from skeleton connections
  |  H3.6M skeleton: 16 edges (see JOINT_CONNECTION in code)
  v
Positional encoding (learnable, spatial_encoding=True not used in practice)
  |
  v
LayerNorm + Dropout
  |
  v
Stacked Transformer Encoder Layers x 4
  |  Each: d_model=128, d_inner=512, n_head=8, d_k=64, d_v=64
  |  With residual connections + ReLU
  |  Intermediate supervision: predicts 3D pose after each layer
  |  Error prediction head at each layer (multi-task)
  v
Flatten: [B, J * d_model] = [B, 17 * 128] = [B, 2176]
  |
  v
3-layer MLP: 2176 -> 128 -> 128 -> d_out
  |  d_out = pose_dim + view_dim = 32 + 4 = 36
  v
Output: embedding [B, 36]
  |  Split into z_pose [B, 32] and z_view [B, 4]
```

**Key hyperparameters (from `config/pretrain.yaml`):**

| Parameter | Value |
|-----------|-------|
| `num_layers` | 4 |
| `hid_dim` (d_model) | 128 |
| `d_inner` | 512 |
| `n_head` | 8 |
| `pose_dim` | 32 |
| `view_dim` | 4 |
| `intermediate` | True |
| `embedding_type` | graph (SemGraphConv) |
| `error_prediction` | True |
| `pred_dropout` | 0.2 |
| `p_dropout` | 0.0 |

**Skeleton definition (17 joints, H3.6M):**

```python
JOINT_CONNECTION = [
    (0, 10), (0, 11), (0, 12),  # Pelvis
    (1, 2), (2, 3),             # Spine -> Thorax
    (3, 4), (3, 5), (3, 10),   # Thorax branches
    (4, 6), (5, 7),             # Shoulders
    (6, 8), (7, 9),             # Elbows
    (11, 13), (12, 14),         # Hips
    (13, 15), (14, 16),         # Knees
]
```

### 2.2 Intermediate Supervision + Error Prediction

After each of the 4 encoder layers, the model:
1. **Predicts 3D pose** via `intermediate_pred[i]`: LayerNorm -> Dropout -> Linear(J*d_model, J*3)
2. **Predicts error** via `intermediate_error[i]`: LayerNorm -> Dropout -> Linear(J*d_model, J*3)
3. **Feeds predicted pose back** via `intermediate_enc[i]`: Linear(J*3, J*d_model) -- this becomes a residual connection

This multi-task setup helps the transformer learn richer joint relationships during pre-training.

---

## 3. 3D Multi-View Contrastive Pre-training

### 3.1 Datasets

Pre-training uses **four** 3D pose datasets, unified to H3.6M 17-keypoint format:

| Dataset | Content | Size | Keypoints |
|---------|---------|------|-----------|
| **Human3.6M** (Ionescu 2013) | Indoor daily activities | ~3.6M frames | 17 (native) |
| **MPI-INF-3DHP** (Mehta 2017) | Indoor + outdoor activities | ~1.3M frames | 28 (mapped to 17) |
| **AIST++** (Li 2021) | Indoor dance motions | ~1.5M frames | 17 (SMPL) |
| **FS-Jump3D** (Tanaka 2024) | Figure skating jumps | 253 sequences | 83 (mapped to 17) |

**FS-Jump3D details:**
- Captured on ice with 12 hardware-synchronized Qualisys Miqus Video cameras
- Markerless motion capture (Theia3D)
- 4 experienced skaters (A-D), 10 trials each for 6 jump types
- 253 jump sequences total
- 83 joints per pose (head:16, torso:16, arms:30, legs:34), mapped to H3.6M 17kp
- Includes triple jumps
- Public download: https://github.com/ryota-skating/FS-Jump3D
- License: CC BY-NC-SA 4.0
- Size: ~9.6 GB (c3d: 303MB, json: 505MB, videos: 8.84GB)
- Integrated into AthletePose3D (CVSports 2025 @ CVPR)

### 3.2 Virtual Camera Projection

For each 3D pose, two 2D projections are generated from random viewpoints:

1. **Ground plane alignment**: RANSAC estimates ground plane from lowest z-coordinates (50% assumed contacts, 50% outliers). Pose rotated so ground plane aligns with xy-plane.

2. **Frontal alignment**: Pose rotated around z-axis so left hip -> positive x, right hip -> negative x.

3. **Normalization**: Center at mid-hip, rescale so `||hip-spine|| + ||spine-thorax|| = 0.4`.

4. **Virtual camera sampling**:
   - Azimuth: uniform from [-180, +180] degrees
   - Elevation: uniform from [-30, +30] degrees
   - Distance: uniform from [5, 10]
   - Camera always looks at origin (mid-hip)
   - Perspective projection to 2D

5. **Data augmentation**:
   - Horizontal flip: multiply x by -1 before projection (50% rate)
   - Jittering: Gaussian noise with variance 0.01 (20% rate)
   - Masking: randomly set 1% of joints to (0,0) per frame (20% rate)

### 3.3 Loss Function: Disentangled Contrastive Loss

The embedding z = [z_pose (32d), z_view (4d)] is split into pose-invariant and view-dependent components.

**Total loss:**

```
L_total = L_pose + 10 * L_view + L_regularization
```

**L_pose -- Barlow Twins loss on pose embeddings:**

```
L_pose = BarlowTwins(z_pose, z'_pose)
```

Standard Barlow Twins (Zbontar et al., ICML 2021):
1. Standardize z_pose and z'_pose (subtract mean, divide by std per dimension)
2. Compute cross-correlation matrix C = (z_pose^T * z'_pose) / batch_size, shape [32, 32]
3. On-diagonal loss: sum((diag(C) - 1)^2) -- push correlations to 1
4. Off-diagonal loss: lambda * sum(C_off_diag^2) -- push correlations to 0
5. lambda = 0.0051

This is **negative-sample-free** (unlike InfoNCE/contrastive triplet losses), which is computationally efficient.

**L_view -- View consistency loss:**

```
L_view = MSE(cossim(z_view, z'_view), cossim(v_c, v'_c))
```

Where v_c, v'_c are unit vectors from origin to the virtual cameras. This explicitly teaches the 4-dimensional z_view to encode the camera viewpoint. The cosine similarity of the view embeddings should match the cosine similarity of the actual camera directions.

Weight: **10x** the pose loss (w_view = 10.0).

**L_regularization -- Anti-collapse terms:**

```
L_reg = VarianceLoss(z) + VarianceLoss(z') + KLUniformLoss(z) + KLUniformLoss(z')
```

- VarianceLoss: MSE between per-dimension variance and target sigma^2 = 1.0
- KLUniformLoss: per-dimension KL divergence toward uniform distribution on [0,1] (after mapping from [-1,1])

Weight: w_R = 1.0

### 3.4 Training Configuration

| Parameter | Value |
|-----------|-------|
| Batch size | 1024 |
| Learning rate | 0.001 (Adam) |
| Epochs | 60 |
| Validation | Every 3 epochs |
| Optimizer | Adam |
| 90/10 train/val split | Within training set |

---

## 4. Fine-tuning: Action Classification

### 4.1 SkatingVerse Dataset

**SkatingVerse** (Gan et al., IET Computer Vision 2024):
- 1,687 competition videos
- 19,993 training clips, 8,586 test clips
- 28 action classes:
  - 23 jump classes: 6 jump types (Axel, Salchow, Toe Loop, Loop, Flip, Lutz) x 4 rotation levels (1T-4T) + 1 extra
  - 4 spin classes: Camel Spin, Sit Spin, Upright Spin, Other Spins
  - 1 "NONE" class
- **Access**: Website (skatingverse.github.io) is under construction. Dataset was distributed via the CVPR 2024 challenge. **No direct public download link found** -- likely requires request to authors (BUPT/TeleAI).

### 4.2 Fine-tuning Architecture

```
Input: sequence of 2D poses [T, 17, 2]
  |
  v
Pre-trained JointFormer encoder (frozen weights from pre-train)
  |  Output: [T, 36] embeddings
  v
2-layer BiGRU (hidden=128, bidirectional=True, batch_first=True)
  |  Output: [T, 256]
  v
Temporal max pooling over time dimension
  |  Output: [1, 256]
  v
FC -> ReLU -> Dropout(0.5) -> FC -> 28 classes
  |  Cross-entropy loss
  v
Output: action class probabilities
```

### 4.3 Fine-tuning Configuration

| Parameter | Value |
|-----------|-------|
| Batch size | Configurable (per GPU) |
| Learning rate | 0.001 (Adam) |
| Epochs | 60 |
| Hidden size (BiGRU) | 128 |
| BiGRU layers | 2 |
| Dropout | 0.5 |
| 80/20 train/val split | Within training set |
| Augmentation | Jitter (50%), Mask (10%), No flip |

---

## 5. Downstream TAS: FACT Model

The learned pose embeddings are used as input features to **FACT** (Lu & Elhamifar, CVPR 2024), a Transformer-based TAS model using cross-attention for joint frame-level and action-level representation learning.

**Evaluation metrics:**
- Frame-wise accuracy (excluding "entry", "landing", "None" labels)
- F1@{10, 25, 50, 75, 90} -- predicted segment correct if overlap >= k% with ground truth

---

## 6. Results

### 6.1 Element-level TAS (23 jump labels, main results)

| Feature | Acc | F1@10 | F1@25 | F1@50 | F1@75 | F1@90 |
|---------|-----|-------|-------|-------|-------|-------|
| 2D pose (baseline) | 71.34 | 78.97 | 78.97 | 78.78 | 75.74 | 35.39 |
| 3D pose (MotionAGFormer) | 70.17 | 77.71 | 77.33 | 76.57 | 71.62 | 29.52 |
| **VIFSS (proposed)** | **85.82** | **92.75** | **92.75** | **92.56** | **90.65** | **49.62** |
| scratch-FSS (ablation) | 82.72 | 89.65 | 89.65 | 89.65 | 86.42 | 41.03 |

### 6.2 Set-level TAS (6 jump type labels)

| Feature | Acc | F1@10 | F1@25 | F1@50 | F1@75 | F1@90 |
|---------|-----|-------|-------|-------|-------|-------|
| 2D pose (baseline) | 78.55 | 85.12 | 84.93 | 84.17 | 81.52 | 35.83 |
| 3D pose | 79.89 | 87.13 | 86.94 | 86.56 | 82.36 | 33.36 |
| **VIFSS** | **89.91** | **95.44** | **95.44** | **94.68** | **93.16** | **51.71** |
| scratch-FSS | 86.38 | 92.48 | 92.29 | 91.72 | 88.87 | 42.44 |

### 6.3 Key Ablation Findings

**Impact of FS-Jump3D on Set-level TAS:**

| Feature | With FS-Jump3D | Without | Delta |
|---------|----------------|---------|-------|
| 3D pose | F1@50: 86.56 | F1@50: 82.51 | -4.05 |
| VIFSS | F1@50: 94.68 | F1@50: 92.78 | -1.90 |

FS-Jump3D contributes +1.9 to +4.0 F1@50. More impactful for 3D coordinate regression than for learned embeddings.

**Impact of procedure-aware annotation (F1@50, Set-level):**

| Feature | Procedure-aware | Coarse (jump only) | Delta |
|---------|----------------|-------------------|-------|
| 2D pose | 84.17 | 76.42 | +7.75 |
| 3D pose | 86.56 | 72.78 | +13.78 |
| VIFSS | 94.68 | 93.93 | +0.75 |

The annotation scheme helps most for raw features. VIFSS is already strong without it (+0.75).

**Pre-training benefit in low-data regime:**

| Fine-tuning data | VIFSS (pre-trained) | scratch-FSS (no pre-train) |
|------------------|---------------------|---------------------------|
| 100% | F1@50: 94.68 (Set), 92.56 (Elem) | 91.72 / 89.65 |
| 50% | ~93 / ~91 | ~89 / ~86 |
| 10% | ~90 / ~87 | ~70 / ~60 |
| 1% | >70 / >60 | ~0 (fails completely) |

**Critical insight:** With only 1% of fine-tuning data (~200 clips), pre-trained VIFSS still achieves >60% F1@50, while training from scratch fails entirely. This validates the approach for real-world scenarios with limited annotations.

---

## 7. What Can Be Reproduced Without Private Data

### 7.1 Publicly Available

| Resource | Access | Notes |
|----------|--------|-------|
| **VIFSS code** | https://github.com/ryota-skating/VIFSS | Apache-2.0, full source |
| **FS-Jump3D** | https://github.com/ryota-skating/FS-Jump3D | CC BY-NC-SA 4.0, Google Drive download |
| **Human3.6M** | Standard academic license | Requires registration |
| **MPI-INF-3DHP** | Publicly available | Standard |
| **AIST++** | Google Research | Publicly available |
| **JointFormer** | https://github.com/seblutz/JointFormer | Based on SemGCN |
| **FACT (TAS model)** | CVPR 2024 paper | Code available |

### 7.2 NOT Publicly Available (Blocking Issues)

| Resource | Status | Impact |
|----------|--------|--------|
| **SkatingVerse** | No public download; challenge-era distribution only | **BLOCKS fine-tuning** (19,993 training clips) |
| **TAS annotations** | Promised at github.com/ryota-skating/VIFSS "upon publication" | **BLOCKS TAS evaluation** |
| **Pre-trained weights** | Not released | Must train from scratch |

### 7.3 What We CAN Do

1. **Pre-training stage is fully reproducible**: We have FS-Jump3D (downloaded), Human3.6M, MPI-INF-3DHP, AIST++ (all public). Can train the JointFormer encoder with contrastive learning using the exact same pipeline.

2. **Architecture is fully available**: The VIFSS code is clean, well-documented, and Apache-2.0 licensed. We can copy the encoder, loss, and training pipeline directly.

3. **Alternative fine-tuning datasets**: We have FSC, MCFS, MMFS, FineFS, Figure-Skating-Classification (HuggingFace). These could potentially substitute for SkatingVerse:
   - **Figure-Skating-Classification** (Mercity/HuggingFace): 5,168 sequences, 64 classes (COCO 17kp) -- closest substitute
   - **MMFS**: 26,198 sequences, 256 categories -- large but different format
   - **FSD-10**: 10-class classification -- too small

4. **Pose format compatibility**: VIFSS uses H3.6M 17kp (same as our system). Our pipeline already produces this format via `halpe26_to_h36m()`. The skeleton connections in the code match our `H36Key` enum.

### 7.4 What We CANNOT Do (Without More Data)

1. **Replicate exact TAS numbers**: The TAS evaluation requires their specific broadcast video annotations (371 Olympic/World Championship videos with frame-level labels). No public substitute exists.

2. **Fine-tune on SkatingVerse**: 28-class action classification requires the 19,993 trimmed clips from SkatingVerse. No download link available.

3. **Pre-trained weights**: Must train from scratch (60 epochs, batch 1024, ~modest GPU requirements).

---

## 8. Practical Implementation Assessment

### 8.1 For Our Project (skating-biomechanics-ml)

**Strengths of VIFSS approach for us:**

- Same skeleton format (H3.6M 17kp) -- zero conversion needed
- View-invariant embeddings solve our core problem: single-camera phone video from arbitrary angles
- Pre-training is data-efficient: even 1% of fine-tuning data works with pre-trained encoder
- The pre-training stage uses only 3D pose data (no video needed) -- we can run it offline
- Apache-2.0 license -- fully permissive
- Code is clean and self-contained (~500 lines total)

**Weaknesses:**

- Pre-training requires 3D pose datasets (Human3.6M, AIST++, FS-Jump3D) -- we have FS-Jump3D but need to acquire/download others
- Fine-tuning requires SkatingVerse or equivalent -- we must use substitute datasets
- The embedding is per-frame (no temporal context in encoder) -- temporal modeling is deferred to BiGRU
- 17kp only (no foot keypoints from HALPE26) -- loses the extra 9 keypoints

**Compute requirements:**

- Pre-training: batch 1024, 60 epochs, modest GPU (paper used single GPU with DataParallel). The JointFormer is small (~2M params). Estimated: 2-4 hours on RTX 3050 Ti.
- Fine-tuning: batch per-GPU, 60 epochs, BiGRU is lightweight. Estimated: 1-2 hours.
- Inference: per-frame, ~1ms per pose on GPU. Real-time capable.

### 8.2 Adaptation Plan

To use VIFSS embeddings in our system:

1. **Train encoder**: Use FS-Jump3D + Human3.6M + AIST++ for contrastive pre-training (exact replication of Stage 1)
2. **Fine-tune encoder**: Use Figure-Skating-Classification (5,168 clips, 64 classes from HuggingFace) as substitute for SkatingVerse. Map COCO 17kp to H3.6M 17kp.
3. **Extract embeddings**: Run encoder on our video frames to get 36-dim embeddings per frame
4. **Use embeddings as features**: Feed into our existing analysis pipeline (phase detection, metrics, recommender) instead of raw 2D/3D coordinates

**Alternative: Skip fine-tuning, use pre-trained encoder directly.** Given the low-data ablation shows >60% F1@50 with 1% data, the pre-trained encoder alone may produce useful embeddings for our biomechanics analysis tasks (which are different from classification/TAS).

### 8.3 Skeleton Compatibility

Our `halpe26_to_h36m()` produces H3.6M 17kp format. VIFSS's `NUM_JOINTS = 17` and `JOINT_CONNECTION` matches our H3.6M skeleton. The normalization (center at mid-hip, scale torso length to 0.4) is identical to our `PoseNormalizer`. **Direct plug-and-play.**

---

## 9. Comparison with Related Work

| Method | Input | View-invariant | Skating-specific | TAS F1@50 (Elem) |
|--------|-------|---------------|------------------|-------------------|
| MCFS (Liu 2021) | 2D pose | No | No | ~70 (est.) |
| Tanaka 2024 (MMSports) | 3D pose | Partial (alignment) | No | ~77 |
| **VIFSS (proposed)** | **2D pose -> embedding** | **Yes (contrastive)** | **Yes (fine-tuned)** | **92.56** |
| Hong 2021 (ICCV) | Pose distillation | Partial | Partial | N/A (classification) |

The key insight: **learned embeddings > raw coordinates**, and **view-invariant pre-training >> training from scratch**, especially with limited data.

---

## 10. Open Questions

1. **SkatingVerse access**: Can we request the dataset from the authors (Gan et al., BUPT/TeleAI)? The challenge website is defunct.
2. **Cross-dataset generalization**: How well do VIFSS embeddings transfer to non-competition skating (practice sessions, lower-level skaters)?
3. **Spin TAS**: Paper focuses on jumps only. Future work planned for spins and step sequences.
4. **Temporal resolution**: The encoder is per-frame. Does adding temporal context (e.g., sliding window) improve embeddings for our biomechanics metrics?
5. **Embedding dimensionality**: 36d (32 pose + 4 view). Is this sufficient for our biomechanics analysis? The 4d view component could be useful for camera angle estimation.

---

## References

- VIFSS: Tanaka, Suzuki, Fujii. "VIFSS: View-Invariant and Figure Skating-Specific Pose Representation Learning for Temporal Action Segmentation." arXiv:2508.10281, 2025.
- JointFormer: Lutz et al. "JointFormer: Single-Frame Lifting Transformer with Error Prediction and Refinement for 3D Human Pose Estimation." ICPR 2022.
- Barlow Twins: Zbontar et al. "Barlow Twins: Self-Supervised Learning via Redundancy Reduction." ICML 2021.
- CV-MIM: Zhao et al. "Learning View-Disentangled Human Pose Representation by Contrastive Cross-View Mutual Information Maximization." CVPR 2021.
- FS-Jump3D: Tanaka, Suzuki, Fujii. ACM MMSports 2024.
- SkatingVerse: Gan et al. IET Computer Vision, 2024.
- FACT: Lu & Elhamifar. "FACT: Frame-Action Cross-Attention Temporal Modeling for Efficient Action Segmentation." CVPR 2024.
- MotionAGFormer: Mehraban et al. "MotionAGFormer: Enhancing 3D Human Pose Estimation with a Transformer-GCNFormer Network." WACV 2024.
