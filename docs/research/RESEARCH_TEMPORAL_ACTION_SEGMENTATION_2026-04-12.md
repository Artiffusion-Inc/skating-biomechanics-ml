# Temporal Action Segmentation (TAS) for Figure Skating — Deep Research
**Date:** 2026-04-12
**Sources:** arXiv papers (2410.20427, 2408.16638, 2110.08568, 2303.17959, 2412.04353, 2006.09220), GitHub repos, survey TPAMI 2023
**Status:** ACTIVE

---

## Summary

Temporal Action Segmentation (TAS) assigns a semantic label to every frame in an untrimmed video, producing start/end timestamps for each action segment. For figure skating, this means automatically finding where each jump, spin, and step sequence begins and ends within a competition program. This research covers the state-of-the-art methods, figure-skating-specific datasets, skeleton-only approaches, and evaluation metrics. The key finding: skeleton-only TAS is not only possible but often **outperforms RGB** for figure skating due to the motion-centric nature of the sport.

---

## Table of Contents

1. [The TAS Task](#1-the-tas-task)
2. [YourSkatingCoach — BIOES Sequence Labeling](#2-yourskatingcoach)
3. [MS-TCN / MS-TCN++ — The Foundational Architecture](#3-ms-tcn--ms-tcn)
4. [ASFormer — Transformer-Based TAS](#4-asformer)
5. [Diffusion-Based Approaches](#5-diffusion-based-approaches)
6. [Skeleton-Only TAS](#6-skeleton-only-tas)
7. [Figure Skating Datasets for TAS](#7-figure-skating-datasets)
8. [Evaluation Metrics](#8-evaluation-metrics)
9. [Key Questions Answered](#9-key-questions-answered)
10. [Open-Source Implementations](#10-open-source-implementations)
11. [Recommended Strategy for Our Project](#11-recommended-strategy)

---

## 1. The TAS Task

Temporal Action Segmentation takes an untrimmed video sequence and assigns a class label to every frame. Formally, given frames x_1:T = (x_1, ..., x_T), the goal is to infer class labels c_1:T = (c_1, ..., c_T) for each frame, where T is the video length.

**Key challenges:**
- **Over-segmentation**: The model produces many short spurious segments instead of clean, contiguous segments. This is the #1 problem in TAS.
- **Class imbalance**: "Background" (transitions, skating between elements) occupies most frames; actual elements (jumps, spins) are rare.
- **Long-range dependencies**: A 4-minute free skate has 12+ elements; the model needs context across thousands of frames.
- **Boundary ambiguity**: Exact start/end of elements is inherently fuzzy (entry transitions blend into the element).

**TAS vs. Action Detection**: TAS densely labels every frame. Action detection only localizes sparse action segments with most frames unlabeled. TAS is the harder, more useful task for figure skating.

---

## 2. YourSkatingCoach — BIOES Sequence Labeling

**Paper:** "YourSkatingCoach: A Figure Skating Video Benchmark for Fine-Grained Element Analysis" (Chen et al., arXiv:2410.20427, Oct 2024)

### BIOES Tagging Scheme

YourSkatingCoach casts air time detection as a **sequence labeling problem** using BIOES format (borrowed from Named Entity Recognition in NLP):

| Label | Meaning | Definition |
|-------|---------|------------|
| **B** (Beginning) | Take-off frame | The exact frame where the skate leaves the ice |
| **I** (Inside) | In-air frame | Any frame during flight |
| **E** (End) | Landing frame | The frame where the skate touches the ice |
| **O** (Outside) | Other | Any non-flight frame |

The air time is computed as: `air_time = (number of I-labels) / fps`

This is a simplified BIOES (no "S" for single-frame segments since jumps always have multiple in-air frames).

### Architecture

```
Input: Skeleton (COCO 17x2 from AlphaPose)
    |
    v
[Human Pose Embedding] -- either:
    |  (a) Pr-VIPE: pretrained view-invariant embedding (fixed weights, H=16)
    |  (b) ST-GCN: trainable graph CNN (weights updated, spatial config partitioning)
    |
    v
+ Positional Embedding
    |
    v
[Encoder-CRF]
    |  2-layer Transformer encoder
    |  Linear projection to K labels (K=4: B, I, E, O)
    |  CRF layer with Viterbi decoding (enforces O->B->I->E->O order)
    |
    v
Output: Per-frame label sequence
```

### Key Design Decision: CRF Layer

The CRF (Conditional Random Field) enforces valid label transitions. Without it, the model might predict I before B, or E before I. The transition matrix A in R^{KxK} models the probability of transitioning from label i to label j. During inference, Viterbi decoding finds the most probable valid label sequence.

### Results

| Method | Accuracy | F1-score | Mean Error % | Edit Distance |
|--------|----------|----------|-------------|---------------|
| Pr-VIPE + Encoder-CRF | 95.3% | 0.564 | 27.17% | 5.674 |
| **GCN + Encoder-CRF** | **96.3%** | **0.671** | **25.05%** | **4.435** |

**Key findings:**
- GCN embeddings (trainable) outperform Pr-VIPE (fixed pretrained) significantly on F1 (0.671 vs 0.564).
- Pr-VIPE is more conservative (predicts fewer air time labels), GCN is more lenient.
- Cross-action experiments show training on `all_jump` dataset generalizes best.
- **Axel jumps confuse other models** because it is the only forward take-off jump.
- **Multiple jumps in one video** degrade performance significantly (61.84% error vs 25.46% for single-jump videos).
- **Spins confuse the model** — rotations on ice vs rotations in air look similar to the model.

### Dataset

- 454 videos of 6 jump types (Axel, Toe Loop, Salchow, Loop, Flip, Lutz)
- Practice clips of a 9-year-old female skater (NOT competition videos)
- 1.3s to 10s duration, 30fps, 1920x1080
- 1-3 jumps per video
- Train: 408, Test: 46 (for all_jump)
- Data augmentation: trim context from beginning/end while keeping 30 frames of context

### Limitations

- Only handles **jumps**, not spins or step sequences
- Only detects **air time** (flight phase), not the full element including approach and exit
- Small dataset (454 videos of a single skater)
- Per-frame accuracy is high (96.3%) but F1 is low (0.671), indicating boundary precision issues
- Mean error percentage of 25% means predictions are off by ~25% of the true air time on average

---

## 3. MS-TCN / MS-TCN++ — The Foundational Architecture

**Paper:** "MS-TCN++: Multi-Stage Temporal Convolutional Network for Action Segmentation" (Li et al., TPAMI 2020, arXiv:2006.09220)

MS-TCN++ is the most widely used baseline for TAS. It is the "ResNet of action segmentation" — the architecture everyone compares against.

### Architecture

```
Input: Frame-wise features (I3D, skeleton embeddings, etc.)
    |
    v
[Stage 1: Prediction Generation]
    |  11 layers of Dual Dilated Layers (DDL)
    |  Each DDL = two parallel dilated convolutions:
    |    - Conv with dilation 2^l (increasing)
    |    - Conv with dilation 2^(L-l) (decreasing)
    |  64 filters, kernel size 3, dropout 0.5
    |  1x1 conv + softmax -> K class probabilities
    |  Loss: Cross-entropy + Smoothing loss (lambda=0.15)
    |
    v
[Stage 2: Refinement] -- input is only Stage 1 probabilities (NO features!)
    |  10 layers of standard dilated convolutions
    |  Refines boundaries, removes over-segmentation
    |
    v
[Stage 3: Refinement] -- parameters SHARED with Stage 2
    |
    v
[Stage 4: Refinement] -- parameters SHARED with Stage 2
    |
    v
Output: Per-frame class probabilities
```

### Key Innovation: Multi-Stage Refinement

The core insight: **refinement stages receive ONLY the probability distribution from the previous stage, NOT the raw features**. This is crucial:

| Input to refinement stages | F1@10 | F1@50 | Edit |
|---------------------------|-------|-------|------|
| Probabilities + features | 56.2 | 45.8 | 47.6 |
| **Probabilities only** | **76.3** | **64.5** | **67.9** |

Adding features to later stages *degrades* performance by ~20% F1 because different action classes share similar appearance/motion. The probability-only input forces the model to focus on label context (neighboring labels), which is exactly what's needed to fix over-segmentation.

### Dual Dilated Layer (DDL)

Standard dilated convolutions use exponentially increasing dilation (1, 2, 4, 8, ...), giving lower layers small receptive fields. DDL runs two parallel convolutions:
- Path 1: dilation 2^l (increasing — captures local context)
- Path 2: dilation 2^(L-l) (decreasing — captures global context)

Both paths are concatenated, giving every layer access to both local and global temporal context.

### Smoothing Loss

The critical loss function that reduces over-segmentation:

```
L = L_cls + lambda * L_T-MSE

L_T-MSE = (1/TC) * sum_t,c clip(|log(y_t,c) - log(y_{t-1,c})|, tau)
```

This penalizes large changes in log-probabilities between consecutive frames, forcing smooth predictions. The truncation threshold tau=4 prevents penalizing genuine action boundaries. This alone improves F1 by up to 5% over cross-entropy alone.

### Results on Standard Benchmarks (I3D features, 15fps)

| Dataset | F1@{10,25,50} | Edit | Acc |
|---------|---------------|------|-----|
| 50Salads | 80.7 / 78.5 / 70.1 | 74.3 | 83.7 |
| GTEA | 88.8 / 85.7 / 76.0 | 83.5 | 80.1 |
| Breakfast | 64.1 / 58.6 / 45.9 | 65.6 | 67.6 |

### Why MS-TCN++ Matters for Us

1. **Feature-agnostic**: Works with any frame-wise feature (I3D, skeleton embeddings, pose features)
2. **Efficient**: Fully convolutional, no sequential processing. 10 min training on 50Salads vs 35 min for Bi-LSTM.
3. **Simple to implement**: ~1000 lines of PyTorch
4. **Battle-tested**: Used as baseline in virtually every TAS paper since 2020

---

## 4. ASFormer — Transformer-Based TAS

**Paper:** "ASFormer: Transformer for Action Segmentation" (Yi et al., BMVC 2021, arXiv:2110.08568)

### Architecture

ASFormer replaces the TCN backbone with a Transformer while addressing three challenges:

1. **Local connectivity inductive bias**: Self-attention lacks the local prior that convolutions have. ASFormer adds explicit local connectivity via a local attention mask (only attend to nearby frames within a window) to constrain the hypothesis space.

2. **Hierarchical representation for long sequences**: Direct self-attention is O(T^2) which is prohibitive for long videos. ASFormer uses a pre-defined hierarchical pattern that reduces sequences at each layer (similar to pooling but learned), enabling efficient processing of minute-long videos.

3. **Decoder for temporal relation refinement**: The decoder explicitly models relationships between multiple action segments (not just individual frames) to refine predictions from the encoder.

### Key Differences from MS-TCN++

| Aspect | MS-TCN++ | ASFormer |
|--------|----------|----------|
| Backbone | 1D dilated convolutions | Self-attention + local mask |
| Inductive bias | Strong (local connectivity built in) | Weak (must be added explicitly) |
| Long sequences | Efficient (linear in T) | Requires hierarchical reduction |
| Small datasets | Works well | May overfit ( Transformer needs more data) |
| State-of-the-art | 2020 baseline | Improves over MS-TCN on all benchmarks |

### Results (from the paper, standard benchmarks)

ASFormer improves over MS-TCN++ by ~1-3% on F1 scores across 50Salads, GTEA, and Breakfast. The improvements are consistent but not dramatic, suggesting the multi-stage refinement paradigm is more important than the specific backbone.

### Open Source

GitHub: `ChinaYi/ASFormer` (135 stars, MIT license)

---

## 5. Diffusion-Based Approaches

### Diffusion Action Segmentation (ICCV 2023)

**Paper:** "Diffusion Action Segmentation" (Liu et al., arXiv:2303.17959)

Instead of discriminative classification, this approach treats action segmentation as a **generative problem**: action predictions are iteratively generated from random noise, conditioned on video features. Each denoising step refines the prediction, similar to how MS-TCN++ refines through stages.

**Unified masking strategy** for three action characteristics:
- **Position prior**: Certain actions tend to occur at certain positions in the video
- **Boundary ambiguity**: Action boundaries are inherently fuzzy
- **Relational dependency**: Actions have strong sequential dependencies (kitchen tasks follow recipes)

### ActFusion (NeurIPS 2024)

**Paper:** "ActFusion: a Unified Diffusion Model for Action Segmentation and Anticipation" (Gong et al., arXiv:2412.04353)

Unifies action segmentation and action anticipation (predicting future actions) in a single diffusion model. Uses **anticipative masking** during training: the late part of the video is masked and replaced with learnable tokens that learn to predict invisible future frames.

**Results:** State-of-the-art on 50Salads, Breakfast, GTEA for both segmentation and anticipation tasks.

### Assessment for Figure Skating

Diffusion approaches are **overkill** for our use case:
- Much slower inference (iterative denoising vs single forward pass)
- More complex to implement and tune
- Gains are marginal over MS-TCN++/ASFormer on standard benchmarks
- Not tested on figure skating or sports datasets
- Better suited for tasks requiring diverse plausible outputs (not our case)

**Verdict:** Interesting research direction, not practical for production use.

---

## 6. Skeleton-Only TAS

### Key Finding: Skeleton Outperforms RGB for Figure Skating

From the MMFS dataset paper (Liu et al., 2023), skeleton-based models **significantly outperform** RGB-based models on fine-grained figure skating recognition:

| Method | Modality | MMFS-3 (3 classes) | MMFS-63 (63 classes) |
|--------|----------|--------------------|--------------------|
| TSM | RGB | 90.3% | 50.9% |
| PAN | RGB | 91.5% | 69.1% |
| ST-GCN | Skeleton | **98.9%** | **77.4%** |
| CTRGCN | Skeleton | **99.4%** | **78.8%** |

Skeleton achieves 78.8% vs RGB's best 69.1% on 63 fine-grained classes. The reason: figure skating is a **motion-centric** sport where body pose dynamics carry far more discriminative information than visual appearance.

### Skeleton-Based TAS Methods

#### 1. LSTM-CNN (mayupei, 2025) — Figure Skating Specific

**Repo:** `mayupei/figure-skating-action-segmentation`

- Two-stage: LSTM (64 units, window=20 frames) -> 1D-CNN (temporal refinement)
- Input: COCO 17kp skeleton at 3fps (downsampled from 30fps)
- Data: 222 routines from MCFS+MMFS overlap
- **Result: 0.89 F1@50** for jump/spin segmentation in full competition videos
- Normalizes skeletons: center at origin, scale by vertical distance
- The CNN stage is the key: it raises F1 from 0.31 (LSTM alone) to 0.89

#### 2. STGA-Net (2023) — Spatial-Temporal Graph Attention

**Paper:** "STGA-Net: Spatial-Temporal Graph Attention Network for Skeleton-Based Temporal Action Segmentation"

Uses graph attention mechanisms on the skeleton graph to capture:
- **Spatial attention**: Which body joints are most relevant for the current action
- **Temporal attention**: Which frames in the temporal window matter most
- Jointly models spatial and temporal relationships

#### 3. Spatial-Temporal Graph Transformer (2023)

**Paper:** "Spatial-temporal graph transformer network for skeleton-based temporal action segmentation" (Springer, 2023)

Applies Transformer architecture directly on the skeleton graph, treating joints as nodes and bones as edges. Uses multi-head attention along both spatial and temporal dimensions.

#### 4. Skeleton Motion Words (ICCV 2025) — Unsupervised

**Paper:** "Skeleton Motion Words for Unsupervised Skeleton-Based Temporal Action Segmentation" (Gokay et al., ICCV 2025)

Novel unsupervised approach that does NOT require frame-level annotations:
1. Sequence-to-sequence temporal autoencoder encodes skeleton sequences
2. Latent sequences divided into non-overlapping patches
3. Patches quantized into "skeleton motion words" (discrete tokens)
4. Clustering discovers semantically meaningful action classes

Evaluated on HuGaDB, LARa, and BABEL datasets. Outperforms previous unsupervised methods.

**Implication:** Could potentially bootstrap a TAS model without expensive frame-level annotations.

### Can Skeleton-Only TAS Work for Our Pipeline?

**Yes, and it should be our primary approach.** Reasons:

1. **We already have skeleton extraction** (RTMPose via rtmlib, H3.6M 17kp)
2. **Skeleton removes visual noise** (background, audience, camera angle changes)
3. **Proven superior for figure skating** (78.8% vs 69.1% on MMFS-63)
4. **Much lighter computation** (17x2 features vs 2048-dim I3D features)
5. **Privacy-preserving** (no raw video needed after extraction)
6. **Camera-agnostic** (works with any camera angle, unlike RGB features)

### Input Format for Skeleton-Based TAS

For a video of T frames with K=17 keypoints (H3.6M format):

```
Input tensor: (T, K * 2) = (T, 34) for 2D poses
           or (T, K * 3) = (T, 51) for 3D poses

Each frame is a flattened vector of joint coordinates:
[x_hip, y_hip, x_knee_r, y_knee_r, ..., x_head, y_head]
```

For GCN-based methods, the input is the adjacency matrix of the skeleton graph plus joint coordinates.

---

## 7. Figure Skating Datasets for TAS

### Dataset Comparison Table

| Dataset | Videos | Duration | Skeleton | Frame Labels | Task | Access |
|---------|--------|----------|----------|-------------|------|--------|
| **MCFS** | 271 | 162-285s | BODY_25 (OpenPose) | Yes (per-frame) | TAS | Available |
| **MMFS** | 1176 | Variable | COCO 17kp (HRNet) | No (routine-level) | AR + AQA | Available |
| **FineFS** | 1167 | 2m40s / 4min | 17kp (3D) | Yes (start/end times) | AQA + TAS | Baidu/GDrive |
| **YourSkatingCoach** | 454 | 1.3-10s | COCO 17kp (AlphaPose) | Yes (BIOES) | Air time detection | In paper |
| **FS-Jump3D** | ~100 | Jump clips | 3D (motion capture) | Yes (procedure-aware) | TAS | GitHub |
| **FSD-10** | 1484 | Variable | No | Clip-level only | AR | **Unavailable** |

### MCFS (Motion-Centered Figure Skating)

- **AAAI 2021**, Dalian University of Technology
- 271 competition routines from 2017-2019 World Championships
- Per-frame labels: each frame assigned one of ~10 element categories (jump, spin, step sequence, etc.)
- Skeleton: OpenPose BODY_25 format
- **Quality issues**: 56% of frames have >=1 missing joint, 19% have >=3 missing joints
- **Label quality**: Start/end frames are ~57 frames (~2s) off from actual take-off/landing (per YourSkatingCoach analysis)
- This imprecision is problematic for fine-grained element analysis but acceptable for coarse TAS

### MMFS (Multi-Modality Multi-Task Figure Skating)

- **ACM MM 2023**, Dalian University of Technology
- 11,671 clips from 107 competition videos, 35.38 hours total
- 256 fine-grained categories (3 sets: Jump, Spin, Sequence; 24 spatial + 22 temporal labels)
- Skeleton: COCO 17kp via HRNet (high quality, no missing joints)
- **No per-frame labels** — only clip-level annotations
- Provides pre-extracted VST features (Kinetics-600 pretrained) in .pkl format
- 222 routines overlap with MCFS (same videos, different skeleton extraction)

### FineFS (Fine-Grained Figure Skating)

- **ACM MM 2023**, same group as MMFS
- 1167 full routines (729 short program + 438 free skate)
- Skeleton: 17kp 3D in camera space coordinates
- **Rich annotations**: fine-grained scores, hierarchical category labels, start/end times of sub-actions
- Pre-extracted VST features provided
- Download: Baidu Drive / Google Drive
- **Ideal for training a TAS model** — has both skeleton data and temporal boundaries

### FS-Jump3D (3D Pose-Based TAS)

- **ACM Multimedia Workshop 2024**, Nagoya University
- Optical markerless motion capture (high-quality 3D)
- **Procedure-aware annotations**: breaks jumps into phases (preparation, approach, take-off, rotation, landing)
- 3D pose features validated as superior to 2D for TAS
- Available at: `github.com/ryota-skating/FS-Jump3D`

### Using FineFS for TAS Training

**Input format:**
```python
# skeleton/0.npz — shape (T, 17, 3) per frame
# annotation/0.json — element boundaries and categories
{
    "elements": [
        {"type": "3Lutz", "start_frame": 120, "end_frame": 245, "score": 4.2},
        {"type": "ChCamelSpin4", "start_frame": 500, "end_frame": 680, "score": 3.8},
        ...
    ]
}
```

**Converting to TAS format:**
```python
# For each video, create a frame-wise label array:
labels = np.zeros(T, dtype=int)  # 0 = background/transitions
for elem in annotation["elements"]:
    labels[elem["start_frame"]:elem["end_frame"]] = class_to_id[elem["type"]]
```

**Number of classes**: FineFS has 256 fine-grained categories, but for practical TAS we'd group into ~10-15 coarse classes (jump, spin, step_sequence, choreo_sequence, etc.)

---

## 8. Evaluation Metrics

### Frame-wise Accuracy (Acc / MoF)

```
Acc = (number of correctly classified frames) / (total frames)
```

**Limitation**: Long action classes dominate. A model that always predicts the most common class can achieve high accuracy. Over-segmentation errors have very low impact (splitting a 100-frame segment into two 50-frame segments only misclassifies ~0 frames).

### Segmental F1 Score (F1@tau)

The most important metric. A predicted segment is a **true positive (TP)** if its IoU with the ground truth exceeds threshold tau/100. Standard thresholds: **F1@10, F1@25, F1@50**.

```
F1@tau = 2 * (precision * recall) / (precision + recall)

Where:
  TP = predicted segments with IoU > tau/100 w.r.t. matching ground truth
  FP = predicted segments that don't match (IoU <= tau/100)
  FN = ground truth segments not matched by any prediction
```

If multiple predictions overlap a single ground truth, only the best match is TP; the rest are FP.

| Metric | What it measures | When to use |
|--------|-----------------|-------------|
| F1@10 | Loose boundary tolerance | Allows predictions shifted by ~10% of segment length |
| F1@25 | Moderate tolerance | Standard for comparing methods |
| **F1@50** | **Strict boundary quality** | **Most commonly reported; punishes over-segmentation** |

### Edit Score

Based on Levenshtein distance (minimum edit operations to transform prediction into ground truth):

```
Edit = (1 - edit_distance(prediction, ground_truth) / max(|pred|, |gt|)) * 100
```

Measures both over-segmentation (extra segment boundaries) and misclassification. An edit operation can be: insert a segment, delete a segment, or substitute a segment label.

### Mean Error Percentage (YourSkatingCoach specific)

```
Error% = mean(|len(pred) - len(gt)| / len(gt)) * 100
```

Only computed for overlapping predictions. Measures how far off the predicted duration is from the ground truth.

### Metric Summary for Our Project

For figure skating TAS, we should track:
1. **F1@50** — primary metric (strict boundary quality)
2. **F1@25** — secondary (allows some boundary slack)
3. **Edit score** — catches over-segmentation
4. **Frame-wise accuracy** — baseline comparison
5. **Boundary F1** (if available) — specifically measures boundary detection quality

---

## 9. Key Questions Answered

### Q: Can we do TAS from skeleton sequences alone (no video RGB)?

**Yes, definitively.** Evidence:

1. MMFS benchmark: ST-GCN skeleton (78.8%) outperforms PAN RGB (69.1%) on 63-class fine-grained recognition
2. mayupei's LSTM-CNN achieves 0.89 F1@50 using only COCO 17kp skeleton for figure skating TAS
3. FS-Jump3D validates 3D pose features as superior to RGB for skating TAS
4. Our pipeline already produces H3.6M 17kp skeleton data via RTMPose

Skeleton-only is actually **preferred** for figure skating because:
- Eliminates visual noise (rink background, audience, camera angle changes)
- Much lower dimensionality (34-dim vs 2048-dim I3D features)
- Privacy-preserving (no raw video needed)
- Works across camera angles

### Q: What's the minimum data needed to train a reasonable TAS model?

Based on published results:

| Approach | Training Data | Performance | Notes |
|----------|--------------|-------------|-------|
| MS-TCN++ (50Salads) | 40 videos | F1@50=70.1 | 17 classes, kitchen activities |
| MS-TCN++ (GTEA) | 21 videos | F1@50=76.0 | 11 classes, egocentric |
| LSTM-CNN (skating) | 222 routines (177 train) | F1@50=0.89 | Jump/Spin/None, skeleton |
| YourSkatingCoach | 408 videos | F1=0.671 | Air time BIOES, skeleton |
| Skeleton Motion Words (unsupervised) | 0 labeled | Lower than supervised | Unsupervised, no labels needed |

**Minimum viable**: ~50-100 annotated routines with per-frame labels should produce a usable model for coarse segmentation (jump/spin/background).

**Good quality**: 200+ annotated routines (available from MCFS+MMFS overlap) should produce F1@50 > 0.85 for coarse segmentation.

**Fine-grained**: FineFS's 1167 routines with temporal boundaries could support 10-15 class TAS.

### Q: How does YourSkatingCoach handle hierarchical elements?

**It doesn't.** YourSkatingCoach only handles individual jump clips (1.3-10s), not full competition programs. It was designed for air time detection in isolated jump videos, not for segmenting continuous programs.

For hierarchical element segmentation (programs containing multiple jumps, spins, sequences), the relevant approaches are:

1. **MCFS + MMFS**: Full competition routines with per-frame labels (10 coarse classes)
2. **FineFS**: Full routines with hierarchical labels (3 sets x 22 temporal categories)
3. **mayupei's LSTM-CNN**: Coarse segmentation (jump/spin/none) on full competition videos
4. **FineGym** (gymnastics): Demonstrates hierarchical segmentation with fine-to-coarse labels — similar structure to figure skating programs

The hierarchical challenge: figure skating programs have a **sequential structure** (elements must appear in a specific order defined by ISU rules). This constraint could be modeled as a grammar (similar to how natural language uses grammars in NER). However, no published method explicitly models ISU program structure constraints.

### Q: What's the best open-source implementation available?

| Project | Stars | License | Language | Figure Skating? |
|---------|-------|---------|----------|----------------|
| **MS-TCN2** (MS-TCN++) | 181 | MIT | Python/PyTorch | No (general TAS) |
| **ASFormer** | 135 | MIT | Python/PyTorch | No (general TAS) |
| **awesome-tas** (survey) | 246 | - | - | Curated list |
| **mayupei/figure-skating** | 2 | - | Jupyter/Python | Yes (skeleton, LSTM-CNN) |
| **FS-Jump3D** | - | - | - | Yes (3D pose TAS) |
| **Skeleton Motion Words** | - | - | - | No (unsupervised) |

**Best starting point**: MS-TCN++ (`sj-li/MS-TCN2`, 181 stars, MIT, well-maintained) adapted with skeleton features instead of I3D.

---

## 10. Open-Source Implementations

### MS-TCN++ (Recommended Baseline)
- **Repo:** `github.com/sj-li/MS-TCN2`
- **Stars:** 181, **License:** MIT
- **Features:** MS-TCN and MS-TCN++, tested on 50Salads/GTEA/Breakfast
- **Input:** Pre-extracted frame features (I3D by default, any feature vector works)
- **Adaptation for skeleton:** Replace I3D features with skeleton joint coordinates

### ASFormer
- **Repo:** `github.com/ChinaYi/ASFormer`
- **Stars:** 135, **License:** MIT
- **Features:** Transformer-based, multi-stage decoder refinement
- **Benchmark:** Improves over MS-TCN++ on standard datasets

### awesome-temporal-action-segmentation
- **Repo:** `github.com/nus-cvml/awesome-temporal-action-segmentation`
- **Stars:** 246
- **Content:** Curated list of 80+ papers with code, organized by supervision type (fully/weakly/unsupervised)

### Figure Skating Specific
- **mayupei/figure-skating-action-segmentation**: LSTM-CNN on MCFS+MMFS, skeleton-only, F1@50=0.89
- **FS-Jump3D** (`github.com/ryota-skating/FS-Jump3D`): 3D pose TAS for figure skating jumps

---

## 11. Recommended Strategy for Our Project

### Phase 1: Coarse Segmentation (Jump/Spin/Background)

**Goal:** Segment full competition programs into jumps, spins, step sequences, and background/transitions.

**Approach:**
1. Use MS-TCN++ as backbone with H3.6M 17kp skeleton features as input
2. Train on MCFS+MMFS 222 shared routines (per-frame labels from MCFS, skeletons from MMFS)
3. Data preparation: normalize skeleton (center at hip, scale by spine length), downsample to 3-5 fps
4. Expected performance: F1@50 > 0.85 (based on mayupei's results)

**Input format per frame:** (17 * 2,) = 34-dim vector of normalized 2D joint coordinates

**Classes:** ~10 (background, jump, spin, step_sequence, choreo_sequence, and element subtypes)

### Phase 2: Fine-Grained Element Classification

**Goal:** After segmenting, classify each element with its specific type (3Lutz, ChCamelSpin4, etc.)

**Approach:**
1. Two-stage pipeline: coarse segmentation -> per-segment classification
2. Use the segment boundaries from Phase 1 to extract element clips
3. Classify each clip using skeleton-based action recognition (ST-GCN, CTRGCN)
4. Train on MMFS (11,671 clips, 256 categories) or FineFS

### Phase 3: Hierarchical / BIOES Labeling

**Goal:** Within each element segment, detect sub-phases (approach, take-off, flight, landing, exit).

**Approach:**
1. Apply YourSkatingCoach's BIOES scheme within each detected element
2. Train on YourSkatingCoach dataset (454 jump clips) or FS-Jump3D (procedure-aware annotations)
3. This enables precise air time measurement per jump

### Data Sources by Phase

| Phase | Dataset | What We Need | Status |
|-------|---------|-------------|--------|
| 1 (coarse TAS) | MCFS + MMFS overlap | 222 routines with per-frame labels + skeletons | Already downloaded |
| 2 (classification) | MMFS | 11,671 clips, 256 categories | Already downloaded |
| 3 (sub-phase BIOES) | FineFS or YourSkatingCoach | Element boundaries with sub-phase labels | Need to download FineFS |

### Why This Strategy Works

1. **We already have the skeleton pipeline** (RTMPose -> H3.6M 17kp)
2. **Training data exists** (MCFS+MMFS for coarse, MMFS for classification, FineFS for fine-grained)
3. **Skeleton-only is proven superior** for figure skating (78.8% vs 69.1%)
4. **MS-TCN++ is simple, fast, and well-understood** — can be implemented in ~500 lines
5. **Modular design** — each phase can be developed and evaluated independently

---

## References

1. Chen et al., "YourSkatingCoach: A Figure Skating Video Benchmark for Fine-Grained Element Analysis", arXiv:2410.20427, 2024
2. Tanaka et al., "3D Pose-Based Temporal Action Segmentation for Figure Skating", arXiv:2408.16638, ACM MM Workshop 2024
3. Li et al., "MS-TCN++: Multi-Stage Temporal Convolutional Network for Action Segmentation", TPAMI 2020, arXiv:2006.09220
4. Yi et al., "ASFormer: Transformer for Action Segmentation", BMVC 2021, arXiv:2110.08568
5. Liu et al., "Diffusion Action Segmentation", ICCV 2023, arXiv:2303.17959
6. Gong et al., "ActFusion: a Unified Diffusion Model for Action Segmentation and Anticipation", NeurIPS 2024, arXiv:2412.04353
7. Ding et al., "Temporal Action Segmentation: An Analysis of Modern Techniques", TPAMI 2023, arXiv:2210.10352
8. Liu et al., "Fine-grained Action Analysis: A Multi-modality and Multi-task Dataset of Figure Skating (MMFS)", ACM MM 2023, arXiv:2307.02730
9. Ji et al., "FineFS: Fine-grained Figure Skating Dataset", ACM MM 2023
10. Liu et al., "MCFS: Temporal Segmentation of Fine-grained Semantic Action", AAAI 2021
11. Gokay et al., "Skeleton Motion Words for Unsupervised Skeleton-Based TAS", ICCV 2025
12. mayupei, "Action Segmentation of Figure Skating Competition Videos: A Skeleton-Based Approach", 2025
13. STGA-Net, "Spatial-Temporal Graph Attention Network for Skeleton-Based TAS", IEEE 2023
