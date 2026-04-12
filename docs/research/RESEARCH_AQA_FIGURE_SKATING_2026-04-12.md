# Action Quality Assessment for Figure Skating: Deep Research

**Date:** 2026-04-12
**Context:** We need to predict GOE (Grade of Execution) scores for figure skating elements from skeleton poses. Current Ridge regression on frame statistics (mean+std position, duration) achieves MAE=0.78 GOE with correlation=0.206 on FineFS (6065 element segments, 3D skeletons). This is barely above baseline (MAE=0.80). We need MUCH better.

---

## 1. Mamba Pyramid (ACM MM 2025) — Two-Stream Mamba Pyramid Network

**Paper:** "Learning Long-Range Action Representation by Two-Stream Mamba Pyramid Network for Figure Skating Assessment" (Wang et al., ACM MM 2025)
**Code:** https://github.com/ycwfs/Figure-Skating-Action-Quality-Assessment (MIT license, Python + CUDA)

### Architecture

The architecture is a **two-stream network** that separates TES (Technical Element Score) and PCS (Program Component Score) evaluation, aligned with actual judging criteria:

1. **Feature Extractors:** Pre-trained I3D (RGB, Kinetics-400) and VGGish (audio, AudioSet). Features projected to 1024-dim embedding space.

2. **Temporal Hierarchical Feature Encoder (THFE):**
   - **Temporal Embedding Module (TEM):** 1D conv + ReLU + LayerNorm, transforms features into high-dimensional temporal embeddings
   - **Temporal Refinement Module (TRM):** Stack of Masked Mamba Blocks that preserve temporal resolution through masking

3. **Multi-scale Mamba Pyramid (MMP):** Stacks MDS (Mamba Down Sampling) blocks with increasing strides:
   - Each MDS: `MaxPool(Mamba(LN(F)) + DropPath(F))` — combines temporal feature extraction, residual connections, and max-pooling
   - Produces multi-scale pyramidal features {F1, F2, ..., FL} at decreasing temporal resolutions
   - 6-level pyramid with score regression ranges: [0,4], [4,8], [8,16], [16,32], [32,64], [64,+inf]

4. **Two-Stream Design:**
   - **TES Stream (visual-only):** I3D features -> THFE -> MMP -> TES Head (1D conv layers predict category, temporal offsets, score per time point)
   - **PCS Stream (visual+audio):** Audio features fused via Multi-level Cross Attention Fusion (MCAF) at each pyramid level; PCS Head on fused features

5. **TES Head:** Shared 1D conv backbone with separate final layers for: (a) action categories, (b) temporal offsets (start/end), (c) action scores

6. **Inference:** Soft-NMS to remove overlapping segments; top-7 (short program) or top-12 (free skating) segments selected; TES = sum of element scores.

### Key Results on FineFS

| Method | Free Skating rho_TES | rho_PCS | Short Program rho_TES | rho_PCS |
|--------|---------------------|---------|----------------------|---------|
| GDLT | 0.63 | 0.80 | 0.64 | 0.78 |
| MS-LSTM | 0.55 | 0.62 | 0.55 | 0.60 |
| TSA | 0.68 | 0.78 | 0.53 | 0.78 |
| LUSD-Net | 0.78 | 0.86 | 0.69 | 0.81 |
| **Mamba Pyramid** | **0.80** | **0.96** | **0.75** | **0.94** |

**Important caveats:**
- These are **Spearman rank correlation** (rho), NOT MAE or R^2
- Task is **whole-program scoring** (TES/PCS for entire skating program), NOT per-element GOE
- Uses **I3D video features** (RGB), not skeleton-only
- Audio modality critical for PCS (music interpretation)
- Training: 50 epochs, batch 8, NVIDIA A800, lr=1e-3, Adam

### Transfer results (zero-shot, no retraining):

| Dataset | rho_TES | rho_PCS |
|---------|---------|---------|
| Fis-V | 0.79 | 0.87 |
| FS1000 | 0.85 | 0.91 |

### Ablation insights:
- More categories = better TES but worse temporal localization (22 categories: TES=0.77, avg tIOU=67.53)
- Audio fusion only helps PCS, not TES (confirming TES is visual-only by judging rules)

---

## 2. OOFSkate (MIT Sports Lab, 2026)

**Founders:** Jerry Lu MFin '24, Prof. Anette "Peko" Hosoi (MIT Sports Lab)
**App:** OOFSkate on iOS App Store, partnered with U.S. Figure Skating (Dec 2025)
**Deployment:** NBC Sports broadcast analytics at 2026 Winter Olympics

### What OOFSkate Does

From the MIT interview and app description, OOFSkate:
- Takes a **video recording** of a figure skater's jump
- Outputs **physical metrics** that drive rotation count:
  - Jump height
  - Angular velocity
  - Peak angular velocity
  - Time to peak angular velocity
- **Automatic jump type classification** (Axel, Lutz, Flip, Loop, Salchow, Toe Loop)
- **Combination jump support** (per-jump stats in combos)
- **Comparison against professionals** ("elite and former elite athletes" database)

### GOE Estimation

Key quote from Jerry Lu: *"the automated classifier, which shows you if you did this trick at World Championships and it were judged by an international panel, this is approximately the grade of execution score they would give you."*

**Critical insight:** OOFSkate does NOT publish their GOE estimation algorithm. It is proprietary. However, from the description:

1. **Input features are physics-based:** height, angular velocity, peak angular velocity, time-to-peak
2. **GOE is approximated** by comparing against a database of elite performances
3. **No depth information needed:** Prof. Hosoi notes that for skating, "How high did this person jump, how many times did they go around, and how well did they land? None of those rely on depth."
4. The system relies on **pose estimators** (which work well for this application despite depth issues)

### Metrics Used (from App Store):
- Jump height (CoM trajectory)
- Angular velocity (deg/s or rev/s or rpm)
- Peak angular velocity
- Time to peak angular velocity

### NOT Used:
- No blade edge detection (consistent with our OOFSkate pivot)
- No 3D depth reconstruction
- No audio analysis (purely visual/pose)

### Limitations:
- **Closed-source / proprietary** — no algorithm details available
- Works best with **side-view or stable camera** (implied by their broadcast use)
- GOE estimation accuracy is **not published** — no MAE/correlation numbers

---

## 3. "From Beats to Scores" (CVPRW 2025) — Multi-Modal Framework

**Paper:** "From Beats to Scores: A Multi-Modal Framework for Comprehensive Figure Skating Assessment" (Wang et al., CVPRW 2025 at CVSPORTS)
**Code:** https://github.com/ycwfs/Figure-Skating-Quality-Assessment

### Architecture

This is the predecessor to the Mamba Pyramid paper, using Multi-modal Large Language Model (MLLM):

1. **Audio-Guide Sub-Action Segmentation:** Uses "hit the beat" phenomenon to segment skating movements by audio features
2. **Video-Audio Feature Fusion:** I3D (video) + VGGish (audio) features, projected to unified embedding
3. **Context Representation Learning:** Transformer decoder processes multi-modal tokens + special tokens + text prompt ("How did the athlete perform in the show?")
4. **Output:** TES regression, PCS regression, AND text-based evaluation description

### Results on FineFS

| Method | TES (SP) | PCS (SP) | TES (MSE) | PCS (MSE) |
|--------|----------|----------|-----------|-----------|
| MS-LSTM | 0.55 | 0.61 | - | - |
| TSA | 0.60 | 0.78 | - | - |
| GDLT | 0.64 | 0.79 | - | - |
| LUSD-Net | 0.73 | 0.84 | - | - |
| **Beats to Scores** | **0.76** | **0.88** | **97.95** | **26.21** |

### Ablation (modality contributions):

| Strategy | TES (SP) | PCS (SP) | TES (MSE) | PCS (MSE) |
|----------|----------|----------|-----------|-----------|
| Audio only | 0.65 | 0.69 | 101.32 | 29.82 |
| Video only | 0.71 | 0.74 | 99.85 | 28.98 |
| Video-guide | 0.72 | 0.84 | 98.52 | 27.89 |
| **Audio-guide** | **0.76** | **0.88** | **97.95** | **26.21** |

Key finding: **Audio-guide segmentation** (using music beats to find element boundaries) significantly helps both TES and PCS.

**No skeleton-only baseline reported.** All methods use I3D video features.

---

## 4. HP-MCoRe (IEEE TIP 2025) — Hierarchical Pose-guided Multi-stage Contrastive Regression

**Paper:** "Action Quality Assessment via Hierarchical Pose-guided Multi-stage Contrastive Regression" (Qi et al., IEEE Transactions on Image Processing, 2025)
**Code:** https://github.com/Lumos0507/HP-MCoRe

### Architecture

This is the most relevant paper for **skeleton-based AQA**:

1. **Multi-scale Dynamic Visual-Skeleton Encoder:**
   - Static visual encoder (processes video frames for context)
   - Dynamic visual encoder + Hierarchical Skeletal Encoder (captures fine-grained spatio-temporal differences)
   - The skeleton branch is the **core innovation** — provides "fine-grained sub-action features" and "physical priors"

2. **Procedure Segmentation Network:** Dynamically separates sub-actions (e.g., take-off, flight, entry in diving). NOT fixed-frame segmentation.

3. **Multi-modal Fusion Module:** Skeletal features serve as **physics structural priors** to guide visual feature learning. This is key — skeleton features don't just add signal, they STRUCTURE the visual representation.

4. **Multi-stage Contrastive Regression:**
   - Stage contrastive loss: learns discriminative representations between sub-actions
   - Unsupervised training on sub-action differences
   - The contrastive approach groups samples by quality level, then learns to rank within groups

### Results on FineDiving (diving, not skating, but demonstrates skeleton-only approach)

| Method (w/ dive numbers) | SRCC (rho) | R_L2 (x100) |
|--------------------------|-----------|-------------|
| CoRe (2021) | 0.9061 | 0.361 |
| TSA (2022) | 0.9203 | 0.342 |
| MCoRe (2024) | 0.9232 | 0.326 |
| **HP-MCoRe** | **0.9365** | **0.244** |

### Key Insight for Our Problem

The skeleton encoder provides **physical structure priors** — not just raw coordinates, but the HIERARCHICAL organization of body parts. The paper uses:
- Hierarchical skeletal encoder (body part tree structure)
- Stage-aware processing (different sub-actions need different attention)
- Contrastive learning (groups of similar quality → learn relative differences)

This is **the best skeleton-based AQA approach** currently published.

---

## 5. CoRe — Group-aware Contrastive Regression (ICCV 2021)

**Paper:** "Group-aware Contrastive Regression for Action Quality Assessment" (Yu et al., ICCV 2021)
**Code:** https://github.com/yuxumin/CoRe (81 stars)

### Core Idea

Instead of directly regressing absolute scores, CoRe:
1. **Groups samples by score range** (e.g., bins of quality levels)
2. **Learns relative differences within groups** using contrastive learning
3. **Uses group-level representations** to make the final regression

### Why It Works

- Absolute scores have high variance (judge subjectivity)
- Relative ranking within a quality group is more stable
- Contrastive loss pushes similar-quality samples together, different-quality samples apart
- The "group-aware" component handles the fact that quality distributions are non-uniform

### Results (FineDiving, with dive numbers):
- SRCC: 0.9061
- R_L2: 0.361

CoRe was the **foundational contrastive approach** that all subsequent methods build upon. HP-MCoRe extends it with pose guidance and multi-stage processing.

---

## 6. Skeleton-Based Figure Skating AQA (IEEE 2022)

**Paper:** "Skeleton Based Action Quality Assessment of Figure Skating Videos" (Xu et al., IEEE 2022/2023)

### Approach
- ST-GCN (Spatial-Temporal Graph Convolutional Network) for skeleton representation
- Pose estimation via OpenPose
- Evaluated on MIT-Skate and FIS-V datasets
- **Better than RGB-based methods** (SENet, C3D) for long-duration skating videos

### Key Finding
Skeleton-based methods are **more effective** than RGB video-based methods for figure skating AQA because they:
1. Exclude ambiguous scene/background information
2. Focus purely on action dynamics
3. Are more robust to camera angle changes

---

## 7. Quality Assessment Methods for Sports in General

### 7.1 Temporal Modeling Approaches

| Approach | Example | Best For | Key Idea |
|----------|---------|----------|----------|
| **TCN (Temporal Conv)** | MS-TCN, TSA | Segmentation | Multi-scale temporal convolutions |
| **ST-GCN** | Basic skeleton AQA | Skeleton data | Graph conv on skeleton graph |
| **Transformer/VST** | GDLT, Skating-Mixer | Long-range deps | Self-attention over time |
| **Mamba** | Mamba Pyramid (2025) | Long sequences | Linear complexity, selective state space |
| **LSTM** | MS-LSTM | Sequential | Recurrent temporal modeling |

### 7.2 Regression vs Ranking

The field has converged on **hybrid approaches**:

1. **Direct Regression** (Case 1): Output continuous score. Simple but suffers from judge subjectivity variance.
2. **Pairwise Comparison** (Case 2): Given two samples, predict which is better. More robust but needs pairs.
3. **Contrastive Regression** (Case 3, CoRe family): Group samples by quality, learn within-group and between-group differences. **Best of both worlds.**

The AQA survey (Zhou et al., 2024) confirms: **contrastive regression consistently outperforms direct regression** across all benchmarks.

### 7.3 Teacher-Student Distillation

- SAP-Net (2022): Teacher branch focuses on actor-centric region, generates pseudo-labels for student
- Effective when labeled data is scarce
- Could be useful for us: train on elements with high-quality GOE annotations, distill to elements with noisier labels

### 7.4 DTW-Based Comparison Against Reference

- QAQA (2025): Uses DTW for skeleton-based AQA with anomaly-aware acceleration
- **Indirect scoring:** Compare against reference execution, use similarity as quality proxy
- Computation: O(n) with multi-resolution approach (coarsening, projection, refinement)
- **This is exactly our DTW alignment module** — we already have this infrastructure

---

## 8. Key Question: What Features Carry Quality Signal Beyond Frame Statistics?

### 8.1 Features That Carry Quality Signal (from literature)

| Feature Type | Quality Signal | Evidence |
|-------------|---------------|----------|
| **Joint angle trajectories** (full time series, not mean+std) | Body posture correctness, technique quality | HP-MCoRe, ST-GCN papers |
| **Velocity profiles** (joint velocities over time) | Movement smoothness, timing precision | OOFSkate (angular velocity), QAQA (acceleration outliers) |
| **Acceleration profiles** | Jerk/smoothness = quality indicator | QAQA anomaly detection |
| **Phase timing** (takeoff duration, airtime, landing prep) | ISU judging criteria | TSA, FineParser |
| **CoM trajectory smoothness** | Jump quality, landing stability | OOFSkate (jump height), PhysicsEngine |
| **Angular momentum conservation** | Rotation quality | Physics literature |
| **Symmetry metrics** (left-right balance) | Technique quality | Biomechanics literature |
| **Sub-action segmentation quality** | Clean transitions between phases | HP-MCoRe procedure segmentation |
| **Relative timing** (ratio of phases) | Whether the element is "well-paced" | TSA temporal-aware modeling |

### 8.2 Why Current Features (mean+std position) Fail

The current Ridge regression uses 69 features:
- mean(x,y) for 17 joints (34 features)
- std(x,y) for 17 joints (34 features)
- duration (1 feature)

**Why this fails:**
1. **Mean position is dominated by camera angle** — same quality jump from different angles has wildly different mean positions
2. **Std captures overall spread, not quality** — a wide-armed skater and a poorly-balanced skater both have high std
3. **Duration doesn't distinguish quality** — same element, same duration, different GOE
4. **No temporal information** — WHEN things happen matters more than average WHERE they are
5. **No inter-joint relationships** — angles between joints (knee bend, arm position relative to torso) carry quality signal

### 8.3 Features That Should Work Better

**Tier 1 (easy to implement, high signal):**
- Joint angle time series (17 angles per frame -> T*17 features)
- Velocity magnitude per joint (smoothness proxy)
- Phase timing: takeoff_frame, peak_frame, landing_frame as ratio of total duration
- Angular velocity (from quaternion or cross-product of joint vectors)

**Tier 2 (moderate effort, strong signal):**
- CoM trajectory (already computed by PhysicsEngine)
- CoM velocity and acceleration profiles
- Jerk (derivative of acceleration) — smoothness metric
- Bone length consistency — proxy for tracking quality / stability
- Left-right symmetry — difference between corresponding left/right joints

**Tier 3 (needs model, but proven effective):**
- DTW distance to reference execution (we have DTW infrastructure)
- Learned embeddings from ST-GCN on skeleton
- Contrastive features from CoRe-style grouping

---

## 9. FineFS Dataset Usage in Literature

### FineFS Statistics
- **1,167 samples** total (729 short program, 438 free skating)
- 570 male, 597 female athletes
- 25 fps, ~5000 frames per video (2-4 minutes)
- Annotations: **per-element** with timing, category, and scores
- Skeletons: 2D and 3D estimated via pose estimation
- Splits: Official train/test split provided

### How Papers Use FineFS

| Paper | Task | Metric | Input | Best Result |
|-------|------|--------|-------|-------------|
| FineFS original (Ji 2023) | Quality + Segmentation | SP, mAP | Video + Skeleton | Skeleton baselines reported |
| LUSD-Net | Whole-program TES/PCS | SP (rho) | I3D video features | TES=0.73, PCS=0.84 |
| From Beats to Scores | Whole-program TES/PCS + text | SP, MSE | I3D + VGGish audio | TES=0.76, PCS=0.88 |
| Mamba Pyramid | Whole-program TES/PCS | SP | I3D + VGGish | TES=0.80, PCS=0.96 |

**CRITICAL GAP:** No paper reports **per-element GOE prediction** on FineFS skeletons. All papers predict **whole-program TES** (sum of element scores) from **video features** (I3D). Our task (per-element GOE from skeleton) is genuinely novel and under-explored.

### FineFS Baselines from Original Paper
The original FineFS paper (Ji et al., 2023) reports:
- RGB-based and skeleton-based baselines for action recognition and action quality assessment
- Skeleton baselines include ST-GCN and similar approaches
- The dataset provides per-element annotations (timing, category, score) which makes per-element prediction possible

---

## 10. Regression vs Ranking: Recommendation

### Evidence from Literature

| Approach | Pros | Cons | Best Performance |
|----------|------|------|-----------------|
| **Direct Regression** | Simple, interpretable | Sensitive to annotation noise | Baseline only |
| **Pairwise Ranking** | Robust to noise | Needs pair construction, O(n^2) | Good for small datasets |
| **Contrastive Regression (CoRe)** | Best of both, handles quality distribution | More complex training | **SOTA across all benchmarks** |

### Recommendation: Contrastive Regression

For our FineFS per-element GOE prediction:
1. **Group elements by type** (axel, lutz, etc.) — quality meaning differs across element types
2. **Within each element type, use CoRe-style contrastive regression:**
   - Group by GOE range (e.g., bins of [-5,-3], [-2,0], [1,3], [4,+])
   - Learn within-group and between-group differences
   - Final regression head on contrastive features
3. **Add ordinal loss** to encourage the model to respect GOE ordering

Rationale: GOE scores from judges have known high variance (different judges give different scores for same performance). Contrastive regression is specifically designed to handle this.

---

## 11. Clear Recommendation: What to Try Next

Given our data (FineFS 6065 segments with 3D skeletons + GOE scores) and the research findings:

### Priority 1: Better Feature Engineering (1-2 days, highest ROI)

**Replace mean+std position features with:**

```
Per-segment features (T frames, 17 joints, 3D):
1. Joint angle time series: 17 angles * T frames
   - knee angles (hip-knee-ankle), elbow, shoulder, torso lean, head tilt
   - Use functional summary: statistical moments + DCT coefficients

2. Phase-relative timing (3 values):
   - takeoff_frame / T
   - peak_height_frame / T
   - landing_frame / T

3. Global quality features:
   - CoM height range (max - min)
   - CoM velocity smoothness (jerk integral)
   - Angular velocity peak and timing
   - Bone length variance (tracking stability)
   - Left-right symmetry score

4. DTW distance to reference:
   - We already have DTW alignment infrastructure
   - Use median GOE execution per element type as reference
   - DTW distance as a quality proxy feature
```

**Expected improvement:** From MAE=0.78, correlation=0.206 to potentially MAE<0.6, correlation>0.4

### Priority 2: ST-GCN or Lightweight Temporal Model (2-3 days)

```
Input: (T, 17, 3) skeleton sequence per segment
Model:
  1. Spatial: Graph conv on H3.6M skeleton graph (17 nodes)
  2. Temporal: TCN or small Transformer (4-6 layers)
  3. Head: Regression with MSE + contrastive loss
  4. Per-element-type models (separate for axel, lutz, etc.)
```

**Reference implementations:**
- ST-GCN: https://github.com/lxtGH/st-gcn
- HP-MCoRe skeleton branch: https://github.com/Lumos0507/HP-MCoRe

**Expected improvement:** MAE<0.5, correlation>0.5

### Priority 3: Contrastive Regression Framework (3-4 days)

```
1. Group FineFS elements by type + GOE bin
2. Use CoRe-style group-aware contrastive loss
3. Combine with ST-GCN encoder from Priority 2
4. Add ordinal regression component
```

**Expected improvement:** MAE<0.4, correlation>0.6

### Priority 4: Reference-Based DTW Quality Score (1-2 days)

```
1. For each element type, find top-10 GOE executions as "references"
2. DTW-align each segment against each reference
3. Use mean DTW distance as quality feature
4. Combine with learned features from Priority 2/3
```

This is the **most biomechanically principled** approach and aligns with how coaches actually evaluate (comparison against ideal technique).

---

## 12. Summary of Key Findings

1. **No paper does per-element GOE prediction from skeletons on FineFS.** All existing work predicts whole-program TES from I3D video features. Our task is novel.

2. **Skeleton-based AQA works better than video for sports** (IEEE 2022 figure skating paper) — this validates our skeleton-only approach.

3. **The quality signal is in temporal dynamics, not spatial averages.** Joint angle trajectories, velocity profiles, phase timing, and CoM smoothness carry the quality signal. Mean+std position is nearly useless.

4. **Contrastive regression (CoRe) is the dominant paradigm** for AQA — it handles judge subjectivity better than direct regression.

5. **Procedure segmentation matters.** HP-MCoRe shows that segmenting sub-actions (takeoff, flight, landing) and processing them separately significantly improves quality prediction.

6. **OOFSkate uses physics-based metrics** (height, angular velocity, peak timing) and compares against an elite athlete database. Their approach validates our proxy-feature direction.

7. **The Mamba Pyramid (SOTA on FineFS)** uses video+audio features, not skeleton. Their architecture is impressive for whole-program scoring but not directly applicable to our per-element skeleton task.

8. **DTW-based comparison against reference** is a proven, principled approach for quality assessment (QAQA 2025) and we already have the infrastructure.

---

## References

1. Wang et al. "Learning Long-Range Action Representation by Two-Stream Mamba Pyramid Network for Figure Skating Assessment" ACM MM 2025. [arXiv:2508.16291](https://arxiv.org/abs/2508.16291)
2. Wang et al. "From Beats to Scores: A Multi-Modal Framework for Comprehensive Figure Skating Assessment" CVPRW 2025. [Code](https://github.com/ycwfs/Figure-Skating-Quality-Assessment)
3. Qi et al. "Action Quality Assessment via Hierarchical Pose-guided Multi-stage Contrastive Regression" IEEE TIP 2025. [Code](https://github.com/Lumos0507/HP-MCoRe)
4. Yu et al. "Group-aware Contrastive Regression for Action Quality Assessment" ICCV 2021. [Code](https://github.com/yuxumin/CoRe)
5. Zhou et al. "A Comprehensive Survey of Action Quality Assessment: Method and Benchmark" 2024. [arXiv:2412.11149](https://arxiv.org/abs/2412.11149)
6. Ji et al. "FineFS: Fine Figure Skating Dataset" 2023. [Code](https://github.com/yanliji/FineFS-dataset)
7. Fu et al. "Skeleton-Based Action Quality Assessment with Anomaly-Aware DTW Optimization" Sensors 2025. [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC12693942/)
8. Chen et al. "YourSkatingCoach: A Figure Skating Video Benchmark for Fine-Grained Element Analysis" 2024. [arXiv:2410.20427](https://arxiv.org/abs/2410.20427)
9. Xu et al. "Skeleton-based deep pose feature learning for action quality assessment on figure skating videos" JVCIR 2022. [DOI](https://doi.org/10.1016/j.jvcir.2022.103625)
10. MIT News. "3 Questions: Using AI to help Olympic skaters land a quint" 2026. [MIT](https://news.mit.edu/2026/3-questions-using-ai-help-olympic-skaters-land-quint-0210)
