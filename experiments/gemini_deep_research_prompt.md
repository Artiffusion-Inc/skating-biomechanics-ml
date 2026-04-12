# Gemini Deep Research Prompt: Figure Skating Element Classification from Pose Skeletons

**Copy everything below this line into Gemini Deep Research.**

---

## Task

I need your help breaking through a 67.5% accuracy ceiling on figure skating element classification from 2D pose skeleton sequences. My target is 80% accuracy on 64 classes. I have exhausted local experimentation and need a comprehensive literature review and analysis of the latest (2024-2026) approaches that could help.

## System Context

I am building an AI coach for figure skating that analyzes video and provides biomechanical feedback. The classification component takes 2D pose skeleton sequences (COCO 17 keypoints) and classifies which figure skating element is being performed (e.g., triple lutz, camel spin, step sequence).

**Architecture:** Video → RTMPose (HALPE26, 26kp) → H3.6M conversion (17kp) → BiGRU classifier → element label

The system is NOT video-based classification. It is skeleton-based classification. The pose estimation is a separate solved component. I only need to classify from skeleton sequences.

## What I Have

### Datasets
1. **FSC (Figure-Skating-Classification)** — 5168 sequences, 64 element classes, COCO 17kp, 150 frames @ 30fps (5s clips), normalized to [-1,1]. Train/test split: 4161/1007. **Highly imbalanced:** min=8, max=406, mean=65 samples per class. This is my primary dataset.
2. **MCFS (Multi-Modal Figure Skating)** — 26198 sequences, 256 categories, COCO 17kp, variable-length, unnormalized. I extracted 2668 segments with frame-level labels (129 classes).
3. **MMFS** — 4915 sequences, 63 element categories, COCO 17kp. Used for cross-dataset evaluation (65.4% BiGRU).
4. **FineFS** — 1167 samples with GOE quality scores (used for regression, MAE=0.78).
5. **AthletePose3D** — 1.3M frames, 12 sports, 71GB. Contains figure skating 3D poses (subset: FS-Jump3D).

### Current Best Model
- **BiGRU (bidirectional GRU)** with packed sequences, 2 layers, hidden=128, dropout=0.3
- Input: raw (x,y) positions, 34-dim per frame (17 keypoints × 2)
- **Best accuracy: 67.5%** on FSC 64-class test set (with joint angle features: 11 angles + 34 raw = 45-dim input)
- Baseline without angles: 63.9%

## Complete Experiment Log (What I've Tried)

### Confirmed Findings
1. **BiGRU with packed sequences** is the correct architecture — 3× better than 1D-CNN with truncation/padding (61.6% vs 21.8%)
2. **Class imbalance is THE primary bottleneck** — Top-10 classes: 81.5%, all-64: 67.5%. Same model, same preprocessing.
3. **Start-crop >> center-crop** for figure skating (elements begin early in clip)
4. **Joint angle features help** (+3.6pp) — internal body angles carry discriminative signal missing from raw (x,y)

### Rejected Approaches
5. **ST-GCN (Spatial-Temporal Graph Convolutional Network)** — 27.4% (broken, no learnable params) and 28.5% (proper, learnable partitions). **GCN is fundamentally broken for this task.** Reason: over-smoothing on 17-node graph, skeleton topology too simple, temporal dynamics >> spatial structure.
6. **Transformer encoder** — 51.2% vs BiGRU 63.9% (−12.7pp). Subsampling long sequences to fixed max_len=300 destroys temporal structure. Strong overfitting (val gap −1.15).
7. **Hierarchical classification** (type→element: jump/spin/step/lift/other → specific element) — 64.2% (+0.3pp, noise). Type information already captured implicitly by flat BiGRU.
8. **Attention pooling** (replace last-hidden-state with attention-weighted sum) — 64.8% (+0.9pp, within noise).
9. **Velocity/acceleration features** (pos+vel+acc = 102-dim) — 64.2% (+0.3pp). BiGRU hidden state already captures temporal dynamics.
10. **Data augmentation** — mirror harmful (direction-dependent elements), joint noise no gain, SkeletonMix no gain. Best augmentation: +0.6pp only.
11. **Contrastive pre-training** (frame-level, on 10% data) — only +1.8pp (22.3% vs 20.6%), far below expected +10pp.
12. **Transfer learning FSC→MCFS** — 49.3% vs 63.9% from scratch. Encoder hurts.
13. **Cosine similarity of raw poses** — within-class similarity 0.557, between-class 0.467, gap only 0.09. Raw pose similarity is useless.

### Cross-Dataset
- **MMFS (63 classes):** 65.4% BiGRU — similar to FSC, confirming dataset quality is comparable
- **MCFS (48 filtered classes, ≥10 samples):** 60.1% BiGRU — lower due to more classes with fewer samples

## Key Constraints

1. **Input format is fixed:** (T, 17, 2) or derived features. Cannot change the pose estimator (RTMPose is already optimal for our setup).
2. **Real-time requirement:** Classification must happen within seconds, not minutes. Heavy models (TCPFormer 422MB) are too slow.
3. **Small dataset:** ~5000 training samples for 64 classes. Any approach must handle small data gracefully.
4. **Variable-length sequences:** 150 frames nominal, but real data varies. Must handle this.
5. **No multi-camera:** Single phone video only. Cannot use multi-view 3D reconstruction.

## What I Need Help With

### Priority 1: How to reach 80% accuracy on 64-class skeleton classification?

Given the constraints above (small data, single modality, 17 keypoints), what approaches from 2024-2026 literature could realistically push accuracy from 67.5% to 80%? Specific questions:

- **Pre-training strategies:** What pre-training paradigms work best for small skeleton classification datasets? VIFSS (arxiv:2508.10281) uses Barlow Twins on 3D poses. MotionBERT pretrains on massive mocap. Are there better approaches?
- **Feature engineering:** Joint angles gave +3.6pp. What other skeleton-derived features could help? Bone lengths, joint velocities, angular momentum proxies, phase-based features?
- **Architecture:** BiGRU works but plateaus at 67.5%. Is there an architecture that handles both variable-length sequences AND small data better? Mamba/SSM? Lightweight attention?
- **Data efficiency:** With ~65 samples/class average, what few-shot or meta-learning approaches could help for tail classes (8-20 samples)?
- **Multi-dataset training:** Can FSC + MCFS + MMFS be combined effectively despite different label spaces? How to handle label mismatch?

### Priority 2: Temporal Action Segmentation for Figure Skating

I need to segment full skating routines into individual elements. Currently I only have pre-segmented 5-second clips. For production use, I need to:
- Detect element boundaries in continuous video (30+ minutes)
- Classify each segment
- Handle transitions (no clear boundary between preparation and element)

What are the best approaches for temporal action segmentation specifically adapted to figure skating? MS-TCN++? ASFormer? Segment-level contrastive learning?

### Priority 3: 3D Pose Reconstruction from Single Camera

I currently use 2D poses. 3D could provide view-invariance. Options I'm aware of:
- MotionBERT (2D→3D lifting, pretrained)
- MotionAGFormer-S (38.4mm MPJPE, currently integrated)
- VIFSS JointFormer encoder (view-invariant features from 3D)

Would 3D significantly help classification accuracy? Is the 2D→3D lifting quality sufficient from single camera phone video?

### Priority 4: Quality Assessment (GOE Estimation)

Beyond classification, I want to estimate element quality (GOE score). Current Ridge regression MAE=0.78 GOE but correlation is weak (0.206). What approaches could improve this? Contrastive regression (CoRe)? Reference-based DTW comparison?

## Specific Papers/Methods I'm Aware Of

Please analyze these and suggest others I may have missed:

- **VIFSS** (arxiv:2508.10281) — View-invariant features for figure skating, JointFormer + Barlow Twins
- **MotionBERT** — 2D→3D lifting pretrained on AMASS/3DPW, strong for downstream tasks
- **InfoGCN** — Information bottleneck for skeleton action recognition, 89.8% NTU120
- **MS-TCN++** — Temporal action segmentation, multi-stage refinement
- **Mamba Pyramid** (ACM MM 2025) — Action quality assessment for figure skating, I3D+audio
- **CoRe** (ICCV 2021) / **HP-MCoRe** (IEEE TIP 2025) — Contrastive regression for quality assessment
- **SkelFormer** — Skeleton action recognition with partition attention
- **YourSkatingCoach** (arxiv:2410.20427) — BIOES tagging for element boundaries
- **OOFSkate (MIT)** — Body kinematics proxy features for quality, deployed at 2026 Olympics

## What Would Change Everything

If you find any of the following, it would be a breakthrough:

1. **A larger figure skating skeleton dataset** (10K+ sequences, labeled elements) that I haven't found
2. **A pre-trained skeleton encoder** specifically for sports/fine motor actions (not just NTU RGB+D which is daily actions)
3. **A method that reliably achieves >75% on 50+ classes with <10K training samples** from skeleton data
4. **An approach to combine multiple datasets** (FSC+MCFS+MMFS) despite label space mismatch that actually improves over single-dataset training
5. **Evidence that 3D reconstruction from single camera** significantly improves classification accuracy (not just theoretically)

## Output Format

Please structure your response as:
1. **Key findings** with paper citations (arxiv IDs where possible)
2. **Ranked recommendations** with estimated effort and expected accuracy improvement
3. **Datasets** I may have missed
4. **Architecture suggestions** with code references (GitHub repos)
5. **Concrete next experiments** I should run, prioritized by expected impact
