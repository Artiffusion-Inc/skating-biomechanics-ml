---
title: "Skeleton-Based Action Recognition: Deep Research (2024-2026)"
date: "2026-04-12"
status: active
---
# Skeleton-Based Action Recognition: Deep Research (2024-2026)

**Date:** 2026-04-12
**Context:** Classifying figure skating elements from H3.6M 17-keypoint pose skeletons. BiGRU 63.9% on 64 classes (FSC), GCN 28.5%. Why does RNN dominate and what can we do better?

---

## 1. NTU RGB+D Leaderboard (Skeleton-Only, 2024-2026)

The canonical benchmark for skeleton action recognition. All methods below use only skeleton data.

### NTU RGB+D 60 (56K videos, 60 classes)

| Method | Venue | X-Sub | X-View | Architecture |
|--------|-------|-------|-------|-------------|
| ST-GCN | AAAI 2018 | 81.5 | 88.3 | Spatio-temporal GCN |
| 2s-AGCN | CVPR 2019 | 88.5 | 95.1 | Adaptive GCN, 2-stream |
| Shift-GCN | CVPR 2020 | 90.7 | 96.5 | Shift graph convolution |
| MS-G3D | CVPR 2020 | 91.5 | 96.2 | Multi-scale 3D GCN |
| CTR-GCN | ICCV 2021 | 92.4 | 96.8 | Channel-wise topology GCN |
| **InfoGCN** | **CVPR 2022** | **93.0** | **97.1** | IB + SA-GC, 6-ensemble |
| **MotionBERT (finetune)** | **ICCV 2023** | **93.0** | **97.2** | Dual-stream Transformer |
| **HD-GCN** | **ICCV 2023** | **~90.5** | **~96.5** | Hierarchical decomposition |
| **SkelFormer** | **PLOS One 2026** | **92.8** | **96.7** | Hierarchical Transformer (4S) |
| **ActionMamba** | **Electronics 2025** | **91.8** | **96.7** | Mamba + GCN hybrid |
| 3MFormer | CVPR 2023 | 94.8 | 98.7 | Multi-order multi-mode Transformer |
| **BlockGCN** | **CVPR 2024** | **~91.0** | **~96.5** | Block partition topology |

### NTU RGB+D 120 (114K videos, 120 classes)

| Method | X-Sub | X-Set | Notes |
|--------|-------|-------|-------|
| ST-GCN | 70.7 | 73.2 | Baseline |
| 2s-AGCN | 82.9 | 84.9 | |
| Shift-GCN | 85.9 | 87.6 | |
| MS-G3D | 86.9 | 88.4 | |
| CTR-GCN | 88.9 | 90.6 | |
| InfoGCN (4-ensemble) | 89.8 | 91.2 | IB + SA-GC |
| SkelFormer (4S) | 89.4 | 91.0 | Hierarchical Transformer |
| ActionMamba (2s) | 86.5 | 88.1 | Mamba+GCN |

**Key observation:** The gap between X-Sub (70.7 -> ~90%) and X-Set (73.2 -> ~91%) is consistent. All top methods require multi-stream ensembles (joint + bone + motion) to reach SOTA.

---

## 2. Method-by-Method Analysis

### 2.1 SkelFormer (PLOS One, 2026)

**Not CVPR 2024** -- this is a 2026 PLOS One paper by Yan et al. (NUDT / Sun Yat-sen University).

**Architecture:**
- **Hierarchical 3-stage pipeline** with SKT (Skeleton Transformer) Blocks
- Stage 1: 8 relay nodes (fine-grained joint interactions like elbow-wrist)
- Stage 2: 4 relay nodes (regional: arm-trunk coupling)
- Stage 3: 2 relay nodes (global: full-body semantics)
- Between stages: temporal merging via 1D conv (stride=2, kernel=2), halving sequence length

**SKT Block components:**
1. **LA-GAT** (Learnable-Adjacent Graph Attention): Learns adjacency matrix initialized from physical skeleton topology. Uses attention-based QK projection (no V projection -- deferred to SKT Block).
2. **Temporal Split**: Divides long sequences into non-overlapping windows of T frames. Alternates segmentation offset across layers to maintain temporal connections.
3. **Node Concentration**: Selects anchor nodes by temporal persistence (consistently high attention scores across frames). Groups remaining nodes via HardMax similarity to anchors. Averages features within groups into relay nodes.
4. **Node Diffusion**: Temporal Relay Attention (TRA) computes dependencies among relay nodes across time. Value matrix comes from *original* node features (not relay nodes). Broadcasts refined context back to all nodes in each group.

**Key innovation:** Dynamic, action-specific node grouping that changes across temporal windows. For "jumping": standing phase groups differently than leaping phase.

**Results:** 92.8% (NTU60 4S), 89.4% (NTU120 4S X-Sub), 96.1% (NW-UCLA Joint)

**Why relevant to skating:** The dynamic node grouping is ideal for figure skating where different body parts dominate different phases (preparation vs. air vs. landing). The hierarchical abstraction (8->4->2 relay nodes) naturally captures the fine-grained vs. coarse motion patterns needed to distinguish similar elements.

**Code:** Not publicly available as of this writing.

---

### 2.2 HD-GCN (ICCV 2023)

**Paper:** "Hierarchically Decomposed Graph Convolutional Networks for Skeleton-Based Action Recognition" (Lee et al., Yonsei University)

**Architecture:**
- **HD-Graph**: Decomposes every joint into multiple sets to extract structurally adjacent AND distant edges within the same semantic space
- **Partition strategy**: Each joint belongs to multiple partitions (e.g., "left arm", "full body", "lower body"). Edges within each partition form a sub-graph.
- **A-HA Module** (Attention-guided Hierarchy Aggregation): Highlights dominant hierarchical edge sets
- **6-way ensemble**: Joint + Bone streams, each with 3 different center-of-mass (CoM) reference points. No motion stream needed.

**Key insight:** Standard GCN only uses physical adjacency (edges = bones). HD-GCN adds semantic edges between joints that are functionally related but physically distant (e.g., left wrist and right ankle in a jumping action). The hierarchical decomposition ensures edges are in the same semantic space.

**Results:** SOTA on NTU60, NTU120, NW-UCLA (2023)

**Why relevant to skating:** The semantic edge concept is crucial. In figure skating, the relationship between the skating foot and the free leg, or between arms and core, changes across elements. Fixed physical adjacency misses these dynamic functional relationships.

**Code:** https://github.com/Jho-yonsei/hd-gcn (MIT license, PyTorch)

---

### 2.3 InfoGCN (CVPR 2022)

**Paper:** "InfoGCN: Representation Learning for Human Skeleton-Based Action Recognition" (Chi et al., Purdue / KAIST)

**Architecture:**
- **Information Bottleneck (IB) objective**: Learns latent representations that are maximally informative about the target (action class) while compressing input information
  - Loss = Cross-entropy + Marginal MMD + Conditional-class MMD
  - MMD (Maximum Mean Discrepancy) forces class-conditional means to be orthogonal in latent space
  - Creates well-separated, discriminative class clusters
- **SA-GC** (Self-Attention Graph Convolution): Context-dependent intrinsic topology
  - Combines learned shared topology (like 2s-AGCN) WITH self-attention map
  - SA-GC infers *asymmetric* attention weights between joints that depend on behavioral context
  - Same pose in different actions gets different topologies
- **Multi-modal representation**: Generalizes joint/bone to K-mode representation (iterative bone-bone-bone). 6-ensemble uses modes {1, 2, K} for both joint and bone.

**Results:** 93.0% NTU60 X-Sub, 89.8% NTU120 X-Sub, 97.0% NW-UCLA

**Over-smoothing prevention:** The IB objective acts as a strong regularizer. The MMD loss prevents the latent space from collapsing to a single point (which is what over-smoothing causes in deep GCNs). Class-conditional means are forced to be orthogonal, maintaining discriminative power even with 9-layer deep encoder.

**Why relevant to skating:** The context-dependent topology is exactly what we need. A "flip" takeoff position looks similar to a "lutz" takeoff, but the behavioral context (which joints lead the motion) differs. InfoGCN's SA-GC would capture this. The IB objective also helps with our small dataset by preventing overfitting.

**Code:** https://github.com/stnoah1/infogcn (137 stars)

**Successor:** InfoGCN++ (arXiv 2310.10547, 2023) adds future prediction for online recognition.

---

### 2.4 MotionBERT (ICCV 2023)

**Paper:** "MotionBERT: A Unified Perspective on Learning Human Motion Representations" (Zhu et al., Peking University)

**Architecture:**
- **DSTformer** (Dual-stream Spatio-temporal Transformer): Two parallel branches with different S-T orderings, fused with adaptive weights
  - Branch 1: Spatial MHSA -> Temporal MHSA (S-T)
  - Branch 2: Temporal MHSA -> Spatial MHSA (T-S)
  - Adaptive fusion weights predicted per-node per-timestep
- **Pretrain-finetune paradigm:**
  - Pretrain: Recover 3D motion from corrupted 2D skeletons (mask + noise)
  - Data: Human3.6M + AMASS (3D mocap) + PoseTrack + InstaVariety (in-the-wild 2D)
  - Finetune: Add 1-2 layer head for downstream task
- **Config:** Depth N=5, heads h=8, feature size Cf=512, embedding Ce=512, sequence length T=243

**For action recognition:**
- Finetune on NTU60: 93.0% X-Sub, 97.2% X-View (comparable to InfoGCN with single-stream!)
- One-shot on NTU120: 67.4% (vs 61.0% scratch, 54.2% previous SOTA)
- **Key advantage:** Pretrained representations transfer extremely well to novel classes with minimal data

**Why relevant to skating:** This is potentially the most impactful approach for our use case. The pretrain-finetune paradigm means we could:
1. Pretrain on large-scale mocap data (Human3.6M, AMASS) which has diverse human motions
2. Finetune on our FSC dataset (5168 sequences, 64 classes) with very few epochs
3. The 2D skeleton input format matches our pipeline perfectly

The one-shot result (67.4% with 1 example per class) is particularly relevant given our limited per-class samples.

**Code:** https://github.com/Walter0807/MotionBERT (1.4K stars, well-maintained)

---

### 2.5 STTFormer (arXiv, 2022)

**Paper:** "Spatio-Temporal Tuples Transformer for Skeleton-Based Action Recognition" (Qiu et al., Xidian University)

**Architecture:**
- Divides skeleton sequence into parts (consecutive frame windows)
- **Spatio-temporal tuples self-attention**: Captures relationships between different joints *across* consecutive frames (not just within-frame or across-time separately)
- **Feature aggregation module**: Enhances ability to distinguish similar actions by aggregating features from non-adjacent frames

**Key innovation:** Standard spatial attention operates within a single frame. Standard temporal attention operates on a single joint across time. STTFormer's "tuple" attention operates on (joint_i, frame_t) x (joint_j, frame_t+k) pairs, capturing cross-joint cross-frame dependencies.

**Results:** Better than ST-GCN, Shift-GCN on NTU benchmarks (2022 era)

**Why relevant to skating:** Cross-joint cross-frame attention is exactly what distinguishes figure skating elements. The timing relationship between the skating foot planting and the free leg swinging is critical for identifying jump types.

**Code:** https://github.com/heleiqiu/STTFormer (62 stars, MIT license)

---

### 2.6 BlockGCN (CVPR 2024)

**Paper:** "BlockGCN: Redefining Topology Awareness for Skeleton-Based Action Recognition" (Zhou et al., Microsoft Research / CMU)

**Architecture:**
- **BlockGC structure**: Divides the feature dimension into multiple groups. Within each group, applies spatial aggregation and feature projection in parallel.
- **Topology-aware design**: Uses graph distance to encode physical topology, preserving important topological details lost in traditional GCNs
- Addresses "topology agnosticism" -- the problem that standard GCN convolutions treat all neighbor contributions equally regardless of their structural role

**Key innovation:** Traditional GCNs mix all channel information during spatial aggregation, losing topology awareness. BlockGCN processes channel groups independently, preserving the structural semantics of different feature dimensions.

**Results:** Higher accuracy with fewer parameters than existing methods on NTU benchmarks

**Why relevant:** The topology preservation aspect is relevant but the method is primarily designed for the standard benchmark datasets. Less directly applicable to our small-data regime.

**Code:** https://github.com/ZhouYuxuanYX/BlockGCN (132 stars, Apache 2.0)

---

### 2.7 ActionMamba (Electronics, 2025)

**Paper:** "ActionMamba: Action Spatial-Temporal Aggregation Network Based on Mamba and GCN" (Wen et al., North University of China)

**Architecture:**
- **ACE** (Action Characteristic Encoder): Combines inner spatio-temporal attention embedding with external space (XYZ coordinate) embedding
- **SM-GCN Block** (Shift Mamba-GCN): Fuses Shift-GCN with Mamba SSM for omnidirectional spatio-temporal mixing. Uses 3 scanning directions (spatial forward/backward + temporal) in Mamba blocks.
- **ST-Mamba Block** (Spatio-Temporal Mamba): Pure Mamba-based spatio-temporal fusion for long-range dependencies

**Key innovation:** First successful integration of Mamba (State Space Model) with GCN for skeleton action recognition. Achieves Transformer-quality results with linear complexity (5.94 GFLOPs for single stream vs 259.4 GFLOPs for ST-TR).

**Results:** 91.8% NTU60 X-Sub (2s), 86.5% NTU120 X-Sub (2s), 44.3% UAV-Human CSv1

**Why relevant:** The linear complexity is attractive for our deployment scenario. However, Mamba is relatively new and less battle-tested than Transformers for this task.

**Code:** Not publicly available.

---

## 3. Why Does BiGRU Outperform GCN for Figure Skating?

### Analysis from Papers and Theory

Based on the research surveyed (including the over-smoothing analysis paper arXiv:2502.10818 "On Vanishing Gradients, Over-Smoothing, and Over-Squashing in GNNs: Bridging Recurrent and Graph Learning") and our specific characteristics:

#### 3.1 Over-smoothing in Deep GCNs on Small Graphs

**The core problem:** With H3.6M's 17 keypoints, the skeleton graph is extremely small. GCN message passing over this tiny graph causes representations to converge to the same value after just 2-3 layers (proven mathematically in the over-smoothing literature). This is called the "over-smoothing threshold" and it is inversely proportional to the spectral gap of the graph's normalized Laplacian. For a 17-node graph, this threshold is very low.

**BiGRU avoids this entirely** because it processes the temporal dimension sequentially without spatial message passing. Each joint's trajectory is modeled independently (or with simple linear mixing), avoiding the graph collapse problem.

#### 3.2 Data Scale: 5168 Sequences is Tiny for GCN

**GCN parameter sharing** requires large datasets to learn meaningful adjacency patterns. On NTU RGB+D (56K+ sequences), GCNs have enough data to learn adaptive topologies. On our FSC dataset (5168 sequences, ~80 per class), GCNs overfit to the limited adjacency patterns they observe.

**BiGRU's sequential inductive bias** is a better fit for small data. The recurrent structure naturally captures temporal dynamics without needing to learn graph structure from limited examples.

#### 3.3 Variable-Length Handling

Figure skating element videos range from ~1s (single jump) to ~5s (combination). Our BiGRU uses **packed sequences** (pack_padded_sequence), which naturally handles variable lengths without padding artifacts.

Standard GCN implementations typically pad to a fixed length (64 or 300 frames). This introduces noise at the padding boundary and wastes computation. The temporal convolutions in GCN also have fixed receptive fields that may not match the variable duration of skating elements.

#### 3.4 Motion Dynamics: Sequential > Spatial for Skating

Figure skating elements are distinguished primarily by **temporal dynamics** (timing of takeoff, rotation speed, landing pattern) rather than spatial pose configurations. A flip and a lutz may have nearly identical pose at the moment of takeoff -- the difference is in the approach edge, which is a temporal pattern.

BiGRU's sequential processing naturally captures these temporal nuances. GCN's spatial message passing adds limited value when the key discriminative information is temporal.

#### 3.5 Skeleton Topology: H3.6M is Too Simple/Informative

With only 17 keypoints, the graph structure is trivially simple (essentially a tree with one branching point at the hip). GCN's main value proposition -- learning complex relational patterns on graphs -- is wasted on such a simple topology. The adjacency matrix contains almost no useful information beyond what a simple linear layer could learn.

**Evidence from SkelFormer paper:** They explicitly note that "the reliance on predefined bone connectivity may hinder adaptability to actions that require precise, context-dependent joint correlations" (p. 14). For skating, the predefined H3.6M connectivity tells us nothing about skating-specific relationships (e.g., free leg vs. skating foot timing).

#### 3.6 The Over-squashing Problem

The over-smoothing analysis paper (arXiv:2502.10818) introduces a unified framework showing that GNNs suffer from three related issues: vanishing gradients, over-smoothing, and over-squashing. Over-squashing is the inability to propagate information between distant nodes in the graph. For H3.6M, the maximum graph distance is ~5 hops (e.g., left wrist to right ankle), and with a simple tree structure, there is essentially one path between any two nodes. This makes over-squashing particularly severe.

---

## 4. Hybrid Approaches (RNN + Attention / RNN + GCN)

### 4.1 GCN-LSTM Hybrids

- **AGC-LSTM** (Si et al., CVPR 2019): Attention-enhanced GCN-LSTM. Uses GCN for spatial, LSTM for temporal, with attention gates. 93.3% on NW-UCLA.
- **Recurrent GCN** (IEEE 2020): Combines recurrent structure with graph convolution for skeleton action recognition.
- **GCN-DevLSTM** (arXiv 2403.15212, 2024): Path development with LSTM for skeleton action recognition.

### 4.2 GCN + Transformer Hybrids (2024-2025)

- **Two-stream GCN-Transformer** (Scientific Reports, 2025): GCN stream for spatial features + Transformer stream for temporal features. On NTU120: comparable to pure Transformer methods.
- **GCN-Former** (MDPI Applied Sciences, 2025): Direct combination of GCN and Transformer modules.
- **SkeleTR** (ICCV 2023): GCN captures individual dynamics, then stacked Transformer encoders model interaction. Designed for in-the-wild skeleton recognition.

### 4.3 The Emerging Pattern

The field is converging on a **"GCN for local spatial, [Transformer/Mamba/RNN] for global temporal"** architecture:
1. Early layers: GCN or attention-based spatial modeling
2. Middle layers: Hybrid or transition
3. Late layers: Transformer/Mamba for long-range temporal modeling

This is exactly what ActionMamba, SkeleTR, and the Two-stream GCN-Transformer do.

---

## 5. Recommendations for Figure Skating Classification

### Priority 1: MotionBERT (Pretrain-Finetune) -- Highest Expected Impact

**Rationale:**
- Pretrained on 3D mocap + in-the-wild video data (massive, diverse motions)
- 2D skeleton input matches our pipeline exactly
- One-shot 67.4% on NTU120 with 1 example per class -- we have ~80 per class
- DSTformer handles variable-length sequences naturally (Transformer advantage)
- The pretrain task (2D-to-3D lifting) forces the model to learn human motion physics, which is directly relevant to skating biomechanics

**Implementation plan:**
1. Start from pretrained MotionBERT weights
2. Finetune on FSC dataset with 1-2 layer classification head
3. Experiment with different sequence lengths (our videos are typically 30-150 frames)
4. Expected accuracy: 75-85% on 64 classes (speculative, based on NTU120 one-shot results)

### Priority 2: InfoGCN or SkelFormer -- Strong GCN Alternative

**Rationale:**
- InfoGCN's IB objective prevents overfitting on small datasets (proven on NW-UCLA with only 1494 videos)
- Context-dependent topology captures action-specific joint relationships
- SkelFormer's hierarchical node concentration is ideal for phase-based skating analysis

**Implementation plan:**
1. Use InfoGCN codebase, adapt for H3.6M 17kp format
2. Start with 2-stream (joint + bone), add motion stream if helpful
3. Apply IB objective with class-conditional orthogonal targets
4. Expected accuracy: 70-80% on 64 classes

### Priority 3: BiGRU + Attention (Low-Risk Improvement)

**Rationale:**
- We already have a working BiGRU at 63.9%
- Adding temporal self-attention after BiGRU captures long-range dependencies
- Minimal code changes, low risk

**Implementation plan:**
1. Add multi-head self-attention layer after BiGRU hidden states
2. Optionally add a small GCN spatial layer before BiGRU for local spatial mixing
3. This creates a "BiGRU + Attention" hybrid that captures both sequential and global patterns

### Priority 4: Hybrid GCN + BiGRU + Attention

**Rationale:**
- Use 1-2 GCN layers for local spatial feature extraction (not deep enough for over-smoothing)
- BiGRU for temporal dynamics (our proven approach)
- Attention for global context

**Architecture:**
```
Input (T, 17, 2) -> GCN (1-2 layers) -> BiGRU -> Self-Attention -> Classification
```

This avoids over-smoothing (shallow GCN), leverages our BiGRU success, and adds global context.

### What NOT to Do

1. **Do not use deep ST-GCN (10 layers)** -- guaranteed over-smoothing on 17-node graph with small dataset
2. **Do not use fixed adjacency matrix** -- H3.6M physical connectivity is too simple; use adaptive/learnable topology
3. **Do not pad all sequences to 300 frames** -- use variable-length handling (packed sequences or transformer-style position encoding)
4. **Do not use 4-stream or 6-stream ensembles** at first -- start with 2-stream (joint + bone), add complexity only if needed

---

## 6. Key Architectural Insights Summary

| Factor | BiGRU Advantage | GCN Disadvantage | Solution |
|--------|----------------|-------------------|----------|
| Over-smoothing | N/A (no graph) | Severe on 17-node graph | Shallow GCN (1-2 layers) or adaptive topology |
| Data scale (5K seq) | Good inductive bias | Overfits | IB objective (InfoGCN), pretrain (MotionBERT) |
| Variable length | pack_padded_sequence | Fixed-length padding | Transformer (MotionBERT), temporal merge (SkelFormer) |
| Temporal dynamics | Core strength | Weak (TCN is limited) | Mamba/Transformer for long-range temporal |
| Skeleton topology | N/A | Too simple (17 nodes) | Learnable adjacency, intrinsic topology |
| Fine-grained distinction | Moderate | Poor (over-smoothed) | Hierarchical node concentration (SkelFormer) |

---

## 7. References

1. SkelFormer: Yan et al., PLOS One 2026. doi:10.1371/journal.pone.0340390
2. HD-GCN: Lee et al., ICCV 2023. arXiv:2208.10741
3. InfoGCN: Chi et al., CVPR 2022. Code: github.com/stnoah1/infogcn
4. MotionBERT: Zhu et al., ICCV 2023. Code: github.com/Walter0807/MotionBERT
5. STTFormer: Qiu et al., arXiv 2022. Code: github.com/heleiqiu/STTFormer
6. BlockGCN: Zhou et al., CVPR 2024. Code: github.com/ZhouYuxuanYX/BlockGCN
7. ActionMamba: Wen et al., Electronics 2025. doi:10.3390/electronics14183610
8. Over-smoothing analysis: Arroyo et al., arXiv:2502.10818, 2025
9. Two-stream GCN-Transformer: Chen et al., Scientific Reports 2025
10. GCN-Former: Zhao et al., Applied Sciences 2025
11. GCN-DevLSTM: arXiv:2403.15212, 2024
12. FSBench: Gao et al., CVPR 2025 (figure skating benchmark with QA pairs)
13. YourSkatingCoach: Chen et al., arXiv:2410.20427, 2024 (BIOES-tagged skating elements)
14. MMFS: Liu et al., arXiv:2307.02730, 2023 (multi-modality figure skating dataset)
