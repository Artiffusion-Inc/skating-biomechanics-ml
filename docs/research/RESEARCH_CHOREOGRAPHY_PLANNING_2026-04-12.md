# Automated Figure Skating Choreography Planning — Research
**Date:** 2026-04-12
**Sources:** Exa web search, GitHub API, Gemini Deep Research (~91 sources)
**Status:** ACTIVE

---

## Summary

Research into building an automated choreography planning system for figure skating — a tool that generates complete programs (short program, free skate, ice dance) synchronized to music, with ISU rule validation, ice coverage planning, and 3D visualization. The recommended architecture is a two-stage approach: (1) deterministic CSP solver for element layout optimization, (2) diffusion-based model for smooth transitional motion generation.

## Key Findings

1. **Two-stage architecture is the way:** CSP solver for layout + diffusion model (EDGE) for transitions. End-to-end music→skating generation is infeasible.
2. **Music analysis stack:** madmom (downbeat/onset), essentia (spectral features), MSAF (structural segmentation). "Musical gravity" concept — jumps on RMS energy peaks, glides on low-energy phrases.
3. **ISU rules engine must be deterministic**, separate from AI. Zayak rule, well-balanced program constraints, back-half bonus optimization via CSP.
4. **Skate Plan** is the direct competitor (iOS, April 2026) but lacks AI generation — only manual program building.
5. **Fraunhofer IDMT** solved the inverse problem (movement→music) used at Olympics 2026. We solve music→choreography.
6. **Skreate** provides an excellent DSL for ice pattern notation — edge codes (LFO, RBI), turns, jumps — can be reused as our internal representation.
7. **SMPL-X + React Three Fiber** for 3D visualization in browser. Pre-bake standard elements as GLTF animations.

---

## 1. Inspiration Projects

### Skreate (github.com/daviddrysdale/skreate)
- **What:** Mermaid-like DSL for skating pattern diagrams (text → SVG)
- **Tech:** Rust + WASM, `nom` parser, `svg` crate, ACE editor for browser input
- **DSL Syntax:** Edge codes (LFO, RBI), turns (3, Br, Rk, Ctr), jumps (1S, 2T, 3A), modifiers (+, >, [angle=120])
- **Strengths:** Well-designed DSL, 30+ example choreographies, real-time preview, Apache-2.0
- **Limitations:** No music sync, no choreography generation, no validation, 2D only, no spins
- **Reuse potential:** DSL as internal program representation format, SVG rink rendering approach

### Figure-Skating-Choreography (github.com/cathzvccc/Figure-Skating-Choreography-)
- **What:** BPM-to-moves lookup table with matplotlib visualization
- **Tech:** Python, librosa, matplotlib, PyQt5. Single file (`6-3.py`, ~430 lines)
- **Verdict:** Student project. No actual algorithm. Useless except as a cautionary example.

### Skate Plan (iOS App, launched April 2026)
- **What:** Commercial app — IJS scoring, animated preview, music sync
- **Model:** Freemium with IAP ($1.99–$34.99)
- **Weakness:** Manual program building only, no AI generation
- **URL:** https://apps.apple.com/us/app/skate-plan/id6759615058

---

## 2. Technical Architecture (from Gemini Deep Research)

### Recommended: Two-Stage Pipeline

**Stage 1: Deterministic CSP Solver** — places elements on timeline + rink
- Variables: sequence of elements from skater's inventory
- Domains: timestamps and spatial coordinates on 60m×30m rink
- Constraints: ISU rules (capacity, Zayak, stamina, spatial connectivity)
- Solver: backtracking with forward checking (python-constraint or OR-Tools)
- Output: top 3 highest-scoring layouts for user selection

**Stage 2: Diffusion Model** — generates smooth transitions between elements
- Based on EDGE architecture (Stanford, CVPR 2023)
- Takes discrete element timestamps as hard keyframes
- Generates continuous 3D transitional motion (footwork, port de bras)
- Must incorporate non-holonomic ice physics (blade velocity alignment)
- Pre-bake standard elements as GLTF animations; diffusion only for transitions

**Stage 3 (optional): LLM for artistic suggestions**
- NOT for layout generation (LLMs fail at Zayak rule combinatorics)
- For suggesting transitional footwork, arm movements, artistic interpretation
- E.g., "Insert forward outside counters and spread eagle during string crescendo"

### JSON Schema for Programs

```json
{
  "program_metadata": {
    "discipline": "womens_singles",
    "level": "senior",
    "segment": "free_skate",
    "music_duration_sec": 240,
    "target_score": 145.50
  },
  "elements": [
    {
      "element_id": "el_01",
      "type": "jump",
      "name": "3Lz",
      "base_value": 5.90,
      "timestamp_sec": 15.2,
      "position": {"x": 25.0, "y": 10.0},
      "exit_edge": "RBO",
      "is_back_half": false
    }
  ],
  "transitions": [
    {
      "transition_id": "tr_01",
      "start_element": "start",
      "end_element": "el_01",
      "moves": ["crossover_bwd", "choctaw", "mohawk"]
    }
  ]
}
```

### CSP Constraints (from Gemini)

| Constraint | Description |
|-----------|-------------|
| `C_capacity` | Total jumping passes ≤ 7 |
| `C_zayak` | Count(J_i) ≤ 2; if == 2, one must be in combo |
| `C_stamina` | Energy depletion curve must not exceed athlete's lactic threshold |
| `C_spatial` | Elements connected by physically feasible B-spline transitions |
| `C_back_half` | Optimize last 3 jumping passes for 10% BV bonus |
| `C_spin_types` | 3 different spin types required |
| `C_well_balanced` | One step sequence + one choreographic sequence required |

---

## 3. Music Analysis (from Gemini Deep Research)

### Feature Extraction Pipeline

| Feature | Library | Purpose |
|---------|---------|---------|
| **Downbeat/Onset** | madmom (RNN-HMM) | Precise jump timing alignment |
| **Spectral features** | essentia (MFCC, CENS, spectral contrast) | Mood, harmonic density, timbre |
| **Structural segmentation** | MSAF (SSM-based) | Verse/chorus/bridge boundaries |
| **Energy curve** | RMS energy + spectral flux | "Musical gravity" mapping |
| **Danceability** | essentia high-level descriptor | Movement intensity guidance |

### "Musical Gravity" Concept (skating-to-music.blog)
- Jumps → placed at RMS energy peaks + spectral flux peaks
- Buildup → accelerating crossovers, deep-edge step sequences
- Breakdown/low-energy → layback spins, spirals, hydroblading
- Climax → highest BV jump on strongest downbeat
- Musical phrases mapped to 8-count / 16-count / 32-count systems

### Music Structure Analysis
- **SSM (Self-Similarity Matrices)** — diagonal stripes = repeated phrases
- **CNN-based boundary detection** (Ullrich et al., ISMIR 2014) — mel-spectrograms
- **SpecTNT model** (Wang et al., ICASSP) — CTL loss for semantic labels ("verseness", "chorusness")
- **Risk:** EDM/ambient tracks lack harmonic structure → SSM fails. Fallback to RMS energy only.

### Server-Side Processing Required
- Python-native (librosa) too slow for real-time web
- Use Essentia.js (WASM) for client-side, or server-side batch processing
- Feature extraction → cached per track, not recalculated on each edit

---

## 4. ISU Rules Engine (from Gemini Deep Research)

### Well-Balanced Program Constraints (Senior Singles Free Skate 2025/26)
1. Maximum 7 jumping passes (at least one Axel-type)
2. Maximum 3 jump combinations/sequences (only one 3-jump combo, Euler only once)
3. Maximum 3 spins (combination, flying, single-position — different codes)
4. One step sequence + one choreographic sequence

### Zayak Rule
- No jump with 3+ revolutions repeated more than once
- If repeated, at least one attempt must be in a combination/sequence
- Violation = element invalidated (zero points)
- Must maintain dynamic ledger during generation

### Back-Half Bonus Optimization
- Last 3 jumping passes in free skate receive +10% base value
- Strategy: high-risk quads in first 60s (fresh muscles), reliable combos in back-half
- Choreographic sequences placed mid-program for cardiovascular recovery

### Edge Compatibility Validation
- Jump exit edge dictates next element's entry edge
- Example: Lutz lands RBO → cannot immediately do forward inside three-turn
- Must enforce edge transition matrices before rendering

### Data Sources for Rules Engine
- ISU Communication 2707 & 2701 (2025/26) — Base Values, Levels, GOE
- ISU Special Regulations & Technical Rules 2024
- BuzzFeedNews/figure-skating-scores — historical protocols for training

### Risk: Ice Dance
- Rules change annually (e.g., 2025/26 Rhythm Dance requires 1990s styles)
- Requires constant manual updates to rules engine
- Recommendation: start with Singles only

---

## 5. Visualization & Animation (from Gemini Deep Research)

### 3D Visualization Stack
- **SMPL-X** (Skinned Multi-Person Linear eXpressive) — parametric body model with hands (54 joints), face, shape
- **React Three Fiber (R3F)** — declarative 3D in React, `useFrame` for 60fps playback
- **Pre-baked GLTF/GLB animations** for standard elements (Triple Axel, sit spin, etc.)
- **Three.js Animation Mixer** — interpolate between pre-baked clips

### UI: Two-Panel Layout
1. **Timeline Editor** (react-timeline-editor style)
   - Audio waveform at top
   - Swimlanes: technical elements, transitions, musical structure
   - Drag-and-drop elements, instant rule violation highlighting
2. **Spatial Playbook** (top-down 2D orthographic)
   - 60m×30m ice rink with Cartesian trajectory
   - B-spline path visualization
   - Dynamic update when elements move on timeline

### Risk Assessment
- Real-time SMPL-X mesh deformation (clothing, soft-tissue) → thermal throttling on mobile
- Low-poly SMPL-X with pre-calculated quaternions is feasible
- Multi-person (pairs/ice dance) collision avoidance → computationally expensive
- Recommendation: start with single skater, low-poly

---

## 6. Choreography Generation Models (from Gemini Deep Research)

### Bailando (CVPR 2022, 431 stars)
- VQ-VAE learns "choreographic memory" (quantized codebook of 3D poses)
- Actor-Critic GPT autoregressively predicts pose codes from music features
- RL reward function for beat alignment
- **Key insight:** Separates upper/lower body — lower body for ice physics, upper body for artistic expression
- Code: github.com/lisiyao21/Bailando

### EDGE (CVPR 2023, Stanford)
- Transformer-based diffusion architecture conditioned on Jukebox audio features
- **In-betweening:** pin elements at specific timestamps, diffusion fills transitions
- Joint-wise conditioning — can pin specific joints while others move freely
- Code: github.com/Stanford-MIND/EDGE

### ChoreoMaster (NetEase)
- Graph-based optimization: transition costs from joint position/rotation/velocity changes
- Finds optimal path through motion graph aligned to musical meters
- Code: github.com/NetEase-GameAI/ChoreoMaster

### Non-Holonomic Ice Physics
- Skating is governed by non-holonomic constraints
- Blade glides forward/backward (near-zero friction), immense lateral resistance
- Velocity vector of grounded foot must align with blade orientation
- Diffusion noise / VQ-VAE decoder must incorporate physics-based loss penalizing lateral slip
- B-spline curves for trajectory optimization (smooth continuous edges)
- CMA-ES optimization within rink boundaries (60m×30m)
- Reference: "Figure Skating Simulation from Video" (Yu et al., Pacific Graphics)
- Reference: "Models of Ice Skating for Robotic Ice Skating Gaits" (UC Berkeley, 2021)

---

## 7. Commercial Landscape (from Gemini Deep Research)

### Direct Competitors
| Product | Platform | Strengths | Weaknesses | Price |
|---------|----------|-----------|------------|-------|
| **Skate Plan** | iOS | IJS scoring, path design, music sync, local privacy | No AI generation, manual only | Free + IAP $1.99–$34.99 |
| **ijsPro** | iOS | Scoring calculator | No visualization, no choreography | $1.99 |
| **Figure Skating Score** | iOS | ISU 2025/26 judging tool | Calculator only | Free |
| **Playasport** | iOS | Career simulation with program building | Game, not a tool | Free + IAP |

### Adjacent Tools
| Product | Purpose | Relevance |
|---------|---------|-----------|
| **StageKeep** | Dance choreography (formations, logistics) | Enterprise pricing $120–$3600/yr shows willingness to pay |
| **SpaceDraft** | Visual planning for choreographers | 2D/3D formation mapping, timeline sync |
| **SportsEngine** | Club management (scheduling, registration) | No creative/biomechanical utility |

### Target Users
1. **Elite Coaches/Choreographers (B2B)** — absolute ISU compliance, high-fidelity editing, privacy
2. **Competitive Skaters (B2C)** — understand programs, optimize layouts, experiment with music
3. **Adult Recreational Skaters** — rapidly growing, no access to elite choreographers, eager for AI

### Recommended Monetization
- **Mid-tier consumer** (~$9.99/month): 3D viewer, music sync, basic IJS calculations
- **Premium Coach/Choreographer** (~$49.99/month): full AI generation engine, batch optimization, PDF protocol export

### Positioning Strategy
- Market as **"Strategic IJS Optimizer"** not "Artistic Replacement"
- Core value: remove mathematical cognitive load (Zayak, BV, back-half bonus)
- Secondary value: 3D animation for mental imagery practice (proven technique for skill acquisition)

---

## 8. Key Research Papers

### Music Analysis
- Serrà et al. (AAAI) — Unsupervised music boundary detection via time series structure features
- Ullrich et al. (ISMIR 2014) — CNN boundary detection on mel-spectrograms
- Wang et al. (ICASSP) — SpecTNT with CTL loss for semantic structural labeling
- Chuan & Chew (ISMIR 2007) — Dynamic programming for phrase boundaries from tempo variations

### Dance/Choreography Generation
- Siyao et al. (CVPR 2022) — Bailando: VQ-VAE + Actor-Critic GPT
- Tseng et al. (CVPR 2023) — EDGE: Editable diffusion-based dance generation
- Chen et al. (ChoreoMaster) — Graph-based motion graph optimization
- Li et al. (ICCV 2021) — AI Choreographer: FACT cross-modal transformer

### Figure Skating Physics & Simulation
- Yu et al. (Pacific Graphics) — Figure Skating Simulation from Video (non-holonomic constraints, CMA-ES, B-splines)
- UC Berkeley EECS 2021 — Models of Ice Skating for Robotic Gaits
- Zou et al. (WACV 2020) — Reducing Footskate with Ground Contact Constraints

### ISU Rules & Scoring
- ISU Communication 2707 & 2701 (2025/26) — Base Values, Levels, GOE
- ISU Special Regulations & Technical Rules 2024
- U.S. Figure Skating — 2025-2026 Singles Technical Requirements Guide

### 3D Visualization
- Pavlakos et al. (CVPR) — SMPL-X: Expressive body capture with hands, face, body
- Siyao et al. — Half-Physics: kinematic 3D human model with physical interactions

---

## 9. Open-Source Projects (Comprehensive)

### Music-to-Dance Generation
| Repo | Stars | Description |
|------|-------|-------------|
| lisiyao21/Bailando | 431 | Actor-Critic GPT, VQ-VAE codebook |
| google/aistplusplus_dataset | — | AIST++ dataset (music-dance pairs) |
| NetEase-GameAI/ChoreoMaster | ~100 | Motion graph optimization |
| asdryau/TransPhase | new | Long motion sequences (NeurIPS 2025 Oral) |

### Music Analysis
| Repo | Stars | Description |
|------|-------|-------------|
| urinieto/msaf | ~100 | Music Structure Analysis Framework |
| carlosholivan/symbolic-music-structure-analysis | — | Graph + changepoint detection |
| morgan76/LinkSeg | — | Pairwise link prediction (ISMIR 2024) |

### Figure Skating Specific
| Repo | Description |
|------|-------------|
| nxxtia/figure_skating_element_parser | ISU notation parser |
| hyperpolymath/anvomidaviser | ISU→formal programs (Rust) |
| BuzzFeedNews/figure-skating-scores | ISU protocols as structured data |
| hjjerrychen/skatecalc | IJS score calculator |
| rleejh/FSCalculator | Program calculator |
| daviddrysdale/skreate | Skating diagram generator (Rust/WASM) |

### 3D & Visualization
| Repo | Description |
|------|-------------|
| pmndrs/react-three-fiber | Declarative 3D for React |
| vchoutas/smplx | SMPL-X body model (Python) |
| Meshcapade/SMPL_blender_addon | SMPL-X in Blender |
| xzdarcy/react-timeline-editor | Timeline/swimlane UI |
| plechanator/Coach-AI | Hockey rink diagrams (UI reference) |

---

## 10. Risks & Mitigation

| Risk | Severity | Mitigation |
|------|----------|------------|
| Foot-skating artifacts on ice (lateral slip) | HIGH | PINN layers in diffusion model, physics-based loss |
| 3D jump generation via neural nets is unstable | HIGH | Pre-bake jumps as GLTF, only generate transitions |
| Ice dance rules change annually | MEDIUM | Start with Singles only, modular rules engine |
| ISU rule error → skater deduction → reputation damage | CRITICAL | Deterministic rules engine, extensive test suite, "not official" disclaimer |
| EDM/ambient music breaks structure analysis | MEDIUM | Fallback to RMS energy + onset detection |
| Real-time SMPL-X on mobile → thermal throttling | MEDIUM | Low-poly mesh, pre-calculated quaternions, server-side rendering |
| Elite choreographers reject AI | LOW | Position as "Strategic IJS Optimizer", not replacement |
| LLM hallucinates edge transitions | MEDIUM | LLM only for artistic suggestions, not layout |
