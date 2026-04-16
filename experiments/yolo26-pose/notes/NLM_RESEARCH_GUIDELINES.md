# NLM Research Guidelines

**Date:** 2026-04-16
**Purpose:** Quality guidelines for NotebookLM research

---

## Critical Rules for NLM Research

### 1. Write Maximally Detailed Queries

**ALWAYS include:**
- Full context of the project (what we're doing, why, current state)
- Specific edge cases and constraints
- All relevant parameters (dataset size, model type, hardware, etc.)
- Current experimental results (what worked, what didn't)
- Specific questions to answer

**Example:**
```
❌ BAD: "YOLO pose fine-tuning best practices"

✅ GOOD: "YOLO26-pose fine-tuning on AthletePose3D dataset (226K train, single subject).
Current HP search results: mosaic=0.0 gives +72% improvement (0.623 vs 0.361 baseline),
freeze=10 recommended by research, but HP search shows f0 (freeze=0) also strong (0.441).
Planning Phase 2: 285K frames at 1280px with 2x RTX 4090 DDP training.
Question: What are the risks of using mosaic=0.0 for full training? Should we use freeze=10 or freeze=5?
How to prevent overfitting on single-subject dataset?"
```

### 2. Describe All Edge Cases

**Include:**
- Dataset characteristics (single subject, small size, domain-specific)
- Hardware constraints (2x GPU, VRAM limits)
- Time constraints (training budget)
- Validation strategy (held-out subject S2)
- Known issues (confirmation bias in pseudo-labeling, overfitting risks)

### 3. Only Use Quality/Verified Sources

**Source quality criteria:**
- ✅ Primary sources: research papers (arXiv, CVPR, ICCV, etc.)
- ✅ Official documentation (Ultralytics docs, GitHub repos)
- ✅ Code examples from production repositories
- ⚠️ Secondary sources: blog posts, tutorials (verify with primary)
- ❌ Unverified claims: "some people say", "common knowledge"
- ❌ Outdated sources: >2 years for fast-moving topics

**Fact verification:**
- Every claim must have a traceable source
- Note contradictions between sources
- Flag low-confidence information

### 4. Facts Must Be Source-Backed

**Required for each fact:**
- Source URL or reference (clickable, accessible)
- Context: which part of source supports the claim
- Recency: when source was published

**Format:**
```
Claim: [factual statement]
Source: [URL or reference]
Context: [relevant quote or section]
```

---

## NLM Notebook Management

### Current Notebooks

**YOLO26 Pose Fine-Tuning Knowledge Base**
- ID: `7d8ff6c7-9bcd-43cb-878d-fa5506851b39`
- Sources: 15 (5 uploaded + 10 from research)
- Artifacts: 1 report
- Report (Google Docs): https://docs.google.com/document/d/1UN26fbdyCKdD8tjXcN0DjlcSLBvW7ctfQ0JKyY_4HNs
- **Local copy:** `experiments/yolo26-pose/notes/NOTEBOOKLM_YOLO26_ANALYSIS.md`

### How to Download Reports (CRITICAL!)

```bash
# ❌ WRONG - positional argument doesn't work
nlm download report <notebook_id> <artifact_id>

# ✅ CORRECT - use --id flag
nlm download report <notebook_id> --id <artifact_id> -o /path/to/report.md

# Example:
nlm download report 7d8ff6c7-9bcd-43cb-878d-fa5506851b39 \
  --id 18c3dcaa-2473-4df3-92b6-423460589ab2 \
  -o experiments/yolo26-pose/notes/NOTEBOOKLM_YOLO26_ANALYSIS.md
```

**Important:** 
- Google Docs URLs are often **private** — use `nlm download report` to get local Markdown copy
- Always specify artifact ID with `--id` flag (not as positional argument)
- Use `-o` to specify output path (default: `./{notebook_id}_report.md`)

### Finding Artifact IDs

```bash
# List artifacts in notebook
nlm list artifacts <notebook_id>

# Get artifact status
nlm studio status <notebook_id>
```

### Sources Loaded

1. **Comprehensive Research Comparison** (generated text)
   - Multi-tool research comparison (tvly, gdr chat, HP search)
   - Parameter recommendations: confidence=0.875, freeze=10, lr0=0.0005

2. **Phase 2 Research-Based Plan** (generated text)
   - Updated plan with overfitting prevention
   - Validation criteria and fallback strategies

3. **HP Search Experimental Results** (generated text)
   - Real-time HP search results (mos0 dominates: 0.623)
   - Epoch progress and observations

4. **Research Hypotheses** (generated text)
   - H1-H5 hypotheses from RESEARCH_REPORT.md
   - Success criteria and validation metrics

5. **Ultralytics Training Tips** (web_page)
   - Official Ultralytics documentation
   - Early stopping, learning rate, augmentation

6. **Research-discovered sources** (10 from web):
   - YOLO26 vs YOLO11 comparison
   - Custom training guides
   - Pseudo-labeling research (NIPS papers)
   - Fine-tuning without forgetting (arXiv)
   - Mosaic augmentation discussions
   - Parameter-Efficient Fine-Tuning (PEFT)

---

## Research Query Templates

### For HP Search Analysis
```
"YOLO26-pose HP search analysis: 10 configs tested on AthletePose3D (79K train, 2.2K val).
Results: mosaic=0.0 dominates (0.623 mAP50-95), freeze=0 strong (0.441), batch64 fails (0.247).
Questions:
1. Why does mosaic=0.0 outperform? Is this specific to vertical sports?
2. freeze=0 vs freeze=10: f0 (no freeze) performs well, but research recommends freeze=10.
3. batch64 convergence: 0.247 at epoch 43 vs 0.361 baseline at epoch 21.
Why slower? Is this expected for pose estimation?
4. Should we use mosaic=0.0 for Phase 2 full training (285K frames, 1280px)?
Context: Figure skating is vertical symmetric sport. Ice rink background is consistent."
```

### For Phase 2 Planning
```
"Phase 2 planning: YOLO26s-pose fine-tuning for figure skating pose estimation.
Dataset: AthletePose3D S1 (226K) + COCO train2017 (59K) = 285K frames.
Hardware: 2x RTX 4090, DDP training.
Validation: S2 held-out subject (critical for generalization).
Current findings:
- mosaic=0.0: +72% improvement in HP search
- freeze=10: recommended by research (Gandhi et al. 2025)
- confidence threshold: 0.875 (balanced from multi-tool research)
- pseudo weight: 0.2 (1:4 GT:pseudo ratio)

Risks to address:
1. Single-subject overfitting: S1 only → body proportion bias
2. Pseudo-labeling confirmation bias: teacher errors → student
3. Mosaic=0.0 for full training: will 285K frames be enough diversity?
4. Freeze depth: freeze=10 vs freeze=5 based on HP results?

Questions:
1. Should we use freeze=10 or freeze=5 for Phase 2?
2. Is mosaic=0.0 safe for 285K frames or should we use mosaic=0.3?
3. How to validate generalization before Phase 2C (pseudo-labeling)?
4. What early stopping metrics for single-subject dataset?"
```

### For Pseudo-Labeling Strategy
```
"Pseudo-labeling strategy for YOLO26-pose on SkatingVerse (28K videos, ~500K pseudo-labels).
Teacher model: YOLO26s-pose trained on AP3D + COCO (285K frames).
Research findings:
- Confidence threshold: 0.875 (balanced, not 0.95 which causes data scarcity)
- Pseudo weight: 0.2 (1:4 GT:pseudo ratio to prevent confirmation bias)
- Mean Teacher: EMA of student weights for stable targets
- Adaptive thresholding: start 0.9, decay to 0.8

Risks:
1. Confirmation bias: model reinforces its own mistakes
2. Domain gap: SkatingVerse (TV broadcasts) vs AP3D (lab environment)
3. Quality threshold: 0.875 might be too low/high

Questions:
1. Should we use adaptive thresholding or fixed 0.875?
2. How to validate pseudo-label quality before student training?
3. Should we use Mean Teacher or standard self-training?
4. What metrics to monitor for confirmation bias during student training?"
```

---

## Quality Checklist

Before running `nlm research`, verify:

- [ ] Query includes full project context
- [ ] Query describes edge cases and constraints
- [ ] Query specifies current experimental results
- [ ] Query asks specific, answerable questions
- [ ] Notebook has relevant sources loaded
- [ ] Sources are from quality publications (papers, official docs)

After research completes:

- [ ] Review sources found (check quality)
- [ ] Verify facts are source-backed
- [ ] Note contradictions between sources
- [ ] Flag unverified or speculative claims
- [ ] Cross-reference with our experimental data

---

## Common Mistakes to Avoid

1. **Too vague:** "YOLO training tips" → ❌
   **Better:** "YOLO26-pose fine-tuning on single-subject dataset (AthletePose3D), 226K frames, preventing overfitting" → ✅

2. **Missing context:** "Should I use freeze=10?" → ❌
   **Better:** "YOLO26-pose on AthletePose3D: HP search shows freeze=0 strong (0.441), but research recommends freeze=10. Should we use freeze=10 or freeze=5 for 285K frame training?" → ✅

3. **No edge cases:** "Pseudo-labeling best practices" → ❌
   **Better:** "Pseudo-labeling for SkatingVerse (28K videos, TV broadcasts, diverse conditions). Teacher trained on AP3D (lab, single subject). Risks: domain gap, confirmation bias. What validation strategy?" → ✅

4. **Trusting all sources:** "Blog post says X" → ❌
   **Better:** "Paper (arXiv:2501.18445) says X, but contradicts Ultralytics docs. Need more sources" → ✅

---

## See Also

- @COMPREHENSIVE_RESEARCH_COMPARISON.md — multi-tool research results
- @PHASE2_RESEARCH_BASED_PLAN.md — updated Phase 2 strategy
- @HP_SEARCH.md — real-time HP search results
- NotebookLM Report: https://docs.google.com/document/d/1UN26fbdyCKdD8tjXcN0DjlcSLBvW7ctfQ0JKyY_4HNs

---

## 📚 CLI Reference (Quick Lookup)

### Download Reports (CRITICAL!)

```bash
# ✅ CORRECT SYNTAX
nlm download report <notebook_id> --id <artifact_id> -o /path/to/report.md

# Example:
nlm download report 7d8ff6c7-9bcd-43cb-878d-fa5506851b39 \
  --id 18c3dcaa-2473-4df3-92b6-423460589ab2 \
  -o experiments/yolo26-pose/notes/NOTEBOOKLM_YOLO26_ANALYSIS.md

# Find artifact IDs first:
nlm list artifacts <notebook_id>
nlm studio status <notebook_id>
```

**Key:** Always use `--id` flag for artifact ID, not positional argument!

### Other Essential Commands

```bash
# Authentication
nlm login
nlm login --check

# Notebook
nlm notebook list
nlm notebook create "Title"

# Sources
nlm source add <nb_id> --url "https://..."
nlm source add <nb_id> --title "Title" --text "..."
nlm source list <nb_id>

# Research
nlm research start "query" --notebook-id <nb_id>
nlm research status <nb_id>
nlm research import <nb_id> <task_id>

# Artifacts
nlm report create <nb_id> --confirm
nlm list artifacts <nb_id>
nlm download report <nb_id> --id <artifact_id> -o file.md

# Export
nlm export to-docs <nb_id> <artifact_id>
```
