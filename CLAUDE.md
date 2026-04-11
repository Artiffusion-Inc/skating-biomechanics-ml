# CLAUDE.md

> **PROJECT ROADMAP:** @ROADMAP.md — SINGLE SOURCE OF TRUTH for implementation status
> **RESEARCH:** @research/RESEARCH_SUMMARY_2026-03-28.md — Exa + Gemini findings (41 papers)

---

## Project Overview

ML-based AI coach for figure skating. Analyzes video, compares attempts to professional references, provides biomechanical feedback in Russian.

**Vision:** AI-тренер по фигурному катанию — анализ видео и рекомендации на русском.

## Architecture

```
Video → RTMPose (rtmlib, CUDA) → HALPE26 (26kp)
  → H3.6M (17kp) conversion → GapFiller → Smoothing
  → [Optional] CorrectiveLens (3D lift → kinematic constraints → project back to 2D)
  → Phase Detection → Biomechanics Metrics → DTW (vs reference)
  → Rule-based Recommender → Russian Text Report
```

**Key decisions:**
- **rtmlib**: sole pose estimation backend — HALPE26 (26kp), ONNX (CPU+GPU), foot keypoints
- **HALPE26 (26kp)** as intermediate format, converted to H3.6M (17kp) for downstream
- **CorrectiveLens**: 3D lifting as corrective layer for 2D skeleton (Kinovea-style angles)
- **PoseTracker**: anatomical biometric Re-ID instead of color (solves black clothing on ice)
- **CoM trajectory** instead of flight time (eliminates 60% error for low jumps)

## Tech Stack

| Component | Technology |
|-----------|-----------|
| **ML Pipeline** | Python, rtmlib, onnxruntime-gpu, scipy |
| **Backend API** | FastAPI, SQLAlchemy, Alembic, arq + Valkey |
| **Frontend** | Next.js 16, React, Tailwind CSS, shadcn/ui, Recharts |
| **Storage** | Cloudflare R2 (S3-compatible) |
| **Remote GPU** | Vast.ai Serverless |
| **Testing** | pytest (backend), tsc + next lint (frontend) |

## Git & GitHub Workflow

### Branches

- **Format**: `feature/<short-name>` (e.g., `feature/onnx-export`)
- **Main branch**: `master`
- **Before push**: `git fetch origin && git merge origin/master`

### Commits

- **Format**: `<type>(<scope>): <description>`
- **Types**: `feat`, `fix`, `refactor`, `chore`, `docs`, `test`, `ci`
- **Scopes**: `pose`, `viz`, `tracking`, `analysis`, `pipeline`, `cli`, `models`, `repo`, `frontend`, `backend`, `dev`, `ci`, `vastai`, `infra`

### Pull Requests

| Field | Value |
|-------|-------|
| Base branch | `master` |
| Title | Same format as commit |
| Description | Must include "Что сделано" and "Как проверить" sections |

## References

- @ROADMAP.md — project status (SINGLE SOURCE OF TRUTH)
- @research/RESEARCH_SUMMARY_2026-03-28.md — research findings (41 papers)
- @research/RESEARCH.md — research memory bank (index)
