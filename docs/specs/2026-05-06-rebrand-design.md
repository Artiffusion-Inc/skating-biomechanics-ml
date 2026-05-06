# SkateLab Rebrand Design

## Summary

Rebrand `skating-biomechanics-ml` → **SkateLab**. Product name for B2B market (federations, sport schools, clubs), international, Russian launch.

## Name

- **Brand:** SkateLab
- **Domain:** skatelab.ru (purchased, NS pending Cloudflare transfer)
- **Pronunciation:** скейт-лаб
- **Etymology:** Skate + Laboratory — skating science lab
- **Audience:** B2B (federations, schools, clubs), international market, Russian launch
- **Tone:** Smart but not boring. Scientific foundation (Lab) + sport (Skate)
- **Conflicts:** sk8lab.com.br (Brazilian skate shop) — different industry, different language zone. No trademark conflicts in target market.

## Naming Convention

**Principle:** Product brand on external artifacts, simple dir-based names on internal packages.

| Artifact | Before | After | Rationale |
|---|---|---|---|
| Root pyproject.toml `name` | `skating-biomechanics-ml` | `skatelab` | Product identity |
| Backend pyproject.toml `name` | `skating-backend` | `backend` | Internal, dir-based |
| ML pyproject.toml `name` | `skating-ml` | `ml` | Internal, dir-based |
| Frontend package.json `name` | `frontend` | `frontend` | No change |
| Docker image | `skating-ml-gpu` | `skatelab-gpu` | Registry = external |
| CLI config dir | `~/.config/skating-cli/` | `~/.config/skatelab/` | External |
| ENV var prefix | `SKATING_*` | `SKATELAB_*` | External |
| Git repo name | `skating-biomechanics-ml` | `skatelab` | External |
| Python imports | `from src.*` | `from src.*` | No change |

## What Does NOT Change

- Internal code: function names, class names, variable names
- ML terminology: pose estimation, phase detection, DTW, etc.
- API endpoints: `/api/v1/sessions`, `/api/v1/metrics` etc.
- Python package structure: `from src.*` imports stay
- Frontend component names

## Implementation Priority

1. `pyproject.toml` (root, backend, ml) — package names
2. Frontend `package.json` — app name (if needed)
3. Docker/GPU image — Containerfile, deploy.sh
4. CLI — config path
5. ENV vars — `SKATING_*` → `SKATELAB_*`
6. Git repo — rename on GitHub (after internal changes)

## Competitors (for reference)

- **OOFSkate** — US Figure Skating partner, AI jump analysis from phone
- **Athlitix RinkUp** — AI video analysis (App Store)
- **SportsReflector** — multi-sport AI/AR coaching