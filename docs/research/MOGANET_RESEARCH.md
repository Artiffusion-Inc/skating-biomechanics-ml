---
title: "MogaNet-B and Alternative 2D Pose Estimators Research"
date: "2026-03-29"
status: historical
citekey: ""
related:
  - RESEARCH_POSE_TOOLS_2026-03-31.md
  - specs/2026-04-02-foot-keypoint-research-prompt.md
---

# MogaNet-B and Alternative 2D Pose Estimators Research

**Date:** 2026-03-29  
**System:** PyTorch 2.11.0, CUDA 13.0, RTX 3050 Ti 4GB  
**Constraint:** Must work with `uv` package manager (NO mmcv/mmdet compilation)

---

## Executive Summary

**Finding:** MogaNet-B is **NOT available as a standalone pose estimator**. It is only a backbone architecture used within MMPose, which requires mmcv compilation.

**Decision:** Use **RTMPose via rtmlib** (ONNX Runtime, no compilation). See [RESEARCH_POSE_TOOLS_2026-03-31.md](RESEARCH_POSE_TOOLS_2026-03-31.md) for final evaluation.

**Rejected alternatives:**
- MMPose/RTMPose direct — requires mmcv compilation
- ViTPose-lib — uncertain pip-installability, large model
- MogaNet standalone — backbone only, needs custom head + training
- ONNX workaround — complex export pipeline

**Trade-offs (chosen RTMPose):**
- HALPE26 (26kp) vs BlazePose 33kp — acceptable, foot keypoints retained
- ONNX Runtime — no compilation, CUDA 13 compatible
- ~10-20ms/frame on RTX 3050 Ti

## Comparison Matrix

| Solution | Accuracy | Speed | Complexity | Works with uv? |
|----------|----------|-------|------------|----------------|
| **RTMPose (rtmlib)** | High | 10-20ms | Easy | Yes |
| YOLOv11-Pose | 50-60% AP | 5-30ms | Easy | Yes |
| BlazePose | Good | 20ms | Easy | Yes |
| ViTPose-lib | High | Slow | Medium | Uncertain |
| MogaNet (timm) | N/A | N/A | Hard | Yes (incomplete) |
| MMPose direct | High | Medium | Hard | No (mmcv) |

## Key Takeaways

1. **MogaNet-B is not a standalone pose estimator** — backbone only within MMPose
2. **RTMPose via rtmlib is the chosen solution** — ONNX, no compilation, CUDA 13 compatible
3. **YOLOv11-Pose is viable fallback** — already installed, fast, easy API
4. **Avoid MMPose/RTMPose direct** — mmcv compilation too fragile on CUDA 13

## References

- YOLOv8/11-Pose: https://github.com/ultralytics/ultralytics
- BlazePose: https://google.github.io/mediapipe/solutions/pose.html
- ViTPose: https://github.com/VITAE-Group/ViTPose
- MogaNet: Multi-order Gated Aggregation Network (2023) — timm backbone only
- MMPose: https://github.com/open-mmlab/mmpose — requires mmcv

---

> **Full historical research with all evaluated alternatives is preserved in git history.**  
> This file is a compressed decision record. See original commit for 390-line detailed analysis.
