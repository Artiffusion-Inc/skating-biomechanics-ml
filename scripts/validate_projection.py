#!/usr/bin/env python3
"""Visual validation of _coco.npy + projected foot keypoints.

Usage:
    uv run python scripts/validate_projection.py
    uv run python scripts/validate_projection.py --sequence Axel_10 --camera 1 --frame 50
    uv run python scripts/validate_projection.py --sequence Axel_10 --camera 1 --frame 50 --save /tmp/halpe26_check.jpg
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.datasets.coco_builder import merge_coco_foot_keypoints
from src.datasets.projector import project_foot_frame, validate_foot_projection

DATA_ROOT = Path("data/datasets/athletepose3d")

HALPE26_SKELETON = [
    (0, 1),
    (0, 2),
    (1, 3),
    (2, 4),
    (5, 6),
    (5, 7),
    (7, 9),
    (6, 8),
    (8, 10),
    (5, 11),
    (6, 12),
    (11, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
    (15, 17),
    (16, 20),
    (17, 18),
    (18, 19),
    (20, 21),
    (21, 22),
    (23, 24),
    (24, 25),
]

HALPE26_NAMES = [
    "nose",
    "L_eye",
    "R_eye",
    "L_ear",
    "R_ear",
    "L_sho",
    "R_sho",
    "L_elb",
    "R_elb",
    "L_wri",
    "R_wri",
    "L_hip",
    "R_hip",
    "L_knee",
    "R_knee",
    "L_ank",
    "R_ank",
    "L_heel",
    "L_bigtoe",
    "L_smtoe",
    "R_heel",
    "R_bigtoe",
    "R_smtoe",
    "L_eye_in",
    "R_eye_in",
    "mouth",
]

# Green for COCO 17, orange for foot 6, magenta for face dupes
KP_COLORS = [(0, 255, 0)] * 17 + [(0, 165, 255)] * 6 + [(255, 0, 255)] * 3


def find_sequence(split: str, sequence: str, camera: int) -> Path | None:
    base = DATA_ROOT / "videos" / split
    for d in sorted(base.iterdir()):
        if not d.is_dir():
            continue
        npy = d / f"{sequence}_cam_{camera}.npy"
        coco_npy = d / f"{sequence}_cam_{camera}_coco.npy"
        mp4 = d / f"{sequence}_cam_{camera}.mp4"
        json_f = d / f"{sequence}_cam_{camera}.json"
        if npy.exists() and coco_npy.exists() and mp4.exists() and json_f.exists():
            return npy
    return None


def draw_halpe26(frame: np.ndarray, pts: np.ndarray, vis: np.ndarray) -> np.ndarray:
    overlay = frame.copy()

    for a, b in HALPE26_SKELETON:
        if vis[a] > 0.1 and vis[b] > 0.1:
            cv2.line(
                overlay,
                (int(pts[a, 0]), int(pts[a, 1])),
                (int(pts[b, 0]), int(pts[b, 1])),
                (180, 180, 180),
                1,
                cv2.LINE_AA,
            )

    for i in range(26):
        if vis[i] < 0.1:
            continue
        color = KP_COLORS[i]
        pt = (int(pts[i, 0]), int(pts[i, 1]))
        radius = 4 if i < 17 else 6
        cv2.circle(overlay, pt, radius, color, -1, cv2.LINE_AA)

    # Label foot points
    for i in range(17, 23):
        if vis[i] > 0.1:
            pt = (int(pts[i, 0]), int(pts[i, 1]))
            cv2.putText(
                overlay,
                HALPE26_NAMES[i][:7],
                (pt[0] + 8, pt[1] - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

    return overlay


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate HALPE26 keypoint projection")
    parser.add_argument("--sequence", default="Axel_10")
    parser.add_argument("--camera", type=int, default=1)
    parser.add_argument("--frame", type=int, default=50)
    parser.add_argument("--split", default="train_set")
    parser.add_argument("--save", help="Save output image")
    args = parser.parse_args()

    npy_path = find_sequence(args.split, args.sequence, args.camera)
    if npy_path is None:
        print("Error: sequence not found")
        return 1

    kp3d = np.load(npy_path)
    coco_kps = np.load(npy_path.parent / f"{npy_path.stem}_coco.npy")

    if args.frame >= len(kp3d):
        print(f"Error: frame {args.frame} out of range (0-{len(kp3d) - 1})")
        return 1

    # Get camera from JSON metadata
    json_path = npy_path.with_suffix(".json")
    with json_path.open() as f:
        meta = json.load(f)

    with (DATA_ROOT / "cam_param.json").open() as f:
        cam_params = json.load(f)

    cam_key = meta["cam"]
    cam = cam_params[cam_key]

    # Project foot keypoints
    foot_2d = project_foot_frame(kp3d[args.frame], cam)
    coco_2d = coco_kps[args.frame]

    # Validate foot projections (sets invalid points to NaN in-place)
    pre_valid = int(np.sum(~np.isnan(foot_2d[:, 0])))
    validate_foot_projection(foot_2d, coco_2d)
    post_valid = int(np.sum(~np.isnan(foot_2d[:, 0])))
    rejected = pre_valid - post_valid

    # Merge
    pts, vis = merge_coco_foot_keypoints(coco_2d, foot_2d)

    # Draw
    mp4_path = npy_path.with_suffix(".mp4")
    cap = cv2.VideoCapture(str(mp4_path))
    ret, frame = cap.read()
    cap.release()
    if not ret:
        print("Error: cannot read frame")
        return 1

    overlay = draw_halpe26(frame, pts, vis)

    # Stats
    valid = (vis > 0.1).sum()
    foot_valid = (vis[17:23] > 0.1).sum()
    print(f"Sequence: {args.sequence} | Camera: {cam_key} | Frame: {args.frame}/{len(kp3d) - 1}")
    print(f"Valid keypoints: {valid}/26 (foot: {foot_valid}/6, {rejected} rejected by validation)")
    print("COCO 17kp from _coco.npy, foot 6kp projected from 3D mocap")

    if args.save:
        cv2.imwrite(args.save, overlay)
        print(f"Saved to: {args.save}")
    else:
        try:
            import matplotlib

            matplotlib.use("TkAgg")
            import matplotlib.pyplot as plt

            plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
            plt.title(f"{args.sequence} cam_{args.camera} frame {args.frame}")
            plt.axis("off")
            plt.show()
        except ImportError:
            cv2.imwrite("/tmp/validation.jpg", overlay)
            print("Saved to /tmp/validation.jpg")

    return 0


if __name__ == "__main__":
    sys.exit(main())
