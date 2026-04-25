#!/usr/bin/env python3
"""Convert MogaNet-B teacher coords from HDF5 to YOLO label format.

Reads teacher_coords.h5 (coords in crop [0,1] space + crop_params)
and writes YOLO pose labels (normalized by original image dimensions).

Usage:
    python3 convert_teacher_labels.py \
        --input data/teacher_coords.h5 \
        --output data/teacher-labels \
        --base /root/skating-biomechanics-ml/experiments/yolo26-pose-kd/data
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from contextlib import suppress
from pathlib import Path

import h5py
import numpy as np
from tqdm import tqdm


def convert_entry(
    teacher_xy: np.ndarray,
    crop_params: np.ndarray,
) -> tuple[list[float], list[float]]:
    x1, y1, cw, ch, iw, ih = crop_params
    orig_x = teacher_xy[:, 0] * cw + x1
    orig_y = teacher_xy[:, 1] * ch + y1
    yolo_x = np.clip(orig_x / iw, 0.0, 1.0)
    yolo_y = np.clip(orig_y / ih, 0.0, 1.0)

    cx = np.clip((x1 + cw / 2) / iw, 0.0, 1.0)
    cy = np.clip((y1 + ch / 2) / ih, 0.0, 1.0)
    w = np.clip(cw / iw, 0.0, 1.0)
    h = np.clip(ch / ih, 0.0, 1.0)
    bbox = [cx, cy, w, h]

    kps = []
    for i in range(17):
        nan_x = np.isnan(yolo_x[i])
        nan_y = np.isnan(yolo_y[i])
        kps.append(0.0 if nan_x else float(yolo_x[i]))
        kps.append(0.0 if nan_y else float(yolo_y[i]))
        kps.append(0 if (nan_x or nan_y) else 1)

    return bbox, kps


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--base", required=True)
    args = parser.parse_args()

    h5_path = Path(args.input)
    output_root = Path(args.output)
    base_path = Path(args.base)

    print(f"Loading {h5_path}...")
    with h5py.File(str(h5_path), "r") as f:
        coords = f["coords"][:]
        crop_params = f["crop_params"][:]
        idx = json.loads(f.attrs["index"])

    print(f"Total entries: {len(idx)}")

    img_entries = defaultdict(list)
    for img_path, row in idx.items():
        rel_path = img_path
        if rel_path.startswith("experiments/yolo26-pose-kd/data/"):
            rel_path = rel_path[len("experiments/yolo26-pose-kd/data/") :]
        img_entries[rel_path].append(row)

    print(f"Unique images: {len(img_entries)}")
    multi_person = sum(1 for v in img_entries.values() if len(v) > 1)
    print(f"Multi-person images: {multi_person}")

    written = 0
    skipped = 0
    nan_entries = 0

    for rel_path, rows in tqdm(img_entries.items(), desc="Converting"):
        lines = []
        for row in rows:
            teacher_xy = coords[row]
            cp = crop_params[row]

            if cp[0] < 0:
                skipped += 1
                continue
            if np.isnan(teacher_xy).all():
                skipped += 1
                continue
            if np.isnan(teacher_xy).any():
                nan_entries += 1

            bbox, kps = convert_entry(teacher_xy, cp)
            parts = ["0"] + [f"{v:.6f}" for v in bbox] + [f"{v:.6f}" for v in kps]
            lines.append(" ".join(parts))

        if not lines:
            continue

        parts = rel_path.split("/")
        split_parts = parts[0].split("/")
        if len(split_parts) == 2:
            ds, split = split_parts
        else:
            continue

        label_dir = output_root / ds / split / "labels"
        label_dir.mkdir(parents=True, exist_ok=True)
        img_stem = Path(rel_path).stem
        label_file = label_dir / f"{img_stem}.txt"

        with open(label_file, "w") as f:
            f.write("\n".join(lines) + "\n")

        written += 1

    print(f"\nImages written: {written}")
    print(f"Skipped (invalid): {skipped}")
    print(f"Entries with NaN coords: {nan_entries}")

    print("\nCreating image symlinks...")
    for rel_path in tqdm(img_entries.keys(), desc="Symlinks"):
        parts = rel_path.split("/")
        split_parts = parts[0].split("/")
        if len(split_parts) != 2:
            continue
        ds, split = split_parts

        src_img = base_path / rel_path
        dst_dir = output_root / ds / split / "images"
        dst_dir.mkdir(parents=True, exist_ok=True)
        dst_file = dst_dir / src_img.name

        if not dst_file.exists() and src_img.exists():
            with suppress(FileExistsError):
                dst_file.symlink_to(src_img.resolve())

    print("Done.")


if __name__ == "__main__":
    main()
