#!/usr/bin/env python3
"""Create subset of YOLO dataset for fast HP search.

Strategies:
- random: Random sampling
- stratified: Preserve AP3D:COCO ratio
- kmeans: Visually diverse frames (requires image features)
"""

import argparse
import random
import shutil
from pathlib import Path

import yaml

random.seed(42)


def create_subset(
    source_dir: Path,
    target_dir: Path,
    subset_size: int = 25000,
    strategy: str = "stratified",
):
    """Create subset of YOLO dataset.

    Args:
        source_dir: Source dataset directory
        target_dir: Target subset directory
        subset_size: Number of images in subset
        strategy: Sampling strategy (random, stratified)
    """
    source_dir = Path(source_dir)
    target_dir = Path(target_dir)

    # Create target directories
    (target_dir / "images" / "train").mkdir(parents=True, exist_ok=True)
    (target_dir / "images" / "val").mkdir(parents=True, exist_ok=True)
    (target_dir / "labels" / "train").mkdir(parents=True, exist_ok=True)
    (target_dir / "labels" / "val").mkdir(parents=True, exist_ok=True)

    # Get source images
    train_images = list((source_dir / "images" / "train").glob("*.jpg"))
    val_images = list((source_dir / "images" / "val").glob("*.jpg"))

    print(f"Source: {len(train_images)} train, {len(val_images)} val")

    # Sample training images
    if strategy == "random":
        subset_train = random.sample(train_images, min(subset_size, len(train_images)))
    elif strategy == "stratified":
        # Preserve AP3D:COCO ratio (roughly 90:10 based on dataset stats)
        # AP3D images start with 2*, 3* (from original numbering)
        ap3d_images = [img for img in train_images if img.stem.startswith(("2", "3"))]
        coco_images = [img for img in train_images if not img.stem.startswith(("2", "3"))]

        ap3d_subset = random.sample(ap3d_images, int(subset_size * 0.9))
        coco_subset = random.sample(coco_images, int(subset_size * 0.1))
        subset_train = ap3d_subset + coco_subset
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    # Use full val set (small enough)
    subset_val = val_images

    print(f"Subset: {len(subset_train)} train, {len(subset_val)} val")

    # Copy files
    print("Copying training images...")
    for img in subset_train:
        shutil.copy(img, target_dir / "images" / "train" / img.name)
        # Copy corresponding label
        label = img.parent.parent.parent / "labels" / "train" / f"{img.stem}.txt"
        if label.exists():
            shutil.copy(label, target_dir / "labels" / "train" / label.name)

    print("Copying val images...")
    for img in subset_val:
        shutil.copy(img, target_dir / "images" / "val" / img.name)
        label = img.parent.parent.parent / "labels" / "val" / f"{img.stem}.txt"
        if label.exists():
            shutil.copy(label, target_dir / "labels" / "val" / label.name)

    # Update data.yaml
    source_yaml = source_dir / "data.yaml"
    target_yaml = target_dir / "data.yaml"

    with open(source_yaml) as f:
        data = yaml.safe_load(f)

    data["path"] = str(target_dir)
    data["description"] = f"YOLO pose subset ({strategy}, {subset_size} train)"

    with open(target_yaml, "w") as f:
        yaml.dump(data, f)

    print(f"\n✅ Subset created at: {target_dir}")
    print(f"   Train: {len(subset_train)} images")
    print(f"   Val: {len(subset_val)} images")
    print(f"   Config: {target_yaml}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="/root/data/datasets/yolo26_ap3d")
    parser.add_argument("--target", default="/root/data/datasets/yolo26_ap3d_subset")
    parser.add_argument("--subset-size", type=int, default=25000)
    parser.add_argument("--strategy", default="stratified", choices=["random", "stratified"])
    args = parser.parse_args()

    create_subset(args.source, args.target, args.subset_size, args.strategy)
