#!/usr/bin/env python3
"""Validate RTMO on AthletePose3D val set for fair comparison with YOLO26n."""

import sys
from pathlib import Path

import cv2
import numpy as np

try:
    from rtmlib import RTMPose
    from rtmlib.tools import cv2_whiten
except ImportError:
    print("❌ rtmlib not installed. Run: pip install rtmlib")
    sys.exit(1)

# Dataset paths
VAL_IMAGES = Path("/root/data/datasets/yolo26_ap3d/images/val")
VAL_LABELS = Path("/root/data/datasets/yolo26_ap3d/labels/val")

# RTMO model
RTMO_MODEL = "rtmo-l"  # Use large model for best comparison

print("🔍 RTMO Validation on AthletePose3D val set")
print(f"Images: {VAL_IMAGES}")
print(f"Labels: {VAL_LABELS}")
print(f"Model: {RTMO_MODEL}")
print()

# Initialize RTMO
print("⏳ Loading RTMO model...")
pose_estimator = RTMPose(RTMO_MODEL, device="cuda")
print("✅ RTMO loaded")
print()

# Validation metrics
total_images = 0
correct_keypoints = 0
total_keypoints = 0
keypoint_distances = []

# IoU threshold for correct detection
IOU_THRESHOLD = 0.5
DISTANCE_THRESHOLD = 0.05  # 5% of image size

print("🔬 Running validation...")
print()

# Process images
image_files = sorted(list(VAL_IMAGES.glob("*.jpg")))[:100]  # First 100 for speed
print(f"Processing {len(image_files)} images...")
print()

for img_idx, img_path in enumerate(image_files, 1):
    if img_idx % 20 == 0:
        print(f"  Progress: {img_idx}/{len(image_files)}")

    # Load image
    image = cv2.imread(str(img_path))
    if image is None:
        continue

    h, w = image.shape[:2]
    img_size = max(h, w)

    # Get corresponding label file
    label_path = VAL_LABELS / (img_path.stem + ".txt")
    if not label_path.exists():
        continue

    # Load GT keypoints
    with open(label_path) as f:
        label_data = f.read().strip().split()

    if len(label_data) < 17 * 3:  # 17 keypoints * 3 (x, y, conf)
        continue

    # Parse GT keypoints (YOLO format: normalized x, y, visibility)
    gt_kpts = []
    for i in range(17):
        x = float(label_data[i * 3 + 0]) * w
        y = float(label_data[i * 3 + 1]) * h
        vis = int(label_data[i * 3 + 2])
        if vis > 0:
            gt_kpts.append([x, y])

    if len(gt_kpts) < 5:  # Need at least 5 visible keypoints
        continue

    # Run RTMO inference
    keypoints = pose_estimator(image)

    if keypoints is None or len(keypoints) == 0:
        continue

    # RTMO returns list of detections, take first person
    pred_kpts = keypoints[0][:17, :2]  # First 17 keypoints, x,y

    # Calculate metrics
    for gt_pt, pred_pt in zip(gt_kpts, pred_kpts):
        total_keypoints += 1

        # Normalize distances
        gt_norm = np.array(gt_pt) / img_size
        pred_norm = np.array(pred_pt) / img_size

        distance = np.linalg.norm(gt_norm - pred_norm)

        if distance < DISTANCE_THRESHOLD:
            correct_keypoints += 1

        keypoint_distances.append(distance)

    total_images += 1

print()
print("=" * 60)
print("📊 RTMO VALIDATION RESULTS")
print("=" * 60)

# Calculate metrics
accuracy = correct_keypoints / total_keypoints if total_keypoints > 0 else 0
mean_distance = np.mean(keypoint_distances) if keypoint_distances else 0
median_distance = np.median(keypoint_distances) if keypoint_distances else 0

print(f"Dataset: AthletePose3D val (first {len(image_files)} images)")
print(f"Model: {RTMO_MODEL}")
print()
print(f"Total images: {total_images}")
print(f"Total keypoints: {total_keypoints}")
print(f"Correct keypoints (5% threshold): {correct_keypoints}")
print()
print(f"** Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%) **")
print(f"Mean distance: {mean_distance:.4f} ({mean_distance * 100:.2f}% of image size)")
print(f"Median distance: {median_distance:.4f} ({median_distance * 100:.2f}%)")
print()

# Compare with YOLO26n result
yolo_map = 0.406
print("=" * 60)
print("🔄 COMPARISON WITH YOLO26n")
print("=" * 60)
print(f"YOLO26n (batch64): mAP50-95(P) = {yolo_map:.3f}")
print(f"RTMO-{RTMO_MODEL}: Accuracy (5%) = {accuracy:.3f}")
print()
print("⚠️  NOTE: Different metrics!")
print("  - YOLO: Detection mAP50-95 (person + keypoints combined)")
print("  - RTMO: Keypoint accuracy (5% threshold)")
print()
print("Conclusion: Need to convert RTMO to YOLO format for direct comparison")
print("=" * 60)
