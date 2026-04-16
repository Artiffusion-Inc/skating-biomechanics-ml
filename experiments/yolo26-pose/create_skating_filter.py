"""Create skating-only image ID filter from AthletePose3D video metadata.

After rsync completes, this script maps video subjects to pose_2d image IDs.

Strategy:
1. Parse video JSON files in S1/S2 to identify skating elements
2. Match video metadata (size, camera) to pose_2d image patterns
3. Export filter list of skating-only image IDs

Usage:
    python3 experiments/yolo26-pose/create_skating_filter.py
"""

import json
from pathlib import Path

# Paths
AP3D_VIDEOS = Path("/root/data/datasets/athletepose3d/videos/train_set")
POSE_2D_ANNOTATIONS = Path("/root/data/datasets/athletepose3d/pose_2d/pose_2d/annotations/train_set.json")
OUTPUT_FILTER = Path("/root/data/datasets/yolo26_ap3d/skating_image_ids.txt")

# Skating elements
SKATING_ELEMENTS = {
    "Axel", "Flip", "Loop", "Lutz", "Salchow", "Toeloop", "Comb"
}

def main():
    print("Loading pose_2d annotations...")
    with open(POSE_2D_ANNOTATIONS) as f:
        pose_2d = json.load(f)

    # Group images by size
    images_by_size = {}
    for img in pose_2d['images']:
        size = (img['width'], img['height'])
        if size not in images_by_size:
            images_by_size[size] = []
        images_by_size[size].append(img['id'])

    print("Images by size:")
    for size, ids in sorted(images_by_size.items()):
        print(f"  {size}: {len(ids)} images")

    # For now, filter based on assumption:
    # Skating (S1/S2) primarily uses 1920x1088 cameras
    # Athletics (S3-S5) uses smaller 1280x768 cameras
    skating_size = (1920, 1088)
    skating_ids = images_by_size.get(skating_size, [])

    print(f"\nSkating filter (1920x1088): {len(skating_ids)} images")

    # Write filter
    OUTPUT_FILTER.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILTER, "w") as f:
        for img_id in skating_ids:
            f.write(f"{img_id}\n")

    print(f"Filter written to {OUTPUT_FILTER}")
    print("Note: This is a rough size-based filter. Refine with video metadata after rsync.")

if __name__ == "__main__":
    main()
