"""Create YOLO pose dataset from AthletePose3D + COCO.

Strategy:
- Train: AP3D train_set (71K) + 15% COCO train2017 person_keypoints
- Val: COCO val2017 person_keypoints (holdout, no leak with AP3D)

Usage:
    python3 experiments/yolo26-pose/create_yolo_dataset.py
"""

import json
import os
import random
import shutil
from collections import defaultdict
from pathlib import Path

random.seed(42)

BASE = Path("/root/data/datasets/athletepose3d")
COCO_BASE = Path("/root/data/datasets/coco")
YOLO_DIR = Path("/root/data/datasets/yolo26_ap3d")


def coco_kp_to_yolo(kps, bbox, img_w, img_h):
    """Convert COCO keypoints + bbox to YOLO pose format.

    COCO: keypoints=[x,y,v,...], bbox=[x,y,w,h]
    YOLO: class cx cy w h kp1_x kp1_y kp1_v ... kp17_x kp17_y kp17_v
    All normalized to [0,1].
    """
    x, y, w, h = bbox
    cx = (x + w / 2) / img_w
    cy = (y + h / 2) / img_h
    nw = w / img_w
    nh = h / img_h

    kp_norm = []
    for i in range(0, len(kps), 3):
        kx, ky, kv = kps[i], kps[i + 1], kps[i + 2]
        if kv > 0:
            kp_norm.extend([kx / img_w, ky / img_h, kv])
        else:
            kp_norm.extend([0.0, 0.0, 0.0])

    return [0.0, cx, cy, nw, nh] + kp_norm


def main():
    # Step 1: Create directory structure
    for d in ["images/train", "images/val", "labels/train", "labels/val"]:
        (YOLO_DIR / d).mkdir(parents=True, exist_ok=True)

    # Step 2: Convert AP3D train_set
    print("Loading AP3D train annotations...")
    with open(BASE / "pose_2d/annotations/train_set.json") as f:
        ap3d = json.load(f)

    img_map = {img["id"]: img for img in ap3d["images"]}
    img_anns = defaultdict(list)
    for ann in ap3d["annotations"]:
        img_anns[ann["image_id"]].append(ann)

    print(f"Converting {len(ap3d['annotations'])} AP3D annotations...")
    converted = 0
    for img_id, anns_list in img_anns.items():
        img_info = img_map.get(img_id)
        if not img_info:
            continue

        src_img = BASE / "pose_2d/train_set" / img_info["file_name"]
        if not src_img.exists():
            continue

        dst_img = YOLO_DIR / "images/train" / img_info["file_name"]
        if not dst_img.exists():
            os.symlink(src_img, dst_img)

        label_path = YOLO_DIR / "labels/train" / (img_info["file_name"].replace(".jpg", ".txt"))
        with open(label_path, "w") as lf:
            for ann in anns_list:
                if ann["num_keypoints"] < 3:
                    continue
                yolo_line = coco_kp_to_yolo(
                    ann["keypoints"], ann["bbox"],
                    img_info["width"], img_info["height"]
                )
                lf.write(" ".join(f"{v:.6f}" for v in yolo_line) + "\n")
        converted += 1

    print(f"AP3D converted: {converted} images")

    # Step 3: Sample 15% COCO person keypoints for train
    print("Loading COCO person keypoints...")
    with open(COCO_BASE / "annotations/person_keypoints_train2017.json") as f:
        coco_train = json.load(f)

    # Filter person images with keypoints
    kp_img_ids = set()
    for ann in coco_train["annotations"]:
        if ann["num_keypoints"] >= 5:
            kp_img_ids.add(ann["image_id"])
    person_imgs = [img for img in coco_train["images"] if img["id"] in kp_img_ids]
    print(f"COCO person images: {len(person_imgs)}")

    sample_size = int(len(person_imgs) * 0.15)
    sampled = random.sample(person_imgs, sample_size)
    print(f"Sampled {sample_size} COCO images for train mix")

    coco_ann_map = defaultdict(list)
    for ann in coco_train["annotations"]:
        if ann["num_keypoints"] >= 5:
            coco_ann_map[ann["image_id"]].append(ann)

    coco_converted = 0
    for img_info in sampled:
        anns_list = coco_ann_map[img_info["id"]]
        src_img = COCO_BASE / "train2017" / img_info["file_name"]
        if not src_img.exists():
            continue

        dst_img = YOLO_DIR / "images/train" / ("coco_" + img_info["file_name"])
        if not dst_img.exists():
            os.symlink(src_img, dst_img)

        label_path = YOLO_DIR / "labels/train" / ("coco_" + img_info["file_name"].replace(".jpg", ".txt"))
        with open(label_path, "w") as lf:
            for ann in anns_list:
                yolo_line = coco_kp_to_yolo(
                    ann["keypoints"], ann["bbox"],
                    img_info["width"], img_info["height"]
                )
                lf.write(" ".join(f"{v:.6f}" for v in yolo_line) + "\n")
        coco_converted += 1

    print(f"COCO converted: {coco_converted} images")

    # Step 4: COCO val2017 as validation set
    print("Loading COCO val2017 person keypoints...")
    with open(COCO_BASE / "annotations/person_keypoints_val2017.json") as f:
        coco_val = json.load(f)

    kp_val_ids = set()
    for ann in coco_val["annotations"]:
        if ann["num_keypoints"] >= 5:
            kp_val_ids.add(ann["image_id"])
    val_imgs = [img for img in coco_val["images"] if img["id"] in kp_val_ids]

    coco_val_ann_map = defaultdict(list)
    for ann in coco_val["annotations"]:
        if ann["num_keypoints"] >= 5:
            coco_val_ann_map[ann["image_id"]].append(ann)

    val_converted = 0
    for img_info in val_imgs:
        anns_list = coco_val_ann_map[img_info["id"]]
        src_img = COCO_BASE / "val2017" / img_info["file_name"]
        if not src_img.exists():
            continue

        dst_img = YOLO_DIR / "images/val" / img_info["file_name"]
        if not dst_img.exists():
            shutil.copy2(src_img, dst_img)

        label_path = YOLO_DIR / "labels/val" / img_info["file_name"].replace(".jpg", ".txt")
        with open(label_path, "w") as lf:
            for ann in anns_list:
                yolo_line = coco_kp_to_yolo(
                    ann["keypoints"], ann["bbox"],
                    img_info["width"], img_info["height"]
                )
                lf.write(" ".join(f"{v:.6f}" for v in yolo_line) + "\n")
        val_converted += 1

    print(f"Val converted: {val_converted} images")

    # Step 5: Write data.yaml
    data_yaml = f"""# YOLO pose dataset: AthletePose3D + COCO mix
path: {YOLO_DIR}
train: images/train
val: images/val

# Keypoints
kpt_shape: [17, 3]  # 17 keypoints, 3 values each (x, y, visibility)
flip_idx: [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]

# Classes
names:
  0: person

# Dataset stats
# Train: AP3D {converted} + COCO {coco_converted} = {converted + coco_converted} images
# Val: COCO {val_converted} images
"""

    yaml_path = YOLO_DIR / "data.yaml"
    yaml_path.write_text(data_yaml)
    print(f"\ndata.yaml written to {yaml_path}")

    # Summary
    train_imgs = len(os.listdir(YOLO_DIR / "images/train"))
    train_labels = len(os.listdir(YOLO_DIR / "labels/train"))
    val_imgs = len(os.listdir(YOLO_DIR / "images/val"))
    val_labels = len(os.listdir(YOLO_DIR / "labels/val"))
    print("\n=== YOLO Dataset Ready ===")
    print(f"Train: {train_imgs} images, {train_labels} labels")
    print(f"Val: {val_imgs} images, {val_labels} labels")


if __name__ == "__main__":
    main()
