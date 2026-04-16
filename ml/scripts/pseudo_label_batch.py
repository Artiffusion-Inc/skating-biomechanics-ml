#!/usr/bin/env python3
"""
Simple batch pseudo-labeling using working MogaNet-B code.
Based on moganet_decode.py structure.

Usage:
    python pseudo_label_batch.py \
        --frames-dir /root/data/datasets/skatingverse/frames \
        --output /root/data/datasets/skatingverse_pseudo/labels.json
"""

import sys

sys.path.insert(0, "/root")

import json
from pathlib import Path

import cv2
import numpy as np

# Mock xtcocotools
import pycocotools.mask as cocomask
import torch
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from tqdm import tqdm


class XtcocoTools:
    class coco:
        COCO = COCO

    class cocoeval:
        COCOeval = COCOeval

    class mask:
        encode = cocomask.encode
        decode = cocomask.decode


sys.modules["xtcocotools"] = XtcocoTools
sys.modules["xtcocotools.coco"] = XtcocoTools.coco
sys.modules["xtcocotools.cocoeval"] = XtcocoTools.cocoeval
sys.modules["xtcocotools.mask"] = XtcocoTools.mask

from moganet_official import MogaNet_feat


# UDP Decoder functions (from mmpose)
def gaussian_blur(heatmaps: np.ndarray, kernel: int = 11) -> np.ndarray:
    assert kernel % 2 == 1
    border = (kernel - 1) // 2
    K, H, W = heatmaps.shape
    for k in range(K):
        origin_max = np.max(heatmaps[k])
        dr = np.zeros((H + 2 * border, W + 2 * border), dtype=np.float32)
        dr[border:-border, border:-border] = heatmaps[k].copy()
        dr = cv2.GaussianBlur(dr, (kernel, kernel), 0)
        heatmaps[k] = dr[border:-border, border:-border].copy()
        heatmaps[k] *= origin_max / np.max(heatmaps[k])
    return heatmaps


def get_heatmap_maximum(heatmaps: np.ndarray):
    K, H, W = heatmaps.shape
    heatmaps_flatten = heatmaps.reshape(K, -1)
    y_locs, x_locs = np.unravel_index(np.argmax(heatmaps_flatten, axis=1), shape=(H, W))
    locs = np.stack((x_locs, y_locs), axis=-1).astype(np.float32)
    vals = np.amax(heatmaps_flatten, axis=1)
    locs[vals <= 0.0] = -1
    return locs, vals


def refine_keypoints_dark_udp(
    keypoints: np.ndarray, heatmaps: np.ndarray, blur_kernel_size: int
) -> np.ndarray:
    N, K = keypoints.shape[:2]
    H, W = heatmaps.shape[1:]
    heatmaps = gaussian_blur(heatmaps, blur_kernel_size)
    np.clip(heatmaps, 1e-3, 50.0, heatmaps)
    np.log(heatmaps, heatmaps)
    heatmaps_pad = np.pad(heatmaps, ((0, 0), (1, 1), (1, 1)), mode="edge").flatten()
    for n in range(N):
        index = keypoints[n, :, 0] + 1 + (keypoints[n, :, 1] + 1) * (W + 2)
        index += (W + 2) * (H + 2) * np.arange(0, K)
        index = index.astype(int).reshape(-1, 1)
        i_ = heatmaps_pad[index]
        ix1 = heatmaps_pad[index + 1]
        iy1 = heatmaps_pad[index + W + 2]
        ix1y1 = heatmaps_pad[index + W + 3]
        ix1_y1_ = heatmaps_pad[index - W - 3]
        ix1_ = heatmaps_pad[index - 1]
        iy1_ = heatmaps_pad[index - 2 - W]
        dx = 0.5 * (ix1 - ix1_)
        dy = 0.5 * (iy1 - iy1_)
        derivative = np.concatenate([dx, dy], axis=1)
        derivative = derivative.reshape(K, 2, 1)
        dxx = ix1 - 2 * i_ + ix1_
        dyy = iy1 - 2 * i_ + iy1_
        dxy = 0.5 * (ix1y1 - ix1 - iy1 + i_ + i_ - ix1_ - iy1_ + ix1_y1_)
        hessian = np.concatenate([dxx, dxy, dxy, dyy], axis=1)
        hessian = hessian.reshape(K, 2, 2)
        hessian = np.linalg.inv(hessian + np.finfo(np.float32).eps * np.eye(2))
        keypoints[n] -= np.einsum("imn,ink->imk", hessian, derivative).squeeze()
    return keypoints


# Deconv Head
class DeconvHead(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.deconv1 = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(True),
        )
        self.deconv2 = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(256, 256, 4, 2, 1, bias=False),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(True),
        )
        self.deconv3 = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(256, 256, 4, 2, 1, bias=False),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(True),
        )
        self.final = torch.nn.Conv2d(256, 17, 1, 1)

    def forward(self, x):
        x = self.deconv1(x)
        x = self.deconv2(x)
        x = self.deconv3(x)
        x = self.final(x)
        return x


# Load MogaNet-B model
print("Loading MogaNet-B...")
checkpoint = torch.load(
    "/root/data/models/athletepose3d/moganet_b_ap2d_384x288.pth",
    weights_only=False,
    map_location="cuda",
)
state_dict = checkpoint["state_dict"]

backbone = MogaNet_feat(arch="base", out_indices=(3,)).cuda()
head = DeconvHead().cuda()

# Load weights with prefix stripping
backbone_state = {}
head_state = {}

for k, v in state_dict.items():
    if "backbone" in k:
        new_k = k.replace("module.", "")
        backbone_state[new_k] = v
    elif "keypoint_head" in k:
        if "deconv_layers" in k:
            suffix = k.replace("keypoint_head.deconv_layers.", "")
            parts = suffix.split(".")
            idx = int(parts[0])
            rest = ".".join(parts[1:])
            if idx == 0:
                new_k = f"deconv1.0.{rest}"
            elif idx == 1:
                new_k = f"deconv1.1.{rest}"
            elif idx == 3:
                new_k = f"deconv2.0.{rest}"
            elif idx == 4:
                new_k = f"deconv2.1.{rest}"
            elif idx == 6:
                new_k = f"deconv3.0.{rest}"
            elif idx == 7:
                new_k = f"deconv3.1.{rest}"
        elif "final_layer" in k:
            new_k = k.replace("keypoint_head.final_layer", "final")
        else:
            continue
        head_state[new_k] = v

backbone.load_state_dict(backbone_state, strict=False)
head.load_state_dict(head_state, strict=False)
backbone.eval()
head.eval()

print("Model loaded")

# ImageNet normalization
mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# Process all frames
frames_dir = Path("/root/data/datasets/skatingverse/frames")
output_file = Path("/root/data/datasets/skatingverse_pseudo/labels.json")
output_file.parent.mkdir(parents=True, exist_ok=True)

all_results = []
annotation_id = 1

# Find all frame directories
frame_dirs = sorted([d for d in frames_dir.iterdir() if d.is_dir()])

print(f"Found {len(frame_dirs)} video directories")

for video_dir in tqdm(frame_dirs, desc="Processing videos"):
    video_id = video_dir.name
    frame_files = sorted(video_dir.glob("frame_*.jpg"))

    for frame_path in frame_files:
        # Read image
        img = cv2.imread(str(frame_path))
        if img is None:
            continue

        h, w = img.shape[:2]

        # Preprocess
        img_resized = cv2.resize(img, (384, 288))
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        img_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 255.0
        img_tensor = (img_tensor - torch.from_numpy(mean).view(3, 1, 1)) / torch.from_numpy(
            std
        ).view(3, 1, 1)
        img_tensor = img_tensor.unsqueeze(0).cuda()

        # Inference
        with torch.no_grad():
            features = backbone(img_tensor)
            heatmaps = head(features[3])  # Use index 3, not -1

        # Postprocess
        heatmaps_np = heatmaps.squeeze(0).cpu().numpy()
        keypoints, scores = get_heatmap_maximum(heatmaps_np)
        keypoints = keypoints[None]
        keypoints = refine_keypoints_dark_udp(keypoints, heatmaps_np, blur_kernel_size=11)

        # Normalize to input size
        W, H = 96, 72
        keypoints = keypoints / [W - 1, H - 1] * [384, 288]

        # Scale to original image size
        scale_x = w / 384
        scale_y = h / 288
        keypoints[:, :, 0] *= scale_x
        keypoints[:, :, 1] *= scale_y

        # Flatten to COCO format
        # NOTE: scores are raw heatmap max values (0-0.95), not calibrated
        # Use empirical thresholds based on actual distribution
        keypoints_flat = []
        for i in range(17):
            x, y = keypoints[0, i]
            # Quick fix: use lower threshold for raw heatmap values
            # TODO: proper softmax normalization
            if scores[i] > 0.001:
                v = 2.0  # High confidence
            elif scores[i] > 0.0005:
                v = 1.0  # Medium confidence
            else:
                v = 0.0  # Low confidence
            keypoints_flat.extend([float(x), float(y), v])

        # Calibrated score for the whole pose
        # Use geometric mean of non-zero scores
        valid_scores = scores[scores > 0.0001]
        if len(valid_scores) > 0:
            calibrated_score = float(np.mean(valid_scores))
        else:
            calibrated_score = 0.0

        # Create result
        result = {
            "image_id": str(frame_path.relative_to(frames_dir)),
            "video_id": video_id,
            "keypoints": keypoints_flat,
            "score": calibrated_score,
        }

        all_results.append(result)
        annotation_id += 1

# Save results
print(f"Processed {len(all_results)} frames")

with open(output_file, "w") as f:
    json.dump({"annotations": all_results, "total": len(all_results)}, f, indent=2)

print(f"Saved to: {output_file}")
