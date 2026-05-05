#!/usr/bin/env python3
"""
Process video with MogaNet-B pose estimation + skeleton overlay.
Simple approach: resize frame to 384x288, run MogaNet-B, draw COCO skeleton.
"""

import sys

sys.path.insert(0, "/root")

from pathlib import Path

import cv2
import numpy as np
import torch
from moganet_official import MogaNet
from torch import nn


class MogaNet_feat(MogaNet):
    def __init__(self, **kwargs):
        super().__init__(fork_feat=True, **kwargs)


class DeconvHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.deconv1 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False), nn.BatchNorm2d(256), nn.ReLU(True)
        )
        self.deconv2 = nn.Sequential(
            nn.ConvTranspose2d(256, 256, 4, 2, 1, bias=False), nn.BatchNorm2d(256), nn.ReLU(True)
        )
        self.deconv3 = nn.Sequential(
            nn.ConvTranspose2d(256, 256, 4, 2, 1, bias=False), nn.BatchNorm2d(256), nn.ReLU(True)
        )
        self.final = nn.Conv2d(256, 17, 1)

    def forward(self, x):
        x = self.deconv1(x)
        x = self.deconv2(x)
        x = self.deconv3(x)
        x = self.final(x)
        return x


def load_model(checkpoint_path, device):
    backbone = MogaNet_feat(arch="base", out_indices=(0, 1, 2, 3)).to(device)
    head = DeconvHead().to(device)

    ckpt = torch.load(checkpoint_path, weights_only=False, map_location=device)

    backbone_state = {}
    for k, v in ckpt["state_dict"].items():
        if k.startswith("backbone."):
            backbone_state[k.replace("backbone.", "")] = v
    backbone.load_state_dict(backbone_state, strict=True)
    backbone.eval()

    head_state = {}
    for k, v in ckpt["state_dict"].items():
        if "keypoint_head" in k:
            if "deconv_layers" in k:
                parts = k.replace("keypoint_head.deconv_layers.", "").split(".")
                idx, rest = int(parts[0]), ".".join(parts[1:])
                mapping = {
                    0: "deconv1.0",
                    1: "deconv1.1",
                    3: "deconv2.0",
                    4: "deconv2.1",
                    6: "deconv3.0",
                    7: "deconv3.1",
                }
                if idx in mapping:
                    head_state[f"{mapping[idx]}.{rest}"] = v
            elif "final_layer" in k:
                head_state[k.replace("keypoint_head.final_layer", "final")] = v
    head.load_state_dict(head_state, strict=True)
    head.eval()

    return backbone, head


def decode_heatmap_simple(heatmaps, input_size, heatmap_size):
    """Simple argmax decoder from heatmaps."""
    K, H, W = heatmaps.shape
    heatmaps_flatten = heatmaps.reshape(K, -1)
    y_locs, x_locs = np.unravel_index(np.argmax(heatmaps_flatten, axis=1), shape=(H, W))
    keypoints = np.stack((x_locs, y_locs), axis=-1).astype(np.float32)
    vals = np.amax(heatmaps_flatten, axis=1)
    # Normalize to input image coordinates
    keypoints[:, 0] = keypoints[:, 0] / (W - 1) * input_size[0]
    keypoints[:, 1] = keypoints[:, 1] / (H - 1) * input_size[1]
    return keypoints, vals


def draw_skeleton(img, keypoints, skeleton, color=(0, 255, 0), pt_color=(0, 0, 255), thickness=2):
    h, w = img.shape[:2]
    for pair in skeleton:
        kp1 = keypoints[pair[0]]
        kp2 = keypoints[pair[1]]
        if kp1[0] > 0 and kp1[1] > 0 and kp2[0] > 0 and kp2[1] > 0:
            pt1 = (int(kp1[0]), int(kp1[1]))
            pt2 = (int(kp2[0]), int(kp2[1]))
            cv2.line(img, pt1, pt2, color, thickness)
    for kp in keypoints:
        if kp[0] > 0 and kp[1] > 0:
            pt = (int(kp[0]), int(kp[1]))
            cv2.circle(img, pt, 3, pt_color, -1)
    return img


def process_video(video_path, output_path, backbone, head, device):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Failed to open {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Video: {width}x{height} @ {fps:.1f}fps, {total_frames} frames")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    skeleton = [
        [15, 13],
        [13, 11],
        [16, 14],
        [14, 12],
        [11, 12],
        [5, 11],
        [6, 12],
        [5, 6],
        [5, 7],
        [6, 8],
        [7, 9],
        [8, 10],
        [1, 2],
        [0, 1],
        [0, 2],
        [1, 3],
        [2, 4],
        [3, 5],
        [4, 6],
    ]

    frame_idx = 0
    processed = 0
    with torch.no_grad():
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Resize frame to model input size
            img_resized = cv2.resize(frame, (384, 288))
            img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)

            # Normalize
            img_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 255.0
            img_tensor = (img_tensor - torch.from_numpy(mean).view(3, 1, 1)) / torch.from_numpy(
                std
            ).view(3, 1, 1)
            img_tensor = img_tensor.unsqueeze(0).to(device)

            # Forward
            feats = backbone(img_tensor)
            heatmaps = head(feats[3])  # (1, 17, 72, 96)
            heatmaps_np = heatmaps[0].cpu().numpy()  # (17, 72, 96)

            # Decode
            keypoints, scores = decode_heatmap_simple(heatmaps_np, (384, 288), (96, 72))

            # Scale keypoints back to original frame size
            scale_x = width / 384.0
            scale_y = height / 288.0
            keypoints[:, 0] *= scale_x
            keypoints[:, 1] *= scale_y

            # Draw skeleton on original frame
            frame_drawn = draw_skeleton(frame.copy(), keypoints, skeleton)

            out.write(frame_drawn)
            processed += 1

            if frame_idx % 30 == 0:
                print(f"  Frame {frame_idx}/{total_frames} ({processed} processed)")
            frame_idx += 1

    cap.release()
    out.release()
    print(f"Done. Output: {output_path} ({processed} frames)")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    print("Loading model...")
    backbone, head = load_model("/root/moganet_b_ap2d_384x288.pth", device)
    print("Model loaded.")

    video_path = "/root/videos/волчок.MOV"
    output_path = "/root/videos/волчок_moganet.mp4"

    Path("/root/videos").mkdir(exist_ok=True)

    print(f"\nProcessing {video_path}...")
    process_video(video_path, output_path, backbone, head, device)


if __name__ == "__main__":
    main()
