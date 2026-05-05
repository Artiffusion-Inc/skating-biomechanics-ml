#!/usr/bin/env python3
"""
Benchmark MogaNet-B inference on GPU.
Measures latency per crop and throughput for batch_size=1.
"""

import sys

sys.path.insert(0, "/root")

import time

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

    # Load backbone
    backbone_state = {}
    for k, v in ckpt["state_dict"].items():
        if k.startswith("backbone."):
            backbone_state[k.replace("backbone.", "")] = v
    backbone.load_state_dict(backbone_state, strict=True)
    backbone.eval()

    # Load head
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


def benchmark(backbone, head, device, num_warmup=50, num_iters=200, input_size=(1, 3, 288, 384)):
    dummy = torch.randn(input_size, device=device)

    # Warmup
    print(f"Warming up {num_warmup} iterations...")
    torch.cuda.synchronize() if device.type == "cuda" else None
    for _ in range(num_warmup):
        with torch.no_grad():
            feats = backbone(dummy)
            _ = head(feats[3])
        if device.type == "cuda":
            torch.cuda.synchronize()

    # Benchmark
    print(f"Benchmarking {num_iters} iterations...")
    latencies = []
    torch.cuda.synchronize() if device.type == "cuda" else None

    for i in range(num_iters):
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()

        with torch.no_grad():
            feats = backbone(dummy)
            out = head(feats[3])

        if device.type == "cuda":
            torch.cuda.synchronize()
        end = time.perf_counter()

        latencies.append((end - start) * 1000)  # ms

    latencies = np.array(latencies)
    return latencies


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    print("\nLoading model...")
    backbone, head = load_model("/root/moganet_b_ap2d_384x288.pth", device)

    print("\nModel loaded. Parameters:")
    total_params = sum(p.numel() for p in backbone.parameters()) + sum(
        p.numel() for p in head.parameters()
    )
    print(f"  Total: {total_params / 1e6:.1f}M")
    print(f"  Backbone: {sum(p.numel() for p in backbone.parameters()) / 1e6:.1f}M")
    print(f"  Head: {sum(p.numel() for p in head.parameters()) / 1e6:.1f}M")

    # Benchmark at input size (1, 3, 288, 384) — MogaNet expects H=288, W=384
    print("\n=== Benchmark: batch=1, input=(3, 288, 384) ===")
    latencies = benchmark(backbone, head, device, num_warmup=50, num_iters=200)

    print("\nResults:")
    print(f"  Mean:   {latencies.mean():.2f} ms")
    print(f"  Median: {np.median(latencies):.2f} ms")
    print(f"  Std:    {latencies.std():.2f} ms")
    print(f"  Min:    {latencies.min():.2f} ms")
    print(f"  Max:    {latencies.max():.2f} ms")
    print(f"  P95:    {np.percentile(latencies, 95):.2f} ms")
    print(f"  P99:    {np.percentile(latencies, 99):.2f} ms")
    print(f"  FPS:    {1000.0 / latencies.mean():.1f}")

    # Also benchmark with torch.compile if available (PyTorch 2.x)
    if hasattr(torch, "compile"):
        print("\n=== Trying torch.compile ===")
        try:
            compiled_backbone = torch.compile(backbone)
            compiled_head = torch.compile(head)
            latencies_compiled = benchmark(
                compiled_backbone, compiled_head, device, num_warmup=50, num_iters=200
            )
            print(
                f"Compiled Mean: {latencies_compiled.mean():.2f} ms ({latencies.mean() / latencies_compiled.mean():.2f}x speedup)"
            )
        except Exception as e:
            print(f"torch.compile failed: {e}")

    print("\nDone.")


if __name__ == "__main__":
    main()
