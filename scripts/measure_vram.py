#!/usr/bin/env python3
"""Measure actual VRAM usage of each ONNX model in the pipeline.

Reports file sizes (including .onnx_data external files) vs the estimated
VRAM values used in ModelRegistry. File size is a lower bound on VRAM —
actual usage includes ORT overhead, execution tensors, and CUDA context
(~200-500MB).

Usage:
    uv run python scripts/measure_vram.py
"""

from __future__ import annotations

from pathlib import Path

import onnxruntime as ort

# Project models with their current estimated VRAM values (from web_helpers.py)
MODELS = {
    "depth_anything": ("data/models/depth_anything_v2_small.onnx", 200),
    "optical_flow": ("data/models/neuflowv2_mixed.onnx", 80),
    "sam2_vision_encoder": ("data/models/sam2/vision_encoder.onnx", 134),
    "sam2_prompt_decoder": ("data/models/sam2/prompt_encoder_mask_decoder.onnx", 21),
    "video_matting": ("data/models/rvm_mobilenetv3.onnx", 40),
    "lama_inpainting": ("data/models/lama_fp32.onnx", 300),
}


def _total_size(path: Path) -> float:
    """Get total file size in MB, including external data files."""
    size_mb = path.stat().st_size / (1024 * 1024)
    # ONNX external data can be .onnx.data or .onnx_data
    for suffix in (".onnx.data", "_data"):
        data_file = (
            Path(str(path) + suffix)
            if suffix == "_data"
            else Path(str(path).replace(".onnx", suffix))
        )
        if data_file.exists():
            size_mb += data_file.stat().st_size / (1024 * 1024)
    return size_mb


def main() -> None:
    has_cuda = "CUDAExecutionProvider" in ort.get_available_providers()

    print(f"{'Model':<25} {'File Size':>10} {'Est. VRAM':>12} {'Delta':>10} {'Status':<10}")
    print("-" * 70)

    total_file = 0.0
    total_est = 0
    missing = []

    for name, (path, est_mb) in MODELS.items():
        p = Path(path)
        total_est += est_mb
        if p.exists():
            file_mb = _total_size(p)
            total_file += file_mb
            delta = est_mb - file_mb
            delta_str = f"{delta:+.0f}MB"
            print(f"{name:<25} {file_mb:>8.1f}MB {est_mb:>10}MB {delta_str:>10} {'OK':<10}")
        else:
            missing.append(name)
            print(f"{name:<25} {'N/A':>10} {est_mb:>10}MB {'---':>10} {'MISSING':<10}")

    print("-" * 70)
    print(f"{'TOTAL':<25} {total_file:>8.1f}MB {total_est:>10}MB")
    print()
    if missing:
        print(f"Missing models: {', '.join(missing)}")
        print()
    if has_cuda:
        print("CUDA: available (for actual VRAM measurement, run with nvidia-smi)")
    else:
        print("CUDA: not available (file sizes shown as lower bound)")
    print()
    print("NOTE: File size is a lower bound on VRAM. Actual VRAM includes")
    print("ORT overhead, execution tensors, and CUDA context (~200-500MB).")
    print("For precise measurement: watch nvidia-smi while loading each model.")


if __name__ == "__main__":
    main()
