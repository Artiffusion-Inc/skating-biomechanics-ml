#!/usr/bin/env python3
"""Normalize video to optimal format for pose estimation pipeline.

Converts any input video (4K HEVC, 60fps, etc.) to a standardized format
that is fast to decode and process:
- H.265 (HEVC) with libx265
- Max width 1280px (preserving aspect ratio)
- 30 fps (sufficient for biomechanics)
- yuv420p 8-bit pixel format
- CRF 23, medium preset (good quality/size balance)

This is a ONE-TIME preprocessing step. After normalization, all subsequent
operations (pose extraction, comparison, analysis) run ~20x faster.

Usage:
    python scripts/normalize_video.py input.mp4
    python scripts/normalize_video.py input.mp4 --output normalized.mp4
    python scripts/normalize_video.py input.mp4 --width 960 --fps 24 --crf 20
"""

import argparse
import subprocess
import sys
from pathlib import Path


def normalize_video(
    input_path: Path,
    output_path: Path | None = None,
    max_width: int = 1280,
    target_fps: float = 30,
    crf: int = 23,
    preset: str = "medium",
    start_sec: float = 0,
    duration_sec: float = 0,
) -> Path:
    """Normalize video to optimal format for pose estimation.

    Args:
        input_path: Path to input video.
        output_path: Path for output (default: <input>_normalized.mp4).
        max_width: Maximum width in pixels (height scaled proportionally).
        target_fps: Target FPS (0 = keep original).
        crf: CRF quality (0-51, lower = better, 23 is default).
        preset: x265 preset (ultrafast/fast/medium/slow).
        start_sec: Start time in seconds.
        duration_sec: Duration in seconds (0 = all).

    Returns:
        Path to normalized video.
    """
    if not input_path.exists():
        print(f"Error: {input_path} not found", file=sys.stderr)
        sys.exit(1)

    if output_path is None:
        output_path = input_path.with_name(f"{input_path.stem}_normalized{input_path.suffix}")

    # Build FFmpeg command
    cmd = [
        "ffmpeg",
        "-y",
        "-hwaccel",
        "cuda",  # NVDEC for fast input decode
        "-i",
        str(input_path),
    ]

    # Trim
    if start_sec > 0 or duration_sec > 0:
        cmd += ["-ss", str(start_sec)]
        if duration_sec > 0:
            cmd += ["-t", str(duration_sec)]

    # Video filters: scale + fps
    vf_parts = [f"scale={max_width}:-2"]  # -2 = auto height, even
    if target_fps > 0:
        vf_parts.append(f"fps={target_fps}")
    cmd += ["-vf", ",".join(vf_parts)]

    # Audio: re-encode to AAC if present
    cmd += [
        "-c:v",
        "libx265",
        "-preset",
        preset,
        "-crf",
        str(crf),
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-b:a",
        "128k",
        "-movflags",
        "+faststart",
        str(output_path),
    ]

    print(f"Normalizing: {input_path}")
    print(f"  Output: {output_path}")
    print(f"  Max width: {max_width}px, FPS: {target_fps if target_fps else 'keep'}, CRF: {crf}")
    print(f"  Preset: {preset}")
    print("Running FFmpeg...", flush=True)

    result = subprocess.run(cmd, capture_output=True, text=True, check=False)

    if result.returncode != 0:
        # If NVDEC fails, retry without hardware accel
        print("  NVDEC failed, retrying CPU decode...", flush=True)
        cmd_no_hw = [c for c in cmd if c not in ("-hwaccel", "cuda")]
        result = subprocess.run(cmd_no_hw, capture_output=True, text=True, check=False)
        if result.returncode != 0:
            print(f"Error: {result.stderr[:500]}", file=sys.stderr)
            sys.exit(1)

    # Report result
    out_size = output_path.stat().st_size / (1024 * 1024)
    in_size = input_path.stat().st_size / (1024 * 1024)
    ratio = out_size / in_size if in_size > 0 else 0
    print(f"  Input:  {in_size:.1f} MB")
    print(f"  Output: {out_size:.1f} MB ({ratio:.0%} of original)")
    print(f"Done: {output_path}")

    return output_path


def is_normalized(path: Path, max_width: int = 1280, target_fps: float = 30) -> bool:
    """Check if video is already in normalized format.

    Returns True if video width <= max_width and fps <= target_fps + 5.
    """
    if not path.exists():
        return False

    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,r_frame_rate,codec_name",
        "-of",
        "csv=p=0",
        str(path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        return False

    try:
        codec_str, width_str, fps_str = result.stdout.strip().split(",")
        width = int(width_str)
        parts = fps_str.split("/")
        fps = float(parts[0]) / float(parts[1]) if len(parts) == 2 else float(parts[0])
    except (ValueError, IndexError):
        return False

    return width <= max_width and fps <= target_fps + 5 and codec_str == "hevc"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Normalize video for fast pose estimation processing",
    )
    parser.add_argument("input", type=Path, help="Input video path")
    parser.add_argument(
        "--output", "-o", type=Path, help="Output path (default: <input>_normalized.mp4)"
    )
    parser.add_argument(
        "--width", type=int, default=1280, help="Max width in pixels (default: 1280)"
    )
    parser.add_argument("--fps", type=float, default=30, help="Target FPS (default: 30)")
    parser.add_argument("--crf", type=int, default=23, help="CRF quality 0-51 (default: 23)")
    parser.add_argument(
        "--preset",
        type=str,
        default="medium",
        choices=["ultrafast", "fast", "medium", "slow"],
        help="x265 preset (default: medium)",
    )
    parser.add_argument("--start", type=float, default=0, help="Start time in seconds")
    parser.add_argument("--duration", type=float, default=0, help="Duration in seconds (0 = all)")
    parser.add_argument("--check", action="store_true", help="Only check if already normalized")
    args = parser.parse_args()

    if args.check:
        if is_normalized(args.input, args.width, args.fps):
            print(f"Already normalized: {args.input}")
            return 0
        else:
            print(f"Not normalized: {args.input}")
            return 1

    normalize_video(
        args.input,
        args.output,
        max_width=args.width,
        target_fps=args.fps,
        crf=args.crf,
        preset=args.preset,
        start_sec=args.start,
        duration_sec=args.duration,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
