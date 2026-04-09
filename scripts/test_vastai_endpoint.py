#!/usr/bin/env python3
"""Smoke test: send a video to Vast.ai endpoint and verify result."""

import os
import sys

from src.vastai.client import process_video_remote

video = os.environ.get("TEST_VIDEO_PATH")
if not video:
    print("Set TEST_VIDEO_PATH env var")
    sys.exit(1)

result = process_video_remote(
    video_path=video,
    person_click={"x": 500, "y": 300},
    frame_skip=1,
    layer=3,
    ml_flags={"depth": True},
)
print(f"Video: {result.video_path}")
print(f"Poses: {result.poses_path}")
print(f"CSV:   {result.csv_path}")
print(f"Stats: {result.stats}")
