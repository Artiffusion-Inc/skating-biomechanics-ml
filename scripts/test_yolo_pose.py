#!/usr/bin/env python3
"""
Quick test script for YOLOv11-Pose as alternative to BlazePose.

This script tests YOLOv11-Pose availability and basic functionality.
"""

import sys
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def test_yolo_pose():
    """Test YOLOv11-Pose installation and basic usage."""
    print("=" * 60)
    print("Testing YOLOv11-Pose")
    print("=" * 60)

    # Test 1: Import
    print("\n[Test 1] Importing ultralytics...")
    try:
        from ultralytics import YOLO
        print("✓ ultralytics imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import: {e}")
        return False

    # Test 2: Load model
    print("\n[Test 2] Loading YOLOv11n-Pose model...")
    try:
        model = YOLO('yolov11n-pose.pt')
        print(f"✓ Model loaded: {type(model).__name__}")
    except Exception as e:
        print(f"✗ Failed to load model: {e}")
        return False

    # Test 3: Create dummy image
    print("\n[Test 3] Testing inference on dummy image...")
    try:
        import numpy as np
        test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

        start = time.time()
        results = model(test_image, verbose=False)
        inference_time = (time.time() - start) * 1000

        print(f"✓ Inference successful: {inference_time:.1f} ms")
        print(f"  Detected {len(results)} result(s)")

        if len(results) > 0 and results[0].keypoints is not None:
            num_keypoints = results[0].keypoints.xy.shape[1]
            num_persons = results[0].keypoints.xy.shape[0]
            print(f"  Keypoints per person: {num_keypoints}")
            print(f"  Persons detected: {num_persons}")
    except Exception as e:
        print(f"✗ Inference failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Test 4: Check model info
    print("\n[Test 4] Model information...")
    try:
        print(f"  Model task: {model.task}")
        print(f"  Model names: {model.names}")
    except Exception as e:
        print(f"  (Could not fetch model info: {e})")

    print("\n" + "=" * 60)
    print("✓ All tests passed!")
    print("=" * 60)

    # Next steps
    print("\nNext steps:")
    print("  1. Test with real video:")
    print("     uv run python scripts/test_yolo_pose.py --video data/test_video.mp4")
    print("  2. Compare with BlazePose results")
    print("  3. Integrate into pipeline if satisfactory")

    return True


def test_with_video(video_path: str):
    """Test YOLOv11-Pose with actual video file."""
    print(f"\n{'=' * 60}")
    print(f"Testing with video: {video_path}")
    print('=' * 60)

    if not Path(video_path).exists():
        print(f"✗ Video file not found: {video_path}")
        return False

    from ultralytics import YOLO
    import cv2

    # Load model
    model = YOLO('yolov11n-pose.pt')

    # Get video info
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    print(f"  Video: {width}x{height} @ {fps:.1f}fps, {frame_count} frames")

    # Process video
    print("\nProcessing video...")
    start = time.time()

    results = model(video_path, verbose=False, stream=True)

    poses = []
    for i, result in enumerate(results):
        if result.keypoints is not None and len(result.keypoints.xy) > 0:
            kp = result.keypoints.xy.cpu().numpy()[0]
            poses.append(kp)

    total_time = time.time() - start

    if len(poses) > 0:
        print(f"\n✓ Successfully processed video")
        print(f"  Frames with poses: {len(poses)}/{frame_count}")
        print(f"  Processing time: {total_time:.1f}s ({total_time/frame_count*1000:.1f} ms/frame)")
        print(f"  Pose shape: {poses[0].shape} (should be (17, 2))")

        # Show sample pose
        print(f"\n  Sample pose (first 5 keypoints):")
        for i, (x, y) in enumerate(poses[0][:5]):
            print(f"    KP {i}: ({x:.1f}, {y:.1f})")
    else:
        print(f"\n✗ No poses detected in video")
        return False

    return True


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test YOLOv11-Pose")
    parser.add_argument("--video", type=str, help="Test with specific video file")
    args = parser.parse_args()

    if args.video:
        success = test_with_video(args.video)
    else:
        success = test_yolo_pose()

    sys.exit(0 if success else 1)
