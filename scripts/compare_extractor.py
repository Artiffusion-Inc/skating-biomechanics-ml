"""Compare our PoseExtractor (RTMO) output with dataset GT poses.

Multiple frames, each as clean side-by-side: GT left, RTMO right.
"""

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

COCO_EDGES = [
    (0, 1),
    (0, 2),
    (1, 3),
    (2, 4),
    (5, 6),
    (5, 7),
    (7, 9),
    (6, 8),
    (8, 10),
    (5, 11),
    (6, 12),
    (11, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
]


def draw_skeleton(frame, kp_xy, conf, color, thickness=3):
    for i, j in COCO_EDGES:
        if conf[i] > 0.3 and conf[j] > 0.3:
            cv2.line(
                frame, kp_xy[i].astype(int), kp_xy[j].astype(int), color, thickness, cv2.LINE_AA
            )
    for i in range(17):
        if conf[i] > 0.3:
            cv2.circle(frame, kp_xy[i].astype(int), 4, color, -1, cv2.LINE_AA)


def main():
    video_path = Path("data/datasets/raw/athletepose3d/videos/train_set/S1/Axel_10_cam_1.mp4")
    gt_all = np.load(video_path.with_name(video_path.stem + "_coco.npy")).astype(np.float32)
    n_frames = len(gt_all)

    # Pick 8 evenly spaced frames
    frame_indices = [int(i * (n_frames - 1) / 7) for i in range(8)]

    # Run RTMO on all frames
    from src.pose_estimation.pose_extractor import PoseExtractor

    extractor = PoseExtractor(mode="lightweight", output_format="pixels")

    cap = cv2.VideoCapture(str(video_path))
    frames = []
    results = []
    for _fi in range(n_frames):
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
        result = extractor.tracker(frame)
        results.append(result)
    cap.release()
    extractor.close()

    h, w = frames[0].shape[:2]

    # Build rows: each row = one side-by-side comparison
    rows = []
    diffs_summary = []

    for fi in frame_indices:
        frame = frames[fi]
        gt_kp = gt_all[fi]
        gt_conf = np.where(np.isnan(gt_kp[:, 0]), 0.0, 1.0)

        left = frame.copy()
        draw_skeleton(left, gt_kp, gt_conf, (100, 150, 255), 3)
        cv2.putText(
            left,
            "GT (AthletePose3D)",
            (20, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (100, 150, 255),
            2,
            cv2.LINE_AA,
        )

        right = frame.copy()
        rtmo_kp = None
        rtmo_conf = None
        if results[fi] is not None and isinstance(results[fi], tuple):
            keypoints, scores = results[fi]
            if keypoints is not None and len(keypoints) > 0:
                rtmo_kp = keypoints[0].astype(np.float32)
                rtmo_conf = scores[0].astype(np.float32)
                draw_skeleton(right, rtmo_kp, rtmo_conf, (100, 255, 100), 3)
        cv2.putText(
            right,
            "RTMO (ours)",
            (20, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (100, 255, 100),
            2,
            cv2.LINE_AA,
        )

        # Frame label at bottom center
        cv2.putText(
            left,
            f"Frame {fi}",
            (20, h - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (200, 200, 200),
            1,
            cv2.LINE_AA,
        )

        # Per-frame avg diff
        if rtmo_kp is not None:
            dists = []
            for j in range(17):
                if gt_conf[j] > 0 and rtmo_conf[j] > 0.3:
                    dists.append(np.linalg.norm(gt_kp[j] - rtmo_kp[j]))
            avg = np.mean(dists) if dists else 0
            diffs_summary.append((fi, avg))
            cv2.putText(
                right,
                f"avg {avg:.0f}px",
                (20, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (200, 200, 200),
                1,
                cv2.LINE_AA,
            )

        # Stack + divider
        combined = np.hstack([left, right])
        cv2.line(combined, (w, 0), (w, h), (255, 255, 255), 2, cv2.LINE_AA)
        rows.append(combined)

    # Vertical stack with thin dividers
    divider = np.zeros((4, w * 2, 3), dtype=np.uint8)
    out = rows[0]
    for row in rows[1:]:
        out = np.vstack([out, divider, row])

    out_path = "scripts/extractor_comparison.png"
    cv2.imwrite(out_path, out)

    print("Per-frame average difference:")
    for fi, avg in diffs_summary:
        print(f"  Frame {fi:3d}: {avg:5.1f} px")
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
