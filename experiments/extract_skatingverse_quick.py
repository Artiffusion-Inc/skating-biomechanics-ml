"""
Quick test: Extract SkatingVerse skeletons + train BiGRU on 28 classes.

Goal: Determine if SkatingVerse data improves figure skating classification.

Step 1 (vast.ai): Extract skeletons from ~500 videos
Step 2 (vast.ai): Train BiGRU on SV 28 classes
Step 3: Compare with FSC equivalent subset

Usage:
    uv run python experiments/extract_skatingverse_quick.py --step extract --num-per-class 100
    uv run python experiments/extract_skatingverse_quick.py --step train
"""

import argparse
import pickle
import random
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

BASE = Path("data/datasets")
SV_TRAIN_VIDEOS = BASE / "skatingverse" / "train_videos"
SV_TRAIN_TXT = BASE / "skatingverse" / "train.txt"
SV_TEST_TXT = BASE / "skatingverse" / "answer.txt"
SV_MAPPING = BASE / "skatingverse" / "mapping.txt"
EXTRACTED_DIR = BASE / "skatingverse" / "extracted"

LR_SWAP = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]


def normalize(p: np.ndarray) -> np.ndarray:
    """Root-center + scale normalize. p: (F, 17, 2) float32."""
    mid = p[:, 11:13, :].mean(axis=1, keepdims=True)
    p = p - mid
    sh = p[:, 5:7, :].mean(axis=1, keepdims=True)
    spine = np.linalg.norm(sh - mid, axis=1, keepdims=True)
    return p / np.maximum(spine, 0.01)


def halpe26_to_h36m_xy(halpe: np.ndarray) -> np.ndarray:
    """HALPE26 first 17 kp → H3.6M (17, 2) xy only."""
    return halpe[:17, :2].copy()


def load_sv_metadata(txt_path):
    """Load SV clip metadata from train.txt or answer.txt."""
    clips = []
    with open(txt_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                try:
                    clips.append((parts[0], int(parts[-1])))
                except ValueError:
                    pass
    return clips


def load_sv_class_names():
    names = {}
    with open(SV_MAPPING) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                name = " ".join(parts[:-1])
                names[int(parts[-1])] = name
    return names


# ─── Extraction ──────────────────────────────────────────────────────────────


def extract_skeletons(num_per_class=100, frame_skip=4):
    """Extract skeletons from SV videos using rtmlib."""
    print("=== Extract SkatingVerse Skeletons ===\n", flush=True)

    sv_clips = load_sv_metadata(SV_TRAIN_TXT)
    sv_names = load_sv_class_names()

    # Group by class, skip No Basic (12) and Sequence (27 — too long)
    by_class = {}
    for fname, label in sv_clips:
        if label in (12, 27):
            continue
        by_class.setdefault(label, []).append(fname)

    # Top classes by count
    sorted_classes = sorted(by_class.keys(), key=lambda c: len(by_class[c]), reverse=True)
    print("SV classes (by count):")
    for c in sorted_classes:
        print(f"  {c:2d} {sv_names.get(c, '?'):20s}: {len(by_class[c]):5d}")

    # Select num_per_class from each class
    selected = []
    for c in sorted_classes:
        clips = by_class[c][:num_per_class]
        selected.extend([(fname, c) for fname in clips])

    print(f"\nSelected: {len(selected)} clips")

    # Check which videos exist locally (train.txt has no .mp4 extension)
    existing = []
    missing = []
    for fname, label in selected:
        p = SV_TRAIN_VIDEOS / f"{fname}.mp4"
        if p.exists():
            existing.append((fname, label))
        else:
            missing.append((fname, label))

    print(f"Found: {len(existing)}, Missing: {len(missing)}")

    if len(existing) == 0:
        print("ERROR: No videos found! Check SV_TRAIN_VIDEOS path.")
        return

    # Extract
    from rtmlib import Wholebody

    print(f"\nLoading RTMPose (frame_skip={frame_skip})...", flush=True)
    wb = Wholebody(mode="balanced", backend="onnxruntime")
    print("Model loaded.", flush=True)

    EXTRACTED_DIR.mkdir(parents=True, exist_ok=True)

    success, failed, empty = 0, 0, 0
    t0 = time.time()
    meta = {"clips": {}, "class_names": sv_names}

    for i, (fname, label) in enumerate(existing):
        video_path = SV_TRAIN_VIDEOS / f"{fname}.mp4"
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            failed += 1
            continue

        frames_poses = []
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % frame_skip == 0:
                keypoints, scores = wb(frame)
                # keypoints: (N_persons, 133, 2), scores: (N_persons, 133)
                if len(keypoints) > 0 and keypoints.shape[1] >= 17:
                    body = keypoints[0, :17, :]  # (17, 2) — COCO body keypoints
                    conf = scores[0, :17] if len(scores) > 0 else None
                    if conf is not None and np.mean(conf) < 0.3:
                        pass  # low confidence, skip
                    else:
                        frames_poses.append(body.copy())
            frame_idx += 1
        cap.release()

        valid_poses = [p for p in frames_poses if p is not None and np.any(p)]
        if len(valid_poses) < 10:
            empty += 1
            continue

        poses = np.array(valid_poses, dtype=np.float32)
        poses = normalize(poses)

        # Save
        safe_name = fname.replace("/", "_").replace(".mp4", "")
        out_path = EXTRACTED_DIR / f"{safe_name}.npy"
        np.save(out_path, poses)
        meta["clips"][safe_name] = {"label": int(label), "frames": len(poses)}
        success += 1

        if (i + 1) % 50 == 0 or i == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / max(elapsed, 1)
            eta = (len(existing) - i - 1) / rate / 60 if rate > 0 else 0
            print(
                f"  [{i + 1}/{len(existing)}] ok={success} empty={empty} "
                f"rate={rate:.1f}/s eta={eta:.0f}m",
                flush=True,
            )

    elapsed = time.time() - t0
    print(f"\nDone: {success} ok, {empty} empty, {failed} missing ({elapsed:.0f}s)")

    with open(EXTRACTED_DIR / "meta.pkl", "wb") as f:
        pickle.dump(meta, f)
    print(f"Meta saved to {EXTRACTED_DIR / 'meta.pkl'}")


# ─── Training ────────────────────────────────────────────────────────────────


class BiGRU(nn.Module):
    def __init__(self, in_features=34, hidden=128, num_layers=2, num_classes=28, dropout=0.3):
        super().__init__()
        self.gru = nn.GRU(
            in_features, hidden, num_layers, batch_first=True, bidirectional=True, dropout=dropout
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden * 2, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x, lengths):
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=True
        )
        _, h = self.gru(packed)
        return self.fc(torch.cat([h[-2], h[-1]], dim=1))


class VarLenDataset(Dataset):
    def __init__(self, poses, labels):
        self.poses, self.labels = poses, labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        p = self.poses[idx]
        return torch.tensor(p.reshape(len(p), -1)), self.labels[idx]


def varlen_collate(batch):
    batch.sort(key=lambda x: x[0].shape[0], reverse=True)
    seqs = [x[0] for x in batch]
    labels = torch.tensor([x[1] for x in batch])
    lengths = torch.tensor([s.shape[0] for s in seqs])
    padded = nn.utils.rnn.pad_sequence(seqs, batch_first=True)
    return padded, lengths, labels


def train_eval(
    model, train_loader, val_loader, test_loader, device, epochs=50, lr=1e-3, patience=10
):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    crit = nn.CrossEntropyLoss(label_smoothing=0.1)

    best_val_loss = float("inf")
    best_test_acc = 0.0
    best_state = None
    wait = 0

    for ep in range(epochs):
        model.train()
        train_loss, train_n = 0.0, 0
        for x, lengths, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = crit(model(x, lengths), y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            train_loss += loss.item() * len(y)
            train_n += len(y)
        sched.step()

        model.eval()
        val_loss, val_n = 0.0, 0
        with torch.no_grad():
            for x, lengths, y in val_loader:
                x, y = x.to(device), y.to(device)
                val_loss += crit(model(x, lengths), y).item() * len(y)
                val_n += len(y)
        val_loss_avg = val_loss / val_n if val_n > 0 else float("inf")

        test_correct, test_n = 0, 0
        with torch.no_grad():
            for x, lengths, y in test_loader:
                x, y = x.to(device), y.to(device)
                test_correct += (model(x, lengths).argmax(1) == y).sum().item()
                test_n += len(y)
        test_acc = test_correct / test_n if test_n > 0 else 0.0

        if val_loss_avg < best_val_loss:
            best_val_loss = val_loss_avg
            best_test_acc = test_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1

        if (ep + 1) % 10 == 0 or ep == 0:
            print(
                f"  Ep {ep + 1:3d}: train={train_loss / train_n:.3f} val={val_loss_avg:.3f} "
                f"test={test_acc:.3f} best={best_test_acc:.3f}"
            )

        if wait >= patience:
            print(f"  Early stop at epoch {ep + 1}")
            break

    if best_state:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    return best_test_acc


def train_on_sv():
    """Train BiGRU on extracted SkatingVerse skeletons."""
    print("=== Train BiGRU on SkatingVerse ===\n")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load extracted data
    meta_path = EXTRACTED_DIR / "meta.pkl"
    if not meta_path.exists():
        print("ERROR: No extracted data. Run --step extract first.")
        return

    meta = pickle.load(open(meta_path, "rb"))

    poses, labels = [], []
    for fname, info in meta["clips"].items():
        npy_path = EXTRACTED_DIR / f"{fname}.npy"
        if npy_path.exists():
            p = np.load(npy_path)
            if len(p) >= 10:
                poses.append(p)
                labels.append(info["label"])

    print(f"Loaded: {len(poses)} samples")
    num_classes = len(set(labels))
    print(f"Classes: {num_classes}")

    # Remap labels to 0..N-1
    unique_labels = sorted(set(labels))
    lmap = {l: i for i, l in enumerate(unique_labels)}
    labels = [lmap[l] for l in labels]
    num_classes = len(unique_labels)

    # Class distribution
    c = Counter(labels)
    print(
        f"Class distribution: min={min(c.values())}, max={max(c.values())}, "
        f"ratio={max(c.values()) / min(c.values()):.0f}:1"
    )

    # Train/val split (90/10 stratified)
    by_class = {}
    for p, l in zip(poses, labels):
        by_class.setdefault(l, []).append(p)
    tr_p, tr_l, va_p, va_l = [], [], [], []
    for cls, samples in by_class.items():
        random.shuffle(samples)
        split = max(1, int(len(samples) * 0.1))
        va_p.extend(samples[:split])
        va_l.extend([cls] * split)
        tr_p.extend(samples[split:])
        tr_l.extend([cls] * (len(samples) - split))

    print(f"Train: {len(tr_p)}, Val: {len(va_p)}")

    # No separate test set (SV test videos not extracted)
    # Use val as test for this quick evaluation
    tr_dl = DataLoader(VarLenDataset(tr_p, tr_l), 64, True, collate_fn=varlen_collate)
    va_dl = DataLoader(VarLenDataset(va_p, va_l), 64, False, collate_fn=varlen_collate)

    model = BiGRU(num_classes=num_classes).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: BiGRU ({n_params:,} params, {num_classes} classes)")

    t0 = time.time()
    acc = train_eval(model, tr_dl, va_dl, va_dl, device)
    elapsed = time.time() - t0

    print(f"\nResult: {acc:.1%} ({elapsed:.0f}s)")
    print(f"Random baseline: {1 / num_classes:.1%}")
    print(f"Improvement over random: {acc - 1 / num_classes:.1%}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--step", choices=["extract", "train", "all"], required=True)
    parser.add_argument("--num-per-class", type=int, default=100)
    parser.add_argument("--frame-skip", type=int, default=4)
    args = parser.parse_args()

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    if args.step in ("extract", "all"):
        extract_skeletons(num_per_class=args.num_per_class, frame_skip=args.frame_skip)

    if args.step in ("train", "all"):
        train_on_sv()


if __name__ == "__main__":
    main()
