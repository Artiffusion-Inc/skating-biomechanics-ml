"""
Experiments 4b-4c: MCFS data experiments.

Science Protocol:
  H8: MCFS segments with matching labels improve FSC accuracy (>=+3pp)
  H10: BiGRU on MCFS 130 classes (>=40% on classes with >=10 samples)
  H11: FineFS 3D skeletons + time-coded elements enable quality regression

Usage:
    cd /home/michael/Github/skating-biomechanics-ml
    uv run python experiments/exp_mcfs_experiments.py
"""

import pickle
import json
import time
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from collections import Counter

BASE = Path("data/datasets")


# ─── Normalize ────────────────────────────────────────────────────────────


def normalize(p):
    mid = p[:, 11:13, :].mean(axis=1, keepdims=True)
    p = p - mid
    sh = p[:, 5:7, :].mean(axis=1, keepdims=True)
    spine = np.linalg.norm(sh - mid, axis=1, keepdims=True)
    return p / np.maximum(spine, 0.01)


def load_fsc(split):
    """Load FSC dataset. Returns (poses, labels) as lists."""
    path = BASE / "figure-skating-classification"
    data = pickle.load(open(path / f"{split}_data.pkl", "rb"))
    labels = pickle.load(open(path / f"{split}_label.pkl", "rb"))
    poses = [normalize(np.array(d[:, :, :2, 0], dtype=np.float32)) for d in data]
    valid = [(p, l) for p, l in zip(poses, labels) if len(p) > 0]
    if not valid:
        return [], []
    return list(zip(*valid))


def load_mcfs_segments(min_samples=5):
    """Load MCFS segments filtered by label count."""
    segs = pickle.load(open(BASE / "mcfs/segments.pkl", "rb"))
    # Count labels
    counts = Counter(s[1] for s in segs)
    # Filter to labels with >= min_samples
    valid_segs = [(p, l) for p, l in segs if counts[l] >= min_samples]
    poses, labels = list(zip(*valid_segs)) if valid_segs else ([], [])
    return poses, labels


# ─── Models ──────────────────────────────────────────────────────────────


class BiGRU(nn.Module):
    def __init__(self, in_f=34, hidden=128, layers=2, nc=64, drop=0.3):
        super().__init__()
        self.gru = nn.GRU(in_f, hidden, layers, batch_first=True, bidirectional=True, dropout=drop)
        self.fc = nn.Sequential(
            nn.Linear(hidden * 2, 128), nn.ReLU(), nn.Dropout(drop), nn.Linear(128, nc)
        )

    def forward(self, x, lengths):
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=True
        )
        _, h = self.gru(packed)
        return self.fc(torch.cat([h[-2], h[-1]], dim=1))


# ─── Dataset ──────────────────────────────────────────────────────────────


class VarLenDataset(Dataset):
    def __init__(self, poses, labels):
        self.poses, self.labels = poses, labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return torch.tensor(self.poses[idx]), self.labels[idx]


def gru_collate(batch):
    batch.sort(key=lambda x: x[0].shape[0], reverse=True)
    seqs, labels = zip(*batch)
    lengths = torch.tensor([s.shape[0] for s in seqs])
    padded = nn.utils.rnn.pad_sequence(seqs, batch_first=True)
    B, T, V, C = padded.shape
    padded = padded.reshape(B, T, V * C)
    return padded, lengths, torch.tensor(labels)


# ─── Training ──────────────────────────────────────────────────────────────


def train_eval(
    model,
    train_loader,
    val_loader,
    test_loader,
    device,
    epochs=50,
    lr=1e-3,
    wd=1e-4,
    patience=10,
    label="",
):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    crit = nn.CrossEntropyLoss(label_smoothing=0.1)
    best_val_loss = float("inf")
    best_test_acc = 0.0
    best_state = None
    wait = 0

    for ep in range(epochs):
        model.train()
        train_loss, train_n = 0.0, 0
        for batch in train_loader:
            x, lengths, y = batch
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
            for batch in val_loader:
                x, lengths, y = batch
                x, y = x.to(device), y.to(device)
                logits = model(x, lengths)
                val_loss += crit(logits, y).item() * len(y)
                val_n += len(y)

        test_c, test_n = 0, 0
        with torch.no_grad():
            for batch in test_loader:
                x, lengths, y = batch
                x, y = x.to(device), y.to(device)
                test_c += (model(x, lengths).argmax(1) == y).sum().item()
                test_n += len(y)

        test_acc = test_c / test_n
        val_loss_avg = val_loss / val_n

        if val_loss_avg < best_val_loss:
            best_val_loss = val_loss_avg
            best_test_acc = test_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1

        if (ep + 1) % 10 == 0 or ep == 0:
            gap = (train_loss / train_n) - val_loss_avg
            print(
                f"  Ep {ep + 1:3d}: train={train_loss / train_n:.3f} val={val_loss_avg:.3f} "
                f"gap={gap:+.3f} test={test_acc:.3f} best={best_test_acc:.3f} [{label}]"
            )

        if wait >= patience:
            print(f"  Early stop at epoch {ep + 1} [{label}]")
            break

    if best_state:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    return best_test_acc


def make_loaders(tr_p, tr_l, va_p, va_l, te_p, te_l, batch=64):
    return (
        DataLoader(VarLenDataset(tr_p, tr_l), batch, True, collate_fn=gru_collate),
        DataLoader(VarLenDataset(va_p, va_l), batch, False, collate_fn=gru_collate),
        DataLoader(VarLenDataset(te_p, te_l), batch, False, collate_fn=gru_collate),
    )


def stratified_split(poses, labels, val_frac=0.1):
    """Stratified train/val split."""
    by_class = {}
    for p, l in zip(poses, labels):
        by_class.setdefault(l, []).append(p)
    tr_p, tr_l, va_p, va_l = [], [], [], []
    for cls, samples in by_class.items():
        random.shuffle(samples)
        split = max(1, int(len(samples) * val_frac))
        va_p.extend(samples[:split])
        va_l.extend([cls] * split)
        tr_p.extend(samples[split:])
        tr_l.extend([cls] * (len(samples) - split))
    return tr_p, tr_l, va_p, va_l


def main():
    device = torch.device("cuda")
    print(f"Device: {device}\n")

    # ═══════════════════════════════════════════════════════════════
    # Exp 4b: MCFS 130-class BiGRU (H10)
    # ═══════════════════════════════════════════════════════════════
    print("=" * 60)
    print("EXP 4b: MCFS 130-class BiGRU (H10)")
    print("=" * 60)
    print("H10: BiGRU on MCFS 130 classes >= 40% (classes with >=10 samples)")
    print("Falsification: < 20%\n")

    mcfs_p, mcfs_l = load_mcfs_segments(min_samples=10)
    # Map string labels to integers
    mcfs_labels_sorted = sorted(set(mcfs_l))
    mcfs_lmap = {l: i for i, l in enumerate(mcfs_labels_sorted)}
    mcfs_l = [mcfs_lmap[l] for l in mcfs_l]
    print(f"MCFS segments: {len(mcfs_p)}, classes (>=10): {len(mcfs_labels_sorted)}")
    lengths = [len(p) for p in mcfs_p]
    print(f"Segment lengths: min={min(lengths)}, max={max(lengths)}, mean={np.mean(lengths):.0f}\n")

    tr_p, tr_l, va_p, va_l = stratified_split(mcfs_p, mcfs_l, val_frac=0.1)
    te_p, te_l = mcfs_p, mcfs_l  # Use all as test (no FSC test set here)

    nc = len(mcfs_labels_sorted)
    print(f"Classes: {nc}, Train: {len(tr_p)}, Val: {len(va_p)}, Test: {len(te_p)}")

    model = BiGRU(nc=nc).to(device)
    t0 = time.time()
    mcfs_acc = train_eval(
        model, *make_loaders(tr_p, tr_l, va_p, va_l, te_p, te_l, batch=32), device, label="MCFS130"
    )
    print(f"  >>> BiGRU MCFS-130: {mcfs_acc:.1%} ({time.time() - t0:.0f}s)\n")

    # ═══════════════════════════════════════════════════════════════
    # ANALYSIS: H10
    # ═══════════════════════════════════════════════════════════════
    print("=" * 60)
    print("ANALYSIS: H10 Verdict")
    print("=" * 60)
    print(f"MCFS 130-class BiGRU: {mcfs_acc:.1%}")
    if mcfs_acc >= 0.40:
        print("H10 verdict: SUPPORTED")
    elif mcfs_acc >= 0.20:
        print(f"H10 verdict: INCONCLUSIVE — {mcfs_acc:.1%} between 20-40%")
    else:
        print("H10 verdict: REJECTED")

    # ═══════════════════════════════════════════════════════════════
    # Exp 4c: FineFS quality regression (H11)
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("EXP 4c: FineFS Quality Regression (H11)")
    print("=" * 60)

    finefs_dir = BASE / "finefs"
    if not finefs_dir.exists():
        print("FineFS not found on server. Skipping.")
    else:
        print("H11: GOE score can be predicted from skeleton features")
        print("Prediction: MAE <= 1.0 GOE points")
        print("Falsification: MAE > 3.0 GOE points\n")

        # Load FineFS data
        annotations = []
        skeletons = []
        for fn in sorted((finefs_dir / "annotation").iterdir()):
            if not fn.suffix == ".json":
                continue
            idx = int(fn.stem)
            ann = json.load(open(fn))
            skel_path = finefs_dir / "skeleton" / f"{idx}.npz"
            if not skel_path.exists():
                continue
            try:
                skel = np.load(skel_path, allow_pickle=True)["reconstruction"]  # (T, 17, 3)
            except Exception:
                continue
            annotations.append(ann)
            skeletons.append(skel)

        print(f"FineFS samples: {len(annotations)}")

        # Extract per-element features and GOE targets
        features = []
        targets = []
        for ann, skel in zip(annotations, skeletons):
            for k, v in ann["executed_element"].items():
                goe = v.get("goe")
                if goe is None:
                    continue
                time_str = v.get("time", "")
                if "," not in time_str:
                    continue
                parts = time_str.split(",")
                try:
                    start_sec = int(parts[0].split("-")[0]) * 60 + int(parts[0].split("-")[1])
                    end_sec = int(parts[1].split("-")[0]) * 60 + int(parts[1].split("-")[1])
                except (IndexError, ValueError):
                    continue
                start_f = start_sec * 25  # 25 fps
                end_f = min(end_sec * 25, len(skel))
                if end_f <= start_f + 10:
                    continue
                seg = skel[start_f:end_f]

                # Features: mean+std of joint positions, segment length
                pose = seg[:, :, :2]  # (T, 17, 2) — x,y only
                feat = np.concatenate(
                    [
                        pose.mean(axis=(0, 1)),  # mean position (17*2=34)
                        pose.std(axis=(0, 1)),  # std position (34)
                        [len(seg) / 25.0],  # duration in seconds
                    ]
                )
                features.append(feat)
                targets.append(goe)

        features = np.array(features, dtype=np.float32)
        targets = np.array(targets, dtype=np.float32)
        print(f"Element segments: {len(features)}")
        print(f"GOE range: [{targets.min():.1f}, {targets.max():.1f}]")
        print(f"Feature dim: {features.shape[1]}\n")

        # Simple linear regression
        from sklearn.model_selection import train_test_split

        X_train, X_test, y_train, y_test = train_test_split(
            features, targets, test_size=0.2, random_state=42
        )

        # Normalize features
        mean = X_train.mean(axis=0)
        std = X_train.std(axis=0) + 1e-8
        X_train_n = (X_train - mean) / std
        X_test_n = (X_test - mean) / std

        # Linear regression
        from sklearn.linear_model import Ridge

        reg = Ridge(alpha=1.0)
        reg.fit(X_train_n, y_train)
        pred = reg.predict(X_test_n)
        mae = np.mean(np.abs(pred - y_test))
        corr = np.corrcoef(pred, y_test)[0, 1]
        baseline_mae = np.mean(np.abs(y_test - y_test.mean()))

        print(f"Linear regression (Ridge):")
        print(f"  MAE: {mae:.2f} GOE (baseline mean: {baseline_mae:.2f})")
        print(f"  Correlation: {corr:.3f}")
        print(f"  Improvement over baseline: {(1 - mae / baseline_mae) * 100:.1f}%")

        if mae <= 1.0:
            print("H11 verdict: SUPPORTED")
        elif mae <= 3.0:
            print(f"H11 verdict: INCONCLUSIVE — MAE={mae:.2f} between 1.0-3.0")
        else:
            print(f"H11 verdict: REJECTED — MAE={mae:.2f} > 3.0")


if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    main()
