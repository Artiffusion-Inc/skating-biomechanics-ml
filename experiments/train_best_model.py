"""
Train the best BiGRU+reg model for production use.

Configuration: Full pipeline augmentation (noise + mirror + tscale + drop + SkeletonMix)
Achieved 67.9% on FSC 64 classes in experiments.

Saves checkpoint to experiments/checkpoints/bigru_best.pt

Usage:
    uv run python experiments/train_best_model.py
"""

import pickle
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

BASE = Path("data/datasets")
CHECKPOINT_DIR = Path("experiments/checkpoints")
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

# COCO 17kp left-right swap
LR_SWAP = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]


def normalize(p: np.ndarray) -> np.ndarray:
    """Root-center + scale normalize. p: (F, 17, 2) float32."""
    mid = p[:, 11:13, :].mean(axis=1, keepdims=True)
    p = p - mid
    sh = p[:, 5:7, :].mean(axis=1, keepdims=True)
    spine = np.linalg.norm(sh - mid, axis=1, keepdims=True)
    return p / np.maximum(spine, 0.01)


def aug_joint_noise(p, sigma=0.02):
    return p + np.random.randn(*p.shape).astype(np.float32) * sigma


def aug_mirror(p):
    p = p.copy()
    p[:, :, 0] = -p[:, :, 0]
    p = p[:, LR_SWAP, :]
    return p


def aug_temporal_scale(p, scale_range=(0.9, 1.1)):
    F = len(p)
    scale = random.uniform(*scale_range)
    new_F = int(F * scale)
    new_F = max(new_F, 10)
    indices = np.linspace(0, F - 1, new_F).astype(int)
    return p[indices]


def aug_joint_drop(p, max_drop=2):
    p = p.copy()
    n_drop = random.randint(1, max_drop)
    joints = random.sample(range(17), n_drop)
    for j in joints:
        p[:, j, :] = 0.0
    return p


def aug_skeleton_mix(p1, p2, alpha=0.3):
    F = min(len(p1), len(p2))
    mixed = alpha * p1[:F] + (1 - alpha) * p2[:F]
    return mixed.astype(np.float32)


def augment(
    p, label, all_poses_by_label, p_mirror=0.5, p_noise=0.8, p_tscale=0.5, p_drop=0.3, p_mix=0.2
):
    results = [(p, label)]
    if random.random() < p_mirror:
        results.append((aug_mirror(p), label))
    if random.random() < p_noise:
        results.append((aug_joint_noise(p), label))
    if random.random() < p_tscale:
        results.append((aug_temporal_scale(p), label))
    if random.random() < p_drop:
        results.append((aug_joint_drop(p), label))
    if (
        random.random() < p_mix
        and label in all_poses_by_label
        and len(all_poses_by_label[label]) > 1
    ):
        other = random.choice(all_poses_by_label[label])
        results.append((aug_skeleton_mix(p, other), label))
    return results


def load_fsc(split: str):
    path = BASE / "figure-skating-classification"
    data = pickle.load(open(path / f"{split}_data.pkl", "rb"))
    labels = pickle.load(open(path / f"{split}_label.pkl", "rb"))
    poses = [normalize(np.array(d[:, :, :2, 0], dtype=np.float32)) for d in data]
    valid = [(p, l) for p, l in zip(poses, labels) if len(p) > 0]
    if not valid:
        return [], []
    poses, labels = zip(*valid)
    return list(poses), list(labels)


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


class BiGRU(nn.Module):
    def __init__(self, in_features=34, hidden=128, num_layers=2, num_classes=64, dropout=0.3):
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


def train_and_save():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load data
    train_p, train_l = load_fsc("train")
    test_p, test_l = load_fsc("test")

    all_labels = sorted(set(train_l + test_l))
    lmap = {l: i for i, l in enumerate(all_labels)}
    train_l = [lmap[l] for l in train_l]
    test_l = [lmap[l] for l in test_l]

    # Split train into train/val (90/10, stratified)
    by_class = {}
    for p, l in zip(train_p, train_l):
        by_class.setdefault(l, []).append(p)
    tr_p, tr_l, va_p, va_l = [], [], [], []
    for cls, samples in by_class.items():
        random.shuffle(samples)
        split = max(1, int(len(samples) * 0.1))
        va_p.extend(samples[:split])
        va_l.extend([cls] * split)
        tr_p.extend(samples[split:])
        tr_l.extend([cls] * (len(samples) - split))

    print(f"Classes: {len(all_labels)}")
    print(f"Train: {len(tr_p)}, Val: {len(va_p)}, Test: {len(test_p)}")

    # Full pipeline augmentation
    poses_by_label = {}
    for p, l in zip(tr_p, tr_l):
        poses_by_label.setdefault(l, []).append(p)

    aug_p, aug_l = [], []
    for p, l in zip(tr_p, tr_l):
        augmented = augment(
            p, l, poses_by_label, p_mirror=0.5, p_noise=0.8, p_tscale=0.5, p_drop=0.3, p_mix=0.2
        )
        for ap, al in augmented:
            aug_p.append(ap)
            aug_l.append(al)
    print(f"Augmented train: {len(aug_p)} ({len(aug_p) / len(tr_p):.1f}x)")

    # DataLoaders
    tr_dl = DataLoader(VarLenDataset(aug_p, aug_l), 64, True, collate_fn=varlen_collate)
    va_dl = DataLoader(VarLenDataset(va_p, va_l), 64, False, collate_fn=varlen_collate)
    te_dl = DataLoader(VarLenDataset(test_p, test_l), 64, False, collate_fn=varlen_collate)

    # Model
    model = BiGRU(num_classes=len(all_labels)).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")

    # Training
    epochs = 50
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    crit = nn.CrossEntropyLoss(label_smoothing=0.1)

    best_val_loss = float("inf")
    best_test_acc = 0.0
    best_state = None
    wait = 0
    patience = 10
    t0 = time.time()

    for ep in range(epochs):
        model.train()
        train_loss, train_n = 0.0, 0
        for x, lengths, y in tr_dl:
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
        val_loss, val_n, val_correct = 0.0, 0, 0
        with torch.no_grad():
            for x, lengths, y in va_dl:
                x, y = x.to(device), y.to(device)
                logits = model(x, lengths)
                val_loss += crit(logits, y).item() * len(y)
                val_n += len(y)
                val_correct += (logits.argmax(1) == y).sum().item()
        val_acc = val_correct / val_n
        val_loss_avg = val_loss / val_n

        test_correct, test_n = 0, 0
        with torch.no_grad():
            for x, lengths, y in te_dl:
                x, y = x.to(device), y.to(device)
                test_correct += (model(x, lengths).argmax(1) == y).sum().item()
                test_n += len(y)
        test_acc = test_correct / test_n

        if val_loss_avg < best_val_loss:
            best_val_loss = val_loss_avg
            best_test_acc = test_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1

        if (ep + 1) % 5 == 0 or ep == 0:
            print(
                f"  Ep {ep + 1:3d}: train={train_loss / train_n:.3f} val={val_loss_avg:.3f} "
                f"val_acc={val_acc:.3f} test_acc={test_acc:.3f} "
                f"best_test={best_test_acc:.3f}"
            )

        if wait >= patience:
            print(f"  Early stop at epoch {ep + 1}")
            break

    elapsed = time.time() - t0
    print(f"\nBest test accuracy: {best_test_acc:.1%} ({elapsed:.0f}s)")

    # Save checkpoint
    if best_state:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
        ckpt_path = CHECKPOINT_DIR / "bigru_best.pt"
        torch.save(
            {
                "model_state_dict": best_state,
                "num_classes": len(all_labels),
                "label_map": lmap,
                "config": {
                    "model": "BiGRU",
                    "in_features": 34,
                    "hidden": 128,
                    "num_layers": 2,
                    "dropout": 0.3,
                    "augmentation": "full_pipeline",
                    "test_accuracy": best_test_acc,
                    "epochs_trained": ep + 1,
                },
            },
            ckpt_path,
        )
        print(f"Checkpoint saved to {ckpt_path}")

    return best_test_acc


if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    train_and_save()
