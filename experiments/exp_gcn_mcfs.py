"""
Experiments 4a: GCN vs BiGRU on FSC 64-class.

Science Protocol:
  Goal: Test if skeleton graph structure helps (H6)
  Prediction: GCN >= BiGRU + 3pp (>= 70.3%)
  Falsification: GCN <= BiGRU + 1pp (<= 68.3%)

Usage:
    cd /home/michael/Github/skating-biomechanics-ml
    uv run python experiments/exp_gcn_mcfs.py
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

# COCO 17kp skeleton edges (undirected)
SKELETON_EDGES = [
    (0, 1),
    (0, 2),
    (1, 3),
    (2, 4),  # head
    (5, 6),  # shoulders
    (5, 7),
    (7, 9),
    (6, 8),
    (8, 10),  # arms
    (5, 11),
    (6, 12),
    (11, 12),  # torso
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),  # legs
]


def build_adjacency(num_joints=17):
    """Build normalized adjacency matrix from skeleton edges."""
    A = np.zeros((num_joints, num_joints), dtype=np.float32)
    for i, j in SKELETON_EDGES:
        A[i, j] = 1.0
        A[j, i] = 1.0
    # Self-loops
    for i in range(num_joints):
        A[i, i] = 1.0
    # Row-normalize
    D_inv_sqrt = np.diag(1.0 / np.sqrt(A.sum(axis=1) + 1e-8))
    A_norm = D_inv_sqrt @ A @ D_inv_sqrt
    return torch.tensor(A_norm)


# ─── Normalize ────────────────────────────────────────────────────────────


def normalize(p):
    mid = p[:, 11:13, :].mean(axis=1, keepdims=True)
    p = p - mid
    sh = p[:, 5:7, :].mean(axis=1, keepdims=True)
    spine = np.linalg.norm(sh - mid, axis=1, keepdims=True)
    return p / np.maximum(spine, 0.01)


def load_fsc(split):
    path = BASE / "figure-skating-classification"
    data = pickle.load(open(path / f"{split}_data.pkl", "rb"))
    labels = pickle.load(open(path / f"{split}_label.pkl", "rb"))
    poses = [normalize(np.array(d[:, :, :2, 0], dtype=np.float32)) for d in data]
    valid = [(p, l) for p, l in zip(poses, labels) if len(p) > 0]
    if not valid:
        return [], []
    return list(zip(*valid))


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


class SpatialGraphConv(nn.Module):
    """Graph convolution: multiply by adjacency matrix A."""

    def __init__(self, adj):
        super().__init__()
        # adj: (V, V) normalized adjacency — register as buffer (not a parameter)
        self.register_buffer("A", adj)

    def forward(self, x):
        # x: (B, C, T, V) → (B, C, T, V)
        B, C, T, V = x.shape
        x = x.permute(0, 3, 1, 2)  # (B, V, C, T)
        x = x.reshape(B, V, C * T)  # (B, V, C*T)
        x = torch.matmul(self.A.unsqueeze(0), x)  # (1, V, V) @ (B, V, C*T) → (B, V, C*T)
        x = x.reshape(B, V, C, T)
        return x.permute(0, 2, 3, 1)  # (B, C, T, V)


class STGCNBlock(nn.Module):
    """Spatial-Temporal Graph Convolution Block with residual."""

    def __init__(self, in_ch, out_ch, adj, kernel_size=9):
        super().__init__()
        self.sgc = SpatialGraphConv(adj)
        self.tgc = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=(1, kernel_size), padding=(0, kernel_size // 2)),
            nn.BatchNorm2d(out_ch),
        )
        self.relu = nn.ReLU(inplace=True)
        self.residual = (
            nn.Conv2d(in_ch, out_ch, kernel_size=1) if in_ch != out_ch else nn.Identity()
        )

    def forward(self, x):
        res = self.residual(x)
        x = self.sgc(x)
        x = self.tgc(x)
        return self.relu(x + res)


class STGCN(nn.Module):
    """Spatial-Temporal Graph Convolutional Network for skeleton action recognition."""

    def __init__(self, adj, num_classes=64, in_ch=2):
        super().__init__()
        self.st1 = STGCNBlock(in_ch, 64, adj)
        self.st2 = STGCNBlock(64, 128, adj)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes),
        )

    def forward(self, x, lengths=None):
        # x: (B, T, V, C) where C=2 (x,y)
        x = x.permute(0, 3, 1, 2)  # (B, C, T, V)
        x = self.st1(x)
        x = self.st2(x)
        x = self.pool(x)
        return self.fc(x)


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
    # (B, T, 17, 2) → (B, T, 34)
    B, T, V, C = padded.shape
    padded = padded.reshape(B, T, V * C)
    return padded, lengths, torch.tensor(labels)


def gcn_collate(batch):
    batch.sort(key=lambda x: x[0].shape[0], reverse=True)
    seqs, labels = zip(*batch)
    # Truncate to max_len to fit in GPU memory
    max_len = 300
    seqs = [s[:max_len] for s in seqs]
    lengths = torch.tensor([s.shape[0] for s in seqs])
    padded = nn.utils.rnn.pad_sequence(seqs, batch_first=True)
    return padded, lengths, torch.tensor(labels)


# ─── Training ──────────────────────────────────────────────────────────────


def train_eval(
    model, train_loader, val_loader, test_loader, device, epochs=50, lr=1e-3, wd=1e-4, patience=10
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
        val_loss, val_n, val_c = 0.0, 0, 0
        with torch.no_grad():
            for batch in val_loader:
                x, lengths, y = batch
                x, y = x.to(device), y.to(device)
                logits = model(x, lengths)
                val_loss += crit(logits, y).item() * len(y)
                val_n += len(y)
                val_c += (logits.argmax(1) == y).sum().item()

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
                f"gap={gap:+.3f} test={test_acc:.3f} best={best_test_acc:.3f}"
            )

        if wait >= patience:
            print(f"  Early stop at epoch {ep + 1}")
            break

    if best_state:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    return best_test_acc


def main():
    device = torch.device("cuda")
    adj = build_adjacency(17).to(device)
    print(f"Device: {device}\n")

    # Load FSC
    train_p, train_l = load_fsc("train")
    test_p, test_l = load_fsc("test")
    all_labels = sorted(set(train_l + test_l))
    lmap = {l: i for i, l in enumerate(all_labels)}
    train_l = [lmap[l] for l in train_l]
    test_l = [lmap[l] for l in test_l]

    # Stratified 90/10 split
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

    nc = len(all_labels)
    print(f"Classes: {nc}, Train: {len(tr_p)}, Val: {len(va_p)}, Test: {len(test_p)}")

    def make_gru_loaders():
        return (
            DataLoader(VarLenDataset(tr_p, tr_l), 64, True, collate_fn=gru_collate),
            DataLoader(VarLenDataset(va_p, va_l), 64, False, collate_fn=gru_collate),
            DataLoader(VarLenDataset(test_p, test_l), 64, False, collate_fn=gru_collate),
        )

    def make_gcn_loaders():
        return (
            DataLoader(VarLenDataset(tr_p, tr_l), 8, True, collate_fn=gcn_collate),
            DataLoader(VarLenDataset(va_p, va_l), 8, False, collate_fn=gcn_collate),
            DataLoader(VarLenDataset(test_p, test_l), 8, False, collate_fn=gcn_collate),
        )

    # ═══════════════════════════════════════════════════════════════
    # Exp 4a: BiGRU Baseline (control)
    # ═══════════════════════════════════════════════════════════════
    print("=" * 60)
    print("EXP 4a: BiGRU Baseline (control)")
    print("=" * 60)
    model = BiGRU(nc=nc).to(device)
    t0 = time.time()
    gru_acc = train_eval(model, *make_gru_loaders(), device)
    print(f"  >>> BiGRU: {gru_acc:.1%} ({time.time() - t0:.0f}s)\n")

    # ═════════════════════════════════════════════════════════════════
    # Exp 4a: ST-GCN (treatment)
    # ═══════════════════════════════════════════════════════════════
    print("=" * 60)
    print("EXP 4a: ST-GCN (treatment)")
    print("=" * 60)
    model = STGCN(adj, num_classes=nc).to(device)
    t0 = time.time()
    gcn_acc = train_eval(model, *make_gcn_loaders(), device)
    print(f"  >>> ST-GCN: {gcn_acc:.1%} ({time.time() - t0:.0f}s)\n")

    # ═════════════════════════════════════════════════════════════════
    # ANALYSIS: H6
    # ═══════════════════════════════════════════════════════════════
    print("=" * 60)
    print("ANALYSIS: H6 Verdict")
    print("=" * 60)
    diff = gcn_acc - gru_acc
    print(f"BiGRU (control):  {gru_acc:.1%}")
    print(f"ST-GCN (treatment): {gcn_acc:.1%}")
    print(f"Difference:        {diff:+.1%}pp")
    print("Prediction:        >= +3.0pp")
    print("Falsification:     <= +1.0pp")
    if diff >= 3.0:
        print("H6 verdict: SUPPORTED")
    elif diff <= 1.0:
        print("H6 verdict: REJECTED — graph structure doesn't help")
    else:
        print(f"H6 verdict: INCONCLUSIVE — {diff:+.1f}pp is between 1-3pp")


if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    main()
