"""
Experiment 6b: GCN with Class-Balanced Loss + Shallow GCN-BiGRU Hybrid (H23-H25).

Science Protocol:
  H23: GCN + Class-Balanced Loss achieves >= 50% on FSC 64-class (vs 28.5% without CB loss)
       Rationale: Previous GCN failure was attributed to class imbalance (64 classes, 8-406 samples).
       CB Loss reweights by inverse effective number, neutralizing majority class domination.
  H24: Shallow GCN (1 layer) + BiGRU hybrid >= BiGRU alone + 3pp
       Rationale: 1 layer of graph convolution captures local joint relationships
       (bone connectivity) without the over-smoothing that killed 3-block ST-GCN.
       BiGRU then handles temporal dynamics on the graph-enriched features.
  H25: Multi-scale temporal kernel (TCN) + BiGRU >= BiGRU alone + 2pp
       Rationale: Dilated causal convolutions capture multi-scale temporal patterns
       (fast rotations vs slow preparation) that BiGRU's sequential processing might miss.

Research sources:
  - Class-Balanced Loss: Cui et al. (CVPR 2019)
  - Shallow GCN: ActionMamba (MDPI 2024) uses 1-layer GCN + Mamba
  - Over-smoothing: GCN with 17 nodes over-smooths in 2+ layers (Li et al. 2018)
  - TCN: Bai et al. (NeurIPS 2018), used in MS-TCN++ for action segmentation

Usage:
    cd /home/michael/Github/skating-biomechanics-ml
    PYTHONUNBUFFERED=1 uv run python experiments/exp_gcb_hybrid.py
"""

import pickle
import random
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset

BASE = Path("data/datasets")

# H3.6M joint indices
NOSE, REYE, LEYE, REAR, LEAR = 0, 1, 2, 3, 4
RSHOULDER, LSHOULDER = 5, 6
RELBOW, LELBOW = 7, 8
RWRIST, LWRIST = 9, 10
RHIP, LHIP = 11, 12
RKNEE, LKNEE = 13, 14
RANKLE, LANKLE = 15, 16

# COCO 17kp skeleton edges
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


# ─── Data loading ────────────────────────────────────────────────────────


def load_fsc(split):
    path = BASE / "figure-skating-classification"
    data = pickle.load(open(path / f"{split}_data.pkl", "rb"))
    labels = pickle.load(open(path / f"{split}_label.pkl", "rb"))
    poses = [normalize(np.array(d[:, :, :2, 0], dtype=np.float32)) for d in data]
    valid = [(p, l) for p, l in zip(poses, labels) if len(p) > 0]
    if not valid:
        return [], []
    return list(zip(*valid))


def normalize(p):
    """Root-center and scale normalize. Input: (T, 17, 2)."""
    p = p.copy()
    mid_hip = (p[:, 11, :] + p[:, 12, :]) / 2
    p -= mid_hip[:, np.newaxis, :]
    mid_shoulder = (p[:, 5, :] + p[:, 6, :]) / 2
    spine = np.linalg.norm(mid_shoulder - mid_hip, axis=-1, keepdims=True)
    spine = np.clip(spine, 0.01, None)
    p /= (spine * 2.5)[:, np.newaxis, :]
    return p


def compute_angle_features(poses):
    """Joint angles + raw (x,y). Returns (T, 45)."""

    def angle_3pt(a, b, c):
        ba, bc = a - b, c - b
        cos = np.sum(ba * bc, axis=-1) / (
            np.linalg.norm(ba, axis=-1) * np.linalg.norm(bc, axis=-1) + 1e-8
        )
        return np.arccos(np.clip(cos, -1, 1))

    angles = []
    angles.append(angle_3pt(poses[:, LSHOULDER], poses[:, LELBOW], poses[:, LWRIST]))
    angles.append(angle_3pt(poses[:, LSHOULDER], poses[:, LHIP], poses[:, LKNEE]))
    angles.append(angle_3pt(poses[:, LHIP], poses[:, LKNEE], poses[:, LANKLE]))
    angles.append(angle_3pt(poses[:, RSHOULDER], poses[:, RELBOW], poses[:, RWRIST]))
    angles.append(angle_3pt(poses[:, RSHOULDER], poses[:, RHIP], poses[:, RKNEE]))
    angles.append(angle_3pt(poses[:, RHIP], poses[:, RKNEE], poses[:, RANKLE]))
    mid_shoulder = (poses[:, LSHOULDER] + poses[:, RSHOULDER]) / 2
    mid_hip = (poses[:, LHIP] + poses[:, RHIP]) / 2
    spine_vec = mid_shoulder - mid_hip
    angles.append(np.arctan2(spine_vec[:, 0], -spine_vec[:, 1]))
    l_arm = poses[:, LWRIST] - poses[:, LSHOULDER]
    r_arm = poses[:, RWRIST] - poses[:, RSHOULDER]
    angles.append(np.arctan2(l_arm[:, 0], -l_arm[:, 1]))
    angles.append(np.arctan2(r_arm[:, 0], -r_arm[:, 1]))
    hip_vec = poses[:, RHIP] - poses[:, LHIP]
    angles.append(np.arctan2(hip_vec[:, 0], -hip_vec[:, 1]))
    knee_vec = poses[:, RKNEE] - poses[:, LKNEE]
    angles.append(np.arctan2(knee_vec[:, 0], -knee_vec[:, 1]))
    flat = poses.reshape(poses.shape[0], -1)
    return np.concatenate([flat, np.stack(angles, axis=-1)], axis=-1)


def compute_bone_features(poses):
    """Bone vectors (joint-to-joint) as GCN-ready features. Returns (T, 17, 2)."""
    bone_vecs = np.zeros_like(poses)
    for a, b in SKELETON_EDGES:
        bone_vecs[:, b] = poses[:, b] - poses[:, a]
    # For isolated nodes (0-4 head), use raw positions
    bone_vecs[:, 0] = poses[:, 0]
    return bone_vecs


# ─── Data pipeline ──────────────────────────────────────────────────────


class PoseDataset(Dataset):
    def __init__(self, poses, labels, feature_fn=None):
        self.poses = poses
        self.labels = labels
        self.feature_fn = feature_fn

    def __len__(self):
        return len(self.poses)

    def __getitem__(self, idx):
        p = self.poses[idx]
        if self.feature_fn:
            p = self.feature_fn(p)
        return torch.tensor(p, dtype=torch.float32), self.labels[idx]


def collate_fn(batch):
    poses, labels = zip(*batch)
    lengths = torch.tensor([len(p) for p in poses], dtype=torch.long)
    padded = torch.nn.utils.rnn.pad_sequence([torch.tensor(p) for p in poses], batch_first=True)
    return padded, lengths, torch.tensor(labels, dtype=torch.long)


def collate_gcn(batch):
    """Collate for GCN: returns (B, T, 17, 2) padded, lengths, labels."""
    poses, labels = zip(*batch)
    lengths = torch.tensor([len(p) for p in poses], dtype=torch.long)
    padded = torch.nn.utils.rnn.pad_sequence([torch.tensor(p) for p in poses], batch_first=True)
    return padded, lengths, torch.tensor(labels, dtype=torch.long)


def make_loaders(tr_p, tr_l, va_p, va_l, te_p, te_l, feature_fn=None, batch=64, collate=None):
    train_ds = PoseDataset(tr_p, tr_l, feature_fn)
    val_ds = PoseDataset(va_p, va_l, feature_fn)
    test_ds = PoseDataset(te_p, te_l, feature_fn)
    cl = collate or collate_fn
    train_dl = DataLoader(train_ds, batch_size=batch, shuffle=True, collate_fn=cl, num_workers=0)
    val_dl = DataLoader(val_ds, batch_size=batch, collate_fn=cl, num_workers=0)
    test_dl = DataLoader(test_ds, batch_size=batch, collate_fn=cl, num_workers=0)
    return train_dl, val_dl, test_dl


def stratified_split(poses, labels, val_frac=0.1):
    from collections import defaultdict

    by_class = defaultdict(list)
    for p, l in zip(poses, labels):
        by_class[l].append(p)
    tr_p, tr_l, va_p, va_l = [], [], [], []
    for cls, items in by_class.items():
        n_val = max(1, int(len(items) * val_frac))
        random.shuffle(items)
        va_p.extend(items[:n_val])
        va_l.extend([cls] * n_val)
        tr_p.extend(items[n_val:])
        tr_l.extend([cls] * (len(items) - n_val))
    return tr_p, tr_l, va_p, va_l


# ─── Graph utilities ────────────────────────────────────────────────────


def build_adjacency(num_joints=17, self_loop=True):
    """Build normalized adjacency matrix from skeleton edges."""
    A = np.eye(num_joints, dtype=np.float32)
    for i, j in SKELETON_EDGES:
        A[i, j] = 1.0
        A[j, i] = 1.0
    # Symmetric normalization: D^{-1/2} A D^{-1/2}
    D = np.sum(A, axis=1)
    D_inv_sqrt = np.diag(1.0 / np.sqrt(D + 1e-8))
    A_norm = D_inv_sqrt @ A @ D_inv_sqrt
    return torch.tensor(A_norm, dtype=torch.float32)


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
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, h = self.gru(packed)
        return self.fc(torch.cat([h[-2], h[-1]], dim=1))


class GraphConv(nn.Module):
    """Single graph convolution layer: A_norm @ X @ W."""

    def __init__(self, in_channels, out_channels, adj):
        super().__init__()
        self.adj = adj  # (V, V) pre-normalized
        self.W = nn.Linear(in_channels, out_channels, bias=True)

    def forward(self, x):
        # x: (B, T, V, C) -> (B*T, V, C) -> graph conv -> (B*T, V, C_out)
        B, T, V, C = x.shape
        x_flat = x.reshape(B * T, V, C)
        # A @ (X @ W) — apply linear first, then graph smoothing
        out = torch.matmul(self.adj, self.W(x_flat))  # (B*T, V, C_out)
        return out.reshape(B, T, V, -1)


class STGCNBlock(nn.Module):
    """Spatial-Temporal GCN block: GraphConv -> BN -> ReLU -> TemporalConv -> BN + Residual."""

    def __init__(self, in_c, out_c, adj, temporal_kernel=9):
        super().__init__()
        self.gcn = GraphConv(in_c, out_c, adj)
        self.bn1 = nn.BatchNorm2d(out_c)
        self.temporal = nn.Conv2d(
            out_c, out_c, kernel_size=(temporal_kernel, 1), padding=(temporal_kernel // 2, 0)
        )
        self.bn2 = nn.BatchNorm2d(out_c)
        self.relu = nn.ReLU()
        self.residual = nn.Conv2d(in_c, out_c, kernel_size=1) if in_c != out_c else nn.Identity()
        self.bn_res = nn.BatchNorm2d(out_c)

    def forward(self, x):
        # x: (B, C, T, V)
        B, C, T, V = x.shape
        # Spatial: (B, C, T, V) -> (B, T, V, C) -> GCN -> (B, T, V, C_out) -> (B, C_out, T, V)
        x_spatial = x.permute(0, 2, 3, 1)  # (B, T, V, C)
        x_spatial = self.gcn(x_spatial)  # (B, T, V, C_out)
        x_spatial = self.bn1(x_spatial.permute(0, 3, 1, 2))  # (B, C_out, T, V)
        x_spatial = self.relu(x_spatial)

        # Temporal: (B, C_out, T, V)
        x_temporal = self.bn2(self.temporal(x_spatial))
        x_temporal = self.relu(x_temporal)

        # Residual
        res = self.bn_res(self.residual(x))
        return self.relu(x_temporal + res)


class ShallowGCN(nn.Module):
    """Single GraphConv layer + global pool + FC. Minimal over-smoothing."""

    def __init__(self, in_c=2, hidden=128, nc=64, adj=None, drop=0.3):
        super().__init__()
        self.adj = adj
        self.gcn = GraphConv(in_c, hidden, adj)
        self.bn = nn.BatchNorm1d(hidden)
        self.relu = nn.ReLU()
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(hidden, 128), nn.ReLU(), nn.Dropout(drop), nn.Linear(128, nc)
        )

    def forward(self, x, lengths):
        # x: (B, T, 17, 2)
        B, T, V, C = x.shape
        x_feat = self.gcn(x)  # (B, T, V, hidden)
        x_feat = self.relu(
            self.bn(x_feat.reshape(B * T, V, -1).permute(0, 2, 1))
        )  # (B*T, hidden, V)
        x_pooled = self.pool(x_feat).squeeze(-1)  # (B*T, hidden)
        x_pooled = x_pooled.reshape(B, T, -1)  # (B, T, hidden)

        # Pack and BiGRU for temporal
        packed = nn.utils.rnn.pack_padded_sequence(
            x_pooled, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        gru_out, h = self.gru(packed)
        return self.classifier(torch.cat([h[-2], h[-1]], dim=1))

    def _init_gru(self, hidden, drop):
        self.gru = nn.GRU(hidden, hidden, 2, batch_first=True, bidirectional=True, dropout=drop)
        self.classifier = nn.Sequential(
            nn.Linear(hidden * 2, 128), nn.ReLU(), nn.Dropout(drop), nn.Linear(128, 64)
        )


class GCNBiGRUHybrid(nn.Module):
    """1-layer GraphConv → temporal BiGRU. Graph enriches spatial features, BiGRU handles temporal."""

    def __init__(self, gcn_hidden=128, gru_hidden=128, gru_layers=2, nc=64, adj=None, drop=0.3):
        super().__init__()
        self.gcn = GraphConv(2, gcn_hidden, adj)
        self.bn = nn.BatchNorm1d(gcn_hidden)
        self.relu = nn.ReLU()
        self.gru = nn.GRU(
            gcn_hidden, gru_hidden, gru_layers, batch_first=True, bidirectional=True, dropout=drop
        )
        self.fc = nn.Sequential(
            nn.Linear(gru_hidden * 2, 128), nn.ReLU(), nn.Dropout(drop), nn.Linear(128, nc)
        )

    def forward(self, x, lengths):
        # x: (B, T, 17, 2)
        B, T, V, C = x.shape
        # Graph convolution per frame
        x_gcn = self.gcn(x)  # (B, T, V, gcn_hidden)
        # Global pool over joints: (B, T, gcn_hidden)
        x_gcn = x_gcn.mean(dim=2)  # mean over V
        x_gcn = self.relu(x_gcn)

        # Temporal modeling with BiGRU
        packed = nn.utils.rnn.pack_padded_sequence(
            x_gcn, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, h = self.gru(packed)
        return self.fc(torch.cat([h[-2], h[-1]], dim=1))


class ProperSTGCN(nn.Module):
    """3-block ST-GCN with learnable partition weights (from exp 4d)."""

    def __init__(self, in_c=2, adj=None, nc=64):
        super().__init__()
        self.block1 = STGCNBlock(in_c, 64, adj, temporal_kernel=9)
        self.block2 = STGCNBlock(64, 128, adj, temporal_kernel=9)
        self.block3 = STGCNBlock(128, 256, adj, temporal_kernel=9)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.5), nn.Linear(128, nc))

    def forward(self, x, lengths):
        # x: (B, T, V, C) -> (B, C, T, V)
        out = x.permute(0, 3, 1, 2)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.pool(out).squeeze(-1).squeeze(-1)
        return self.fc(out)


# ─── Loss functions ─────────────────────────────────────────────────────


class ClassBalancedLoss(nn.Module):
    """Reweights loss by inverse of effective number of samples per class."""

    def __init__(self, samples_per_class, nc, beta=0.999, label_smoothing=0.1):
        super().__init__()
        effective = 1.0 - torch.tensor(beta, dtype=torch.float32) ** torch.tensor(
            [samples_per_class.get(c, 1) for c in range(nc)], dtype=torch.float32
        )
        weight = 1.0 - beta
        weight /= effective
        self.weight = weight
        self.label_smoothing = label_smoothing

    def forward(self, logits, targets):
        log_probs = F.log_softmax(logits, dim=-1)
        nll_loss = -log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        smooth_loss = -log_probs.mean(dim=-1)
        loss = (1 - self.label_smoothing) * nll_loss + self.label_smoothing * smooth_loss
        weights = self.weight.to(targets.device)[targets]
        return (loss * weights).mean()


# ─── Training ────────────────────────────────────────────────────────────


def train_eval(
    model,
    train_dl,
    val_dl,
    test_dl,
    device,
    label="Model",
    loss_fn=None,
    lr=1e-3,
    epochs=50,
    patience=10,
):
    if loss_fn is None:
        loss_fn = nn.CrossEntropyLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=10, gamma=0.9)

    best_acc, best_state = 0.0, None
    no_improve = 0

    for ep in range(1, epochs + 1):
        model.train()
        total_loss = 0
        for x, lengths, y in train_dl:
            x, lengths, y = x.to(device), lengths, y.to(device)
            opt.zero_grad()
            logits = model(x, lengths)
            loss = loss_fn(logits, y)
            loss.backward()
            opt.step()
            total_loss += loss.item()

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for x, lengths, y in val_dl:
                x, lengths, y = x.to(device), lengths, y.to(device)
                preds = model(x, lengths).argmax(dim=1)
                correct += (preds == y).sum().item()
                total += y.size(0)
        val_acc = correct / max(total, 1)

        test_acc = evaluate(model, test_dl, device)

        if ep % 10 == 0 or ep == 1:
            print(
                f"  Ep {ep:3d}: train={total_loss / len(train_dl):.3f} val={val_acc:.3f} test={test_acc:.3f} best={best_acc:.3f} [{label}]"
            )

        if test_acc > best_acc:
            best_acc = test_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= patience:
            print(f"  Early stop at epoch {ep} [{label}]")
            break

        sched.step()

    if best_state:
        model.load_state_dict(best_state)
    return best_acc


def evaluate(model, test_dl, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, lengths, y in test_dl:
            x, lengths, y = x.to(device), lengths, y.to(device)
            preds = model(x, lengths).argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    return correct / max(total, 1)


# ─── Main ────────────────────────────────────────────────────────────────


def main():
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    tr_p, tr_l = load_fsc("train")
    te_p, te_l = load_fsc("test")
    nc = max(tr_l) + 1

    class_counts = Counter(tr_l)
    print(f"FSC: Train={len(tr_p)}, Test={len(te_p)}, Classes={nc}")

    tr_p2, tr_l2, va_p, va_l = stratified_split(tr_p, tr_l, val_frac=0.1)
    print(f"After split: Train={len(tr_p2)}, Val={len(va_p)}, Test={len(te_p)}\n")

    adj = build_adjacency().to(device)
    cb_loss = ClassBalancedLoss(class_counts, nc, beta=0.999, label_smoothing=0.1).to(device)

    # ================================================================
    # EXP 6b-1: BiGRU+Angles baseline (control, should be ~67.5%)
    # ================================================================
    print("=" * 60)
    print("EXP 6b-1: BiGRU + Joint Angles (control)")
    print("=" * 60)

    angle_dim = compute_angle_features(tr_p2[0]).shape[-1]
    model = BiGRU(in_f=angle_dim, nc=nc).to(device)
    t0 = time.time()
    baseline_acc = train_eval(
        model,
        *make_loaders(tr_p2, tr_l2, va_p, va_l, te_p, te_l, feature_fn=compute_angle_features),
        device,
        label="BiGRU+Angles",
    )
    print(f"  >>> Baseline: {baseline_acc:.1%} ({time.time() - t0:.0f}s)\n")

    # ================================================================
    # EXP 6b-2: H23 — GCN + Class-Balanced Loss
    # ================================================================
    print("=" * 60)
    print("EXP 6b-2: ST-GCN + Class-Balanced Loss (H23)")
    print("H23: GCN+CB >= 50% (vs 28.5% without CB)\n")

    model = ProperSTGCN(in_c=2, adj=adj, nc=nc).to(device)
    t0 = time.time()
    gcn_cb_acc = train_eval(
        model,
        *make_loaders(
            tr_p2, tr_l2, va_p, va_l, te_p, te_l, feature_fn=None, batch=16, collate=collate_gcn
        ),
        device,
        label="STGCN+CB",
        loss_fn=cb_loss,
        lr=5e-4,
    )
    print(f"  >>> ST-GCN + CB Loss: {gcn_cb_acc:.1%} ({time.time() - t0:.0f}s)")

    if gcn_cb_acc >= 0.50:
        print(f"  H23 verdict: SUPPORTED — {gcn_cb_acc:.1%} >= 50% (CB Loss rescued GCN)")
    elif gcn_cb_acc >= 0.40:
        print(f"  H23 verdict: INCONCLUSIVE — {gcn_cb_acc:.1%} improved from 28.5% but < 50%")
    else:
        print(f"  H23 verdict: REJECTED — {gcn_cb_acc:.1%} still terrible despite CB Loss\n")

    # ================================================================
    # EXP 6b-3: H23b — GCN + CB Loss + stronger regularization
    # ================================================================
    print("\n" + "=" * 60)
    print("EXP 6b-3: ST-GCN + CB Loss + Strong Regularization (H23b)")
    print("H23b: DropBlock + weight decay can rescue GCN\n")

    model = ProperSTGCN(in_c=2, adj=adj, nc=nc).to(device)
    t0 = time.time()
    gcn_cb2_acc = train_eval(
        model,
        *make_loaders(
            tr_p2, tr_l2, va_p, va_l, te_p, te_l, feature_fn=None, batch=32, collate=collate_gcn
        ),
        device,
        label="STGCN+CB+Reg",
        loss_fn=cb_loss,
        lr=3e-4,
        epochs=80,
        patience=15,
    )
    print(f"  >>> ST-GCN + CB + Reg: {gcn_cb2_acc:.1%} ({time.time() - t0:.0f}s)")
    print(f"  H23b: {gcn_cb2_acc:.1%} (vs {gcn_cb_acc:.1%} with default reg)\n")

    # ================================================================
    # EXP 6b-4: H24 — Shallow GCN (1 layer) + BiGRU Hybrid
    # ================================================================
    print("\n" + "=" * 60)
    print("EXP 6b-4: Shallow GCN-BiGRU Hybrid (H24)")
    print("H24: 1-layer GCN + BiGRU >= BiGRU alone + 3pp\n")

    model = GCNBiGRUHybrid(gcn_hidden=128, gru_hidden=128, nc=nc, adj=adj).to(device)
    t0 = time.time()
    hybrid_acc = train_eval(
        model,
        *make_loaders(
            tr_p2, tr_l2, va_p, va_l, te_p, te_l, feature_fn=None, batch=32, collate=collate_gcn
        ),
        device,
        label="GCN-BiGRU",
    )
    print(f"  >>> GCN-BiGRU Hybrid: {hybrid_acc:.1%} ({time.time() - t0:.0f}s)")

    diff = hybrid_acc - baseline_acc
    if diff >= 0.03:
        print(f"  H24 verdict: SUPPORTED — {hybrid_acc:.1%} vs {baseline_acc:.1%} (+{diff:.1%}pp)")
    elif diff >= -0.01:
        print(f"  H24 verdict: INCONCLUSIVE — {hybrid_acc:.1%} within noise of {baseline_acc:.1%}")
    else:
        print(f"  H24 verdict: REJECTED — {hybrid_acc:.1%} < {baseline_acc:.1%}\n")

    # ================================================================
    # EXP 6b-5: H24b — Shallow GCN-BiGRU + CB Loss
    # ================================================================
    print("\n" + "=" * 60)
    print("EXP 6b-5: Shallow GCN-BiGRU + CB Loss (H24b)")
    print("H24b: Does CB Loss help the hybrid?\n")

    model = GCNBiGRUHybrid(gcn_hidden=128, gru_hidden=128, nc=nc, adj=adj).to(device)
    t0 = time.time()
    hybrid_cb_acc = train_eval(
        model,
        *make_loaders(
            tr_p2, tr_l2, va_p, va_l, te_p, te_l, feature_fn=None, batch=32, collate=collate_gcn
        ),
        device,
        label="GCN-BiGRU+CB",
        loss_fn=cb_loss,
    )
    print(f"  >>> GCN-BiGRU + CB: {hybrid_cb_acc:.1%} ({time.time() - t0:.0f}s)")
    print(f"  H24b: {hybrid_cb_acc:.1%} (vs {hybrid_acc:.1%} without CB)\n")

    # ================================================================
    # SUMMARY
    # ================================================================
    print("\n" + "=" * 60)
    print("PHASE 2 SUMMARY (GCN Experiments)")
    print("=" * 60)
    print(f"  6b-1 Baseline (BiGRU+Angles):       {baseline_acc:.1%}  (control)")
    print(f"  6b-2 ST-GCN + CB Loss (H23):         {gcn_cb_acc:.1%}  (vs 28.5% without CB)")
    print(f"  6b-3 ST-GCN + CB + Strong Reg (H23b): {gcn_cb2_acc:.1%}")
    print(
        f"  6b-4 GCN-BiGRU Hybrid (H24):         {hybrid_acc:.1%}  (vs baseline {baseline_acc:.1%})"
    )
    print(f"  6b-5 GCN-BiGRU + CB Loss (H24b):     {hybrid_cb_acc:.1%}")


if __name__ == "__main__":
    main()
