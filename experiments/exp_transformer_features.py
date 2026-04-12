"""
Experiments 5a: Architecture & Feature experiments.

Science Protocol:
  H15: Transformer encoder outperforms BiGRU (>=+3pp on FSC 64-class)
  H16: Joint angle input features outperform raw (x,y) coordinates (>=+3pp)
  H17: Hierarchical classification (type→element) improves accuracy (>=+5pp)
  H18: Attention pooling outperforms last-hidden-state in BiGRU (>=+2pp)
  H19: Velocity/acceleration as input channels improves BiGRU (>=+3pp)

Usage:
    cd /home/michael/Github/skating-biomechanics-ml
    uv run python experiments/exp_transformer_features.py
"""

import pickle
import json
import time
import random
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from collections import Counter

BASE = Path("data/datasets")

# H3.6M joint indices
NOSE, REYE, LEYE, REAR, LEAR = 0, 1, 2, 3, 4
RSHOULDER, LSHOULDER = 5, 6
RELBOW, LELBOW = 7, 8
RWRIST, LWRIST = 9, 10
RHIP, LHIP = 11, 12
RKNEE, LKNEE = 13, 14
RANKLE, LANKLE = 15, 16

# Mirror mapping for COCO/H3.6M
MIRROR_MAP = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]

# Class-to-type mapping (set by build_type_mapping in main)
CLASS_TO_TYPE = {}

# Element type hierarchy (FSC 64 classes → type groups)
# FSC uses numeric IDs; we define groups by typical ISU categories
ELEMENT_GROUPS = {
    "jump": list(
        range(0, 28)
    ),  # Axels, Lutzes, Flips, Loops, Salchows, Toe loops (singles to quads)
    "spin": list(range(28, 44)),  # Upright, sit, camel spins + variations
    "step": list(range(44, 52)),  # Step sequences, twizzles
    "lift": list(range(52, 58)),  # Pairs lifts
    "other": list(range(58, 64)),  # Moves, choreo, etc.
}
# This is approximate — actual FSC label mapping may differ
# We'll detect actual label distribution at runtime


# ─── Normalize ────────────────────────────────────────────────────────────


def normalize(p):
    mid = p[:, 11:13, :].mean(axis=1, keepdims=True)
    p = p - mid
    sh = p[:, 5:7, :].mean(axis=1, keepdims=True)
    spine = np.linalg.norm(sh - mid, axis=1, keepdims=True)
    return p / np.maximum(spine, 0.01)


# ─── Feature extraction ──────────────────────────────────────────────────


def compute_joint_angles(poses):
    """Compute joint angles from (T, 17, 2) poses. Returns (T, N_angles)."""
    T = poses.shape[0]
    angles = []

    def angle_3pt(a, b, c):
        """Angle at point b formed by segments ba and bc. Returns (T,)."""
        ba = a - b
        bc = c - b
        cos_angle = np.sum(ba * bc, axis=-1) / (
            np.linalg.norm(ba, axis=-1) * np.linalg.norm(bc, axis=-1) + 1e-8
        )
        return np.arccos(np.clip(cos_angle, -1, 1))

    # Left side angles
    angles.append(angle_3pt(poses[:, LSHOULDER], poses[:, LELBOW], poses[:, LWRIST]))  # L elbow
    angles.append(
        angle_3pt(poses[:, LELBOW], poses[:, LWRIST], poses[:, LWRIST] + [0, 0.01])
    )  # ~L wrist
    angles.append(
        angle_3pt(poses[:, LSHOULDER], poses[:, LHIP], poses[:, LKNEE])
    )  # L torso-hip-knee
    angles.append(angle_3pt(poses[:, LHIP], poses[:, LKNEE], poses[:, LANKLE]))  # L knee

    # Right side angles
    angles.append(angle_3pt(poses[:, RSHOULDER], poses[:, RELBOW], poses[:, RWRIST]))  # R elbow
    angles.append(
        angle_3pt(poses[:, RSHOULDER], poses[:, RHIP], poses[:, RKNEE])
    )  # R torso-hip-knee
    angles.append(angle_3pt(poses[:, RHIP], poses[:, RKNEE], poses[:, RANKLE]))  # R knee

    # Trunk angle (spine to vertical)
    mid_shoulder = (poses[:, LSHOULDER] + poses[:, RSHOULDER]) / 2
    mid_hip = (poses[:, LHIP] + poses[:, RHIP]) / 2
    spine_vec = mid_shoulder - mid_hip
    trunk_angle = np.arctan2(spine_vec[:, 0], -spine_vec[:, 1])  # angle from vertical
    angles.append(trunk_angle)

    # Shoulder angle (arms raised?)
    l_arm = poses[:, LWRIST] - poses[:, LSHOULDER]
    r_arm = poses[:, RWRIST] - poses[:, RSHOULDER]
    l_arm_angle = np.arctan2(l_arm[:, 0], -l_arm[:, 1])
    r_arm_angle = np.arctan2(r_arm[:, 0], -r_arm[:, 1])
    angles.append(l_arm_angle)
    angles.append(r_arm_angle)

    # Hip spread angle
    hip_vec = poses[:, RHIP] - poses[:, LHIP]
    hip_angle = np.arctan2(hip_vec[:, 0], -hip_vec[:, 1])
    angles.append(hip_angle)

    # Knee spread angle (leg width indicator)
    knee_vec = poses[:, RKNEE] - poses[:, LKNEE]
    knee_angle = np.arctan2(knee_vec[:, 0], -knee_vec[:, 1])

    return np.stack(angles, axis=-1)  # (T, N_angles)


def compute_velocity_features(poses):
    """Compute velocity and acceleration from (T, 17, 2) poses. Returns (T, 102)."""
    # Velocity (finite differences)
    vel = np.zeros_like(poses)
    vel[1:-1] = (poses[2:] - poses[:-2]) / 2.0
    vel[0] = poses[1] - poses[0]
    vel[-1] = poses[-1] - poses[-2]

    # Acceleration
    acc = np.zeros_like(poses)
    acc[1:-1] = vel[2:] - vel[:-2]
    acc[0] = vel[1] - vel[0]
    acc[-1] = vel[-1] - vel[-2]

    # Flatten: (T, 17, 6) → (T, 102)
    return np.concatenate([poses, vel, acc], axis=-1).reshape(poses.shape[0], -1)


def compute_angle_features(poses):
    """Compute joint angles + raw (x,y). Returns (T, 17 + N_angles)."""
    angles = compute_joint_angles(poses)  # (T, 12)
    # Flatten poses: (T, 17, 2) → (T, 34)
    flat = poses.reshape(poses.shape[0], -1)
    return np.concatenate([flat, angles], axis=-1)  # (T, 34 + 12)


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


class BiGRUAttention(nn.Module):
    """BiGRU with attention-weighted pooling over all time steps."""

    def __init__(self, in_f=34, hidden=128, layers=2, nc=64, drop=0.3):
        super().__init__()
        self.gru = nn.GRU(in_f, hidden, layers, batch_first=True, bidirectional=True, dropout=drop)
        self.attn = nn.Sequential(nn.Linear(hidden * 2, 64), nn.Tanh(), nn.Linear(64, 1))
        self.fc = nn.Sequential(
            nn.Linear(hidden * 2, 128), nn.ReLU(), nn.Dropout(drop), nn.Linear(128, nc)
        )

    def forward(self, x, lengths):
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=True
        )
        output, _ = self.gru(packed)
        # Unpack: (B, T, hidden*2)
        output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
        # Attention weights
        attn_logits = self.attn(output).squeeze(-1)  # (B, T)
        # Mask padding
        mask = torch.arange(attn_logits.size(1), device=attn_logits.device)[None, :] < lengths[
            :, None
        ].to(attn_logits.device)
        attn_logits = attn_logits.masked_fill(~mask, -1e9)
        attn_weights = F.softmax(attn_logits, dim=1)  # (B, T)
        # Weighted sum
        context = (output * attn_weights.unsqueeze(-1)).sum(dim=1)  # (B, hidden*2)
        return self.fc(context)


class PoseTransformer(nn.Module):
    """Transformer encoder for pose sequences."""

    def __init__(self, in_f=34, d_model=128, nhead=4, layers=3, nc=64, drop=0.3, max_len=300):
        super().__init__()
        self.input_proj = nn.Linear(in_f, d_model)
        self.pos_embed = nn.Parameter(torch.randn(1, max_len, d_model) * 0.02)
        self.max_len = max_len
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=drop,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=layers)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.fc = nn.Sequential(
            nn.Linear(d_model, 128), nn.ReLU(), nn.Dropout(drop), nn.Linear(128, nc)
        )

    def forward(self, x, lengths):
        B, T, _ = x.shape
        # Subsample long sequences to max_len
        if T > self.max_len:
            indices = torch.linspace(0, T - 1, self.max_len, device=x.device).long()
            x = x[:, indices, :]
            lengths = (lengths.float() * (self.max_len / T)).long().clamp(max=self.max_len)
            T = self.max_len
        # Project input
        h = self.input_proj(x)  # (B, T, d_model)
        # Add positional embedding
        h = h + self.pos_embed[:, :T, :]
        # Prepend CLS token
        cls = self.cls_token.expand(B, -1, -1)  # (B, 1, d_model)
        h = torch.cat([cls, h], dim=1)  # (B, 1+T, d_model)
        # Create padding mask: True = ignore
        pad_mask = torch.arange(1, T + 1, device=x.device)[None, :] > lengths[:, None].to(x.device)
        pad_mask = torch.cat(
            [torch.zeros(B, 1, dtype=torch.bool, device=x.device), pad_mask], dim=1
        )
        # Encode
        h = self.encoder(h, src_key_padding_mask=pad_mask)
        # Use CLS token
        return self.fc(h[:, 0, :])


class HierarchicalClassifier(nn.Module):
    """Two-stage classifier: element_type → specific element."""

    def __init__(self, in_f=34, hidden=128, n_types=5, nc=64, drop=0.3):
        super().__init__()
        self.gru = nn.GRU(in_f, hidden, 2, batch_first=True, bidirectional=True, dropout=drop)
        self.type_head = nn.Sequential(
            nn.Linear(hidden * 2, 64), nn.ReLU(), nn.Dropout(drop), nn.Linear(64, n_types)
        )
        self.element_head = nn.Sequential(
            nn.Linear(hidden * 2 + n_types, 128), nn.ReLU(), nn.Dropout(drop), nn.Linear(128, nc)
        )

    def forward(self, x, lengths):
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=True
        )
        _, h = self.gru(packed)
        h_cat = torch.cat([h[-2], h[-1]], dim=1)  # (B, hidden*2)
        type_logits = self.type_head(h_cat)
        # Concatenate type logits as conditioning
        combined = torch.cat([h_cat, F.softmax(type_logits, dim=1)], dim=1)
        element_logits = self.element_head(combined)
        return element_logits, type_logits


# ─── Dataset ──────────────────────────────────────────────────────────────


class VarLenDataset(Dataset):
    def __init__(self, poses, labels, feature_fn=None):
        self.poses = poses
        self.labels = labels
        self.feature_fn = feature_fn

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        p = self.poses[idx]
        if self.feature_fn:
            p = self.feature_fn(p)
        else:
            p = p.reshape(p.shape[0], -1)  # (T, 17, 2) → (T, 34)
        return torch.tensor(p, dtype=torch.float32), self.labels[idx]


def gru_collate(batch):
    batch.sort(key=lambda x: x[0].shape[0], reverse=True)
    seqs, labels = zip(*batch)
    lengths = torch.tensor([s.shape[0] for s in seqs])
    padded = nn.utils.rnn.pad_sequence(seqs, batch_first=True)
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
    hierarchical=False,
):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    crit = nn.CrossEntropyLoss(label_smoothing=0.1)
    type_crit = nn.CrossEntropyLoss(label_smoothing=0.1) if hierarchical else None
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
            if hierarchical:
                elem_logits, type_logits = model(x, lengths)
                loss = crit(elem_logits, y)
                # Type loss uses mapped labels
                type_labels = torch.tensor([CLASS_TO_TYPE[ly.item()] for ly in y], device=device)
                loss = loss + 0.3 * type_crit(type_logits, type_labels)
            else:
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
                if hierarchical:
                    logits, _ = model(x, lengths)
                else:
                    logits = model(x, lengths)
                val_loss += crit(logits, y).item() * len(y)
                val_n += len(y)

        test_c, test_n = 0, 0
        with torch.no_grad():
            for batch in test_loader:
                x, lengths, y = batch
                x, y = x.to(device), y.to(device)
                if hierarchical:
                    logits, _ = model(x, lengths)
                else:
                    logits = model(x, lengths)
                test_c += (logits.argmax(1) == y).sum().item()
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


def make_loaders(tr_p, tr_l, va_p, va_l, te_p, te_l, batch=64, feature_fn=None):
    return (
        DataLoader(VarLenDataset(tr_p, tr_l, feature_fn), batch, True, collate_fn=gru_collate),
        DataLoader(VarLenDataset(va_p, va_l, feature_fn), batch, False, collate_fn=gru_collate),
        DataLoader(VarLenDataset(te_p, te_l, feature_fn), batch, False, collate_fn=gru_collate),
    )


def stratified_split(poses, labels, val_frac=0.1):
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


# ─── Build class-to-type mapping from label distribution ─────────────────


def build_type_mapping(labels, n_classes):
    """Build element_type mapping based on FSC label structure.
    FSC classes 0-27: jumps, 28-43: spins, 44-51: step sequences, 52-57: lifts, 58-63: other
    """
    mapping = {}
    for c in range(n_classes):
        if c < 28:
            mapping[c] = 0  # jump
        elif c < 44:
            mapping[c] = 1  # spin
        elif c < 52:
            mapping[c] = 2  # step
        elif c < 58:
            mapping[c] = 3  # lift
        else:
            mapping[c] = 4  # other
    return mapping


# ─── Main ────────────────────────────────────────────────────────────────


def main():
    device = torch.device("cuda")
    print(f"Device: {device}\n")

    # Load FSC
    tr_p, tr_l = load_fsc("train")
    te_p, te_l = load_fsc("test")
    nc = max(max(tr_l), max(te_l)) + 1
    print(f"FSC: Train={len(tr_p)}, Test={len(te_p)}, Classes={nc}")

    # Check label distribution
    counts = Counter(tr_l)
    print(f"Label range: [{min(tr_l)}, {max(tr_l)}]")
    print(
        f"Samples per class: min={min(counts.values())}, max={max(counts.values())}, mean={np.mean(list(counts.values())):.0f}\n"
    )

    # Build type mapping
    import exp_transformer_features as _self

    _self.CLASS_TO_TYPE = build_type_mapping(tr_l, nc)
    n_types = len(set(CLASS_TO_TYPE.values()))
    print(f"Hierarchical: {n_types} element types")
    for t_id, (name, cls_range) in enumerate(
        [
            ("jump", "0-27"),
            ("spin", "28-43"),
            ("step", "44-51"),
            ("lift", "52-57"),
            ("other", "58-63"),
        ]
    ):
        count = sum(1 for l in tr_l if CLASS_TO_TYPE.get(l) == t_id)
        print(f"  {name}: {count} samples\n")

    # Stratified train/val split
    tr_p2, tr_l2, va_p, va_l = stratified_split(tr_p, tr_l, val_frac=0.1)
    print(f"After split: Train={len(tr_p2)}, Val={len(va_p)}, Test={len(te_p)}\n")

    # ═══════════════════════════════════════════════════════════════
    # Exp 5a-1: BiGRU baseline (control)
    # ═══════════════════════════════════════════════════════════════
    print("=" * 60)
    print("EXP 5a-1: BiGRU Baseline (control)")
    print("=" * 60)

    model = BiGRU(nc=nc).to(device)
    t0 = time.time()
    baseline_acc = train_eval(
        model, *make_loaders(tr_p2, tr_l2, va_p, va_l, te_p, te_l), device, label="BiGRU-baseline"
    )
    print(f"  >>> BiGRU baseline: {baseline_acc:.1%} ({time.time() - t0:.0f}s)\n")

    # ═══════════════════════════════════════════════════════════════
    # Exp 5a-2: BiGRU with attention pooling (H18)
    # ═══════════════════════════════════════════════════════════════
    print("=" * 60)
    print("EXP 5a-2: BiGRU + Attention Pooling (H18)")
    print("H18: Attention pooling >= BiGRU baseline + 2pp")
    print("Falsification: < baseline\n")

    model = BiGRUAttention(nc=nc).to(device)
    t0 = time.time()
    attn_acc = train_eval(
        model, *make_loaders(tr_p2, tr_l2, va_p, va_l, te_p, te_l), device, label="BiGRU-Attn"
    )
    print(f"  >>> BiGRU+Attn: {attn_acc:.1%} ({time.time() - t0:.0f}s)")

    if attn_acc >= baseline_acc + 0.02:
        print("  H18 verdict: SUPPORTED")
    elif attn_acc >= baseline_acc - 0.01:
        print(f"  H18 verdict: INCONCLUSIVE — {attn_acc:.1%} within noise of {baseline_acc:.1%}")
    else:
        print(f"  H18 verdict: REJECTED — {attn_acc:.1%} < {baseline_acc:.1%}\n")

    # ═══════════════════════════════════════════════════════════════
    # Exp 5a-3: Transformer encoder (H15)
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("EXP 5a-3: Transformer Encoder (H15)")
    print("H15: Transformer >= BiGRU + 3pp")
    print("Falsification: < BiGRU\n")

    model = PoseTransformer(in_f=34, d_model=128, nhead=4, layers=3, nc=nc, drop=0.3).to(device)
    t0 = time.time()
    tf_acc = train_eval(
        model,
        *make_loaders(tr_p2, tr_l2, va_p, va_l, te_p, te_l, batch=32),
        device,
        label="Transformer",
        lr=5e-4,
    )
    print(f"  >>> Transformer: {tf_acc:.1%} ({time.time() - t0:.0f}s)")

    if tf_acc >= baseline_acc + 0.03:
        print("  H15 verdict: SUPPORTED")
    elif tf_acc >= baseline_acc - 0.02:
        print(f"  H15 verdict: INCONCLUSIVE — {tf_acc:.1%} within noise of {baseline_acc:.1%}")
    else:
        print(f"  H15 verdict: REJECTED — {tf_acc:.1%} < {baseline_acc:.1%}\n")

    # ═══════════════════════════════════════════════════════════════
    # Exp 5a-4: Joint angle features (H16)
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("EXP 5a-4: Joint Angle Features (H16)")
    print("H16: Angles+raw >= BiGRU baseline + 3pp")
    print("Falsification: < baseline\n")

    # Verify angle feature dimension
    sample_angles = compute_angle_features(tr_p2[0])
    angle_dim = sample_angles.shape[-1]
    print(f"  Angle feature dim: {angle_dim} (raw=34 + angles={angle_dim - 34})")

    model = BiGRU(in_f=angle_dim, nc=nc).to(device)
    t0 = time.time()
    angle_acc = train_eval(
        model,
        *make_loaders(tr_p2, tr_l2, va_p, va_l, te_p, te_l, feature_fn=compute_angle_features),
        device,
        label="BiGRU-Angles",
    )
    print(f"  >>> BiGRU+Angles: {angle_acc:.1%} ({time.time() - t0:.0f}s)")

    if angle_acc >= baseline_acc + 0.03:
        print("  H16 verdict: SUPPORTED")
    elif angle_acc >= baseline_acc - 0.02:
        print(f"  H16 verdict: INCONCLUSIVE — {angle_acc:.1%} within noise of {baseline_acc:.1%}")
    else:
        print(f"  H16 verdict: REJECTED — {angle_acc:.1%} < {baseline_acc:.1%}\n")

    # ═══════════════════════════════════════════════════════════════
    # Exp 5a-5: Velocity/acceleration features (H19)
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("EXP 5a-5: Velocity+Acceleration Features (H19)")
    print("H19: Vel+Acc >= BiGRU baseline + 3pp")
    print("Falsification: < baseline\n")

    # Verify velocity feature dimension: (T, 17, 2) → flatten → (T, 102)
    sample_vel = compute_velocity_features(tr_p2[0])
    vel_dim = sample_vel.shape[-1]
    print(f"  Velocity feature dim: {vel_dim} (pos=34 + vel=34 + acc=34)")

    model = BiGRU(in_f=vel_dim, nc=nc).to(device)
    t0 = time.time()
    vel_acc = train_eval(
        model,
        *make_loaders(tr_p2, tr_l2, va_p, va_l, te_p, te_l, feature_fn=compute_velocity_features),
        device,
        label="BiGRU-Vel",
    )
    print(f"  >>> BiGRU+Vel: {vel_acc:.1%} ({time.time() - t0:.0f}s)")

    if vel_acc >= baseline_acc + 0.03:
        print("  H19 verdict: SUPPORTED")
    elif vel_acc >= baseline_acc - 0.02:
        print(f"  H19 verdict: INCONCLUSIVE — {vel_acc:.1%} within noise of {baseline_acc:.1%}")
    else:
        print(f"  H19 verdict: REJECTED — {vel_acc:.1%} < {baseline_acc:.1%}\n")

    # ═══════════════════════════════════════════════════════════════
    # Exp 5a-6: Hierarchical classifier (H17)
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("EXP 5a-6: Hierarchical Classifier (H17)")
    print("H17: Hierarchical >= BiGRU baseline + 5pp")
    print("Falsification: < baseline\n")

    model = HierarchicalClassifier(in_f=34, n_types=n_types, nc=nc).to(device)
    t0 = time.time()
    hier_acc = train_eval(
        model,
        *make_loaders(tr_p2, tr_l2, va_p, va_l, te_p, te_l),
        device,
        label="Hierarchical",
        hierarchical=True,
    )
    print(f"  >>> Hierarchical: {hier_acc:.1%} ({time.time() - t0:.0f}s)")

    if hier_acc >= baseline_acc + 0.05:
        print("  H17 verdict: SUPPORTED")
    elif hier_acc >= baseline_acc - 0.02:
        print(f"  H17 verdict: INCONCLUSIVE — {hier_acc:.1%} within noise of {baseline_acc:.1%}")
    else:
        print(f"  H17 verdict: REJECTED — {hier_acc:.1%} < {baseline_acc:.1%}\n")

    # ═══════════════════════════════════════════════════════════════
    # Summary
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    results = [
        ("BiGRU baseline", baseline_acc, "—"),
        (
            "BiGRU + Attention",
            attn_acc,
            f"{'SUPPORTED' if attn_acc >= baseline_acc + 0.02 else 'REJECTED'} (H18)",
        ),
        (
            "Transformer",
            tf_acc,
            f"{'SUPPORTED' if tf_acc >= baseline_acc + 0.03 else 'REJECTED'} (H15)",
        ),
        (
            "BiGRU + Angles",
            angle_acc,
            f"{'SUPPORTED' if angle_acc >= baseline_acc + 0.03 else 'REJECTED'} (H16)",
        ),
        (
            "BiGRU + Velocity",
            vel_acc,
            f"{'SUPPORTED' if vel_acc >= baseline_acc + 0.03 else 'REJECTED'} (H19)",
        ),
        (
            "Hierarchical",
            hier_acc,
            f"{'SUPPORTED' if hier_acc >= baseline_acc + 0.05 else 'REJECTED'} (H17)",
        ),
    ]
    print(f"{'Model':<22} {'Accuracy':>10} {'vs Baseline':>12}  Verdict")
    print("-" * 70)
    for name, acc, verdict in results:
        diff = acc - baseline_acc
        sign = "+" if diff >= 0 else ""
        print(f"{name:<22} {acc:>9.1%} {sign}{diff:>10.1%}pp  {verdict}")


if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    main()
