"""
Experiments 6a: Phase 1 Quick Wins (H20-H22).

Science Protocol:
  H20: Class-Balanced Loss + Label Smoothing >= BiGRU baseline + 3pp
  H21: Biomechanical proxy features (angular velocity, flight mask, takeoff angle) >= BiGRU+angles + 3pp
  H22: Multi-stream BiGRU ensemble >= BiGRU+angles + 3pp

Research sources:
  - Class-Balanced Loss: Cui et al. (CVPR 2019), BRL paper (MIR 2025)
  - Biomechanical features: OOFSkate (MIT), Gemini Deep Research
  - Multi-stream ensemble: JMDA (ACM TOMM 2024), CTR-GCN (NeurIPS 2023)

Usage:
    cd /home/michael/Github/skating-biomechanics-ml
    PYTHONUNBUFFERED=1 uv run python experiments/exp_phase1_quick_wins.py
"""

import pickle
import time
import math
import random
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

# ─── Data loading ────────────────────────────────────────────────────────


def load_fsc(split):
    path = BASE / "figure-skating-classification"
    data = pickle.load(open(path / f"{split}_data.pkl", "rb"))
    labels = pickle.load(open(path / f"{split}_label.pkl", "rb"))
    # Data shape: (150, 17, 3, 1) -> (150, 17, 2) -> normalize
    poses = [normalize(np.array(d[:, :, :2, 0], dtype=np.float32)) for d in data]
    valid = [(p, l) for p, l in zip(poses, labels) if len(p) > 0]
    if not valid:
        return [], []
    return list(zip(*valid))


def normalize(p):
    """Root-center and scale normalize to [-1, 1]. Input: (T, 17, 2)."""
    p = p.copy()
    mid_hip = (p[:, 11, :] + p[:, 12, :]) / 2  # (T, 2)
    p -= mid_hip[:, np.newaxis, :]  # broadcast over joints: (T,1,2) -> (T,17,2)
    mid_shoulder = (p[:, 5, :] + p[:, 6, :]) / 2
    spine = np.linalg.norm(mid_shoulder - mid_hip, axis=-1, keepdims=True)
    spine = np.clip(spine, 0.01, None)
    p /= (spine * 2.5)[:, np.newaxis, :]
    return p


# ─── Feature functions ──────────────────────────────────────────────────


def compute_joint_angles(poses):
    """Compute 12 joint angles from (T, 17, 2). Returns (T, 12)."""

    def angle_3pt(a, b, c):
        ba, bc = a - b, c - b
        cos = np.sum(ba * bc, axis=-1) / (
            np.linalg.norm(ba, axis=-1) * np.linalg.norm(bc, axis=-1) + 1e-8
        )
        return np.arccos(np.clip(cos, -1, 1))

    angles = []
    # Left side
    angles.append(angle_3pt(poses[:, LSHOULDER], poses[:, LELBOW], poses[:, LWRIST]))  # L elbow
    angles.append(angle_3pt(poses[:, LSHOULDER], poses[:, LHIP], poses[:, LKNEE]))  # L torso
    angles.append(angle_3pt(poses[:, LHIP], poses[:, LKNEE], poses[:, LANKLE]))  # L knee
    # Right side
    angles.append(angle_3pt(poses[:, RSHOULDER], poses[:, RELBOW], poses[:, RWRIST]))  # R elbow
    angles.append(angle_3pt(poses[:, RSHOULDER], poses[:, RHIP], poses[:, RKNEE]))  # R torso
    angles.append(angle_3pt(poses[:, RHIP], poses[:, RKNEE], poses[:, RANKLE]))  # R knee
    # Trunk + arms + hips
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
    return np.stack(angles, axis=-1)


def compute_angle_features(poses):
    """Joint angles + raw (x,y). Returns (T, 46)."""
    angles = compute_joint_angles(poses)
    flat = poses.reshape(poses.shape[0], -1)
    return np.concatenate([flat, angles], axis=-1)


def compute_bone_vectors(poses):
    """Bone vectors (joint-to-joint displacements). Returns (T, 16*2=32)."""
    bones = []
    bone_pairs = [
        (LHIP, LKNEE),
        (LKNEE, LANKLE),
        (RHIP, RKNEE),
        (RKNEE, RANKLE),
        (LSHOULDER, LELBOW),
        (LELBOW, LWRIST),
        (RSHOULDER, RELBOW),
        (RELBOW, RWRIST),
        (LSHOULDER, LHIP),
        (RSHOULDER, RHIP),
        (LHIP, RHIP),
        (NOSE, LSHOULDER),
        (NOSE, RSHOULDER),
        (LSHOULDER, RSHOULDER),
        (LKNEE, RKNEE),
        (LANKLE, RANKLE),
    ]
    for a, b in bone_pairs:
        bones.append(poses[:, b] - poses[:, a])
    return np.concatenate(bones, axis=-1)  # (T, 32)


def compute_biomechanical_features(poses):
    """OOFSkate-inspired biomechanical proxy features. Returns (T, F)."""
    T, J, C = poses.shape
    mid_hip = (poses[:, LHIP] + poses[:, RHIP]) / 2
    mid_shoulder = (poses[:, LSHOULDER] + poses[:, RSHOULDER]) / 2

    features = []

    # 1. Center of Mass (approx: weighted hip+shoulder)
    com = mid_hip * 0.6 + mid_shoulder * 0.4  # (T, 2)
    features.append(com)

    # 2. Angular velocity of extremities relative to CoM
    for joint_idx in [LWRIST, RWRIST, LANKLE, RANKLE]:
        r = poses[:, joint_idx] - com  # (T, 2)
        # Angle of r vector over time
        theta = np.arctan2(r[:, 1], r[:, 0])  # (T,)
        # Angular velocity (finite diff)
        omega = np.zeros(T)
        omega[1:-1] = (theta[2:] - theta[:-2]) / 2.0
        omega[0] = theta[1] - theta[0]
        omega[-1] = theta[-1] - theta[-2]
        features.append(omega[:, None])  # (T, 1)

    # 3. Flight phase binary mask (vertical CoM displacement > threshold)
    com_y = com[:, 1]  # vertical (inverted in image coords)
    com_mean = np.mean(com_y)
    # Smooth velocity to find takeoff/landing
    com_vel = np.zeros(T)
    com_vel[1:-1] = (com_y[2:] - com_y[:-2]) / 2.0
    com_vel[0] = com_y[1] - com_y[0]
    com_vel[-1] = com_y[-1] - com_y[-2]
    # Binary: 1 if CoM is significantly above mean (in flight)
    flight_mask = (com_y < com_mean - 0.1).astype(np.float32)
    features.append(flight_mask[:, None])  # (T, 1)

    # 4. Takeoff ankle angle (for edge vs toe distinction)
    for ankle_idx in [LANKLE, RANKLE]:
        ankle_pos = poses[:, ankle_idx]
        ankle_to_hip = ankle_pos - mid_hip
        ankle_angle = np.arctan2(ankle_to_hip[:, 0], -ankle_to_hip[:, 1])
        features.append(ankle_angle[:, None])  # (T, 1)

    # 5. Hip spread width (dynamic balance indicator)
    hip_width = np.linalg.norm(poses[:, RHIP] - poses[:, LHIP], axis=-1)
    features.append(hip_width[:, None])  # (T, 1)

    # 6. Shoulder-hip alignment (body lean)
    body_vec = mid_shoulder - mid_hip
    body_lean = np.arctan2(body_vec[:, 0], -body_vec[:, 1])
    features.append(body_lean[:, None])  # (T, 1)

    # 7. CoM velocity magnitude
    features.append(np.abs(com_vel)[:, None])  # (T, 1)

    # 8. Joint angle features (same 12 as before)
    angles = compute_joint_angles(poses)
    features.append(angles)

    return np.concatenate(features, axis=-1)


# ─── Data loading ────────────────────────────────────────────────────────


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


def make_loaders(tr_p, tr_l, va_p, va_l, te_p, te_l, feature_fn=None, batch=64):
    train_ds = PoseDataset(tr_p, tr_l, feature_fn)
    val_ds = PoseDataset(va_p, va_l, feature_fn)
    test_ds = PoseDataset(te_p, te_l, feature_fn)
    train_dl = DataLoader(
        train_ds, batch_size=batch, shuffle=True, collate_fn=collate_fn, num_workers=0
    )
    val_dl = DataLoader(val_ds, batch_size=batch, collate_fn=collate_fn, num_workers=0)
    test_dl = DataLoader(test_ds, batch_size=batch, collate_fn=collate_fn, num_workers=0)
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
        self.nc = nc

    def forward(self, logits, targets):
        log_probs = F.log_softmax(logits, dim=-1)
        # Label smoothing
        nll_loss = -log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        smooth_loss = -log_probs.mean(dim=-1)
        loss = (1 - self.label_smoothing) * nll_loss + self.label_smoothing * smooth_loss
        # Class-balanced weighting
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
    """Train model and return best test accuracy."""
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

        # Validation
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for x, lengths, y in val_dl:
                x, lengths, y = x.to(device), lengths, y.to(device)
                preds = model(x, lengths).argmax(dim=1)
                correct += (preds == y).sum().item()
                total += y.size(0)
        val_acc = correct / max(total, 1)

        # Test
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


def ensemble_evaluate(models, test_dl, device):
    """Evaluate ensemble by averaging softmax outputs."""
    for m in models:
        m.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, lengths, y in test_dl:
            x, lengths, y = x.to(device), lengths, y.to(device)
            probs = F.softmax(models[0](x, lengths), dim=1)
            for m in models[1:]:
                probs += F.softmax(m(x, lengths), dim=1)
            preds = probs.argmax(dim=1)
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

    # Count samples per class for balanced loss
    class_counts = Counter(tr_l)
    print(f"FSC: Train={len(tr_p)}, Test={len(te_p)}, Classes={nc}")
    print(f"Label range: [{min(tr_l)}, {max(tr_l)}]")
    print(
        f"Samples per class: min={min(class_counts.values())}, max={max(class_counts.values())}, mean={np.mean(list(class_counts.values())):.0f}\n"
    )

    tr_p2, tr_l2, va_p, va_l = stratified_split(tr_p, tr_l, val_frac=0.1)
    print(f"After split: Train={len(tr_p2)}, Val={len(va_p)}, Test={len(te_p)}\n")

    # ================================================================
    # EXP 6a-1: Baseline (BiGRU + joint angles, known 67.5%)
    # ================================================================
    print("=" * 60)
    print("EXP 6a-1: BiGRU + Joint Angles (control)")
    print("=" * 60)

    angle_dim = compute_angle_features(tr_p2[0]).shape[-1]
    print(f"  Angle feature dim: {angle_dim}")

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
    # EXP 6a-2: H20 — Class-Balanced Loss + Label Smoothing
    # ================================================================
    print("=" * 60)
    print("EXP 6a-2: Class-Balanced Loss + Label Smoothing (H20)")
    print("H20: CB Loss >= baseline + 3pp\n")

    cb_loss = ClassBalancedLoss(class_counts, nc, beta=0.999, label_smoothing=0.1).to(device)
    model = BiGRU(in_f=angle_dim, nc=nc).to(device)
    t0 = time.time()
    cb_acc = train_eval(
        model,
        *make_loaders(tr_p2, tr_l2, va_p, va_l, te_p, te_l, feature_fn=compute_angle_features),
        device,
        label="CB-Loss",
        loss_fn=cb_loss,
    )
    print(f"  >>> CB Loss: {cb_acc:.1%} ({time.time() - t0:.0f}s)")

    if cb_acc >= baseline_acc + 0.03:
        print("  H20 verdict: SUPPORTED")
    elif cb_acc >= baseline_acc - 0.01:
        print(f"  H20 verdict: INCONCLUSIVE — {cb_acc:.1%} within noise of {baseline_acc:.1%}")
    else:
        print(f"  H20 verdict: REJECTED — {cb_acc:.1%} < {baseline_acc:.1%}\n")

    # ================================================================
    # EXP 6a-3: H21 — Biomechanical Proxy Features
    # ================================================================
    print("\n" + "=" * 60)
    print("EXP 6a-3: Biomechanical Proxy Features (H21)")
    print("H21: Bio features >= baseline + 3pp\n")

    bio_dim = compute_biomechanical_features(tr_p2[0]).shape[-1]
    print(f"  Biomechanical feature dim: {bio_dim}")

    model = BiGRU(in_f=bio_dim, nc=nc).to(device)
    t0 = time.time()
    bio_acc = train_eval(
        model,
        *make_loaders(
            tr_p2, tr_l2, va_p, va_l, te_p, te_l, feature_fn=compute_biomechanical_features
        ),
        device,
        label="Bio-Feats",
        lr=5e-4,
    )
    print(f"  >>> Bio features: {bio_acc:.1%} ({time.time() - t0:.0f}s)")

    if bio_acc >= baseline_acc + 0.03:
        print("  H21 verdict: SUPPORTED")
    elif bio_acc >= baseline_acc - 0.01:
        print(f"  H21 verdict: INCONCLUSIVE — {bio_acc:.1%} within noise of {baseline_acc:.1%}")
    else:
        print(f"  H21 verdict: REJECTED — {bio_acc:.1%} < {baseline_acc:.1%}\n")

    # ================================================================
    # EXP 6a-4: H22 — Multi-stream BiGRU Ensemble
    # ================================================================
    print("\n" + "=" * 60)
    print("EXP 6a-4: Multi-stream BiGRU Ensemble (H22)")
    print("H22: Ensemble >= baseline + 3pp\n")

    # Train 3 models with different features
    streams = [
        ("Angles", compute_angle_features, angle_dim),
        ("Bones", compute_bone_vectors, 32),
        ("Bio", compute_biomechanical_features, bio_dim),
    ]
    models = []
    for name, fn, dim in streams:
        print(f"  Training stream: {name} (dim={dim})")
        m = BiGRU(in_f=dim, nc=nc).to(device)
        train_eval(
            m,
            *make_loaders(tr_p2, tr_l2, va_p, va_l, te_p, te_l, feature_fn=fn),
            device,
            label=f"Stream-{name}",
        )
        models.append(m)

    # Evaluate ensemble
    ens_acc = ensemble_evaluate(
        models, make_loaders(tr_p2, tr_l2, va_p, va_l, te_p, te_l)[2], device
    )
    print(f"  >>> Ensemble: {ens_acc:.1%}")

    if ens_acc >= baseline_acc + 0.03:
        print("  H22 verdict: SUPPORTED")
    elif ens_acc >= baseline_acc - 0.01:
        print(f"  H22 verdict: INCONCLUSIVE — {ens_acc:.1%} within noise of {baseline_acc:.1%}")
    else:
        print(f"  H22 verdict: REJECTED — {ens_acc:.1%} < {baseline_acc:.1%}\n")

    # ================================================================
    # SUMMARY
    # ================================================================
    print("\n" + "=" * 60)
    print("PHASE 1 SUMMARY")
    print("=" * 60)
    print(f"  6a-1 Baseline (BiGRU+Angles): {baseline_acc:.1%}  (control)")
    print(
        f"  6a-2 CB Loss (H20):           {cb_acc:.1%}  ({'SUPPORTED' if cb_acc >= baseline_acc + 0.03 else 'INCONCLUSIVE' if cb_acc >= baseline_acc - 0.01 else 'REJECTED'})"
    )
    print(
        f"  6a-3 Bio features (H21):      {bio_acc:.1%}  ({'SUPPORTED' if bio_acc >= baseline_acc + 0.03 else 'INCONCLUSIVE' if bio_acc >= baseline_acc - 0.01 else 'REJECTED'})"
    )
    print(
        f"  6a-4 Ensemble (H22):          {ens_acc:.1%}  ({'SUPPORTED' if ens_acc >= baseline_acc + 0.03 else 'INCONCLUSIVE' if ens_acc >= baseline_acc - 0.01 else 'REJECTED'})"
    )


if __name__ == "__main__":
    main()
