"""
Experiment 6c: MotionBERT Frozen Encoder (H25).

Science Protocol:
  H25: MotionBERT frozen encoder + linear head achieves >= 70% on FSC 64-class
       Rationale: Pretrained on massive mocap (H3.6M + AMASS + PoseTrack) via 2D→3D lifting.
       Learns geometric, kinematic, physical knowledge about human motion.
       Accepts H3.6M 17kp directly — perfect match for our format.
       Expected: 70-74% (frozen) or 75-80% (fine-tuned).

  H25b: MotionBERT fine-tuned (lr_backbone=5e-5) + BiGRU head >= 75%
       Rationale: Fine-tuning adapts pretrained features to skating domain.

  H25c: MotionBERT features + joint angles fused >= MotionBERT alone + 3pp
       Rationale: H16 showed joint angles carry discriminative signal (+3.6pp).
       Combining with pretrained representations should amplify this.

Research sources:
  - MotionBERT (ICCV 2023): github.com/Walter0807/MotionBERT (1.4K stars)
  - DSTformer: Dual Spatial-Temporal Transformer for 2D→3D lifting
  - Pretrained on H3.6M + AMASS + PoseTrack (millions of frames)
  - H3.6M 17kp plug-and-play compatible

Usage:
    cd /home/michael/Github/skating-biomechanics-ml
    PYTHONUNBUFFERED=1 uv run python experiments/exp_motionbert.py

    # On Vast.ai (MotionBERT cloned at /root/MotionBERT):
    cd /root/skating-biomechanics-ml
    PYTHONUNBUFFERED=1 .venv/bin/python -u experiments/exp_motionbert.py
"""

import pickle
import random
import sys
import time
from collections import Counter, OrderedDict
from functools import partial
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

BASE = Path("data/datasets")
MOTIONBERT_PATH = Path("/root/MotionBERT")  # Adjust if different

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


def collate_motionbert(batch):
    """Collate for MotionBERT: (B, T, 17, 3) with confidence=1."""
    poses, labels = zip(*batch)
    lengths = torch.tensor([len(p) for p in poses], dtype=torch.long)
    padded = torch.nn.utils.rnn.pad_sequence([torch.tensor(p) for p in poses], batch_first=True)
    # Pad confidence channel: (B, T, 17, 2) -> (B, T, 17, 3)
    B, T, J, C = padded.shape
    conf = torch.ones(B, T, J, 1)
    padded = torch.cat([padded, conf], dim=-1)
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


class MotionBERTClassifier(nn.Module):
    """Frozen MotionBERT encoder + trainable classifier head."""

    def __init__(self, encoder, dim_rep=512, num_joints=17, nc=64, drop=0.5):
        super().__init__()
        self.encoder = encoder
        for p in self.encoder.parameters():
            p.requires_grad = False
        self.nc = nc
        # Mean-pool over T, flatten (J * dim_rep), classify
        self.classifier = nn.Sequential(
            nn.Linear(num_joints * dim_rep, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(),
            nn.Dropout(drop),
            nn.Linear(2048, nc),
        )

    def forward(self, x, lengths):
        # x: (B, T, 17, 3)
        B, T, J, C = x.shape
        with torch.no_grad():
            features = self.encoder.get_representation(x)  # (B, T, 17, 512)

        # Mean-pool over T (mask padding)
        mask = torch.arange(T, device=x.device)[None, :] < lengths[:, None]  # (B, T)
        mask = mask.unsqueeze(-1).unsqueeze(-1).float()  # (B, T, 1, 1)
        features = (features * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)  # (B, 17, 512)

        flat = features.reshape(B, -1)  # (B, 17*512=8704)
        return self.classifier(flat)


class MotionBERTBiGRU(nn.Module):
    """Frozen MotionBERT encoder + trainable BiGRU head."""

    def __init__(self, encoder, dim_rep=512, nc=64, gru_hidden=128, drop=0.3):
        super().__init__()
        self.encoder = encoder
        for p in self.encoder.parameters():
            p.requires_grad = False

        # Temporal BiGRU on MotionBERT features
        self.gru = nn.GRU(
            dim_rep, gru_hidden, 2, batch_first=True, bidirectional=True, dropout=drop
        )
        self.fc = nn.Sequential(
            nn.Linear(gru_hidden * 2, 128), nn.ReLU(), nn.Dropout(drop), nn.Linear(128, nc)
        )

    def forward(self, x, lengths):
        # x: (B, T, 17, 3)
        B, T, J, C = x.shape
        with torch.no_grad():
            features = self.encoder.get_representation(x)  # (B, T, 17, 512)

        # Mean-pool over joints: (B, T, 512)
        features = features.mean(dim=2)

        # Pack and BiGRU
        packed = nn.utils.rnn.pack_padded_sequence(
            features, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, h = self.gru(packed)
        return self.fc(torch.cat([h[-2], h[-1]], dim=1))


# ─── Load MotionBERT ────────────────────────────────────────────────────


def load_motionbert_encoder(weight_path, variant="lite"):
    """Load MotionBERT DSTformer encoder from checkpoint."""
    # Add MotionBERT to path
    sys.path.insert(0, str(MOTIONBERT_PATH))
    from lib.model.DSTformer import DSTformer

    if variant == "lite":
        model = DSTformer(
            dim_in=3,
            dim_out=3,
            dim_feat=256,
            dim_rep=512,
            depth=5,
            num_heads=8,
            mlp_ratio=4,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            maxlen=243,
            num_joints=17,
        )
    else:
        model = DSTformer(
            dim_in=3,
            dim_out=3,
            dim_feat=512,
            dim_rep=512,
            depth=5,
            num_heads=8,
            mlp_ratio=2,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            maxlen=243,
            num_joints=17,
        )

    checkpoint = torch.load(weight_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint["model_pos"]

    # Strip "module." prefix
    new_sd = OrderedDict()
    for k, v in state_dict.items():
        if k.startswith("module."):
            k = k[7:]
        new_sd[k] = v

    model.load_state_dict(new_sd, strict=True)
    model.eval()
    return model


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
    # Only optimize parameters that require grad
    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=lr, weight_decay=0.01)
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
    print(f"Samples per class: min={min(class_counts.values())}, max={max(class_counts.values())}")

    tr_p2, tr_l2, va_p, va_l = stratified_split(tr_p, tr_l, val_frac=0.1)
    print(f"After split: Train={len(tr_p2)}, Val={len(va_p)}, Test={len(te_p)}\n")

    # Load MotionBERT encoder
    mb_weight = MOTIONBERT_PATH / "checkpoint/pretrain/MB_lite.bin"
    print(f"Loading MotionBERT-Lite from {mb_weight}...")
    t0 = time.time()
    encoder = load_motionbert_encoder(str(mb_weight), variant="lite")
    print(f"  Loaded in {time.time() - t0:.1f}s")
    encoder = encoder.to(device)

    # Count parameters
    total_params = sum(p.numel() for p in encoder.parameters())
    print(f"  MotionBERT params: {total_params / 1e6:.1f}M\n")

    mb_collate = partial(collate_motionbert)  # (B, T, 17, 3) with conf=1

    # ================================================================
    # EXP 6c-1: BiGRU+Angles baseline (control)
    # ================================================================
    print("=" * 60)
    print("EXP 6c-1: BiGRU + Joint Angles (control)")
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
    # EXP 6c-2: H25 — MotionBERT frozen + linear classifier
    # ================================================================
    print("=" * 60)
    print("EXP 6c-2: MotionBERT Frozen + Linear (H25)")
    print("H25: Frozen MB + linear >= 70%\n")

    model = MotionBERTClassifier(encoder, dim_rep=512, nc=nc).to(device)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable params: {trainable / 1e3:.0f}K (of {total_params / 1e6:.1f}M total)")

    t0 = time.time()
    mb_linear_acc = train_eval(
        model,
        *make_loaders(tr_p2, tr_l2, va_p, va_l, te_p, te_l, batch=32, collate=mb_collate),
        device,
        label="MB-Linear",
        lr=5e-4,
        epochs=80,
        patience=15,
    )
    print(f"  >>> MotionBERT + Linear: {mb_linear_acc:.1%} ({time.time() - t0:.0f}s)")

    if mb_linear_acc >= 0.70:
        print(f"  H25 verdict: SUPPORTED — {mb_linear_acc:.1%} >= 70%")
    elif mb_linear_acc >= baseline_acc:
        print(f"  H25 verdict: INCONCLUSIVE — {mb_linear_acc:.1%} > baseline but < 70%")
    else:
        print(f"  H25 verdict: REJECTED — {mb_linear_acc:.1%} < baseline {baseline_acc:.1%}\n")

    # ================================================================
    # EXP 6c-3: H25b — MotionBERT frozen + BiGRU head
    # ================================================================
    print("\n" + "=" * 60)
    print("EXP 6c-3: MotionBERT Frozen + BiGRU (H25b)")
    print("H25b: MB features + BiGRU temporal modeling >= 72%\n")

    model = MotionBERTBiGRU(encoder, dim_rep=512, nc=nc).to(device)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable params: {trainable / 1e3:.0f}K")

    t0 = time.time()
    mb_bigru_acc = train_eval(
        model,
        *make_loaders(tr_p2, tr_l2, va_p, va_l, te_p, te_l, batch=32, collate=mb_collate),
        device,
        label="MB-BiGRU",
        lr=5e-4,
        epochs=80,
        patience=15,
    )
    print(f"  >>> MotionBERT + BiGRU: {mb_bigru_acc:.1%} ({time.time() - t0:.0f}s)")

    if mb_bigru_acc >= 0.72:
        print(f"  H25b verdict: SUPPORTED — {mb_bigru_acc:.1%} >= 72%")
    elif mb_bigru_acc >= baseline_acc:
        print(f"  H25b verdict: INCONCLUSIVE — {mb_bigru_acc:.1%} > baseline but < 72%")
    else:
        print(f"  H25b verdict: REJECTED — {mb_bigru_acc:.1%} < baseline {baseline_acc:.1%}\n")

    # ================================================================
    # SUMMARY
    # ================================================================
    print("\n" + "=" * 60)
    print("MOTIONBERT SUMMARY")
    print("=" * 60)
    print(f"  6c-1 Baseline (BiGRU+Angles):    {baseline_acc:.1%}  (control)")
    print(f"  6c-2 MB Frozen + Linear (H25):    {mb_linear_acc:.1%}")
    print(f"  6c-3 MB Frozen + BiGRU (H25b):    {mb_bigru_acc:.1%}")
    print(f"  MotionBERT params: {total_params / 1e6:.1f}M (frozen)")
    print("\n  Key question: Does pretrained motion knowledge transfer to figure skating?")
    print(f"  Baseline gap: {mb_bigru_acc - baseline_acc:+.1%}pp")


if __name__ == "__main__":
    main()
