"""
Experiments 4d-4e: Proper GCN with learnable partitions + VIFSS-inspired contrastive pre-training.

Science Protocol:
  H12: Proper ST-GCN (with distance partitioning + learnable weights) outperforms BiGRU on FSC 64-class
       Prediction: GCN >= BiGRU + 3pp (>= 70.3%). Falsification: GCN <= BiGRU (<= 67.3%)
  H13: Contrastive pre-training on skeleton poses improves classification under low-data regime
       (10% of training data). Prediction: +10pp over no pre-training. Falsification: < +3pp
  H14: Transfer learning from FSC (64 classes) → MCFS (48 classes) beats training MCFS from scratch
       Prediction: +5pp improvement. Falsification: < +1pp

Why H12 matters: Previous ST-GCN (exp_gcn_mcfs.py) used fixed adjacency with no learnable
spatial parameters — essentially a spatial smoothing filter, not a graph neural network. The
real ST-GCN (Yan et al. 2018) uses partition strategies that split the adjacency into subsets
and applies separate learnable convolutions to each subset.

Why H13 matters: VIFSS (Tanaka et al. 2025, arxiv:2508.10281) showed contrastive pre-training
on 3D poses gives +7pp on TAS. We test if a simpler version (no 3D, just 2D skeletons) helps
for classification under data scarcity.

Why H14 matters: FSC has 5168 sequences across 64 classes. MCFS has 2668 across 130 classes.
If features learned on FSC transfer to MCFS, we can leverage larger datasets for rare classes.

Usage:
    cd /home/michael/Github/skating-biomechanics-ml
    uv run python experiments/exp_gcn_proper_mcfs.py
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

# Natural node grouping for distance partitioning
# 0: root (hip center), 1: left arm, 2: right arm, 3: left leg, 4: right leg
DISTANCE_PARTITION = {
    0: [0, 1, 2, 3, 4],  # head → root
    1: [5, 7, 9],  # left shoulder, left elbow, left wrist
    2: [6, 8, 10],  # right shoulder, right elbow, right wrist
    3: [11, 13, 15],  # left hip, left knee, left ankle
    4: [12, 14, 16],  # right hip, right knee, right ankle
}


def build_partitioned_adjacency(num_joints=17):
    """Build 3-partition adjacency matrices following ST-GCN distance partitioning.

    Partition 0: self-loops (root = 1.0)
    Partition 1: centripetal (closer to root, distance < D)
    Partition 2: centrifugal (farther from root, distance >= D)
    """
    # Build full adjacency + BFS distances
    A_full = np.zeros((num_joints, num_joints), dtype=np.float32)
    for i, j in SKELETON_EDGES:
        A_full[i, j] = 1.0
        A_full[j, i] = 1.0

    # Build adjacency list
    adj_list = {i: [] for i in range(num_joints)}
    for i, j in SKELETON_EDGES:
        adj_list[i].append(j)
        adj_list[j].append(i)

    # BFS distances from body center (joint 0 in H3.6M = nose, but body is 5-16)
    # H3.6M has two disconnected components: head (0-4) and body (5-16)
    # Distance partitioning only makes sense for connected subgraphs
    # We'll use BFS from the torso center, assigning max distance to disconnected nodes
    root = 11  # left hip (body center)
    dist = {}
    queue = [(root, 0)]
    seen = {root}
    while queue:
        node, d = queue.pop(0)
        dist[node] = d
        for nb in adj_list[node]:
            if nb not in seen:
                seen.add(nb)
                queue.append((nb, d + 1))
    # Assign remaining disconnected nodes (head) distance 99
    for i in range(num_joints):
        if i not in dist:
            dist[i] = 99

    # Build partition masks
    A_self = np.zeros((num_joints, num_joints), dtype=np.float32)
    A_cent = np.zeros((num_joints, num_joints), dtype=np.float32)
    A_centrif = np.zeros((num_joints, num_joints), dtype=np.float32)

    for i in range(num_joints):
        for j in range(num_joints):
            if A_full[i, j] > 0:
                if i == j:
                    A_self[i, j] = 1.0
                elif dist[j] <= dist[i]:
                    A_cent[i, j] = 1.0  # neighbor closer to root
                else:
                    A_centrif[i, j] = 1.0  # neighbor farther from root

    # Normalize each partition matrix
    def normalize(A):
        D = A.sum(axis=1, keepdims=True) + 1e-8
        return A / D

    return (
        torch.tensor(normalize(A_self)),
        torch.tensor(normalize(A_cent)),
        torch.tensor(normalize(A_centrif)),
    )


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


def load_mcfs_segments(min_samples=10):
    segs = pickle.load(open(BASE / "mcfs/segments.pkl", "rb"))
    counts = Counter(s[1] for s in segs)
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


class PartitionedGraphConv(nn.Module):
    """ST-GCN spatial convolution with distance partitioning + learnable weights.

    Splits adjacency into 3 subsets (self, centripetal, centrifugal) and applies
    separate learnable convolutions to each. This is the key difference from the
    broken implementation in exp_gcn_mcfs.py which had NO learnable spatial parameters.
    """

    def __init__(self, in_ch, out_ch, A_self, A_cent, A_centrif):
        super().__init__()
        self.register_buffer("A_self", A_self)
        self.register_buffer("A_cent", A_cent)
        self.register_buffer("A_centrif", A_centrif)
        # Separate learnable weights per partition
        self.w_self = nn.Parameter(torch.zeros(1, out_ch, in_ch))
        self.w_cent = nn.Parameter(torch.zeros(1, out_ch, in_ch))
        self.w_centrif = nn.Parameter(torch.zeros(1, out_ch, in_ch))
        # Xavier init
        nn.init.xavier_uniform_(self.w_self)
        nn.init.xavier_uniform_(self.w_cent)
        nn.init.xavier_uniform_(self.w_centrif)
        self.bias = nn.Parameter(torch.zeros(1, out_ch, 1, 1))

    def forward(self, x):
        # x: (B, C_in, T, V)
        B, C_in, T, V = x.shape
        # Aggregate neighbors per partition: (V, V) @ (B, V, C*T) → (B, V, C*T)
        x_vct = x.permute(0, 3, 1, 2).reshape(B, V, C_in * T)

        agg_self = (
            torch.matmul(self.A_self.unsqueeze(0), x_vct).reshape(B, V, C_in, T).permute(0, 2, 3, 1)
        )
        agg_cent = (
            torch.matmul(self.A_cent.unsqueeze(0), x_vct).reshape(B, V, C_in, T).permute(0, 2, 3, 1)
        )
        agg_centrif = (
            torch.matmul(self.A_centrif.unsqueeze(0), x_vct)
            .reshape(B, V, C_in, T)
            .permute(0, 2, 3, 1)
        )

        # Apply learnable weights: (1, C_out, C_in) @ (B, C_in, T, V) → (B, C_out, T, V)
        out = (
            torch.matmul(self.w_self, agg_self.reshape(B, C_in, T * V)).reshape(B, -1, T, V)
            + torch.matmul(self.w_cent, agg_cent.reshape(B, C_in, T * V)).reshape(B, -1, T, V)
            + torch.matmul(self.w_centrif, agg_centrif.reshape(B, C_in, T * V)).reshape(B, -1, T, V)
            + self.bias
        )
        return out


class ProperSTGCNBlock(nn.Module):
    """ST-GCN block with learnable spatial convolution + temporal convolution."""

    def __init__(self, in_ch, out_ch, A_self, A_cent, A_centrif, t_kernel=9):
        super().__init__()
        self.sgc = PartitionedGraphConv(in_ch, out_ch, A_self, A_cent, A_centrif)
        self.tgc = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, kernel_size=(1, t_kernel), padding=(0, t_kernel // 2)),
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


class ProperSTGCN(nn.Module):
    """Proper ST-GCN with learnable partition weights."""

    def __init__(self, A_self, A_cent, A_centrif, num_classes=64, in_ch=2):
        super().__init__()
        self.st1 = ProperSTGCNBlock(in_ch, 64, A_self, A_cent, A_centrif)
        self.st2 = ProperSTGCNBlock(64, 128, A_self, A_cent, A_centrif)
        self.st3 = ProperSTGCNBlock(128, 256, A_self, A_cent, A_centrif)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x, lengths=None):
        x = x.permute(0, 3, 1, 2)  # (B, T, V, C) → (B, C, T, V)
        x = self.st1(x)
        x = self.st2(x)
        x = self.st3(x)
        x = self.pool(x)
        return self.fc(x)


class PoseEncoder(nn.Module):
    """Simple pose encoder for contrastive pre-training.

    Takes a single frame pose (V, 2) and maps to embedding space.
    Used for contrastive learning: same pose from different augmentations
    should map close together.
    """

    def __init__(self, in_dim=34, embed_dim=128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, embed_dim),
        )

    def forward(self, x):
        return self.encoder(x)


class ContrastivePretrainer(nn.Module):
    """Contrastive pre-training for pose encoder (simplified VIFSS approach).

    Uses InfoNCE loss: positive pairs = same pose with different augmentations,
    negatives = other poses in the batch.
    """

    def __init__(self, pose_encoder, embed_dim=128, temperature=0.1):
        super().__init__()
        self.encoder = pose_encoder
        self.temperature = temperature
        # Projection head
        self.projector = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.ReLU(),
            nn.Linear(64, embed_dim),
        )

    def forward(self, x1, x2):
        """x1, x2: (B, V*C) — augmented versions of same poses."""
        z1 = self.projector(self.encoder(x1))
        z2 = self.projector(self.encoder(x2))
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)
        # InfoNCE
        B = z1.shape[0]
        sim = torch.cat(
            [
                torch.exp(torch.sum(z1 * z2, dim=1) / self.temperature).unsqueeze(1),  # (B, 1)
                torch.mm(z1, z2.t()) / self.temperature,  # (B, B)
            ],
            dim=1,
        )  # (B, B+1) — column 0 is the positive
        # Mask out self-similarity in the negative block
        mask = ~torch.eye(B, dtype=torch.bool, device=z1.device)
        sim[:, 1:][~mask] = -1e9
        labels = torch.zeros(B, dtype=torch.long, device=z1.device)
        return F.cross_entropy(sim, labels)


# ─── Dataset ──────────────────────────────────────────────────────────────


class VarLenDataset(Dataset):
    def __init__(self, poses, labels):
        self.poses, self.labels = poses, labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return torch.tensor(self.poses[idx]), self.labels[idx]


class ContrastivePoseDataset(Dataset):
    """Dataset for contrastive pre-training. Returns pairs of augmented poses."""

    def __init__(self, poses, aug_noise=0.02, aug_mask_prob=0.05, aug_scale_range=(0.8, 1.2)):
        self.poses = poses
        self.aug_noise = aug_noise
        self.aug_mask_prob = aug_mask_prob
        self.aug_scale_range = aug_scale_range

    def __len__(self):
        return len(self.poses)

    def __getitem__(self, idx):
        pose = self.poses[idx]  # (T, V, 2)
        # Sample a random frame
        t = random.randint(0, len(pose) - 1)
        frame = pose[t].flatten()  # (V*2,)

        aug1 = self._augment(frame)
        aug2 = self._augment(frame)
        return aug1, aug2

    def _augment(self, frame):
        """Apply random augmentations to a pose frame."""
        frame = frame.copy()  # numpy array
        # Gaussian noise
        frame += np.random.randn(len(frame)) * self.aug_noise
        # Random masking
        mask = np.random.rand(len(frame)) < self.aug_mask_prob
        frame[mask] = 0.0
        # Random scale
        scale = random.uniform(*self.aug_scale_range)
        frame *= scale
        return torch.tensor(frame, dtype=torch.float32)


def gru_collate(batch):
    batch.sort(key=lambda x: x[0].shape[0], reverse=True)
    seqs, labels = zip(*batch)
    lengths = torch.tensor([s.shape[0] for s in seqs])
    padded = nn.utils.rnn.pad_sequence(seqs, batch_first=True)
    B, T, V, C = padded.shape
    padded = padded.reshape(B, T, V * C)
    return padded, lengths, torch.tensor(labels)


def gcn_collate(batch, max_len=300):
    batch.sort(key=lambda x: x[0].shape[0], reverse=True)
    seqs, labels = zip(*batch)
    seqs = [s[:max_len] for s in seqs]
    lengths = torch.tensor([s.shape[0] for s in seqs])
    padded = nn.utils.rnn.pad_sequence(seqs, batch_first=True)
    return padded, lengths, torch.tensor(labels)


def contrastive_collate(batch):
    augs1, augs2 = zip(*batch)
    return torch.stack(augs1), torch.stack(augs2)


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


def pretrain_contrastive(encoder, train_poses, device, epochs=30, lr=1e-3, batch=256):
    """Pre-train pose encoder with contrastive learning."""
    print("  Pre-training pose encoder with contrastive learning...")
    model = ContrastivePretrainer(encoder, embed_dim=128).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    dataset = ContrastivePoseDataset(train_poses)
    loader = DataLoader(dataset, batch, True, num_workers=2, collate_fn=contrastive_collate)

    for ep in range(epochs):
        model.train()
        total_loss, n = 0.0, 0
        for x1, x2 in loader:
            x1, x2 = x1.to(device), x2.to(device)
            opt.zero_grad()
            loss = model(x1, x2)
            loss.backward()
            opt.step()
            total_loss += loss.item() * len(x1)
            n += len(x1)
        sched.step()
        if (ep + 1) % 10 == 0 or ep == 0:
            print(f"    Ep {ep + 1:3d}: loss={total_loss / n:.4f}")

    print("  Contrastive pre-training done.")
    return model.encoder


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


def make_loaders(tr_p, tr_l, va_p, va_l, te_p, te_l, batch=64, collate_fn=gru_collate):
    return (
        DataLoader(VarLenDataset(tr_p, tr_l), batch, True, collate_fn=collate_fn),
        DataLoader(VarLenDataset(va_p, va_l), batch, False, collate_fn=collate_fn),
        DataLoader(VarLenDataset(te_p, te_l), batch, False, collate_fn=collate_fn),
    )


def main():
    device = torch.device("cuda")
    print(f"Device: {device}\n")

    # Build partitioned adjacency
    A_self, A_cent, A_centrif = build_partitioned_adjacency(17)
    A_self, A_cent, A_centrif = A_self.to(device), A_cent.to(device), A_centrif.to(device)

    # ═══════════════════════════════════════════════════════════════
    # Exp 4d: Proper ST-GCN vs BiGRU on FSC 64-class (H12)
    # ═══════════════════════════════════════════════════════════════
    print("=" * 60)
    print("EXP 4d: Proper ST-GCN vs BiGRU (H12)")
    print("=" * 60)
    print("H12: Proper ST-GCN (learnable partitions) >= BiGRU + 3pp")
    print("Prediction: GCN >= 70.3% | Falsification: GCN <= 67.3%\n")

    train_p, train_l = load_fsc("train")
    test_p, test_l = load_fsc("test")
    all_labels = sorted(set(train_l + test_l))
    lmap = {l: i for i, l in enumerate(all_labels)}
    train_l = [lmap[l] for l in train_l]
    test_l = [lmap[l] for l in test_l]

    tr_p, tr_l, va_p, va_l = stratified_split(train_p, train_l)
    nc = len(all_labels)
    print(f"FSC: {nc} classes, Train={len(tr_p)}, Val={len(va_p)}, Test={len(test_p)}")

    # BiGRU baseline
    model = BiGRU(nc=nc).to(device)
    t0 = time.time()
    gru_acc = train_eval(
        model, *make_loaders(tr_p, tr_l, va_p, va_l, test_p, test_l), device, label="BiGRU-FSC"
    )
    print(f"  >>> BiGRU: {gru_acc:.1%} ({time.time() - t0:.0f}s)\n")

    # Proper ST-GCN
    model = ProperSTGCN(A_self, A_cent, A_centrif, num_classes=nc).to(device)
    t0 = time.time()
    gcn_acc = train_eval(
        model,
        *make_loaders(
            tr_p,
            tr_l,
            va_p,
            va_l,
            test_p,
            test_l,
            batch=16,
            collate_fn=lambda b: gcn_collate(b, max_len=300),
        ),
        device,
        label="STGCN-Proper-FSC",
    )
    print(f"  >>> Proper ST-GCN: {gcn_acc:.1%} ({time.time() - t0:.0f}s)\n")

    # H12 verdict
    diff = gcn_acc - gru_acc
    print(f"  H12 Analysis: BiGRU={gru_acc:.1%}, ST-GCN={gcn_acc:.1%}, diff={diff:+.1%}pp")
    if diff >= 3.0:
        print("  H12 verdict: SUPPORTED")
    elif diff <= 0.0:
        print("  H12 verdict: REJECTED")
    else:
        print(f"  H12 verdict: INCONCLUSIVE ({diff:+.1f}pp)")

    # ═══════════════════════════════════════════════════════════════
    # Exp 4e: Contrastive Pre-training (H13)
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("EXP 4e: Contrastive Pre-training on Low Data (H13)")
    print("=" * 60)
    print("H13: Contrastive pre-training helps under 10% data regime")
    print("Prediction: +10pp | Falsification: < +3pp\n")

    # Use 10% of training data
    by_class = {}
    for p, l in zip(tr_p, tr_l):
        by_class.setdefault(l, []).append(p)
    tr_p_10, tr_l_10 = [], []
    for cls, samples in by_class.items():
        random.shuffle(samples)
        n_10 = max(1, int(len(samples) * 0.1))
        tr_p_10.extend(samples[:n_10])
        tr_l_10.extend([cls] * n_10)
    print(f"10% train: {len(tr_p_10)} samples (from {len(tr_p)})")

    # Without pre-training
    model_no_pre = BiGRU(nc=nc).to(device)
    t0 = time.time()
    acc_no_pre = train_eval(
        model_no_pre,
        *make_loaders(tr_p_10, tr_l_10, va_p, va_l, test_p, test_l),
        device,
        epochs=80,
        label="BiGRU-10%-no-pre",
    )
    print(f"  >>> BiGRU 10% no pretrain: {acc_no_pre:.1%} ({time.time() - t0:.0f}s)\n")

    # With contrastive pre-training
    # Pre-train on FULL training data, then fine-tune on 10%
    pose_encoder = PoseEncoder(in_dim=34, embed_dim=128)
    pose_encoder = pretrain_contrastive(pose_encoder, tr_p, device, epochs=30)

    # Build BiGRU with pretrained encoder
    class PretrainedBiGRU(nn.Module):
        def __init__(self, encoder, hidden=128, layers=2, nc=64, drop=0.3):
            super().__init__()
            self.encoder = encoder
            embed_dim = 128
            self.gru = nn.GRU(
                embed_dim, hidden, layers, batch_first=True, bidirectional=True, dropout=drop
            )
            self.fc = nn.Sequential(
                nn.Linear(hidden * 2, 128), nn.ReLU(), nn.Dropout(drop), nn.Linear(128, nc)
            )

        def forward(self, x, lengths):
            # x: (B, T, 34) — pass each frame through encoder
            B, T, C = x.shape
            x = x.reshape(B * T, C)
            x = self.encoder(x)  # (B*T, 128)
            x = x.reshape(B, T, -1)  # (B, T, 128)
            packed = nn.utils.rnn.pack_padded_sequence(
                x, lengths.cpu(), batch_first=True, enforce_sorted=True
            )
            _, h = self.gru(packed)
            return self.fc(torch.cat([h[-2], h[-1]], dim=1))

    model_pre = PretrainedBiGRU(pose_encoder, nc=nc).to(device)
    t0 = time.time()
    acc_pre = train_eval(
        model_pre,
        *make_loaders(tr_p_10, tr_l_10, va_p, va_l, test_p, test_l),
        device,
        epochs=80,
        label="BiGRU-10%-pretrained",
    )
    print(f"  >>> BiGRU 10% pretrained: {acc_pre:.1%} ({time.time() - t0:.0f}s)\n")

    boost = acc_pre - acc_no_pre
    print(
        f"  H13 Analysis: no-pre={acc_no_pre:.1%}, pretrained={acc_pre:.1%}, boost={boost:+.1%}pp"
    )
    if boost >= 10.0:
        print("  H13 verdict: SUPPORTED")
    elif boost < 3.0:
        print("  H13 verdict: REJECTED")
    else:
        print(f"  H13 verdict: INCONCLUSIVE ({boost:+.1%}pp)")

    # ═══════════════════════════════════════════════════════════════
    # Exp 4f: Transfer Learning FSC → MCFS (H14)
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("EXP 4f: Transfer Learning FSC → MCFS (H14)")
    print("=" * 60)
    print("H14: Pre-training on FSC improves MCFS classification")
    print("Prediction: +5pp | Falsification: < +1pp\n")

    mcfs_p, mcfs_l = load_mcfs_segments(min_samples=10)
    mcfs_labels = sorted(set(mcfs_l))
    mcfs_lmap = {l: i for i, l in enumerate(mcfs_labels)}
    mcfs_l_idx = [mcfs_lmap[l] for l in mcfs_l]
    mcfs_nc = len(mcfs_labels)
    print(f"MCFS: {mcfs_nc} classes, {len(mcfs_p)} segments")

    mtr_p, mtr_l, mva_p, mva_l = stratified_split(mcfs_p, mcfs_l_idx)
    mte_p, mte_l = mcfs_p, mcfs_l_idx

    # MCFS from scratch (BiGRU)
    model_scratch = BiGRU(nc=mcfs_nc).to(device)
    t0 = time.time()
    acc_scratch = train_eval(
        model_scratch,
        *make_loaders(mtr_p, mtr_l, mva_p, mva_l, mte_p, mte_l, batch=32),
        device,
        label="BiGRU-MCFS-scratch",
    )
    print(f"  >>> MCFS from scratch: {acc_scratch:.1%} ({time.time() - t0:.0f}s)\n")

    # MCFS with FSC-pretrained encoder
    # Pre-train on FSC first
    fsc_encoder = PoseEncoder(in_dim=34, embed_dim=128)
    fsc_encoder = pretrain_contrastive(fsc_encoder, tr_p, device, epochs=30)

    model_transfer = PretrainedBiGRU(fsc_encoder, nc=mcfs_nc).to(device)
    t0 = time.time()
    acc_transfer = train_eval(
        model_transfer,
        *make_loaders(mtr_p, mtr_l, mva_p, mva_l, mte_p, mte_l, batch=32),
        device,
        label="BiGRU-MCFS-transfer",
    )
    print(f"  >>> MCFS with FSC transfer: {acc_transfer:.1%} ({time.time() - t0:.0f}s)\n")

    boost = acc_transfer - acc_scratch
    print(
        f"  H14 Analysis: scratch={acc_scratch:.1%}, transfer={acc_transfer:.1%}, boost={boost:+.1%}pp"
    )
    if boost >= 5.0:
        print("  H14 verdict: SUPPORTED")
    elif boost < 1.0:
        print("  H14 verdict: REJECTED")
    else:
        print(f"  H14 verdict: INCONCLUSIVE ({boost:+.1%}pp)")

    # ═══════════════════════════════════════════════════════════════
    # Summary
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"H12 (Proper GCN):  BiGRU={gru_acc:.1%} ST-GCN={gcn_acc:.1%} diff={diff:+.1%}pp")
    print(f"H13 (Contrastive): no-pre={acc_no_pre:.1%} pre={acc_pre:.1%} boost={boost:+.1%}pp")
    print(
        f"H14 (Transfer):    scratch={acc_scratch:.1%} transfer={acc_transfer:.1%} boost={boost:+.1%}pp"
    )


if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    main()
