"""Data feeder for FSC dataset → InfoGCN format.

Converts FSC pkl data (N, 150, 17, 3) to InfoGCN format (N, C, T, V, M).
Supports joint, bone, and joint_motion modalities.
"""

import pickle

import numpy as np
import torch
from torch.utils.data import Dataset


def compute_joint_angles(data):
    """Compute 11 joint angles as additional channels.

    Appended to xy coordinates: (N, T, V, 2+11, M).
    Angles: L/R knee, L/R elbow, L/R shoulder, trunk, L/R hip, L/R ankle spread.
    """
    N, C, T, V, M = data.shape
    angles = np.zeros((N, T, V, 11, M), dtype=np.float32)

    # Keypoint pairs for angle computation (COCO 17kp indices)
    angle_defs = [
        # (joint, p1, p2) — angle at `joint` between segments to p1 and p2
        (13, 11, 15),  # L knee: hip-knee-ankle
        (14, 12, 16),  # R knee
        (7, 5, 9),  # L elbow: shoulder-elbow-wrist
        (8, 6, 10),  # R elbow
        (5, 7, 11),  # L shoulder: elbow-shoulder-hip
        (6, 8, 12),  # R shoulder
        (0, 5, 6),  # Trunk: Lshoulder-nose-Rshoulder
        (11, 5, 13),  # L hip: shoulder-hip-knee
        (12, 6, 14),  # R hip
        (11, 12, 13),  # L ankle spread: Rhip-Lhip-Lknee
        (12, 11, 14),  # R ankle spread
    ]

    for j_idx, (joint, p1, p2) in enumerate(angle_defs):
        for m in range(M):
            j = data[:, :, joint, :, m]  # (N, T, 2)
            a = data[:, :, p1, :, m] - j  # vector 1
            b = data[:, :, p2, :, m] - j  # vector 2
            cos_angle = np.sum(a * b, axis=-1) / (
                np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1) + 1e-8
            )
            angles[:, :, j_idx, m] = np.arccos(np.clip(cos_angle, -1, 1))

    return angles


class FSCFeeder(Dataset):
    """FSC dataset loader for InfoGCN.

    Input format: pickle files with (N, 150, 17, 3) arrays.
    Output format: (N, C, T, V, M) where M=1 (single person).

    Modalities:
        'joint' — raw (x, y) coordinates [C=2]
        'bone'  — bone vectors (joint[i] - joint[parent]) [C=2]
        'motion' — frame difference joint[t] - joint[t-1] [C=2]
        'joint_angle' — xy + 11 angles [C=13]
    """

    def __init__(
        self,
        data_path: str,
        split: str = "train",
        modality: str = "joint",
        window_size: int = 64,
        p_interval=(0.5, 1.0),
        random_rot: bool = False,
        normalize: bool = True,
    ):
        self.modality = modality
        self.window_size = window_size
        self.p_interval = p_interval
        self.random_rot = random_rot

        # Load data
        data = pickle.load(open(data_path / f"{split}_data.pkl", "rb"))
        labels = pickle.load(open(data_path / f"{split}_label.pkl", "rb"))

        # Filter empty sequences
        labels = np.array(labels)
        valid = np.array([d.shape[0] > 0 for d in data])
        self.data = [data[i] for i in range(len(data)) if valid[i]]
        self.label = labels[valid]

        # Normalize: root-center + spine-length
        if normalize:
            self.data = [self._normalize(d) for d in self.data]

        # Build modality-specific data
        self.samples = []
        for i in range(len(self.data)):
            d = self.data[i]  # (T, 17, 3, 1) or (T, 17, 3)
            if d.ndim == 4:
                d = d.squeeze(-1)  # → (T, 17, 3)
            xy = d[:, :, :2]  # (T, 17, 2) — x, y only

            if modality == "joint":
                feat = xy
            elif modality == "bone":
                feat = self._get_bones(xy)
            elif modality == "motion":
                feat = self._get_motion(xy)
            elif modality == "joint_angle":
                angles = self._get_angles(xy)
                feat = np.concatenate([xy, angles], axis=-1)  # (T, 17, 13)
            else:
                raise ValueError(f"Unknown modality: {modality}")

            # Reshape to (C, T, V, M=1)
            # feat is (T, V, C) → transpose to (C, T, V) → add M dim
            feat = feat.transpose(2, 0, 1)[..., np.newaxis]  # (C, T, V, 1)
            self.samples.append(feat.astype(np.float32))

        self.n_per_cls = self._compute_n_per_cls()

    @staticmethod
    def _normalize(data):
        """Root-center + spine-length normalization."""
        d = data.copy().astype(np.float32)
        if d.ndim == 4:
            d = d.squeeze(-1)  # (T, 17, 3, 1) → (T, 17, 3)
        mid_hip = (d[:, 11:12, :2] + d[:, 12:13, :2]) / 2  # (T, 1, 2)
        d[:, :, :2] -= mid_hip
        mid_shoulder = (d[:, 5:6, :2] + d[:, 6:7, :2]) / 2  # (T, 1, 2)
        spine_length = np.linalg.norm(mid_shoulder - mid_hip, axis=-1, keepdims=True)
        spine_length = np.maximum(spine_length, 0.01)
        d[:, :, :2] /= spine_length
        return d

    @staticmethod
    def _get_bones(xy):
        """Compute bone vectors: joint - parent for each edge."""
        T, V, C = xy.shape
        # Parent mapping (COCO 17kp): -1 = no parent
        parent = [-1, 0, 0, 1, 2, 0, 0, 5, 6, 7, 8, 5, 6, 11, 12, 13, 14]
        bones = np.zeros_like(xy)
        for j in range(V):
            if parent[j] >= 0:
                bones[:, j, :] = xy[:, j, :] - xy[:, parent[j], :]
        return bones

    @staticmethod
    def _get_motion(xy):
        """Frame difference: joint[t] - joint[t-1]."""
        motion = np.zeros_like(xy)
        motion[1:] = xy[1:] - xy[:-1]
        return motion

    @staticmethod
    def _get_angles(xy):
        """Compute 11 joint angles."""
        T, V, C = xy.shape
        angle_defs = [
            (13, 11, 15),
            (14, 12, 16),
            (7, 5, 9),
            (8, 6, 10),
            (5, 7, 11),
            (6, 8, 12),
            (0, 5, 6),
            (11, 5, 13),
            (12, 6, 14),
            (11, 12, 13),
            (12, 11, 14),
        ]
        angles = np.zeros((T, V, 11), dtype=np.float32)
        for a_idx, (joint, p1, p2) in enumerate(angle_defs):
            j = xy[:, joint, :]
            a = xy[:, p1, :] - j
            b = xy[:, p2, :] - j
            cos_a = np.sum(a * b, axis=-1) / (
                np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1) + 1e-8
            )
            angles[:, joint, a_idx] = np.arccos(np.clip(cos_a, -1, 1))
        return angles

    def _compute_n_per_cls(self):
        labels = np.array(self.label)
        return np.array([np.sum(labels == c) for c in range(max(labels) + 1)], dtype=int)

    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        data = self.samples[index]  # (C, T, V, 1)
        label = self.label[index]
        C, T, V, M = data.shape

        # Random crop (only during training with p_interval of length 2)
        if self.p_interval and len(self.p_interval) == 2 and self.window_size < T:
            crop_len = np.random.randint(
                max(int(self.p_interval[0] * T), self.window_size),
                min(int(self.p_interval[1] * T), T) + 1,
            )
            start = np.random.randint(0, T - crop_len + 1)
            data = data[:, start : start + crop_len, :, :]
            T = crop_len

        # Always resize to window_size via bilinear interpolation
        if self.window_size != T:
            data_t = torch.from_numpy(data)
            data_t = data_t.permute(0, 2, 3, 1).reshape(C * V * M, T)
            data_t = data_t[None, None, :, :]
            data_t = torch.nn.functional.interpolate(
                data_t,
                size=(C * V * M, self.window_size),
                mode="bilinear",
                align_corners=False,
            ).squeeze()
            data_t = data_t.reshape(C, V, M, self.window_size).permute(0, 3, 1, 2)
            data = data_t.numpy()

        # Random rotation augmentation
        if self.random_rot:
            theta = np.random.uniform(-0.3, 0.3)
            cos_t, sin_t = np.cos(theta), np.sin(theta)
            rot = np.array([[cos_t, -sin_t], [sin_t, cos_t]], dtype=np.float32)
            data_flat = data[:, :, :, 0].reshape(2, -1).T  # (T*V, 2)
            data_flat = (rot @ data_flat.T).T
            data[:, :, :, 0] = data_flat.reshape(2, self.window_size, V)

        return torch.from_numpy(data), label, index
