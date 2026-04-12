"""Loss functions for InfoGCN: MMD bottleneck loss, label smoothing CE, BalancedSampler."""

from collections.abc import Sized

import numpy as np
import torch
import torch.nn.functional as F
from torch import linalg as LA
from torch import nn
from torch.utils.data import Sampler


def get_mmd_loss(z, z_prior, y, num_cls):
    """MMD loss: align per-class latent means with learnable z_prior.

    This is the Information Bottleneck objective — forces compact,
    class-separated latent representations.
    """
    y_valid = [i_cls in y for i_cls in range(num_cls)]
    z_mean = torch.stack([z[y == i_cls].mean(dim=0) for i_cls in range(num_cls)], dim=0)
    l2_z_mean = LA.norm(z.mean(dim=0), ord=2)
    mmd_loss = F.mse_loss(z_mean[y_valid], z_prior[y_valid].to(z.device))
    return mmd_loss, l2_z_mean, z_mean[y_valid]


class LabelSmoothingCE(nn.Module):
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing

    def forward(self, x, target):
        confidence = 1.0 - self.smoothing
        logprobs = F.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1)).squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        return (confidence * nll_loss + self.smoothing * smooth_loss).mean()


class BalancedSampler(Sampler[int]):
    """Class-balanced sampler: repeats minority class samples to match majority.

    Unlike Class-Balanced Loss (which reweights the loss and can underfit),
    this keeps the loss landscape intact and just balances the batches.
    """

    def __init__(self, data_source: Sized, num_classes: int):
        self.data_source = data_source
        self.num_classes = num_classes
        self.n = len(data_source)
        # Count per-class samples
        labels = np.array(data_source.label)
        self.n_per_cls = np.array([np.sum(labels == c) for c in range(num_classes)], dtype=int)
        self.n_cls_wise_desired = max(self.n // num_classes, 1)
        self.n_repeat = np.ceil(self.n_cls_wise_desired / np.maximum(self.n_per_cls, 1)).astype(int)
        self.n_samples = self.n_cls_wise_desired * num_classes
        # Precompute class start indices (sorted by label)
        sorted_idx = labels.argsort()
        self.cls_idx = []
        for c in range(num_classes):
            mask = sorted_idx[labels[sorted_idx] == c]
            self.cls_idx.append(torch.from_numpy(mask.copy()))

    def __iter__(self):
        batches = []
        for c in range(self.num_classes):
            idx = self.cls_idx[c]
            n = len(idx)
            if n == 0:
                continue
            repeat = self.n_repeat[c]
            rand = torch.rand(repeat, n)
            brp = rand.argsort(dim=-1).reshape(-1)[: self.n_cls_wise_desired]
            batches.append(idx[brp])
        batch = torch.stack(batches, 0).permute(1, 0).reshape(-1)
        return iter(batch.tolist())

    def __len__(self):
        return self.n_samples
