"""InfoGCN building blocks: SelfAttention, SA_GC (attention graph conv), MS-TCN, EncodingBlock.

Adapted from stnoah1/infogcn (CVPR 2022). Key change: removed einops dependency,
replaced with plain PyTorch permute/reshape operations.
"""

import math

import numpy as np
import torch
from torch import nn


def conv_init(conv):
    if conv.weight is not None:
        nn.init.kaiming_normal_(conv.weight, mode="fan_out")
    if conv.bias is not None:
        nn.init.constant_(conv.bias, 0)


def bn_init(bn, scale):
    nn.init.constant_(bn.weight, scale)
    nn.init.constant_(bn.bias, 0)


def conv_branch_init(conv, branches):
    weight = conv.weight
    n, k1, k2 = weight.size(0), weight.size(1), weight.size(2)
    nn.init.normal_(weight, 0, math.sqrt(2.0 / (n * k1 * k2 * branches)))
    if conv.bias is not None:
        nn.init.constant_(conv.bias, 0)


class SelfAttention(nn.Module):
    """Multi-head self-attention over joints (V dimension).

    Computes Q, K projections per head, then scaled dot-product attention.
    Returns attention weights (N*T, num_heads, V, V) for use in SA_GC.
    """

    def __init__(self, in_channels, hidden_dim, n_heads):
        super().__init__()
        self.scale = hidden_dim**-0.5
        self.hidden_dim = hidden_dim
        inner_dim = hidden_dim * n_heads
        self.to_qk = nn.Linear(in_channels, inner_dim * 2)
        self.n_heads = n_heads
        self.ln = nn.LayerNorm(in_channels)
        nn.init.normal_(self.to_qk.weight, 0, 1)

    def forward(self, x):
        # x: (N, C, T, V)
        N, C, T, V = x.size()
        # (N, T, V, C) → LayerNorm → QK projection
        y = x.permute(0, 2, 3, 1).contiguous()  # (N, T, V, C)
        y = self.ln(y)
        y = self.to_qk(y)  # (N, T, V, inner_dim*2)
        q, k = y.chunk(2, dim=-1)  # each (N, T, V, inner_dim)

        # Reshape for multi-head: (N*T, heads, V, hidden_dim)
        B = N * T
        hd = self.hidden_dim  # per-head dimension
        q = q.reshape(B, V, self.n_heads, hd).permute(0, 2, 1, 3)  # (B, H, V, hd)
        k = k.reshape(B, V, self.n_heads, hd).permute(0, 2, 1, 3)

        # Scaled dot-product attention
        dots = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # (B, H, V, V)
        attn = dots.softmax(dim=-1).float()
        return attn  # (N*T, H, V, V)


class SA_GC(nn.Module):
    """Self-Attention Graph Convolution.

    Combines learnable shared topology with dynamic attention weights:
        A_effective = attention * shared_topology
    Then per-head graph convolution: A_h @ features → 1x1 conv.
    """

    def __init__(self, in_channels, out_channels, A):
        super().__init__()
        self.out_c = out_channels
        self.in_c = in_channels
        self.num_head = A.shape[0]
        self.shared_topology = nn.Parameter(
            torch.from_numpy(A.astype(np.float32)), requires_grad=True
        )

        # Per-head 1x1 convolutions
        self.conv_d = nn.ModuleList()
        for _ in range(self.num_head):
            self.conv_d.append(nn.Conv2d(in_channels, out_channels, 1))

        # Residual connection
        if in_channels != out_channels:
            self.down = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.down = lambda x: x

        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)
        bn_init(self.bn, 1e-6)
        for i in range(self.num_head):
            conv_branch_init(self.conv_d[i], self.num_head)

        # Attention mechanism
        rel_channels = max(in_channels // 8, 8)
        self.attn = SelfAttention(in_channels, rel_channels, self.num_head)

    def forward(self, x, attn=None):
        N, C, T, V = x.size()

        if attn is None:
            attn = self.attn(x)

        # Modulate topology with attention
        A = attn * self.shared_topology.unsqueeze(0)  # (N*T, H, V, V)

        out = None
        for h in range(self.num_head):
            A_h = A[:, h, :, :]  # (N*T, V, V)
            # Graph convolution: A_h @ features
            feature = x.permute(0, 2, 3, 1).reshape(N * T, V, C)  # (N*T, V, C)
            z = torch.bmm(A_h, feature)  # (N*T, V, C)
            z = z.reshape(N, T, V, C).permute(0, 3, 1, 2)  # (N, C, T, V)
            z = self.conv_d[h](z)
            out = z + out if out is not None else z

        out = self.bn(out)
        out += self.down(x)
        out = self.relu(out)
        return out


class UnitTCN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5, stride=1):
        super().__init__()
        pad = int((kernel_size - 1) / 2)
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(kernel_size, 1),
            padding=(pad, 0),
            stride=(stride, 1),
        )
        self.bn = nn.BatchNorm2d(out_channels)
        conv_init(self.conv)
        bn_init(self.bn, 1)

    def forward(self, x):
        return self.bn(self.conv(x))


class MultiScale_TemporalConv(nn.Module):
    """Multi-Scale Temporal Convolution (MS-TCN).

    Parallel branches with different dilations + max pool + 1x1 skip.
    """

    def __init__(
        self, in_channels, out_channels, kernel_size=5, stride=1, dilations=(1, 2), residual=True
    ):
        super().__init__()
        num_branches = len(dilations) + 2
        branch_channels = out_channels // num_branches
        assert out_channels % num_branches == 0, (
            f"out_channels ({out_channels}) must be divisible by {num_branches}"
        )

        self.branches = nn.ModuleList()
        # Dilated conv branches
        for d in dilations:
            self.branches.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, branch_channels, 1),
                    nn.BatchNorm2d(branch_channels),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(
                        branch_channels,
                        branch_channels,
                        kernel_size=(kernel_size, 1),
                        padding=((kernel_size + (kernel_size - 1) * (d - 1) - 1) // 2, 0),
                        stride=(stride, 1),
                        dilation=(d, 1),
                    ),
                )
            )
        # MaxPool branch
        self.branches.append(
            nn.Sequential(
                nn.Conv2d(in_channels, branch_channels, 1),
                nn.BatchNorm2d(branch_channels),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=(3, 1), stride=(stride, 1), padding=(1, 0)),
                nn.BatchNorm2d(branch_channels),
            )
        )
        # 1x1 skip branch
        self.branches.append(
            nn.Sequential(
                nn.Conv2d(in_channels, branch_channels, 1, stride=(stride, 1)),
                nn.BatchNorm2d(branch_channels),
            )
        )

        # Residual
        if not residual:
            self.residual = lambda x: 0
        elif in_channels == out_channels and stride == 1:
            self.residual = lambda x: x
        else:
            self.residual = UnitTCN(in_channels, out_channels, kernel_size=1, stride=stride)

        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        res = self.residual(x)
        out = torch.cat([b(x) for b in self.branches], dim=1)
        return self.act(out + res)


class EncodingBlock(nn.Module):
    """InfoGCN encoding block: SA_GC → MS-TCN + residual."""

    def __init__(self, in_channels, out_channels, A, stride=1, residual=True):
        super().__init__()
        self.agcn = SA_GC(in_channels, out_channels, A)
        self.tcn = MultiScale_TemporalConv(
            out_channels,
            out_channels,
            kernel_size=5,
            stride=stride,
            dilations=[1, 2],
            residual=False,
        )
        self.relu = nn.ReLU(inplace=True)

        if not residual:
            self.residual = lambda x: 0
        elif in_channels == out_channels and stride == 1:
            self.residual = lambda x: x
        else:
            self.residual = UnitTCN(in_channels, out_channels, kernel_size=1, stride=stride)

    def forward(self, x, attn=None):
        return self.relu(self.tcn(self.agcn(x, attn)) + self.residual(x))
