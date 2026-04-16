"""InfoGCN model for skeleton-based action recognition.

Adapted from stnoah1/infogcn (CVPR 2022).
Key changes: no Apex dependency, COCO 17kp support, no einops.
"""

import math

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .graph import Graph
from .modules import EncodingBlock, bn_init


class InfoGCN(nn.Module):
    """InfoGCN: Attention-based GCN with Information Bottleneck.

    Architecture:
        A_vector @ x → Linear embedding + pos_embed → BN
        → 9× EncodingBlock (SA_GC + MS-TCN)
        → Global pool → FC → VAE latent (μ, logvar) → decoder → classes

    The Information Bottleneck forces compact, class-separated representations
    via MMD loss between per-class latent means and learnable z_prior.
    """

    def __init__(
        self,
        num_class=64,
        num_point=17,
        num_person=1,
        graph=None,
        in_channels=2,
        drop_out=0.0,
        num_head=3,
        noise_ratio=0.5,
        k=0,
        gain=3,
    ):
        super().__init__()

        if graph is None:
            graph = Graph()
        elif isinstance(graph, str):
            # Support "coco17" string
            graph = Graph()

        A = np.stack([np.eye(num_point)] * num_head, axis=0)

        base_channel = 64
        self.num_class = num_class
        self.num_point = num_point
        self.noise_ratio = noise_ratio
        self.gain = gain

        # Learnable class prior for IB loss
        self.z_prior = nn.Parameter(torch.empty(num_class, base_channel * 4))
        nn.init.orthogonal_(self.z_prior, gain=gain)

        # Graph topology: I - A_outward^k (with k=0 this is zero — learned from scratch)
        A_outward = graph.A_outward_binary
        I = np.eye(graph.num_node)
        self.A_vector = nn.Parameter(
            torch.from_numpy((I - np.linalg.matrix_power(A_outward, k)).astype(np.float32)),
            requires_grad=True,
        )

        # Input projection + positional embedding
        self.to_joint_embedding = nn.Linear(in_channels, base_channel)
        self.pos_embedding = nn.Parameter(torch.randn(1, num_point, base_channel))

        # BatchNorm over (M*V*C, T)
        self.data_bn = nn.BatchNorm1d(num_person * base_channel * num_point)

        # 9 EncodingBlocks with channel expansion
        self.l1 = EncodingBlock(base_channel, base_channel, A)
        self.l2 = EncodingBlock(base_channel, base_channel, A)
        self.l3 = EncodingBlock(base_channel, base_channel, A)
        self.l4 = EncodingBlock(base_channel, base_channel * 2, A, stride=2)
        self.l5 = EncodingBlock(base_channel * 2, base_channel * 2, A)
        self.l6 = EncodingBlock(base_channel * 2, base_channel * 2, A)
        self.l7 = EncodingBlock(base_channel * 2, base_channel * 4, A, stride=2)
        self.l8 = EncodingBlock(base_channel * 4, base_channel * 4, A)
        self.l9 = EncodingBlock(base_channel * 4, base_channel * 4, A)

        # Latent space
        self.fc = nn.Linear(base_channel * 4, base_channel * 4)
        self.fc_mu = nn.Linear(base_channel * 4, base_channel * 4)
        self.fc_logvar = nn.Linear(base_channel * 4, base_channel * 4)
        self.decoder = nn.Linear(base_channel * 4, num_class)

        # Initialize
        nn.init.xavier_uniform_(self.fc.weight, gain=nn.init.calculate_gain("relu"))
        nn.init.xavier_uniform_(self.fc_mu.weight, gain=nn.init.calculate_gain("relu"))
        nn.init.xavier_uniform_(self.fc_logvar.weight, gain=nn.init.calculate_gain("relu"))
        nn.init.normal_(self.decoder.weight, 0, math.sqrt(2.0 / num_class))
        bn_init(self.data_bn, 1)

        self.drop_out = nn.Dropout(drop_out) if drop_out else lambda x: x

    @staticmethod
    def _bn_init(bn, scale):
        nn.init.constant_(bn.weight, scale)
        nn.init.constant_(bn.bias, 0)

    def latent_sample(self, mu, logvar):
        if self.training:
            std = logvar.mul(self.noise_ratio).exp()
            std = torch.clamp(std, max=100)
            eps = torch.empty_like(std).normal_()
            return eps.mul(std) + mu
        return mu

    def forward(self, x):
        # x: (N, C, T, V, M)
        N, C, T, V, M = x.size()

        # Graph modulation + joint embedding
        x = x.permute(0, 3, 2, 1, 4).reshape(N * M * T, V, C)  # (NM T, V, C)
        x = torch.matmul(self.A_vector.expand(N * M * T, -1, -1), x)
        x = self.to_joint_embedding(x)
        x = x + self.pos_embedding[:, :V]
        x = x.reshape(N, M, T, V, -1).permute(0, 1, 4, 2, 3)  # (N, M, C, T, V)
        x = x.reshape(N, M * V * x.size(2), T)  # (N, M*V*C, T)
        x = self.data_bn(x)
        x = x.reshape(N, M, -1, T, V).permute(0, 1, 2, 3, 4)  # (N, M, C, T, V)
        x = x.reshape(N * M, -1, T, V)  # (NM, C, T, V)

        # 9 EncodingBlocks
        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        x = self.l4(x)
        x = self.l5(x)
        x = self.l6(x)
        x = self.l7(x)
        x = self.l8(x)
        x = self.l9(x)

        # Global pool over T and V → (N, M, C)
        c_new = x.size(1)
        x = x.view(N, M, c_new, -1).mean(3).mean(1)  # (N, C)

        # Latent space
        x = F.relu(self.fc(x))
        x = self.drop_out(x)
        z_mu = self.fc_mu(x)
        z_logvar = self.fc_logvar(x)
        z = self.latent_sample(z_mu, z_logvar)

        # Classification
        y_hat = self.decoder(z)
        return y_hat, z
