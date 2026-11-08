# -*- coding: utf-8 -*-
"""Minimal modules used by the diagnostic harness (vendored)."""
import numpy as np
import torch
import torch.nn as nn


class GaRALinear(nn.Module):
    """Frozen base projection + input-conditioned low-rank update."""

    def __init__(self, base, rank, K):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False
        self.in_features = base.in_features
        self.out_features = base.out_features
        self.L = nn.Parameter(torch.zeros(self.out_features, rank))
        self.R = nn.Parameter(torch.randn(self.in_features, rank) * 0.02)
        self.G = nn.Parameter(torch.randn(K, self.out_features, rank) * 0.02)
        self.alpha = None

    def set_alpha(self, alpha):
        self.alpha = alpha

    def forward(self, x):
        y = self.base(x)
        if self.alpha is None:
            return y
        dt = x.dtype
        R, L, G = self.R.to(dt), self.L.to(dt), self.G.to(dt)
        a = self.alpha.to(dt)
        xR = x @ R
        Geff = torch.einsum("bk,kpr->bpr", a, G)
        W = L.unsqueeze(0) + Geff
        return y + torch.einsum("bsr,brp->bsp", xR, W.transpose(1, 2))


def batches(idx, bs, g):
    if g is None:
        for i in range(0, len(idx), bs):
            yield idx[i:i + bs]
        return
    idx = idx[torch.randperm(len(idx), generator=g)]
    for i in range(0, len(idx), bs):
        yield idx[i:i + bs]


def normalize_adj(n, edges, device):
    """Symmetric normalised adjacency with self-loops."""
    row = np.concatenate([edges[:, 0], edges[:, 1], np.arange(n)])
    col = np.concatenate([edges[:, 1], edges[:, 0], np.arange(n)])
    idx = torch.tensor(np.stack([row, col]), dtype=torch.long, device=device)
    deg = torch.zeros(n, device=device).scatter_add_(
        0, idx[0], torch.ones(idx.shape[1], device=device))
    dinv = deg.pow(-0.5)
    dinv[torch.isinf(dinv)] = 0
    return torch.sparse_coo_tensor(idx, dinv[idx[0]] * dinv[idx[1]],
                                   (n, n), device=device).coalesce()
