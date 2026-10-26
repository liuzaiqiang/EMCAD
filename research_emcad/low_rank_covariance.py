"""Image-specific channel interaction from compact second-order statistics."""

import torch
from torch import nn
from torch.nn import functional as F


class LowRankCovarianceInteraction(nn.Module):
    """Mix correlated and anticorrelated feature channels at a decoder stage."""

    def __init__(self, channels, rank=32, eps=1e-5):
        super().__init__()
        if channels <= 0 or rank <= 0 or eps <= 0:
            raise ValueError("channels, rank and eps must be positive")
        rank = min(channels, rank)
        self.rank = rank
        self.eps = eps

        # A deterministic grouped projection avoids changing the initialization
        # of the existing decoder, output heads, or class scale fusion module.
        reduce = torch.zeros(rank, channels, 1, 1)
        for index in range(rank):
            members = torch.arange(index, channels, rank)
            reduce[index, members, 0, 0] = 1.0 / len(members)
        self.reduce = nn.Parameter(reduce)
        self.expand = nn.Parameter(torch.zeros(channels, rank, 1, 1))

    def forward(self, feature):
        batch, channels, height, width = feature.shape
        compressed = F.conv2d(feature, self.reduce)
        flat = compressed.flatten(2).float()
        centered = flat - flat.mean(dim=-1, keepdim=True)
        covariance = torch.bmm(centered, centered.transpose(1, 2)) / max(height * width - 1, 1)
        inverse_std = torch.rsqrt(covariance.diagonal(dim1=1, dim2=2).clamp_min(0) + self.eps)
        correlation = covariance * inverse_std[:, :, None] * inverse_std[:, None, :]
        correlation = correlation.clamp(-1.0, 1.0)
        mixed = torch.bmm(correlation, centered).reshape(batch, self.rank, height, width)
        mixed = (mixed / self.rank).to(dtype=compressed.dtype)
        correction = F.conv2d(mixed, self.expand)
        return feature + torch.tanh(correction)
