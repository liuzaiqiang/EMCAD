"""Per-pixel selection of EMCAD's existing depthwise kernel responses."""

import torch
from torch import nn
from torch.nn import functional as F


class SpatialKernelSelection(nn.Module):
    """Predict a normalized kernel mixture at each location, with an identity start."""

    def __init__(self, channels, num_kernels):
        super().__init__()
        if num_kernels < 2:
            raise ValueError("SpatialKernelSelection requires at least two kernels")
        self.num_kernels = num_kernels
        # Direct zero parameters do not consume the RNG used by the original decoder.
        self.weight = nn.Parameter(torch.zeros(num_kernels, channels, 1, 1))
        self.bias = nn.Parameter(torch.zeros(num_kernels))

    def forward(self, expanded_feature, branches):
        if len(branches) != self.num_kernels:
            raise ValueError("Kernel response count does not match the selector")
        # Pooling supplies local context without introducing another learned kernel.
        context = F.avg_pool2d(expanded_feature, kernel_size=3, stride=1, padding=1)
        probabilities = F.softmax(F.conv2d(context, self.weight, self.bias), dim=1)
        # Mean-one factors preserve the original sum (or concatenation) at initialization.
        return [branch * (self.num_kernels * probabilities[:, i:i + 1])
                for i, branch in enumerate(branches)]
