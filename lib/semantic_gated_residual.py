"""模块 B：用 d1 的语义信息选择性引入原图高频残差。"""

import torch
from torch import nn
from torch.nn import functional as F


class SemanticGatedResidual(nn.Module):
    """输入原图和 d1，输出 [B,K,H,W] 的语义门控残差 logits。"""

    def __init__(self, decoder_channels, num_classes, gate_channels=16):
        super().__init__()
        if min(decoder_channels, num_classes, gate_channels) < 1:
            raise ValueError("all channel counts must be positive")
        self.semantic_gate = nn.Sequential(
            nn.Conv2d(decoder_channels, gate_channels, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(gate_channels, 1, 1),
        )
        self.detail_projection = nn.Conv2d(1, num_classes, 1)
        nn.init.zeros_(self.detail_projection.weight)
        nn.init.zeros_(self.detail_projection.bias)

    def forward(self, image, d1):
        if image.ndim != 4 or image.shape[1] not in (1, 3):
            raise ValueError("image must have shape [B,1/3,H,W]")
        if d1.ndim != 4 or image.shape[0] != d1.shape[0]:
            raise ValueError("d1 must have shape [B,C,H/4,W/4]")
        if image.shape[-2] != 4 * d1.shape[-2] or image.shape[-1] != 4 * d1.shape[-1]:
            raise ValueError("d1 spatial size must be exactly one quarter of input")
        # 门先在 1/4 网格学习，再插值到输入网格，控制高分辨率计算量。
        gate = torch.sigmoid(self.semantic_gate(d1))
        gate = F.interpolate(gate, size=image.shape[-2:], mode="bilinear", align_corners=False)
        intensity = image.mean(dim=1, keepdim=True)
        # 固定局部高通：I - AvgPool3x3(I)。它没有可训练的边界提取器，
        # B 的可学习部分只决定类别投影与语义许可位置。
        local_mean = F.avg_pool2d(F.pad(intensity, (1, 1, 1, 1), mode="replicate"),
                                  3, stride=1)
        high_frequency = intensity - local_mean
        return gate * self.detail_projection(high_frequency)
