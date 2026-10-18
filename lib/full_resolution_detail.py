"""模块 A：从原图提取全分辨率细节，生成最终预测的残差 logits。"""

import torch
from torch import nn


class FullResolutionDetailBranch(nn.Module):
    """输入 [B,1/3,H,W]，输出 [B,K,H,W]；不改变 EMCAD 主干。"""

    def __init__(self, num_classes, hidden_channels=8):
        super().__init__()
        if num_classes < 1 or hidden_channels < 1:
            raise ValueError("num_classes and hidden_channels must be positive")
        self.stem = nn.Sequential(
            nn.Conv2d(1, hidden_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1,
                      groups=hidden_channels, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
        )
        self.to_logits = nn.Conv2d(hidden_channels, num_classes, 1)
        # 初始预测严格等于基础 p1；训练再学习细节修正。
        nn.init.zeros_(self.to_logits.weight)
        nn.init.zeros_(self.to_logits.bias)

    def forward(self, image):
        if image.ndim != 4 or image.shape[1] not in (1, 3):
            raise ValueError("detail branch expects [B,1/3,H,W]")
        # 灰度与 RGB 共用同一支路，避免为不同数据集维护两套参数。
        intensity = image.mean(dim=1, keepdim=True)
        return self.to_logits(self.stem(intensity))
