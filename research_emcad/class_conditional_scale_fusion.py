"""逐类别选择 d3/d2/d1 预测感受野，权重由图像特征及类别表示决定。"""
import torch
from torch import nn
from torch.nn import functional as F


class ClassConditionalScaleFusion(nn.Module):
    """输入 d3/d2/d1 特征与 p3/p2/p1 logits，输出同尺寸 p1 logits。"""

    def __init__(self, channels, num_classes, embedding_dim=32):
        super().__init__()
        if len(channels) != 3 or min(channels) <= 0 or num_classes <= 0 or embedding_dim <= 0:
            raise ValueError("channels 必须为三个正整数，类别数与嵌入维度必须为正")
        self.channels = tuple(channels)
        self.num_classes = num_classes
        self.projections = nn.ModuleList(nn.Linear(c, embedding_dim, bias=False) for c in channels)
        self.class_queries = nn.Parameter(torch.empty(num_classes, embedding_dim))
        self.scale_bias = nn.Parameter(torch.tensor([0.0, 0.0, 2.0]))
        self.fusion_gain = nn.Parameter(torch.tensor(0.1))
        nn.init.normal_(self.class_queries, std=embedding_dim ** -0.5)

    def forward(self, features, logits):
        if len(features) != 3 or len(logits) != 3:
            raise ValueError("需要 d3,d2,d1 特征及相应三个 logits")
        batch = features[0].shape[0]
        target_size = logits[-1].shape[-2:]
        for feature, logit, channels in zip(features, logits, self.channels):
            if (feature.ndim != 4 or logit.ndim != 4 or feature.shape[:2] != (batch, channels)
                    or logit.shape[:2] != (batch, self.num_classes)
                    or feature.shape[-2:] != logit.shape[-2:]):
                raise ValueError("尺度融合的特征与 logits 尺寸或通道不匹配")
        query = F.normalize(self.class_queries, dim=-1).unsqueeze(0).expand(batch, -1, -1)
        pooled = [F.normalize(proj(feature.mean(dim=(-2, -1))), dim=-1)
                  for feature, proj in zip(features, self.projections)]
        score = torch.stack([torch.einsum('bkd,bd->bk', query, value) for value in pooled], dim=-1)
        # w_{b,k,s}=softmax_s(<q_{b,k},g_{b,s}>+bias_s)。每个类别独立选择尺度。
        weights = torch.softmax(score + self.scale_bias, dim=-1)
        resized = [F.interpolate(logit, size=target_size, mode='bilinear', align_corners=False)
                   if logit.shape[-2:] != target_size else logit for logit in logits]
        fused = sum(weights[:, :, index, None, None] * item for index, item in enumerate(resized))
        return logits[-1] + self.fusion_gain * (fused - logits[-1])
