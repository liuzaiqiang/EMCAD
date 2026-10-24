"""从 d3/d2/d1 提取图像条件类别原型，并修正最高分辨率 logits。"""
import torch
from torch import nn
from torch.nn import functional as F


class ClassPrototypeQuery(nn.Module):
    """输入三个解码特征，返回 [B,K,H1,W1] 修正 logits 和 [B,K,D] 原型。"""

    def __init__(self, channels, num_classes, embedding_dim=32):
        super().__init__()
        if len(channels) != 3 or min(channels) <= 0 or num_classes <= 0 or embedding_dim <= 0:
            raise ValueError("channels 必须为三个正整数，类别数与嵌入维度必须为正")
        self.channels = tuple(channels)
        self.num_classes = num_classes
        self.projections = nn.ModuleList(nn.Conv2d(c, embedding_dim, 1, bias=False) for c in channels)
        self.queries = nn.Parameter(torch.empty(num_classes, embedding_dim))
        self.logit_gain = nn.Parameter(torch.tensor(0.01))
        nn.init.normal_(self.queries, std=embedding_dim ** -0.5)

    def forward(self, features):
        if len(features) != 3:
            raise ValueError("需要按 d3,d2,d1 顺序提供三个特征")
        batch = features[0].shape[0]
        for feature, channels in zip(features, self.channels):
            if feature.ndim != 4 or feature.shape[0] != batch or feature.shape[1] != channels:
                raise ValueError("类别原型查询的 batch 或通道数不匹配")
        query = F.normalize(self.queries, dim=-1)
        prototypes = []
        projected = []
        for feature, projection in zip(features, self.projections):
            embedding = projection(feature)
            projected.append(embedding)
            flat = embedding.flatten(2)
            # q_k 与每个空间 token 的相似度，经空间 softmax 汇聚类别原型。
            attention = torch.softmax(torch.einsum('kd,bdn->bkn', query, F.normalize(flat, dim=1)), dim=-1)
            prototypes.append(torch.einsum('bkn,bdn->bkd', attention, flat))
        prototype = F.normalize(torch.stack(prototypes, dim=0).mean(dim=0), dim=-1)
        pixel = F.normalize(projected[-1], dim=1)
        correction = self.logit_gain * torch.einsum('bkd,bdhw->bkhw', prototype, pixel)
        return correction, prototype
