# PyTorch 张量运算；注意力矩阵乘法、linspace 和 reshape 流程依赖它。
import torch
# nn 提供 Linear、Conv2d、LayerNorm、Dropout 以及 ModuleList。
import torch.nn as nn
# F 在当前有效前向中未调用，保留自上游 PVT 实现的导入。
import torch.nn.functional as F
# partial 用来预设 LayerNorm 的 eps 等构造参数。
from functools import partial

# DropPath 实现随机深度；to_2tuple 统一尺寸参数；trunc_normal_ 用于 Transformer 权重初始化。
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
# register_model 把各 PVTv2 变体注册到 timm 模型注册表。
from timm.models.registry import register_model

# math.sqrt 用于卷积 fan-out 初始化。
import math


# PVTv2 的 Mix-FFN：两层全连接之间插入 3x3 depth-wise convolution，引入局部空间信息。
# EMCAD 论文第5页 Sec.3.2 只把 PVTv2 当层级编码器；本类细节源自 PVTv2，而非 EMCAD 解码创新。
class Mlp(nn.Module):
    # 输入和输出通常都是当前 stage 的嵌入维 C，hidden_features=C*mlp_ratio。
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        # 初始化基础模块。
        super().__init__()
        # 调用者不指定输出维时保持 C_out=C_in，便于 Transformer 残差相加。
        out_features = out_features or in_features
        # 未指定隐藏维时退化为不扩张；各 PVT 变体实际会传入 C*mlp_ratio。
        hidden_features = hidden_features or in_features
        # 第一层线性映射按 token 独立执行：C -> C_hidden。
        self.fc1 = nn.Linear(in_features, hidden_features)
        # 在隐藏通道上使用深度卷积；它需要 H、W 才能把 token 还原为空间特征。
        self.dwconv = DWConv(hidden_features)
        # 默认 GELU 激活。
        self.act = act_layer()
        # 第二层线性映射：C_hidden -> C_out。
        self.fc2 = nn.Linear(hidden_features, out_features)
        # Dropout 同时用于激活后和第二层线性后；默认 drop=0，不随机丢弃。
        self.drop = nn.Dropout(drop)

        # 递归初始化当前 MLP 的 Linear、LayerNorm 和 Conv2d 子层。
        self.apply(self._init_weights)

    # PVT 编码器专用初始化函数。
    def _init_weights(self, m):
        # Linear 权重使用标准差 0.02 的截断正态分布。
        if isinstance(m, nn.Linear):
            # 初始化线性层权重。
            trunc_normal_(m.weight, std=.02)
            # 线性层存在偏置时将其清零；内层 isinstance 判断是原实现的冗余保护。
            if isinstance(m, nn.Linear) and m.bias is not None:
                # 清零偏置。
                nn.init.constant_(m.bias, 0)
        # LayerNorm 按恒等仿射变换初始化。
        elif isinstance(m, nn.LayerNorm):
            # beta=0。
            nn.init.constant_(m.bias, 0)
            # gamma=1。
            nn.init.constant_(m.weight, 1.0)
        # Conv2d 按 fan-out 缩放正态分布初始化。
        elif isinstance(m, nn.Conv2d):
            # 计算卷积核面积乘输出通道。
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            # 分组卷积需除以 groups；DWConv 中 groups=channels。
            fan_out //= m.groups
            # 按 sqrt(2/fan_out) 初始化卷积权重。
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            # 若卷积含偏置则清零。
            if m.bias is not None:
                # 清零卷积偏置。
                m.bias.data.zero_()

    # x 输入形状 (B,N,C)，且 N=H*W。
    def forward(self, x, H, W):
        # 对每个 token 扩张通道：(B,N,C)->(B,N,C_hidden)。
        x = self.fc1(x)
        # 暂时恢复为二维特征做 3x3 深度卷积，再变回 token；形状仍为 (B,N,C_hidden)。
        x = self.dwconv(x, H, W)
        # GELU 非线性。
        x = self.act(x)
        # 第一次 dropout。
        x = self.drop(x)
        # 压回输出通道，通常 C_hidden->C。
        x = self.fc2(x)
        # 第二次 dropout。
        x = self.drop(x)
        # 返回与 Block 输入相同的 (B,N,C)，以便残差相加。
        return x


# 多头空间降采样注意力：Q 保留全部 N 个查询，K/V 可按 sr_ratio 降低 token 数以节省计算。
class Attention(nn.Module):
    # dim=C；num_heads=h；每头维度 d=C/h；sr_ratio 控制 K/V 的空间降采样倍数。
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1):
        # 初始化基础模块。
        super().__init__()
        # 多头拆分要求 C 能被 head 数整除。
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        # 保存总嵌入维。
        self.dim = dim
        # 保存注意力头数。
        self.num_heads = num_heads
        # 单头通道 d=C/h。
        head_dim = dim // num_heads
        # 默认缩放因子 1/sqrt(d)，防止 QK 点积随维度增大而数值过大。
        self.scale = qk_scale or head_dim ** -0.5

        # Q 投影保持 C 维。
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        # K 和 V 一次联合投影为 2C，之后再拆成两份。
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        # 注意力权重 dropout。
        self.attn_drop = nn.Dropout(attn_drop)
        # 多头结果拼接后的输出投影。
        self.proj = nn.Linear(dim, dim)
        # 输出投影后的 dropout。
        self.proj_drop = nn.Dropout(proj_drop)

        # 保存空间降采样比例；B2 四阶段默认 [8,4,2,1]。
        self.sr_ratio = sr_ratio
        # sr_ratio>1 时只压缩 K/V 分支，Q 仍覆盖每个原始位置。
        if sr_ratio > 1:
            # kernel=stride=sr_ratio 的卷积把 (H,W) 降到约 (H/sr,W/sr)，通道保持 C。
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            # 降采样后的 K/V token 在投影前做 LayerNorm。
            self.norm = nn.LayerNorm(dim)

        # 初始化本注意力模块及其子层。
        self.apply(self._init_weights)

    # 与 MLP 相同风格的初始化函数。
    def _init_weights(self, m):
        # Linear 使用截断正态。
        if isinstance(m, nn.Linear):
            # 初始化权重。
            trunc_normal_(m.weight, std=.02)
            # 存在偏置时清零。
            if isinstance(m, nn.Linear) and m.bias is not None:
                # 清零线性偏置。
                nn.init.constant_(m.bias, 0)
        # LayerNorm 初始化为恒等仿射。
        elif isinstance(m, nn.LayerNorm):
            # beta=0。
            nn.init.constant_(m.bias, 0)
            # gamma=1。
            nn.init.constant_(m.weight, 1.0)
        # Conv2d 使用 fan-out 初始化。
        elif isinstance(m, nn.Conv2d):
            # 计算 fan-out。
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            # 考虑分组数。
            fan_out //= m.groups
            # 初始化卷积权重。
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            # 检查可选偏置。
            if m.bias is not None:
                # 清零偏置。
                m.bias.data.zero_()

    # 输入 x=(B,N,C)，N=H*W；输出保持相同形状。
    def forward(self, x, H, W):
        # 提取 batch、token 数和通道数。
        B, N, C = x.shape
        # Q: (B,N,C)->(B,N,h,d)->(B,h,N,d)。
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        # 前三个 stage 通常 sr_ratio>1，通过减少 K/V token 控制注意力复杂度。
        if self.sr_ratio > 1:
            # token 恢复为图像布局：(B,N,C)->(B,C,H,W)。
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
            # 空间降采样并重新展平为 N' 个 token：(B,C,H',W')->(B,N',C)。
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
            # 对降采样 token 做 LayerNorm。
            x_ = self.norm(x_)
            # 联合生成 K/V，并整理为 (2,B,h,N',d)。
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        # 最后一阶段 sr_ratio=1，不压缩空间，K/V 直接来自原 N 个 token。
        else:
            # 整理为 (2,B,h,N,d)。
            kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        # 沿首维拆出 K 和 V；形状分别为 (B,h,N',d)。
        k, v = kv[0], kv[1]

        # Q 与 K^T 相乘得到 (B,h,N,N') 注意力 logits，并乘缩放因子。
        attn = (q @ k.transpose(-2, -1)) * self.scale
        # 对每个查询位置沿 K/V token 维归一化为概率。
        attn = attn.softmax(dim=-1)
        # 对注意力概率执行 dropout。
        attn = self.attn_drop(attn)

        # 权重与 V 相乘得 (B,h,N,d)，再拼回 (B,N,C)。
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        # 输出线性投影混合各头信息。
        x = self.proj(x)
        # 输出 dropout。
        x = self.proj_drop(x)

        # 返回 (B,N,C)。
        return x


# 单个 PVTv2 Transformer Block：Pre-Norm 注意力残差 + Pre-Norm Mix-FFN 残差。
class Block(nn.Module):

    # drop_path 为该 block 的随机深度概率，越深的 block 通常概率越大。
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1):
        # 初始化基础模块。
        super().__init__()
        # 注意力前的 LayerNorm。
        self.norm1 = norm_layer(dim)
        # 构造空间降采样多头注意力。
        self.attn = Attention(
            # 传入当前 stage 通道维及注意力配置。
            dim,
            # qk_scale 可覆盖默认 1/sqrt(d)，drop 和 sr_ratio 控制正则与 K/V 长度。
            num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        # 概率大于 0 时使用 DropPath，否则用 Identity 保持完整分支。
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        # Mix-FFN 前的第二个 LayerNorm。
        self.norm2 = norm_layer(dim)
        # 隐藏通道 C_hidden=int(C*mlp_ratio)。
        mlp_hidden_dim = int(dim * mlp_ratio)
        # 构造带 DWConv 的 Mix-FFN，输出仍为 C。
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        # 初始化本 block；嵌套模块也曾各自初始化，最终参数值以最后一次递归初始化为准。
        self.apply(self._init_weights)

    # Block 的初始化规则。
    def _init_weights(self, m):
        # Linear 截断正态初始化。
        if isinstance(m, nn.Linear):
            # 初始化权重。
            trunc_normal_(m.weight, std=.02)
            # 可选偏置清零。
            if isinstance(m, nn.Linear) and m.bias is not None:
                # 清零偏置。
                nn.init.constant_(m.bias, 0)
        # LayerNorm 恒等仿射初始化。
        elif isinstance(m, nn.LayerNorm):
            # beta=0。
            nn.init.constant_(m.bias, 0)
            # gamma=1。
            nn.init.constant_(m.weight, 1.0)
        # Conv2d fan-out 初始化。
        elif isinstance(m, nn.Conv2d):
            # 计算 fan-out。
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            # 考虑 groups。
            fan_out //= m.groups
            # 初始化权重。
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            # 检查偏置。
            if m.bias is not None:
                # 清零偏置。
                m.bias.data.zero_()

    # 输入输出都是 (B,N,C)，因此两条残差都可直接逐元素相加。
    def forward(self, x, H, W):
        # 第一条残差：x + DropPath(Attention(LN(x)))。
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        # 第二条残差：x + DropPath(MixFFN(LN(x)))。
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))

        # 返回当前 block 编码后的 token。
        return x


# Overlapping Patch Embedding：用带 padding 的卷积生成互相重叠的 patch 特征，而非无重叠切块。
class OverlapPatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    # stage1 通常 kernel=7,stride=4；stage2-4 通常 kernel=3,stride=2。
    def __init__(self, img_size=224, patch_size=7, stride=4, in_chans=3, embed_dim=768):
        # 初始化基础模块。
        super().__init__()
        # 将整数图像尺寸标准化为 (H,W)。
        img_size = to_2tuple(img_size)
        # 将整数 patch 尺寸标准化为 (kH,kW)。
        patch_size = to_2tuple(patch_size)

        # 保存声明的输入尺寸元数据。
        self.img_size = img_size
        # 保存卷积核尺寸元数据。
        self.patch_size = patch_size
        # 这里按 patch_size 计算的 H/W 是遗留元数据，不等于 stride 卷积的真实输出；有效 forward 会重新读取实际 H、W。
        self.H, self.W = img_size[0] // patch_size[0], img_size[1] // patch_size[1]
        # 同样属于元数据，当前 EMCAD 前向不依赖该 num_patches。
        self.num_patches = self.H * self.W
        # 卷积完成空间降采样和通道投影；padding=k//2 使相邻 patch 感受野重叠。
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                              # 二维对称 padding。
                              padding=(patch_size[0] // 2, patch_size[1] // 2))
        # 卷积结果展平为 token 后，对每个 token 的 embed_dim 通道做 LayerNorm。
        self.norm = nn.LayerNorm(embed_dim)

        # 初始化投影卷积和 LayerNorm。
        self.apply(self._init_weights)

    # Patch embedding 初始化规则。
    def _init_weights(self, m):
        # Linear 分支保留统一规则，当前类本身没有 Linear。
        if isinstance(m, nn.Linear):
            # 初始化线性权重。
            trunc_normal_(m.weight, std=.02)
            # 可选偏置清零。
            if isinstance(m, nn.Linear) and m.bias is not None:
                # 清零偏置。
                nn.init.constant_(m.bias, 0)
        # LayerNorm 初始化。
        elif isinstance(m, nn.LayerNorm):
            # beta=0。
            nn.init.constant_(m.bias, 0)
            # gamma=1。
            nn.init.constant_(m.weight, 1.0)
        # Conv2d 初始化。
        elif isinstance(m, nn.Conv2d):
            # 计算 fan-out。
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            # 考虑 groups。
            fan_out //= m.groups
            # 初始化卷积权重。
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            # 可选偏置清零。
            if m.bias is not None:
                # 清零偏置。
                m.bias.data.zero_()

    # 输入是二维特征 (B,C_in,H_in,W_in)，输出 token、实际 H_out、W_out。
    def forward(self, x):
        # 重叠卷积投影后得到 (B,C_embed,H_out,W_out)。
        x = self.proj(x)
        # 从真实卷积输出读取 H、W，后续 reshape 以它们为准。
        _, _, H, W = x.shape
        # 展平空间并交换维度：(B,C,H,W)->(B,C,N)->(B,N,C)。
        x = x.flatten(2).transpose(1, 2)
        # 对每个 token 做 LayerNorm。
        x = self.norm(x)

        # 返回 token 及其二维布局，供 Attention 和 DWConv 临时恢复空间结构。
        return x, H, W


# 四阶段 PVTv2 主干：每个阶段依次执行重叠 patch embedding、若干 Transformer Block 和 LayerNorm。
# 对 EMCAD 而言，最重要的接口是 forward 返回 [x1,x2,x3,x4] 四张 NCHW 特征图。
class PyramidVisionTransformerImpr(nn.Module):
    # embed_dims、num_heads、mlp_ratios、depths、sr_ratios 都是长度为 4 的逐阶段配置。
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dims=[64, 128, 256, 512],
                 # drop_path_rate 会按所有 block 的全局深度线性递增。
                 num_heads=[1, 2, 4, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=False, qk_scale=None, drop_rate=0.,
                 # norm_layer 默认 LayerNorm；PVT 变体通常通过 partial 把 eps 设为 1e-6。
                 attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                 # sr_ratios 控制四阶段 K/V 空间降采样，最后阶段通常为 1。
                 depths=[3, 4, 6, 3], sr_ratios=[8, 4, 2, 1]):
        # 初始化基础模块。
        super().__init__()
        # 分类类别数是上游分类模型遗留元数据；EMCAD 分割前向不使用分类 head。
        self.num_classes = num_classes
        # 保存四阶段 block 数，reset_drop_path 会按它们遍历。
        self.depths = depths

        # patch_embed
        # stage1：7x7、stride4，把 RGB 输入降到约 1/4 并投影为 embed_dims[0] 通道。
        self.patch_embed1 = OverlapPatchEmbed(img_size=img_size, patch_size=7, stride=4, in_chans=in_chans,
                                              # stage1 的输出嵌入通道。
                                              embed_dim=embed_dims[0])
        # stage2：3x3、stride2，从 x1 的 1/4 分辨率降到 x2 的 1/8。
        self.patch_embed2 = OverlapPatchEmbed(img_size=img_size // 4, patch_size=3, stride=2, in_chans=embed_dims[0],
                                              # 通道 embed_dims[0] -> embed_dims[1]。
                                              embed_dim=embed_dims[1])
        # stage3：再降到 1/16。
        self.patch_embed3 = OverlapPatchEmbed(img_size=img_size // 8, patch_size=3, stride=2, in_chans=embed_dims[1],
                                              # 通道 embed_dims[1] -> embed_dims[2]。
                                              embed_dim=embed_dims[2])
        # stage4：再降到 1/32，得到送入 EMCAD 主路的最深特征 x4。
        self.patch_embed4 = OverlapPatchEmbed(img_size=img_size // 16, patch_size=3, stride=2, in_chans=embed_dims[2],
                                              # 通道 embed_dims[2] -> embed_dims[3]。
                                              embed_dim=embed_dims[3])

        # transformer encoder
        # 为所有 block 生成从 0 到 drop_path_rate 的线性随机深度概率表。
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        # cur 指向当前 stage 在全局 dpr 表中的起始位置。
        cur = 0
        # stage1 包含 depths[0] 个 Block。
        self.block1 = nn.ModuleList([Block(
            # stage1 使用第一组通道、头数、MLP 扩张倍率和空间降采样比例。
            dim=embed_dims[0], num_heads=num_heads[0], mlp_ratio=mlp_ratios[0], qkv_bias=qkv_bias, qk_scale=qk_scale,
            # 每个 block 取得自己的随机深度概率 dpr[cur+i]。
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            # stage1 的 sr_ratio 通常为 8，大幅减少 K/V token。
            sr_ratio=sr_ratios[0])
            # 循环创建 stage1 的全部 block。
            for i in range(depths[0])])
        # stage1 全部 block 后的最终 LayerNorm。
        self.norm1 = norm_layer(embed_dims[0])

        # 将 dpr 游标移动到 stage2 起点。
        cur += depths[0]
        # stage2 Block 列表。
        self.block2 = nn.ModuleList([Block(
            # stage2 通道和注意力头配置。
            dim=embed_dims[1], num_heads=num_heads[1], mlp_ratio=mlp_ratios[1], qkv_bias=qkv_bias, qk_scale=qk_scale,
            # 使用对应的全局随机深度概率。
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            # stage2 sr_ratio 通常为 4。
            sr_ratio=sr_ratios[1])
            # 循环创建 stage2 blocks。
            for i in range(depths[1])])
        # stage2 最终 LayerNorm。
        self.norm2 = norm_layer(embed_dims[1])

        # 将 dpr 游标移动到 stage3 起点。
        cur += depths[1]
        # stage3 Block 列表；B2 的这一阶段有 6 个 block，B3-B5 会明显更深。
        self.block3 = nn.ModuleList([Block(
            # stage3 通道、头数和 MLP 配置。
            dim=embed_dims[2], num_heads=num_heads[2], mlp_ratio=mlp_ratios[2], qkv_bias=qkv_bias, qk_scale=qk_scale,
            # stage3 随机深度概率。
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            # stage3 sr_ratio 通常为 2。
            sr_ratio=sr_ratios[2])
            # 循环创建 stage3 blocks。
            for i in range(depths[2])])
        # stage3 最终 LayerNorm。
        self.norm3 = norm_layer(embed_dims[2])

        # 将 dpr 游标移动到 stage4 起点。
        cur += depths[2]
        # stage4 Block 列表。
        self.block4 = nn.ModuleList([Block(
            # stage4 通道和多头配置。
            dim=embed_dims[3], num_heads=num_heads[3], mlp_ratio=mlp_ratios[3], qkv_bias=qkv_bias, qk_scale=qk_scale,
            # stage4 随机深度概率。
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            # sr_ratio=1 时最后阶段 K/V 不再做空间降采样。
            sr_ratio=sr_ratios[3])
            # 循环创建 stage4 blocks。
            for i in range(depths[3])])
        # stage4 最终 LayerNorm。
        self.norm4 = norm_layer(embed_dims[3])

        # classification head
        # 分类 head 被原工程注释掉，因为 EMCAD 需要四级特征而不是图像分类 logits。
        # self.head = nn.Linear(embed_dims[3], num_classes) if num_classes > 0 else nn.Identity()

        # 对整个四阶段主干再次执行统一初始化。
        self.apply(self._init_weights)

    # 主干级初始化函数，与子模块规则保持一致。
    def _init_weights(self, m):
        # Linear 截断正态初始化。
        if isinstance(m, nn.Linear):
            # 初始化权重。
            trunc_normal_(m.weight, std=.02)
            # 可选偏置清零。
            if isinstance(m, nn.Linear) and m.bias is not None:
                # 清零偏置。
                nn.init.constant_(m.bias, 0)
        # LayerNorm 恒等仿射初始化。
        elif isinstance(m, nn.LayerNorm):
            # beta=0。
            nn.init.constant_(m.bias, 0)
            # gamma=1。
            nn.init.constant_(m.weight, 1.0)
        # Conv2d fan-out 初始化。
        elif isinstance(m, nn.Conv2d):
            # 计算卷积 fan-out。
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            # 分组卷积修正。
            fan_out //= m.groups
            # 初始化卷积权重。
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            # 检查可选偏置。
            if m.bias is not None:
                # 清零卷积偏置。
                m.bias.data.zero_()

    # 上游框架兼容接口；当前函数不会真正加载 pretrained 文件，正式权重加载在 lib/networks.py 完成。
    def init_weights(self, pretrained=None):
        # 只有传入字符串时进入该遗留分支。
        if isinstance(pretrained, str):
            # logger 只是占位变量，当前主路径不读取它。
            logger = 1

    # 原 checkpoint 加载调用已被现有代码注释，因此本方法当前没有加载效果。
    # load_checkpoint(self, pretrained, map_location='cpu', strict=False, logger=logger)

    # 工程遗留的动态随机深度调整接口；当前训练入口没有调用它。
    def reset_drop_path(self, drop_path_rate):
        # 重新生成全部 block 的 drop-path 概率表。
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(self.depths))]
        # 从 stage1 起点开始。
        cur = 0
        # 遍历 stage1 blocks。
        for i in range(self.depths[0]):
            # 直接改写 DropPath.drop_prob；首个 block 若是 Identity，则该遗留接口可能不适用。
            self.block1[i].drop_path.drop_prob = dpr[cur + i]

        # 移动到 stage2 起点。
        cur += self.depths[0]
        # 遍历 stage2 blocks。
        for i in range(self.depths[1]):
            # 更新 stage2 概率。
            self.block2[i].drop_path.drop_prob = dpr[cur + i]

        # 移动到 stage3 起点。
        cur += self.depths[1]
        # 遍历 stage3 blocks。
        for i in range(self.depths[2]):
            # 更新 stage3 概率。
            self.block3[i].drop_path.drop_prob = dpr[cur + i]

        # 移动到 stage4 起点。
        cur += self.depths[2]
        # 遍历 stage4 blocks。
        for i in range(self.depths[3]):
            # 更新 stage4 概率。
            self.block4[i].drop_path.drop_prob = dpr[cur + i]

    # 工程遗留冻结接口；当前训练入口没有调用。
    def freeze_patch_emb(self):
        # 这里只给模块对象设置属性，并未逐个把 Parameter.requires_grad 设为 False，不能视作已真正冻结参数。
        self.patch_embed1.requires_grad = False

    # 告诉某些优化器哪些参数不应 weight decay；当前返回名称来自含位置编码的旧版接口。
    @torch.jit.ignore
    # TorchScript 忽略这个 Python 集合返回方法。
    def no_weight_decay(self):
        # 当前模型并没有启用这些 pos_embed/cls_token 参数，因此它是兼容性遗留信息。
        return {'pos_embed1', 'pos_embed2', 'pos_embed3', 'pos_embed4', 'cls_token'}  # has pos_embed may be better

    # 分类模型兼容接口；EMCAD 路径不调用。
    def get_classifier(self):
        # self.head 在当前构造器中已被注释，直接调用该遗留方法可能找不到属性。
        return self.head

    # 分类 head 重设接口；EMCAD 分割任务不调用。
    def reset_classifier(self, num_classes, global_pool=''):
        # 更新分类类别元数据。
        self.num_classes = num_classes
        # 当前类没有设置 self.embed_dim，且分类 head 原本被禁用，因此该行属于未接入主路径的遗留 API。
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    # def _get_pos_embed(self, pos_embed, patch_embed, H, W):
    #     if H * W == self.patch_embed1.num_patches:
    #         return pos_embed
    #     else:
    #         return F.interpolate(
    #             pos_embed.reshape(1, patch_embed.H, patch_embed.W, -1).permute(0, 3, 1, 2),
    #             size=(H, W), mode="bilinear").reshape(1, -1, H * W).permute(0, 2, 1)

    # 核心四阶段前向；输入 (B,3,H,W)，输出四个从浅到深的 NCHW 特征。
    def forward_features(self, x):
        # 保存 batch 大小，后续把 token 恢复成二维特征时使用。
        B = x.shape[0]
        # 收集 x1、x2、x3、x4。
        outs = []

        # stage 1
        # 重叠 patch embedding：RGB -> embed_dims[0]，空间约降到 1/4；返回 token 和实际 H1、W1。
        x, H, W = self.patch_embed1(x)
        # 依次执行 stage1 的所有 Transformer blocks。
        for i, blk in enumerate(self.block1):
            # 每个 block 保持 token 形状 (B,H1*W1,C1)。
            x = blk(x, H, W)
        # stage1 最终 LayerNorm。
        x = self.norm1(x)
        # token 恢复为 NCHW：(B,N,C1)->(B,H1,W1,C1)->(B,C1,H1,W1)。
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        # 保存 x1，供 EMCAD 最高分辨率 skip 使用。
        outs.append(x)

        # stage 2
        # x1 直接作为二维输入，经过 stride2 patch embedding 得到 stage2 token。
        x, H, W = self.patch_embed2(x)
        # 执行 stage2 blocks。
        for i, blk in enumerate(self.block2):
            # 保持 (B,H2*W2,C2)。
            x = blk(x, H, W)
        # stage2 最终 LayerNorm。
        x = self.norm2(x)
        # 恢复 x2=(B,C2,H2,W2)，空间约为输入 1/8。
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        # 保存 x2，供 EMCAD 第二条 skip 使用。
        outs.append(x)

        # stage 3
        # stride2 patch embedding 得到 stage3 token，空间约为输入 1/16。
        x, H, W = self.patch_embed3(x)
        # 执行 stage3 blocks。
        for i, blk in enumerate(self.block3):
            # B2 在此循环 6 次，B3/B4/B5 更深。
            x = blk(x, H, W)
        # stage3 最终 LayerNorm。
        x = self.norm3(x)
        # 恢复 x3=(B,C3,H3,W3)。
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        # 保存 x3，供 EMCAD 最深的一条 skip 使用。
        outs.append(x)

        # stage 4
        # 最后一次 stride2 patch embedding，空间约为输入 1/32。
        x, H, W = self.patch_embed4(x)
        # 执行 stage4 blocks；此阶段 sr_ratio=1，注意力不压缩 K/V。
        for i, blk in enumerate(self.block4):
            # 保持 (B,H4*W4,C4)。
            x = blk(x, H, W)
        # stage4 最终 LayerNorm。
        x = self.norm4(x)
        # 恢复最深特征 x4=(B,C4,H4,W4)。
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        # 保存 x4，它会进入 EMCAD 解码主路。
        outs.append(x)

        # 返回顺序 [x1,x2,x3,x4]；默认 B2 通道 [64,128,320,512]、尺度 [1/4,1/8,1/16,1/32]。
        return outs

        # return x.mean(dim=1)

    # 对外 forward 仅转调四阶段特征提取，不执行分类池化或分类 head。
    def forward(self, x):
        # 获取四级 NCHW 特征列表。
        x = self.forward_features(x)
        # x = self.head(x)

        # 返回给 EMCADNet 解包为 x1,x2,x3,x4。
        return x


# Mix-FFN 内的深度卷积：在不混合通道的前提下为 token 注入 3x3 局部空间关系。
class DWConv(nn.Module):
    # dim 等于 MLP 隐藏通道数。
    def __init__(self, dim=768):
        # 初始化基础模块。
        super(DWConv, self).__init__()
        # groups=dim 使每个通道单独执行 3x3 卷积，padding=1 保持 H、W。
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    # 输入 token (B,N,C)，N 必须等于 H*W。
    def forward(self, x, H, W):
        # 读取 token 形状。
        B, N, C = x.shape
        # (B,N,C)->(B,C,N)->(B,C,H,W)。
        x = x.transpose(1, 2).view(B, C, H, W)
        # 在每个隐藏通道内提取局部 3x3 上下文。
        x = self.dwconv(x)
        # 恢复 (B,N,C)，供后续 GELU 和 fc2 使用。
        x = x.flatten(2).transpose(1, 2)

        # 返回 token 布局。
        return x


# 旧式 checkpoint 兼容辅助：把手工 patchify 的线性投影权重改成卷积核形状。
def _conv_filter(state_dict, patch_size=16):
    """ convert patch embedding weight from manual patchify + linear proj to conv"""
    # 新建输出字典，避免遍历时原地改写输入映射。
    out_dict = {}
    # 遍历 checkpoint 的全部参数。
    for k, v in state_dict.items():
        # 只处理旧命名中的 patch embedding 投影权重。
        if 'patch_embed.proj.weight' in k:
            # 把二维 Linear 权重还原为 (C_out,3,patch_size,patch_size) 卷积权重。
            v = v.reshape((v.shape[0], 3, patch_size, patch_size))
        # 无论是否转换，都保存到输出字典。
        out_dict[k] = v

    # 返回转换后的 state_dict；当前 EMCAD 权重加载路径没有调用此函数。
    return out_dict


# 注册 PVTv2-B0：tiny 版本，适配论文 PVT-EMCAD-B0。
@register_model
class pvt_v2_b0(PyramidVisionTransformerImpr):
    # kwargs 是上游注册接口的兼容参数；当前构造代码没有继续传给父类。
    def __init__(self, **kwargs):
        # B0 通道 [32,64,160,256]、block 深度 [2,2,2,2]。
        super(pvt_v2_b0, self).__init__(
            # 头数必须整除各阶段通道；前两阶段 MLP 扩张 8 倍。
            patch_size=4, embed_dims=[32, 64, 160, 256], num_heads=[1, 2, 5, 8], mlp_ratios=[8, 8, 4, 4],
            # 四阶段 K/V 空间压缩比例 [8,4,2,1]，最大 drop-path 为 0.1。
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[2, 2, 2, 2], sr_ratios=[8, 4, 2, 1],
            # 普通 dropout 为 0，只使用逐层递增的随机深度。
            drop_rate=0.0, drop_path_rate=0.1)


# 注册 PVTv2-B1：标准通道、较浅深度。
@register_model
class pvt_v2_b1(PyramidVisionTransformerImpr):
    # 当前 kwargs 未透传，模型配置由下方常量固定。
    def __init__(self, **kwargs):
        # B1 通道 [64,128,320,512]，各 stage 两个 block。
        super(pvt_v2_b1, self).__init__(
            # 头数 [1,2,5,8]，MLP ratios [8,8,4,4]。
            patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[8, 8, 4, 4],
            # LayerNorm eps=1e-6，sr ratios 固定。
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[2, 2, 2, 2], sr_ratios=[8, 4, 2, 1],
            # dropout 和 drop-path 配置。
            drop_rate=0.0, drop_path_rate=0.1)


# 注册 PVTv2-B2：本项目和论文 PVT-EMCAD-B2 的默认标准编码器。
@register_model
class pvt_v2_b2(PyramidVisionTransformerImpr):
    # 当前 kwargs 未透传。
    def __init__(self, **kwargs):
        # B2 通道 [64,128,320,512]，深度 [3,4,6,3]。
        super(pvt_v2_b2, self).__init__(
            # 多头和 MLP 配置。
            patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[8, 8, 4, 4],
            # sr ratios 和随机深度配置。
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 4, 6, 3], sr_ratios=[8, 4, 2, 1],
            # 普通 dropout 为 0，最大 drop-path 0.1。
            drop_rate=0.0, drop_path_rate=0.1)


# 注册 PVTv2-B3：主要把第三阶段从 6 个 block 增加到 18 个。
@register_model
class pvt_v2_b3(PyramidVisionTransformerImpr):
    # 当前 kwargs 未透传。
    def __init__(self, **kwargs):
        # B3 深度 [3,4,18,3]，通道接口仍与 B2 相同。
        super(pvt_v2_b3, self).__init__(
            # 多头和 MLP 配置。
            patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[8, 8, 4, 4],
            # sr ratios 保持 [8,4,2,1]。
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 4, 18, 3], sr_ratios=[8, 4, 2, 1],
            # dropout 配置。
            drop_rate=0.0, drop_path_rate=0.1)


# 注册 PVTv2-B4：进一步加深 stage2 和 stage3。
@register_model
class pvt_v2_b4(PyramidVisionTransformerImpr):
    # 当前 kwargs 未透传。
    def __init__(self, **kwargs):
        # B4 深度 [3,8,27,3]，四级输出通道不变。
        super(pvt_v2_b4, self).__init__(
            # 多头和 MLP 配置。
            patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[8, 8, 4, 4],
            # LayerNorm、深度和 sr ratios。
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 8, 27, 3], sr_ratios=[8, 4, 2, 1],
            # dropout 配置。
            drop_rate=0.0, drop_path_rate=0.1)


# 注册 PVTv2-B5：本文件中最深变体，stage3 有 40 个 block。
@register_model
class pvt_v2_b5(PyramidVisionTransformerImpr):
    # 当前 kwargs 未透传。
    def __init__(self, **kwargs):
        # B5 保持标准通道，但把所有 stage 的 MLP ratio 设为 4。
        super(pvt_v2_b5, self).__init__(
            # 通道、头数和 MLP ratios。
            patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4],
            # 深度 [3,6,40,3]，sr ratios 不变。
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 6, 40, 3], sr_ratios=[8, 4, 2, 1],
            # dropout 配置。
            drop_rate=0.0, drop_path_rate=0.1)
