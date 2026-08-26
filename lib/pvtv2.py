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


# ==================================================================================================
# 【本文件先读这一段：它在 EMCAD 中究竟负责什么】
# ==================================================================================================
# 1. 本文件实现的是 PVTv2（Pyramid Vision Transformer v2）编码器，不负责最终分割预测。
#    它把一张输入图像逐级编码成 4 张不同分辨率、不同通道数的特征图。
# 2. `lib/networks.py` 中的 `EMCADNet` 会实例化这里的 `pvt_v2_b2`，然后调用其 `forward`。
#    本文件最终返回 `[x1, x2, x3, x4]`；`EMCADNet` 再把这些特征交给 `lib/decoders.py` 中的 EMCAD 解码器。
# 3. 因此，阅读时可以把本文件分成三层：
#    - 最小算子层：`DWConv`、`Mlp`、`Attention`；
#    - Transformer 基本块层：`Block`；
#    - 四阶段金字塔层：`OverlapPatchEmbed`、`PyramidVisionTransformerImpr`；
#    - 模型规格层：文件末尾的 `pvt_v2_b0` 到 `pvt_v2_b5`。
# 4. 本文件有两种常见张量布局，必须先区分：
#    - 图像/特征图布局 `(B, C, H, W)`：卷积层使用，C 是通道数；
#    - token 布局 `(B, N, C)`：Transformer 使用，其中 `N = H * W`。
#    `OverlapPatchEmbed` 把 NCHW 变成 token；每个 stage 结束时又把 token 变回 NCHW。
# 5. 默认 PVTv2-B2、输入 `352 x 352` 时，四阶段的真实尺寸是：
#    - x1: `(B,  64, 88, 88)`，token 数 N1=7744，约为原图 1/4；
#    - x2: `(B, 128, 44, 44)`，token 数 N2=1936，约为原图 1/8；
#    - x3: `(B, 320, 22, 22)`，token 数 N3= 484，约为原图 1/16；
#    - x4: `(B, 512, 11, 11)`，token 数 N4= 121，约为原图 1/32。
# 6. 默认 B2 的四阶段 block 数是 `[3, 4, 6, 3]`，共 16 个 Block；注意力头数是 `[1, 2, 5, 8]`。
#    四阶段每个注意力头的维度恰好都是 64：64/1、128/2、320/5、512/8 都等于 64。
# 7. 默认 B2 的 `sr_ratios=[8,4,2,1]` 只压缩 K/V，不压缩 Q。对于 352 输入：
#    - stage1 的 K/V: 88x88 经 sr=8 变成 11x11；
#    - stage2 的 K/V: 44x44 经 sr=4 变成 11x11；
#    - stage3 的 K/V: 22x22 经 sr=2 变成 11x11；
#    - stage4 的 K/V: sr=1，原本就是 11x11。
#    这样前三阶段不必在全部高分辨率 token 之间做完整两两注意力，显著降低显存和计算量。
# 8. 本文件保留了一些上游分类模型的兼容 API，例如 `get_classifier`、`reset_classifier`、
#    `freeze_patch_emb` 和 `_conv_filter`。它们并非 EMCAD 当前训练主路径，部分接口甚至缺少配套属性；
#    阅读时应把“当前分割前向实际会执行的代码”和“历史兼容代码”分开，不要误以为后者已在训练中生效。
# 9. 下文所有形状说明都采用：B=batch size，C=通道/嵌入维度，H/W=空间尺寸，N=H*W，
#    h=注意力头数，d=C/h，N'=空间降采样后的 K/V token 数。


# PVTv2 的 Mix-FFN：两层全连接之间插入 3x3 depth-wise convolution，引入局部空间信息。
# EMCAD 论文第5页 Sec.3.2 只把 PVTv2 当层级编码器；本类细节源自 PVTv2，而非 EMCAD 解码创新。
# 【Mlp 的位置】每个 `Block` 都有一个本类实例，它位于注意力子层之后，是第二条残差分支的主体。
# 【为什么不只用普通 MLP】普通 Transformer MLP 对每个 token 独立运算，不直接感知相邻像素；
# 这里在两层 Linear 之间插入 `DWConv`，先把 token 恢复成 HxW，再做逐通道 3x3 卷积，
# 于是既保留 MLP 的通道变换能力，又显式补入局部空间先验，这就是 PVTv2 常说的 Mix-FFN。
class Mlp(nn.Module):
    # 输入和输出通常都是当前 stage 的嵌入维 C，hidden_features=C*mlp_ratio。
    # 参数逐项说明：
    # - `in_features`：输入 token 的通道 C，也是 Block 的 `dim`；
    # - `hidden_features`：中间通道 C_hidden；B2 四阶段分别为 512、1024、1280、2048；
    # - `out_features`：输出通道，Block 中未显式传入，所以回退为 C，以便和残差 x 相加；
    # - `act_layer`：激活层的“类/构造器”，默认是 GELU，下面需要用 `act_layer()` 创建实例；
    # - `drop`：两处 dropout 的丢弃概率；B0-B5 的当前固定配置均传 0.0。
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        # 初始化基础模块。
        # `nn.Module.__init__` 会建立参数、子模块、hook 等内部容器；任何 `self.xxx = nn.Module` 前都必须先调用。
        super().__init__()
        # 调用者不指定输出维时保持 C_out=C_in，便于 Transformer 残差相加。
        # Python 的 `a or b` 在 a 为 None/0/False 时选择 b；这里正常的维度必须是正整数。
        out_features = out_features or in_features
        # 未指定隐藏维时退化为不扩张；各 PVT 变体实际会传入 C*mlp_ratio。
        # 该兜底让 Mlp 可以单独构造；但在 `Block.__init__` 中会明确传入 `int(dim * mlp_ratio)`。
        hidden_features = hidden_features or in_features
        # 第一层线性映射按 token 独立执行：C -> C_hidden。
        # Linear 只作用于最后一维，因此不会改变 B 和 N，也不会让不同空间 token 在此处互相混合。
        self.fc1 = nn.Linear(in_features, hidden_features)
        # 在隐藏通道上使用深度卷积；它需要 H、W 才能把 token 还原为空间特征。
        # 这里引用的 `DWConv` 虽然定义在文件后部，但 Python 在真正实例化 Mlp 时已经执行完整个模块，所以可正常找到。
        self.dwconv = DWConv(hidden_features)
        # 默认 GELU 激活。
        # GELU 相比 ReLU 是平滑门控，在 Transformer 中很常见；这里创建的是层对象，不是立即处理张量。
        self.act = act_layer()
        # 第二层线性映射：C_hidden -> C_out。
        # 该层把扩张后的隐藏表示投影回残差分支所要求的通道数。
        self.fc2 = nn.Linear(hidden_features, out_features)
        # Dropout 同时用于激活后和第二层线性后；默认 drop=0，不随机丢弃。
        # 同一个无状态 Dropout 模块可以在 forward 中调用两次；训练态随机丢弃，eval 态自动变为恒等映射。
        self.drop = nn.Dropout(drop)

        # 递归初始化当前 MLP 的 Linear、LayerNorm 和 Conv2d 子层。
        # `apply` 会深度优先访问 self 及所有子模块，并把每个模块传给 `_init_weights`；它不是执行一次前向。
        self.apply(self._init_weights)

    # PVT 编码器专用初始化函数。
    # 参数 `m` 不是整张模型，而是 `self.apply` 当前遍历到的某一个子模块。
    def _init_weights(self, m):
        # Linear 权重使用标准差 0.02 的截断正态分布。
        # 该分支会命中 fc1 和 fc2；截断正态能避免初始化时出现绝对值特别大的少数权重。
        if isinstance(m, nn.Linear):
            # 初始化线性层权重。
            # 这是原地写入 Parameter 数据，不创建新的可训练参数。
            trunc_normal_(m.weight, std=.02)
            # 线性层存在偏置时将其清零；内层 isinstance 判断是原实现的冗余保护。
            # 由于已经进入 Linear 分支，再判断一次 `isinstance` 在逻辑上没有必要，但保留原代码不做清理。
            if isinstance(m, nn.Linear) and m.bias is not None:
                # 清零偏置。
                # 初始时不人为偏向任何输出通道，之后偏置会随反向传播正常学习。
                nn.init.constant_(m.bias, 0)
        # LayerNorm 按恒等仿射变换初始化。
        # 当前 Mlp 本身没有 LayerNorm；保留该分支是为了与其他模块使用统一初始化模板。
        elif isinstance(m, nn.LayerNorm):
            # beta=0。
            # LayerNorm 输出公式中的可学习平移量初始化为 0。
            nn.init.constant_(m.bias, 0)
            # gamma=1。
            # 可学习缩放量初始化为 1，因此初始仿射部分不额外缩放标准化结果。
            nn.init.constant_(m.weight, 1.0)
        # Conv2d 按 fan-out 缩放正态分布初始化。
        # 该分支会命中 `self.dwconv.dwconv`，即 Mix-FFN 中实际的深度卷积。
        elif isinstance(m, nn.Conv2d):
            # 计算卷积核面积乘输出通道。
            # 对普通卷积，fan_out 可理解为一个输入位置向后影响的输出连接规模。
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            # 分组卷积需除以 groups；DWConv 中 groups=channels。
            # 深度卷积每组只有一个通道，除以 groups 后不会误把彼此独立的通道都算进同一组连接数。
            fan_out //= m.groups
            # 按 sqrt(2/fan_out) 初始化卷积权重。
            # `.data.normal_` 原地采样均值 0 的正态分布；这是初始化阶段，不进入 autograd 计算图。
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            # 若卷积含偏置则清零。
            # 本类 DWConv 明确 `bias=True`，所以实际会执行下面这一行。
            if m.bias is not None:
                # 清零卷积偏置。
                # `.zero_()` 直接把已有偏置张量全部写成 0。
                m.bias.data.zero_()

    # x 输入形状 (B,N,C)，且 N=H*W。
    # 【调用者】`Block.forward` 在 `self.mlp(self.norm2(x), H, W)` 处调用本方法。
    # 【形状示例】B2 stage1、352 输入时：`x=(B,7744,64)`，fc1 后为 `(B,7744,512)`；
    # DWConv 内会短暂变成 `(B,512,88,88)`，最后 fc2 再回到 `(B,7744,64)`。
    def forward(self, x, H, W):
        # 对每个 token 扩张通道：(B,N,C)->(B,N,C_hidden)。
        # 此时每个位置只做通道线性组合，空间 token 数 N 完全不变。
        x = self.fc1(x)
        # 暂时恢复为二维特征做 3x3 深度卷积，再变回 token；形状仍为 (B,N,C_hidden)。
        # H、W 不能随便填：若 H*W 不等于 N，`DWConv.view` 会因元素总数不匹配而报错。
        x = self.dwconv(x, H, W)
        # GELU 非线性。
        # 激活不改变形状，只改变数值，使多层线性/卷积组合具备非线性表达能力。
        x = self.act(x)
        # 第一次 dropout。
        # B2 固定 drop=0 时数值不变；代码仍保留该层以支持其他调用方传入非零配置。
        x = self.drop(x)
        # 压回输出通道，通常 C_hidden->C。
        # 这一步是能够与 Block 原输入做逐元素残差相加的关键：最后一维必须恢复为 dim。
        x = self.fc2(x)
        # 第二次 dropout。
        # 同样只在训练模式且 drop>0 时随机置零；推理时不会引入随机性。
        x = self.drop(x)
        # 返回与 Block 输入相同的 (B,N,C)，以便残差相加。
        # 本方法本身不做残差加法；残差连接位于外层 `Block.forward`。
        return x


# 多头空间降采样注意力：Q 保留全部 N 个查询，K/V 可按 sr_ratio 降低 token 数以节省计算。
# 【与标准自注意力的差别】标准注意力的 Q、K、V 都有 N 个 token，注意力矩阵为 NxN；
# 本类让 Q 保持 N 个位置，但把 K/V 压缩到 N'，注意力矩阵变为 NxN'，输出仍能覆盖原来的每个 Q 位置。
# 【为什么适合金字塔前几层】浅层 H、W 大，完整 NxN 成本很高；越靠后分辨率越小，sr_ratio 也逐级减小。
class Attention(nn.Module):
    # dim=C；num_heads=h；每头维度 d=C/h；sr_ratio 控制 K/V 的空间降采样倍数。
    # 参数中的 `qkv_bias` 控制 Q 和 KV 的 Linear 是否含偏置；B0-B5 均设 True。
    # `qk_scale` 若为 None 就采用理论常用值 1/sqrt(d)；`attn_drop` 丢注意力概率，`proj_drop` 丢最终输出。
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1):
        # 初始化基础模块。
        # 与所有 nn.Module 子类一样，这一步必须在注册 `self.q` 等子层前完成。
        super().__init__()
        # 多头拆分要求 C 能被 head 数整除。
        # 否则下面的 reshape 无法把 C 无损拆成 `(num_heads, head_dim)` 两维，因此构造阶段就尽早报错。
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        # 保存总嵌入维。
        # 当前 forward 实际从 `x.shape` 再读取 C；保留该属性主要用于模块配置记录和外部检查。
        self.dim = dim
        # 保存注意力头数。
        # forward 的 reshape 和 permute 都会使用它。
        self.num_heads = num_heads
        # 单头通道 d=C/h。
        # B2 四阶段都设计成 d=64，有利于保持每头容量一致。
        head_dim = dim // num_heads
        # 默认缩放因子 1/sqrt(d)，防止 QK 点积随维度增大而数值过大。
        # Python `or` 表示当 qk_scale 是 None 或 0 时使用默认值；通常不会有意传入 0。
        self.scale = qk_scale or head_dim ** -0.5

        # Q 投影保持 C 维。
        # 对每个 token 独立产生 query，后面拆成 h 个头；空间 token 数保持 N。
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        # K 和 V 一次联合投影为 2C，之后再拆成两份。
        # 用一个 Linear 同时生成 K/V 与两个独立 Linear 在功能上等价，但便于一次矩阵乘法完成。
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        # 注意力权重 dropout。
        # 它作用于 softmax 后的概率矩阵，而不是直接作用在输入 token 上。
        self.attn_drop = nn.Dropout(attn_drop)
        # 多头结果拼接后的输出投影。
        # 注意力头拼回 C 后，再用该层混合来自不同头的信息。
        self.proj = nn.Linear(dim, dim)
        # 输出投影后的 dropout。
        # 它对应 Transformer 注意力子层输出端的常规 dropout。
        self.proj_drop = nn.Dropout(proj_drop)

        # 保存空间降采样比例；B2 四阶段默认 [8,4,2,1]。
        # 该值同时决定构造时是否创建 `self.sr/self.norm`，以及 forward 选择哪条分支。
        self.sr_ratio = sr_ratio
        # sr_ratio>1 时只压缩 K/V 分支，Q 仍覆盖每个原始位置。
        # sr=1 时不创建这两个成员，这是有意的；forward 的 else 分支也不会访问它们。
        if sr_ratio > 1:
            # kernel=stride=sr_ratio 的卷积把 (H,W) 降到约 (H/sr,W/sr)，通道保持 C。
            # 无 padding，所以一般公式是 floor((H-sr)/sr)+1；对当前可整除尺寸恰好等于 H/sr。
            # 这里不是 depth-wise 卷积：groups 默认 1，因此降采样同时允许通道混合。
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            # 降采样后的 K/V token 在投影前做 LayerNorm。
            # 卷积后不同 token 的数值尺度可能改变，归一化有助于稳定后续 KV 线性投影。
            self.norm = nn.LayerNorm(dim)

        # 初始化本注意力模块及其子层。
        # 包括 q、kv、proj、可能存在的 sr 卷积和 norm；Dropout/Module 自身不会命中任何参数分支。
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
    # 【352/B2 的关键形状】
    # - stage1: Q=(B,1,7744,64)，K/V=(B,1,121,64)，注意力矩阵=(B,1,7744,121)；
    # - stage2: Q=(B,2,1936,64)，K/V=(B,2,121,64)，注意力矩阵=(B,2,1936,121)；
    # - stage3: Q=(B,5, 484,64)，K/V=(B,5,121,64)，注意力矩阵=(B,5, 484,121)；
    # - stage4: Q=(B,8, 121,64)，K/V=(B,8,121,64)，注意力矩阵=(B,8, 121,121)。
    def forward(self, x, H, W):
        # 提取 batch、token 数和通道数。
        # `x.shape` 解包也起到一致性前提说明：输入必须严格是三维 token 张量。
        B, N, C = x.shape
        # Q: (B,N,C)->(B,N,h,d)->(B,h,N,d)。
        # `permute(0,2,1,3)` 把 head 维移到 token 维之前，便于后面的批量矩阵乘法按每个头独立进行。
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        # 前三个 stage 通常 sr_ratio>1，通过减少 K/V token 控制注意力复杂度。
        # 此条件与构造器保持一致，保证进入分支时 `self.sr` 和 `self.norm` 一定存在。
        if self.sr_ratio > 1:
            # token 恢复为图像布局：(B,N,C)->(B,C,H,W)。
            # 先 permute 是因为 PyTorch 卷积规定通道维必须位于第二维；reshape 依赖 N=H*W。
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
            # 空间降采样并重新展平为 N' 个 token：(B,C,H',W')->(B,N',C)。
            # `-1` 让 PyTorch 根据输出元素数自动推导 N'=H'*W'，避免手工重复计算。
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
            # 对降采样 token 做 LayerNorm。
            # LayerNorm 默认规范化最后一维 C，所以此处必须先排成 `(B,N',C)`。
            x_ = self.norm(x_)
            # 联合生成 K/V，并整理为 (2,B,h,N',d)。
            # 中间 `(B,N',2,h,d)` 中的 2 表示 K/V 两份；permute 后把它提到最前，便于下一行索引拆分。
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        # 最后一阶段 sr_ratio=1，不压缩空间，K/V 直接来自原 N 个 token。
        else:
            # 整理为 (2,B,h,N,d)。
            # 除了 token 数使用 N 而非 N'，其余布局与上方分支完全相同，所以后续代码可以共用。
            kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        # 沿首维拆出 K 和 V；形状分别为 (B,h,N',d)。
        # `kv[0]` 和 `kv[1]` 是张量视图/索引结果，不会引入新的可学习参数。
        k, v = kv[0], kv[1]

        # Q 与 K^T 相乘得到 (B,h,N,N') 注意力 logits，并乘缩放因子。
        # `transpose(-2,-1)` 把 K 的 `(N',d)` 变成 `(d,N')`；`@` 对 B 和 h 两维执行批量矩阵乘法。
        attn = (q @ k.transpose(-2, -1)) * self.scale
        # 对每个查询位置沿 K/V token 维归一化为概率。
        # `dim=-1` 表示每个 Q 对全部 K 的权重和为 1，而不是在注意力头或查询之间归一化。
        attn = attn.softmax(dim=-1)
        # 对注意力概率执行 dropout。
        # 训练时随机去掉部分 Q-K 联系作为正则化；当前变体 attn_drop_rate=0，所以默认不改变数值。
        attn = self.attn_drop(attn)

        # 权重与 V 相乘得 (B,h,N,d)，再拼回 (B,N,C)。
        # transpose 把 N 放回第二维；reshape 把 h 和 d 合并，因为 h*d=C。
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        # 输出线性投影混合各头信息。
        # 输出仍是 `(B,N,C)`，不会改变空间 token 数或通道总数。
        x = self.proj(x)
        # 输出 dropout。
        # 在外层 Block 做残差相加前进行；eval 模式自动关闭随机丢弃。
        x = self.proj_drop(x)

        # 返回 (B,N,C)。
        # 注意力模块不负责 LayerNorm 和残差，它们由 `Block.forward` 包在外面。
        return x


# 单个 PVTv2 Transformer Block：Pre-Norm 注意力残差 + Pre-Norm Mix-FFN 残差。
# 【一个 Block 的完整数据流】
# 输入 x -> LN1 -> 空间降采样注意力 -> DropPath -> 与原 x 相加
#       -> LN2 -> Mix-FFN（Linear + DWConv + GELU + Linear）-> DropPath -> 再与上一结果相加。
# “Pre-Norm”表示归一化发生在注意力/MLP 之前，而不是残差相加之后；这通常有利于深层 Transformer 训练稳定。
# 两个子层都保持 `(B,N,C)`，所以残差是同形状逐元素加法，不做拼接，也不增加通道。
class Block(nn.Module):

    # drop_path 为该 block 的随机深度概率，越深的 block 通常概率越大。
    # `drop` 是普通元素级 dropout 概率，`drop_path` 是整条残差分支按样本随机丢弃的概率，二者不要混淆。
    # `norm_layer` 接收 dim 后返回 LayerNorm；B2 传入的是 `partial(nn.LayerNorm, eps=1e-6)`。
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1):
        # 初始化基础模块。
        # 完成 Module 内部状态建立，随后定义的 norm、attn、mlp 才会被正确注册进 state_dict 和优化器。
        super().__init__()
        # 注意力前的 LayerNorm。
        # LayerNorm 只规范化最后一维 C；B、N 不变，所以输出仍是 `(B,N,C)`。
        self.norm1 = norm_layer(dim)
        # 构造空间降采样多头注意力。
        # 本 Block 不直接实现 Q/K/V 细节，而是把当前 stage 的配置传给上面的 `Attention` 类。
        self.attn = Attention(
            # 传入当前 stage 通道维及注意力配置。
            # `dim` 作为第一个位置参数传入，对应 `Attention.__init__` 的 dim。
            dim,
            # qk_scale 可覆盖默认 1/sqrt(d)，drop 和 sr_ratio 控制正则与 K/V 长度。
            # 这里的 `proj_drop=drop` 表示注意力输出投影后复用该 stage 的普通 dropout 概率。
            num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        # 概率大于 0 时使用 DropPath，否则用 Identity 保持完整分支。
        # 为什么要用条件表达式：当概率为 0 时 Identity 没有 `drop_prob` 等状态，也省去一次随机操作；
        # 这也导致后面的遗留 `reset_drop_path` 若直接访问首个 Identity 的 `.drop_prob`，可能出现属性错误。
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        # Mix-FFN 前的第二个 LayerNorm。
        # 它接收的是已经完成第一条注意力残差相加后的 x，而不是最初输入。
        self.norm2 = norm_layer(dim)
        # 隐藏通道 C_hidden=int(C*mlp_ratio)。
        # `int` 把可能为浮点数的倍率乘积转成 Linear 所需整数；当前配置乘积本身都是整数值。
        mlp_hidden_dim = int(dim * mlp_ratio)
        # 构造带 DWConv 的 Mix-FFN，输出仍为 C。
        # 未传 out_features，因此 Mlp 会让输出通道回退为 `in_features=dim`，满足第二条残差连接。
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        # 初始化本 block；嵌套模块也曾各自初始化，最终参数值以最后一次递归初始化为准。
        # 构造 Attention/Mlp 时它们各自已 `apply` 一次；这里再次遍历属于原实现行为，不影响参数形状或模块连接。
        self.apply(self._init_weights)

    # Block 的初始化规则。
    # 这段与 Mlp/Attention 中的规则重复，是上游实现为了每个类可独立构造而保留的局部模板。
    def _init_weights(self, m):
        # Linear 截断正态初始化。
        # 会覆盖初始化本 Block 内 Attention 与 Mlp 的全部 Linear 权重。
        if isinstance(m, nn.Linear):
            # 初始化权重。
            # 标准差 0.02，不改变 Parameter 的形状、dtype 或 requires_grad。
            trunc_normal_(m.weight, std=.02)
            # 可选偏置清零。
            # 内层再次判断 Linear 是冗余但无害的原始逻辑。
            if isinstance(m, nn.Linear) and m.bias is not None:
                # 清零偏置。
                # 使用 PyTorch 初始化 API 原地写值。
                nn.init.constant_(m.bias, 0)
        # LayerNorm 恒等仿射初始化。
        # norm1、norm2，以及 Attention 在 sr>1 时的 norm 都会命中。
        elif isinstance(m, nn.LayerNorm):
            # beta=0。
            # 平移参数清零。
            nn.init.constant_(m.bias, 0)
            # gamma=1。
            # 缩放参数设 1。
            nn.init.constant_(m.weight, 1.0)
        # Conv2d fan-out 初始化。
        # 会命中空间降采样卷积 `Attention.sr` 和 Mix-FFN 的深度卷积。
        elif isinstance(m, nn.Conv2d):
            # 计算 fan-out。
            # 卷积核高、宽和输出通道共同决定未分组时的 fan-out。
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            # 考虑 groups。
            # 普通 sr 卷积 groups=1；DWConv groups=C，二者都能使用同一公式。
            fan_out //= m.groups
            # 初始化权重。
            # 方差按连接规模缩放，避免层间信号尺度快速放大或缩小。
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            # 检查偏置。
            # 某些卷积可能关闭 bias，所以先判断再访问。
            if m.bias is not None:
                # 清零偏置。
                # 初始卷积输出不附加常量偏移。
                m.bias.data.zero_()

    # 输入输出都是 (B,N,C)，因此两条残差都可直接逐元素相加。
    # 【调用者】`PyramidVisionTransformerImpr.forward_features` 在每个 stage 的 for 循环中调用。
    # 【H/W 的作用】注意力需要 H/W 做 K/V 空间降采样，Mlp 中 DWConv 也需要 H/W 恢复二维布局；
    # Block 本身不改变 H、W 或 N，一个 stage 内的所有 Block 均共享同一组 H、W。
    def forward(self, x, H, W):
        # 第一条残差：x + DropPath(Attention(LN(x)))。
        # Python 会先算右侧：norm1 不改形状，attn 不改形状，drop_path 不改形状，最后才与原 x 逐元素相加。
        # 若训练时 DropPath 丢掉此分支，残差主干仍直接保留 x，因此深网络仍有稳定的信息/梯度通路。
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        # 第二条残差：x + DropPath(MixFFN(LN(x)))。
        # 这里左侧 x 已包含注意力结果；同一个 DropPath 模块被调用两次，但每次调用会独立生成随机掩码。
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))

        # 返回当前 block 编码后的 token。
        # 后续可能进入同 stage 的下一个 Block；最后一个 Block 后才由 stage 对应的 norm1/2/3/4 统一归一化。
        return x


# Overlapping Patch Embedding：用带 padding 的卷积生成互相重叠的 patch 特征，而非无重叠切块。
# 【它承担两个任务】一是用 stride 做空间下采样，二是把输入通道投影为当前 stage 的嵌入通道。
# stage1 接收原始图像；stage2-4 接收上一个 stage 已恢复成 NCHW 的特征图。
# 所谓 overlap 来自 `kernel_size > stride` 且有 padding：相邻输出位置看到的输入区域互相重叠，
# 相比硬切成互不重叠 patch，边界处的信息连续性更好，也更像 CNN 的逐层特征提取。
class OverlapPatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    # stage1 通常 kernel=7,stride=4；stage2-4 通常 kernel=3,stride=2。
    # `img_size` 只用于记录预期尺寸元数据，forward 不会强制输入必须等于它；
    # `patch_size` 实际作为卷积核大小，`stride` 决定真正降采样倍率，二者不是同一个概念。
    def __init__(self, img_size=224, patch_size=7, stride=4, in_chans=3, embed_dim=768):
        # 初始化基础模块。
        # 先初始化 Module，才能注册下面的 Conv2d 和 LayerNorm。
        super().__init__()
        # 将整数图像尺寸标准化为 (H,W)。
        # 若调用方已经传二元组，`to_2tuple` 会保留；这样后面统一用 `[0]`、`[1]` 访问。
        img_size = to_2tuple(img_size)
        # 将整数 patch 尺寸标准化为 (kH,kW)。
        # 支持正方形整数，也兼容未来传入非正方形卷积核。
        patch_size = to_2tuple(patch_size)

        # 保存声明的输入尺寸元数据。
        # 该值不会在 forward 中做 assert，因此 EMCAD 可输入 224、352 等不同大小图像。
        self.img_size = img_size
        # 保存卷积核尺寸元数据。
        # 调试或打印模型时可以看到每阶段使用的 patch 核大小。
        self.patch_size = patch_size
        # 这里按 patch_size 计算的 H/W 是遗留元数据，不等于 stride 卷积的真实输出；有效 forward 会重新读取实际 H、W。
        # 例如 stage1 声明 img_size=224、patch_size=7 时这里得到 32，而 stride=4 卷积实际输出 56；
        # 好在本项目后续绝不依赖这里的 `self.H/self.W` 来 reshape 有效前向张量。
        self.H, self.W = img_size[0] // patch_size[0], img_size[1] // patch_size[1]
        # 同样属于元数据，当前 EMCAD 前向不依赖该 num_patches。
        # 它只可能被文件中已经注释掉的位置编码兼容代码引用，不能用来判断当前实际 token 数。
        self.num_patches = self.H * self.W
        # 卷积完成空间降采样和通道投影；padding=k//2 使相邻 patch 感受野重叠。
        # Conv2d 输入/输出布局都是 NCHW；输出尺寸公式为 floor((H+2p-k)/stride)+1。
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                              # 二维对称 padding。
                              # 奇数核 7 或 3 使用 k//2，可在 stride=1 时保持空间尺寸；此处 stride>1 用来规则降采样。
                              padding=(patch_size[0] // 2, patch_size[1] // 2))
        # 卷积结果展平为 token 后，对每个 token 的 embed_dim 通道做 LayerNorm。
        # LayerNorm 的 normalized_shape 是 embed_dim，因此 forward 必须先把通道移到最后一维。
        self.norm = nn.LayerNorm(embed_dim)

        # 初始化投影卷积和 LayerNorm。
        # 统一规则会递归访问 self.proj 与 self.norm；当前类没有 Linear，但保留其分支以复用模板。
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
    # 【352/B2 示例】四次调用分别为：
    # `(B,3,352,352)->(B,7744,64),88,88`；
    # `(B,64,88,88)->(B,1936,128),44,44`；
    # `(B,128,44,44)->(B,484,320),22,22`；
    # `(B,320,22,22)->(B,121,512),11,11`。
    def forward(self, x):
        # 重叠卷积投影后得到 (B,C_embed,H_out,W_out)。
        # 这一行同时完成局部感受野提取、通道变换和按 stride 的降采样。
        x = self.proj(x)
        # 从真实卷积输出读取 H、W，后续 reshape 以它们为准。
        # 前两个下划线接收 B、C 但表示“本行不需要使用”；实际 H/W 比构造器中的元数据可靠。
        _, _, H, W = x.shape
        # 展平空间并交换维度：(B,C,H,W)->(B,C,N)->(B,N,C)。
        # `flatten(2)` 从第 2 维开始合并 H、W；`transpose(1,2)` 再把 token 维放中间、通道放最后。
        x = x.flatten(2).transpose(1, 2)
        # 对每个 token 做 LayerNorm。
        # 只在 C_embed 维上求均值/方差，不会把不同像素位置或不同 batch 样本混在一起归一化。
        x = self.norm(x)

        # 返回 token 及其二维布局，供 Attention 和 DWConv 临时恢复空间结构。
        # H、W 单独返回是因为展平后仅从 N 未必能唯一推回长宽，尤其输入不是正方形时。
        return x, H, W


# 四阶段 PVTv2 主干：每个阶段依次执行重叠 patch embedding、若干 Transformer Block 和 LayerNorm。
# 对 EMCAD 而言，最重要的接口是 forward 返回 [x1,x2,x3,x4] 四张 NCHW 特征图。
# 【四阶段之间不是并行关系，而是串行关系】
# 原图 -> patch_embed1 + block1 -> x1
# x1   -> patch_embed2 + block2 -> x2
# x2   -> patch_embed3 + block3 -> x3
# x3   -> patch_embed4 + block4 -> x4
# 每一阶段都先降采样并升通道，再在固定分辨率上反复执行若干 Block；因此越深的特征空间越小、语义越强。
# 【与 EMCAD 解码器的连接】`networks.py` 取得本类输出后，把 x4 作为解码主输入，
# 把 `[x3,x2,x1]` 作为三个由深到浅的跳跃连接，让解码器逐级恢复分辨率和边界细节。
class PyramidVisionTransformerImpr(nn.Module):
    # embed_dims、num_heads、mlp_ratios、depths、sr_ratios 都是长度为 4 的逐阶段配置。
    # 参数详细含义：
    # - `img_size`：构造 patch embedding 时记录的基准输入边长，不限制实际 forward 尺寸；
    # - `patch_size`：保留的上游统一参数，当前四个 OverlapPatchEmbed 实际分别硬编码 7/3/3/3；
    # - `in_chans`：原图通道，EMCADNet 通常保证传入这里的是 3 通道；
    # - `num_classes`：分类版遗留元数据，当前分割 forward 不产生分类 logits；
    # - `embed_dims[i]`：stage i+1 的输出通道 C；
    # - `num_heads[i]`：stage i+1 的注意力头数 h；
    # - `mlp_ratios[i]`：stage i+1 的 Mix-FFN 隐藏通道倍率；
    # - `depths[i]`：stage i+1 重复多少个 Block；
    # - `sr_ratios[i]`：stage i+1 的 K/V 空间降采样倍率；
    # - 三种 drop rate 分别控制普通输出 dropout、注意力概率 dropout、随机深度最大概率。
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dims=[64, 128, 256, 512],
                 # drop_path_rate 会按所有 block 的全局深度线性递增。
                 # `num_heads` 和 `mlp_ratios` 的第 i 项必须与 `embed_dims` 第 i 项相容。
                 num_heads=[1, 2, 4, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=False, qk_scale=None, drop_rate=0.,
                 # norm_layer 默认 LayerNorm；PVT 变体通常通过 partial 把 eps 设为 1e-6。
                 # `attn_drop_rate` 只作用于 softmax 后注意力矩阵，`drop_path_rate` 则按 Block 深度分配。
                 attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                 # sr_ratios 控制四阶段 K/V 空间降采样，最后阶段通常为 1。
                 # 默认参数是通用基类默认值；真正 B2 配置由文件末尾 `pvt_v2_b2.__init__` 传入覆盖。
                 depths=[3, 4, 6, 3], sr_ratios=[8, 4, 2, 1]):
        # 初始化基础模块。
        # 完成 nn.Module 内部初始化，后面赋给 self 的层会自动注册为子模块。
        super().__init__()
        # 分类类别数是上游分类模型遗留元数据；EMCAD 分割前向不使用分类 head。
        # 保留它使部分 timm 风格工具仍能查询模型的 num_classes，但它不会决定 EMCAD 的分割类别数。
        self.num_classes = num_classes
        # 保存四阶段 block 数，reset_drop_path 会按它们遍历。
        # forward 本身通过 ModuleList 迭代，不直接使用 self.depths；主要服务构造后的管理/兼容接口。
        self.depths = depths

        # patch_embed
        # stage1：7x7、stride4，把 RGB 输入降到约 1/4 并投影为 embed_dims[0] 通道。
        # 注意这里没有使用形参 `patch_size=16`；当前有效 stage1 核固定为 7，文件末尾变体传入的 patch_size=4 也不会改它。
        self.patch_embed1 = OverlapPatchEmbed(img_size=img_size, patch_size=7, stride=4, in_chans=in_chans,
                                              # stage1 的输出嵌入通道。
                                              # B2 中 embed_dims[0]=64，所以 352 输入变为 `(B,7744,64)` token。
                                              embed_dim=embed_dims[0])
        # stage2：3x3、stride2，从 x1 的 1/4 分辨率降到 x2 的 1/8。
        # `img_size // 4` 只是传递预期元数据；真实输入尺寸由上一阶段实际输出决定。
        self.patch_embed2 = OverlapPatchEmbed(img_size=img_size // 4, patch_size=3, stride=2, in_chans=embed_dims[0],
                                              # 通道 embed_dims[0] -> embed_dims[1]。
                                              # B2 中 64->128；352 示例空间 88x88->44x44。
                                              embed_dim=embed_dims[1])
        # stage3：再降到 1/16。
        # 输入通道必须等于 stage2 的输出通道，否则 Conv2d 会在运行时报告通道不匹配。
        self.patch_embed3 = OverlapPatchEmbed(img_size=img_size // 8, patch_size=3, stride=2, in_chans=embed_dims[1],
                                              # 通道 embed_dims[1] -> embed_dims[2]。
                                              # B2 中 128->320；352 示例空间 44x44->22x22。
                                              embed_dim=embed_dims[2])
        # stage4：再降到 1/32，得到送入 EMCAD 主路的最深特征 x4。
        # 这是编码器最后一次空间降采样，之后不再创建 stage5。
        self.patch_embed4 = OverlapPatchEmbed(img_size=img_size // 16, patch_size=3, stride=2, in_chans=embed_dims[2],
                                              # 通道 embed_dims[2] -> embed_dims[3]。
                                              # B2 中 320->512；352 示例空间 22x22->11x11。
                                              embed_dim=embed_dims[3])

        # transformer encoder
        # 为所有 block 生成从 0 到 drop_path_rate 的线性随机深度概率表。
        # `sum(depths)` 是所有 stage 的 Block 总数；B2 为 3+4+6+3=16。
        # `torch.linspace(0,0.1,16)` 让最浅 Block 概率为 0、最深 Block 为 0.1，中间线性递增；
        # `.item()` 把每个 0 维张量转换为普通 Python 浮点数，以便传给 DropPath 构造器。
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        # cur 指向当前 stage 在全局 dpr 表中的起始位置。
        # 它不是网络中的可训练量，只是构造 ModuleList 时切分概率列表的 Python 游标。
        cur = 0
        # stage1 包含 depths[0] 个 Block。
        # `nn.ModuleList` 很关键：若用普通 Python list，内部 Block 参数不会被模型注册，优化器和 `.to(device)` 都可能遗漏它们。
        self.block1 = nn.ModuleList([Block(
            # stage1 使用第一组通道、头数、MLP 扩张倍率和空间降采样比例。
            # B2 为 dim=64、num_heads=1、mlp_ratio=8，因此隐藏通道为 512。
            dim=embed_dims[0], num_heads=num_heads[0], mlp_ratio=mlp_ratios[0], qkv_bias=qkv_bias, qk_scale=qk_scale,
            # 每个 block 取得自己的随机深度概率 dpr[cur+i]。
            # stage1 的 i=0,1,2 对应全局 dpr 的前三项；普通 drop 和 attn_drop 在 B2 都是 0。
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            # stage1 的 sr_ratio 通常为 8，大幅减少 K/V token。
            # 对 352 输入，Q 有 7744 个位置，而 K/V 仅 121 个位置。
            sr_ratio=sr_ratios[0])
            # 循环创建 stage1 的全部 block。
            # `range(depths[0])` 只在模型构造时运行一次，并不是每次 forward 动态创建层。
            for i in range(depths[0])])
        # stage1 全部 block 后的最终 LayerNorm。
        # 该 norm 不属于任何单个 Block；所有 stage1 Block 完成后再统一规范化一次输出 token。
        self.norm1 = norm_layer(embed_dims[0])

        # 将 dpr 游标移动到 stage2 起点。
        # B2 此时 cur=3，所以下一个 stage 从 dpr[3] 开始取概率。
        cur += depths[0]
        # stage2 Block 列表。
        # B2 会创建 4 个 Block，每个输入/输出 token 通道均为 128。
        self.block2 = nn.ModuleList([Block(
            # stage2 通道和注意力头配置。
            # B2 为 dim=128、num_heads=2、每头 64 维、mlp_ratio=8、隐藏通道 1024。
            dim=embed_dims[1], num_heads=num_heads[1], mlp_ratio=mlp_ratios[1], qkv_bias=qkv_bias, qk_scale=qk_scale,
            # 使用对应的全局随机深度概率。
            # i 从 0 到 depths[1]-1，索引全局 dpr[cur+i]，确保概率随整网深度而不是每 stage 重新从 0 开始。
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            # stage2 sr_ratio 通常为 4。
            # 352 输入时 44x44 的 K/V 被卷积降成 11x11。
            sr_ratio=sr_ratios[1])
            # 循环创建 stage2 blocks。
            # ModuleList 保留创建顺序，forward 会严格按同一顺序逐个执行。
            for i in range(depths[1])])
        # stage2 最终 LayerNorm。
        # normalized_shape=128（B2），只规范化每个 token 的通道维。
        self.norm2 = norm_layer(embed_dims[1])

        # 将 dpr 游标移动到 stage3 起点。
        # B2 此时 cur=3+4=7。
        cur += depths[1]
        # stage3 Block 列表；B2 的这一阶段有 6 个 block，B3-B5 会明显更深。
        # stage3 通常是 PVTv2 计算和表征的核心阶段：空间已经缩小，但深度开始增加。
        self.block3 = nn.ModuleList([Block(
            # stage3 通道、头数和 MLP 配置。
            # B2 为 dim=320、num_heads=5、每头 64 维、mlp_ratio=4、隐藏通道 1280。
            dim=embed_dims[2], num_heads=num_heads[2], mlp_ratio=mlp_ratios[2], qkv_bias=qkv_bias, qk_scale=qk_scale,
            # stage3 随机深度概率。
            # B2 的 6 个 Block 使用 dpr[7] 到 dpr[12]。
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            # stage3 sr_ratio 通常为 2。
            # 352 输入时 22x22 的 K/V 被降成 11x11；Q 仍保留 22x22 的每个位置。
            sr_ratio=sr_ratios[2])
            # 循环创建 stage3 blocks。
            # 修改 B2 的 depths[2] 会改变模型深度、参数量和预训练权重键数量，不能随意改后继续严格加载原权重。
            for i in range(depths[2])])
        # stage3 最终 LayerNorm。
        # B2 对最后一个 stage3 Block 的 `(B,484,320)` 输出做归一化。
        self.norm3 = norm_layer(embed_dims[2])

        # 将 dpr 游标移动到 stage4 起点。
        # B2 此时 cur=7+6=13，最后 3 个 Block 使用 dpr[13:16]。
        cur += depths[2]
        # stage4 Block 列表。
        # 此阶段分辨率最低、通道最多，输出直接成为 EMCAD 解码器的最深语义特征。
        self.block4 = nn.ModuleList([Block(
            # stage4 通道和多头配置。
            # B2 为 dim=512、num_heads=8、每头 64 维、mlp_ratio=4、隐藏通道 2048。
            dim=embed_dims[3], num_heads=num_heads[3], mlp_ratio=mlp_ratios[3], qkv_bias=qkv_bias, qk_scale=qk_scale,
            # stage4 随机深度概率。
            # 最深 Block 获得最大 drop_path_rate=0.1，训练时正则化强于浅层。
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            # sr_ratio=1 时最后阶段 K/V 不再做空间降采样。
            # 由于 352 输入此时只有 121 个 token，完整 121x121 注意力已经可接受。
            sr_ratio=sr_ratios[3])
            # 循环创建 stage4 blocks。
            # B2 创建 3 个；B0/B1/B3/B4/B5 的数量由各自 depths[3] 指定。
            for i in range(depths[3])])
        # stage4 最终 LayerNorm。
        # 归一化后才恢复为 `(B,C4,H4,W4)` 并加入输出列表。
        self.norm4 = norm_layer(embed_dims[3])

        # classification head
        # 分类 head 被原工程注释掉，因为 EMCAD 需要四级特征而不是图像分类 logits。
        # self.head = nn.Linear(embed_dims[3], num_classes) if num_classes > 0 else nn.Identity()

        # 对整个四阶段主干再次执行统一初始化。
        # 这一步会访问 patch embedding、所有 Block 和所有 stage norm；模型随后若加载预训练 state_dict，
        # checkpoint 中匹配到的权重会覆盖这些随机初值，未匹配参数则仍保留这里的初始化值。
        self.apply(self._init_weights)

    # 主干级初始化函数，与子模块规则保持一致。
    # 【调用时机】只在模型构造阶段通过上一行 `self.apply(self._init_weights)` 被递归调用；
    # 它不是训练循环中的参数更新，也不是加载 `.pth` 文件。训练时真正更新参数的是优化器。
    def _init_weights(self, m):
        # Linear 截断正态初始化。
        # 命中所有 Q/K/V、注意力输出投影以及 Mix-FFN 的 fc1/fc2。
        if isinstance(m, nn.Linear):
            # 初始化权重。
            # `trunc_normal_` 原地写入，典型绝对值不会远离均值太多，有利于初始数值稳定。
            trunc_normal_(m.weight, std=.02)
            # 可选偏置清零。
            # qkv_bias=True 的 B0-B5 会让相关 Linear 存在偏置，因此这些偏置会从 0 开始学习。
            if isinstance(m, nn.Linear) and m.bias is not None:
                # 清零偏置。
                # 内层类型判断重复，但我们保留原实现不作代码重构。
                nn.init.constant_(m.bias, 0)
        # LayerNorm 恒等仿射初始化。
        # 命中四个 stage norm、每个 Block 的 norm1/norm2，以及 sr 分支的 norm。
        elif isinstance(m, nn.LayerNorm):
            # beta=0。
            # 归一化后的可学习平移初始为零。
            nn.init.constant_(m.bias, 0)
            # gamma=1。
            # 归一化后的可学习缩放初始为一。
            nn.init.constant_(m.weight, 1.0)
        # Conv2d fan-out 初始化。
        # 命中四个 patch embedding 卷积、前三阶段 Attention.sr 卷积和所有 Mix-FFN 深度卷积。
        elif isinstance(m, nn.Conv2d):
            # 计算卷积 fan-out。
            # kernel_size 是二元组，分别取高和宽；out_channels 是该卷积输出通道数。
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            # 分组卷积修正。
            # 除以 groups 后同一公式可正确处理普通卷积和 groups=通道数的 depth-wise 卷积。
            fan_out //= m.groups
            # 初始化卷积权重。
            # 权重张量形状保持原样，只改变初始数值分布。
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            # 检查可选偏置。
            # PatchEmbed 和 Attention.sr 使用 Conv2d 默认 bias=True，DWConv 也显式 True，因此通常会执行清零。
            if m.bias is not None:
                # 清零卷积偏置。
                # 仍是构造期原地操作，不被记录为一次训练计算。
                m.bias.data.zero_()

    # 上游框架兼容接口；当前函数不会真正加载 pretrained 文件，正式权重加载在 lib/networks.py 完成。
    # 【重要】不要因为方法名叫 init_weights 就认为传路径后会自动恢复预训练参数；当前函数体只赋了一个整数局部变量。
    # 在本项目的有效路径中，`EMCADNet` 会先构造 `pvt_v2_b2()`，再由 `torch.load` 和 `load_state_dict` 加载编码器权重。
    def init_weights(self, pretrained=None):
        # 只有传入字符串时进入该遗留分支。
        # 非字符串（包括 None、dict、Path）都会直接返回，不做任何事情。
        if isinstance(pretrained, str):
            # logger 只是占位变量，当前主路径不读取它。
            # 这是普通 Python 局部变量，不会注册到模型，也不会输出日志或触发文件 I/O。
            logger = 1

    # 原 checkpoint 加载调用已被现有代码注释，因此本方法当前没有加载效果。
    # load_checkpoint(self, pretrained, map_location='cpu', strict=False, logger=logger)

    # 工程遗留的动态随机深度调整接口；当前训练入口没有调用它。
    # 【设计意图】模型构造后若想改变最大 drop-path，可重新为各 Block 分配线性概率；
    # 【实际风险】每个 stage 的第一个甚至全网第一个 Block 可能在构造时因概率为 0 使用 `nn.Identity`，
    # Identity 没有 `.drop_prob` 属性，所以该方法并不是对所有初始配置都稳健。除非先修复实现，否则不要在主训练中盲目调用。
    def reset_drop_path(self, drop_path_rate):
        # 重新生成全部 block 的 drop-path 概率表。
        # 长度仍由原始 self.depths 决定；这不会增删 Block，只尝试改每个现有 DropPath 的概率属性。
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(self.depths))]
        # 从 stage1 起点开始。
        # cur 仍是 Python 索引游标，不属于 state_dict。
        cur = 0
        # 遍历 stage1 blocks。
        # 循环次数必须与构造时 block1 长度一致；这里依赖 self.depths 未被外部错误改写。
        for i in range(self.depths[0]):
            # 直接改写 DropPath.drop_prob；首个 block 若是 Identity，则该遗留接口可能不适用。
            # 赋值只改变后续 forward 的随机深度概率，不会重新初始化或加载任何权重。
            self.block1[i].drop_path.drop_prob = dpr[cur + i]

        # 移动到 stage2 起点。
        # B2 从索引 3 开始处理 stage2。
        cur += self.depths[0]
        # 遍历 stage2 blocks。
        # `self.block2[i]` 是构造时注册在 ModuleList 中的原 Block 对象。
        for i in range(self.depths[1]):
            # 更新 stage2 概率。
            # 对应全局网络深度继续递增，而不是在 stage2 重新从 0 开始。
            self.block2[i].drop_path.drop_prob = dpr[cur + i]

        # 移动到 stage3 起点。
        # B2 中加上 4 后 cur=7。
        cur += self.depths[1]
        # 遍历 stage3 blocks。
        # B2 循环 6 次。
        for i in range(self.depths[2]):
            # 更新 stage3 概率。
            # 这里只改公开属性；是否允许这样运行取决于当前 timm DropPath 实现。
            self.block3[i].drop_path.drop_prob = dpr[cur + i]

        # 移动到 stage4 起点。
        # B2 中 cur 变为 13。
        cur += self.depths[2]
        # 遍历 stage4 blocks。
        # 最后一个元素将获得调用参数 `drop_path_rate` 本身。
        for i in range(self.depths[3]):
            # 更新 stage4 概率。
            # 不改变模型 train/eval 状态；DropPath 只有在 train 模式才真正随机丢分支。
            self.block4[i].drop_path.drop_prob = dpr[cur + i]

    # 工程遗留冻结接口；当前训练入口没有调用。
    # 【风险说明】PyTorch 冻结参数的标准做法是遍历 `self.patch_embed1.parameters()` 并设置每个参数的
    # `requires_grad=False`。下面代码仅在 Module 对象上新增一个普通同名属性，不会自动递归修改其 Parameter。
    def freeze_patch_emb(self):
        # 这里只给模块对象设置属性，并未逐个把 Parameter.requires_grad 设为 False，不能视作已真正冻结参数。
        # 因此即便调用该方法，优化器仍可能看到并更新 patch_embed1 的卷积和 LayerNorm 参数。
        self.patch_embed1.requires_grad = False

    # 告诉某些优化器哪些参数不应 weight decay；当前返回名称来自含位置编码的旧版接口。
    # 装饰器使 TorchScript 编译模型时忽略该辅助方法，因为 Python set 返回值不参与张量前向。
    @torch.jit.ignore
    # TorchScript 忽略这个 Python 集合返回方法。
    # 当前 EMCAD 训练代码没有查询这个集合，所以它不会改变现有优化器参数组。
    def no_weight_decay(self):
        # 当前模型并没有启用这些 pos_embed/cls_token 参数，因此它是兼容性遗留信息。
        # 返回的是参数“名字”的集合，不是 Parameter 对象；而且这些名字在当前 state_dict 中通常不存在。
        return {'pos_embed1', 'pos_embed2', 'pos_embed3', 'pos_embed4', 'cls_token'}  # has pos_embed may be better

    # 分类模型兼容接口；EMCAD 路径不调用。
    # 因构造器中的 `self.head = ...` 已被注释掉，本实例通常没有 `head` 属性；直接调用会触发 AttributeError。
    def get_classifier(self):
        # self.head 在当前构造器中已被注释，直接调用该遗留方法可能找不到属性。
        # 这不影响分割训练，因为 `forward` 从未访问 get_classifier 或 self.head。
        return self.head

    # 分类 head 重设接口；EMCAD 分割任务不调用。
    # 该方法看似会补建 self.head，但又依赖构造器从未设置的 `self.embed_dim`，所以 num_classes>0 时同样可能报错。
    def reset_classifier(self, num_classes, global_pool=''):
        # 更新分类类别元数据。
        # `global_pool` 形参在函数体没有使用，仅为兼容上游调用签名而保留。
        self.num_classes = num_classes
        # 当前类没有设置 self.embed_dim，且分类 head 原本被禁用，因此该行属于未接入主路径的遗留 API。
        # 只有 num_classes<=0 走 `nn.Identity()` 分支时，Python 条件表达式才不会求值 `self.embed_dim`。
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    # def _get_pos_embed(self, pos_embed, patch_embed, H, W):
    #     if H * W == self.patch_embed1.num_patches:
    #         return pos_embed
    #     else:
    #         return F.interpolate(
    #             pos_embed.reshape(1, patch_embed.H, patch_embed.W, -1).permute(0, 3, 1, 2),
    #             size=(H, W), mode="bilinear").reshape(1, -1, H * W).permute(0, 2, 1)

    # 核心四阶段前向；输入 (B,3,H,W)，输出四个从浅到深的 NCHW 特征。
    # 【本方法是整份文件最重要的阅读入口】它把构造器中四套 patch_embed/block/norm 真正串起来运行。
    # 它没有分类池化、没有全连接分类头、没有 sigmoid/softmax，也不会直接产生分割 mask。
    # 对非 32 整除输入，四次 stride 卷积仍会按卷积公式取整；后续解码时是否精确对齐要由整体网络尺寸策略保证。
    def forward_features(self, x):
        # 保存 batch 大小，后续把 token 恢复成二维特征时使用。
        # 只读取第 0 维；输入的通道/空间合法性会由第一个 Conv2d 在运行时检查。
        B = x.shape[0]
        # 收集 x1、x2、x3、x4。
        # 每次 append 的对象都是 NCHW 张量，顺序固定从浅到深；列表长度最终必须为 4。
        outs = []

        # stage 1
        # 重叠 patch embedding：RGB -> embed_dims[0]，空间约降到 1/4；返回 token 和实际 H1、W1。
        # 352/B2：输入 `(B,3,352,352)`，得到 `x=(B,7744,64)`、H=88、W=88。
        x, H, W = self.patch_embed1(x)
        # 依次执行 stage1 的所有 Transformer blocks。
        # `enumerate` 同时给出索引 i 和 Block 对象 blk；当前函数只使用 blk，i 是保留的遍历信息。
        for i, blk in enumerate(self.block1):
            # 每个 block 保持 token 形状 (B,H1*W1,C1)。
            # 352/B2 时循环 3 次，每次输入输出都为 `(B,7744,64)`；内部数值和感受野会更新。
            x = blk(x, H, W)
        # stage1 最终 LayerNorm。
        # 沿最后一维 64 做归一化，形状仍为 `(B,7744,64)`。
        x = self.norm1(x)
        # token 恢复为 NCHW：(B,N,C1)->(B,H1,W1,C1)->(B,C1,H1,W1)。
        # `-1` 自动推导 C1；permute 后内存通常不连续，`.contiguous()` 创建连续布局供后续 Conv2d 高效读取。
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        # 保存 x1，供 EMCAD 最高分辨率 skip 使用。
        # 352/B2 此时 append 的 x1 是 `(B,64,88,88)`，包含较多边缘、纹理和局部细节。
        outs.append(x)

        # stage 2
        # x1 直接作为二维输入，经过 stride2 patch embedding 得到 stage2 token。
        # 这里复用变量名 x：上一行还是 NCHW，执行后变回 `(B,N2,C2)` token；要结合每行注释判断当前布局。
        x, H, W = self.patch_embed2(x)
        # 执行 stage2 blocks。
        # ModuleList 中的参数已注册并会随整个模型一起移动到 GPU、保存 checkpoint、参与优化。
        for i, blk in enumerate(self.block2):
            # 保持 (B,H2*W2,C2)。
            # 352/B2 时循环 4 次，形状一直为 `(B,1936,128)`。
            x = blk(x, H, W)
        # stage2 最终 LayerNorm。
        # 对 128 通道归一化，不跨 token 位置。
        x = self.norm2(x)
        # 恢复 x2=(B,C2,H2,W2)，空间约为输入 1/8。
        # 352/B2：`(B,1936,128)->(B,44,44,128)->(B,128,44,44)`。
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        # 保存 x2，供 EMCAD 第二条 skip 使用。
        # 相比 x1，x2 空间面积变为 1/4、通道翻倍，语义抽象程度更高。
        outs.append(x)

        # stage 3
        # stride2 patch embedding 得到 stage3 token，空间约为输入 1/16。
        # 352/B2：`(B,128,44,44)->(B,484,320)`，H=W=22。
        x, H, W = self.patch_embed3(x)
        # 执行 stage3 blocks。
        # B2 stage3 是 6 个 Block；B3/B4/B5 分别是 18/27/40 个，是各大变体深度差异的主要来源。
        for i, blk in enumerate(self.block3):
            # B2 在此循环 6 次，B3/B4/B5 更深。
            # 每次保持 `(B,484,320)`，其 Attention 用 5 个头，K/V 空间从 22x22 降成 11x11。
            x = blk(x, H, W)
        # stage3 最终 LayerNorm。
        # 对每个 token 的 320 维嵌入归一化。
        x = self.norm3(x)
        # 恢复 x3=(B,C3,H3,W3)。
        # 352/B2：`(B,484,320)->(B,320,22,22)`，contiguous 保证后续卷积使用连续内存。
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        # 保存 x3，供 EMCAD 最深的一条 skip 使用。
        # 解码器从 x4 第一次上采样到 22x22 时，会利用这个同尺度 x3 补充信息。
        outs.append(x)

        # stage 4
        # 最后一次 stride2 patch embedding，空间约为输入 1/32。
        # 352/B2：`(B,320,22,22)->(B,121,512)`，H=W=11。
        x, H, W = self.patch_embed4(x)
        # 执行 stage4 blocks；此阶段 sr_ratio=1，注意力不压缩 K/V。
        # 虽然是完整自注意力，但 N=121 很小，注意力矩阵规模只有 121x121/头。
        for i, blk in enumerate(self.block4):
            # 保持 (B,H4*W4,C4)。
            # B2 循环 3 次，形状始终 `(B,121,512)`，注意力头数为 8。
            x = blk(x, H, W)
        # stage4 最终 LayerNorm。
        # 对最深 token 的 512 通道归一化。
        x = self.norm4(x)
        # 恢复最深特征 x4=(B,C4,H4,W4)。
        # 352/B2：`(B,121,512)->(B,512,11,11)`。
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        # 保存 x4，它会进入 EMCAD 解码主路。
        # x4 分辨率最低、感受野和语义最强，是解码器逐级恢复目标结构的起点。
        outs.append(x)

        # 返回顺序 [x1,x2,x3,x4]；默认 B2 通道 [64,128,320,512]、尺度 [1/4,1/8,1/16,1/32]。
        # 返回的是 Python list，不是沿通道拼接的单个张量；四项空间尺寸不同，本来也不能直接 stack。
        return outs

        # return x.mean(dim=1)

    # 对外 forward 仅转调四阶段特征提取，不执行分类池化或分类 head。
    # `nn.Module.__call__` 会间接进入本方法，因此外部应写 `features = backbone(image)`，通常不直接手调 `forward`。
    def forward(self, x):
        # 获取四级 NCHW 特征列表。
        # 输入 x 应为浮点图像张量，通道数必须与构造时 in_chans 一致；EMCAD 主路径是 3。
        x = self.forward_features(x)
        # x = self.head(x)

        # 返回给 EMCADNet 解包为 x1,x2,x3,x4。
        # 变量名仍叫 x，但此时类型已经从 Tensor 变为含 4 个 Tensor 的 list。
        return x


# Mix-FFN 内的深度卷积：在不混合通道的前提下为 token 注入 3x3 局部空间关系。
# 【为什么单独封装成类】`Mlp` 的主要数据布局是 `(B,N,C)`，而 Conv2d 要求 `(B,C,H,W)`；
# 本类把“token 转图像 -> 卷积 -> 图像转 token”的固定流程集中起来，使 Mlp.forward 保持清晰。
# 【depth-wise 的含义】groups=dim 后，第 c 个输出通道只卷积第 c 个输入通道；
# 所以它负责同一通道内的局部空间混合，不负责通道间混合。通道间混合已经由前后的 fc1/fc2 完成。
class DWConv(nn.Module):
    # dim 等于 MLP 隐藏通道数。
    # B2 四阶段对应 dim=512、1024、1280、2048，而不是 stage 本身的 64、128、320、512。
    def __init__(self, dim=768):
        # 初始化基础模块。
        # 这里使用带类名的旧式 `super(DWConv, self)` 写法，与 Python 3 的 `super()` 效果相同。
        super(DWConv, self).__init__()
        # groups=dim 使每个通道单独执行 3x3 卷积，padding=1 保持 H、W。
        # 参数顺序依次是 in_channels、out_channels、kernel_size、stride、padding；两者都为 dim，空间步长为 1。
        # 每个通道仅有 3x3=9 个卷积权重，相比普通 dim->dim 的 3x3 卷积大幅减少参数和计算量。
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    # 输入 token (B,N,C)，N 必须等于 H*W。
    # 【调用者】只由 `Mlp.forward` 调用，传入的是 fc1 扩张后的隐藏表示。
    # 【输出约束】返回形状必须与输入完全一致，后面的 GELU、Dropout 和 fc2 才能继续按 token 运算。
    def forward(self, x, H, W):
        # 读取 token 形状。
        # N 虽然下文没有单独参与计算，但解包能明确输入必须为三维，并为形状检查提供语义。
        B, N, C = x.shape
        # (B,N,C)->(B,C,N)->(B,C,H,W)。
        # `transpose(1,2)` 把通道移到 Conv2d 所需位置；`view` 要求元素总数 B*C*H*W 等于 B*N*C。
        # 若上游 H/W 与 token 数不一致，这一行会立即报 shape 错误，避免悄悄使用错误空间邻接关系。
        x = x.transpose(1, 2).view(B, C, H, W)
        # 在每个隐藏通道内提取局部 3x3 上下文。
        # padding=1、stride=1，所以输出仍为 `(B,C,H,W)`；边缘位置使用零填充参与卷积。
        x = self.dwconv(x)
        # 恢复 (B,N,C)，供后续 GELU 和 fc2 使用。
        # flatten(2) 合并 H/W 为 N，transpose 再把 C 放回最后一维。
        x = x.flatten(2).transpose(1, 2)

        # 返回 token 布局。
        # 卷积改变数值和局部感受野，但不改变 B、N、C 三个形状维度。
        return x


# 旧式 checkpoint 兼容辅助：把手工 patchify 的线性投影权重改成卷积核形状。
# 【当前项目是否会调用】正常 EMCAD 编码器预训练加载路径没有调用 `_conv_filter`；它主要来自早期 PVT/timm 兼容实现。
# 【适用前提】旧 checkpoint 的目标键名必须包含精确子串 `patch_embed.proj.weight`，并且权重元素总数必须能
# 重排为 `(输出通道, 3, patch_size, patch_size)`。当前 PVTv2 有 patch_embed1/2/3/4 命名，不能假定都能匹配。
def _conv_filter(state_dict, patch_size=16):
    """ convert patch embedding weight from manual patchify + linear proj to conv"""
    # 新建输出字典，避免遍历时原地改写输入映射。
    # 浅复制参数映射时 Tensor 对象本身仍复用；只有命中的 v 会被 reshape 成新视图。
    out_dict = {}
    # 遍历 checkpoint 的全部参数。
    # `k` 是 state_dict 字符串键，`v` 通常是权重 Tensor，也可能是其他持久状态张量。
    for k, v in state_dict.items():
        # 只处理旧命名中的 patch embedding 投影权重。
        # 使用 `in` 子串判断而不是完全相等，是为了兼容可能存在的模型前缀，例如 `module.patch_embed...`。
        if 'patch_embed.proj.weight' in k:
            # 把二维 Linear 权重还原为 (C_out,3,patch_size,patch_size) 卷积权重。
            # 这里把输入通道硬编码为 3，只适合 RGB 的第一层；若旧权重不是 RGB 或元素数不符，reshape 会报错。
            v = v.reshape((v.shape[0], 3, patch_size, patch_size))
        # 无论是否转换，都保存到输出字典。
        # 键名不做替换，所以这个函数只解决权重形状，不解决新旧模型命名差异。
        out_dict[k] = v

    # 返回转换后的 state_dict；当前 EMCAD 权重加载路径没有调用此函数。
    # 返回后仍需由调用方显式执行 `model.load_state_dict(...)` 才会真正写入模型参数。
    return out_dict


# 注册 PVTv2-B0：tiny 版本，适配论文 PVT-EMCAD-B0。
# `register_model` 是类装饰器：模块导入时把该名字加入 timm 注册表，便于按字符串名称发现/创建模型。
# 本项目 `networks.py` 也可以直接通过 Python 类名实例化，所以注册行为不是前向计算的一部分。
@register_model
class pvt_v2_b0(PyramidVisionTransformerImpr):
    # kwargs 是上游注册接口的兼容参数；当前构造代码没有继续传给父类。
    # 这意味着调用 `pvt_v2_b0(img_size=352)` 等自定义参数并不会覆盖父类配置；kwargs 被接收后直接忽略。
    def __init__(self, **kwargs):
        # B0 通道 [32,64,160,256]、block 深度 [2,2,2,2]。
        # 下面显式调用父类构造器，创建四个 patch embedding、8 个 Block 和四个 stage norm。
        super(pvt_v2_b0, self).__init__(
            # 头数必须整除各阶段通道；前两阶段 MLP 扩张 8 倍。
            # 每头维度仍为 32：32/1、64/2、160/5、256/8；这与 B1-B5 每头 64 不同。
            # 此处 `patch_size=4` 会传入父类形参，但父类有效 patch embedding 核仍固定为 7/3/3/3。
            patch_size=4, embed_dims=[32, 64, 160, 256], num_heads=[1, 2, 5, 8], mlp_ratios=[8, 8, 4, 4],
            # 四阶段 K/V 空间压缩比例 [8,4,2,1]，最大 drop-path 为 0.1。
            # qkv_bias=True 让 Attention 的 Q 和联合 KV 投影都带可学习偏置；eps=1e-6 比 nn.LayerNorm 默认值更小。
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[2, 2, 2, 2], sr_ratios=[8, 4, 2, 1],
            # 普通 dropout 为 0，只使用逐层递增的随机深度。
            # drop_rate=0 不等于完全没有正则：drop_path_rate=0.1 仍会让较深残差分支训练时随机丢弃。
            drop_rate=0.0, drop_path_rate=0.1)


# 注册 PVTv2-B1：标准通道、较浅深度。
# B1 开始使用 `[64,128,320,512]` 标准四级通道，因此其输出接口与 B2-B5 相同，但 Block 更少。
@register_model
class pvt_v2_b1(PyramidVisionTransformerImpr):
    # 当前 kwargs 未透传，模型配置由下方常量固定。
    # 因此通过 kwargs 传 `drop_path_rate`、`in_chans`、`img_size` 等不会生效，这是阅读/二次开发时的重要限制。
    def __init__(self, **kwargs):
        # B1 通道 [64,128,320,512]，各 stage 两个 block。
        # 总 Block 数为 8，参数量和计算量低于 B2；输出尺度规律仍是约 1/4、1/8、1/16、1/32。
        super(pvt_v2_b1, self).__init__(
            # 头数 [1,2,5,8]，MLP ratios [8,8,4,4]。
            # 四阶段每头维度均为 64；隐藏通道分别 512、1024、1280、2048。
            patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[8, 8, 4, 4],
            # LayerNorm eps=1e-6，sr ratios 固定。
            # `partial` 先固定 eps，等父类调用 `norm_layer(dim)` 时再补入 normalized_shape=dim。
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[2, 2, 2, 2], sr_ratios=[8, 4, 2, 1],
            # dropout 和 drop-path 配置。
            # 最大随机深度概率仍为 0.1，并在线性概率表中分配给 8 个 Block。
            drop_rate=0.0, drop_path_rate=0.1)


# 注册 PVTv2-B2：本项目和论文 PVT-EMCAD-B2 的默认标准编码器。
# `lib/networks.py` 在选择 PVTv2-B2 时直接构造本类；预训练文件中的 encoder 参数也应与这套固定规格匹配。
# 对 352 输入，最终返回通道/尺寸依次为 `[64x88x88, 128x44x44, 320x22x22, 512x11x11]`（省略 B）。
@register_model
class pvt_v2_b2(PyramidVisionTransformerImpr):
    # 当前 kwargs 未透传。
    # 即使外部传入参数，下面仍只使用写死的父类实参；若要支持定制，需要改实现，但本次仅添加注释不改代码。
    def __init__(self, **kwargs):
        # B2 通道 [64,128,320,512]，深度 [3,4,6,3]。
        # 总 Block 数 16，是默认模型最需要记住的层数分布；stage3 的 6 个 Block 占主要编码深度。
        super(pvt_v2_b2, self).__init__(
            # 多头和 MLP 配置。
            # num_heads 与通道相除都得到每头 64 维；前两 stage 的 MLP 扩张 8 倍，后两 stage 扩张 4 倍。
            patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[8, 8, 4, 4],
            # sr ratios 和随机深度配置。
            # qkv_bias=True，LayerNorm eps=1e-6；sr 逐层减半，使 352 输入下各 stage 的 K/V 都约为 11x11。
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 4, 6, 3], sr_ratios=[8, 4, 2, 1],
            # 普通 dropout 为 0，最大 drop-path 0.1。
            # 16 个 Block 的 drop-path 概率由 0 线性增加到 0.1；推理 eval 模式下 DropPath 不随机丢弃。
            drop_rate=0.0, drop_path_rate=0.1)


# 注册 PVTv2-B3：主要把第三阶段从 6 个 block 增加到 18 个。
# 由于通道接口与 B2 完全相同，理论上 EMCAD 解码器通道配置无需改变；但参数量、显存和预训练权重必须匹配 B3。
@register_model
class pvt_v2_b3(PyramidVisionTransformerImpr):
    # 当前 kwargs 未透传。
    # 自定义 kwargs 被忽略的限制与 B0-B2 相同。
    def __init__(self, **kwargs):
        # B3 深度 [3,4,18,3]，通道接口仍与 B2 相同。
        # 总 Block 数为 28；相较 B2 多出的 12 个 Block 全部位于 1/16 尺度的 stage3。
        super(pvt_v2_b3, self).__init__(
            # 多头和 MLP 配置。
            # 通道、头数、MLP 扩张倍率均保持 B2 规格，所以单个 Block 的形状规则相同。
            patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[8, 8, 4, 4],
            # sr ratios 保持 [8,4,2,1]。
            # 28 个 Block 的随机深度概率仍从 0 到 0.1，只是相邻 Block 之间的概率增量更小。
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 4, 18, 3], sr_ratios=[8, 4, 2, 1],
            # dropout 配置。
            # 普通 dropout 为零，注意力 dropout 也沿用父类默认零。
            drop_rate=0.0, drop_path_rate=0.1)


# 注册 PVTv2-B4：进一步加深 stage2 和 stage3。
# B4 仍输出标准四级通道，因此“能否接 EMCAD 解码器”与“是否有匹配 B4 预训练权重”是两个独立问题。
@register_model
class pvt_v2_b4(PyramidVisionTransformerImpr):
    # 当前 kwargs 未透传。
    # 不能依赖 timm 常见的 `pretrained=True` kwargs 在这里自动加载权重；本类不会读取它。
    def __init__(self, **kwargs):
        # B4 深度 [3,8,27,3]，四级输出通道不变。
        # 总 Block 数 41；相较 B3，stage2 和 stage3 都进一步加深，训练/推理开销更大。
        super(pvt_v2_b4, self).__init__(
            # 多头和 MLP 配置。
            # 单个 stage 的维度、注意力头和 Mix-FFN 隐藏宽度仍与 B2/B3 相同。
            patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[8, 8, 4, 4],
            # LayerNorm、深度和 sr ratios。
            # `depths` 只改变 Block 重复数，不改变四张输出特征图的通道数和理论空间尺度。
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 8, 27, 3], sr_ratios=[8, 4, 2, 1],
            # dropout 配置。
            # 41 个 Block 的 drop-path 从 0 线性到 0.1。
            drop_rate=0.0, drop_path_rate=0.1)


# 注册 PVTv2-B5：本文件中最深变体，stage3 有 40 个 block。
# B5 不仅更深，还把前两 stage 的 MLP ratio 从 8 改为 4，因此“更大”不代表每个局部模块都更宽。
@register_model
class pvt_v2_b5(PyramidVisionTransformerImpr):
    # 当前 kwargs 未透传。
    # 与其他变体一样，构造参数由本类内部固定，外部 kwargs 目前只是为了兼容注册器调用签名。
    def __init__(self, **kwargs):
        # B5 保持标准通道，但把所有 stage 的 MLP ratio 设为 4。
        # 隐藏通道变为 256、512、1280、2048；前两 stage 比 B1-B4 的 512、1024 更窄。
        super(pvt_v2_b5, self).__init__(
            # 通道、头数和 MLP ratios。
            # 四级输出通道仍为 [64,128,320,512]，所以形状接口不变。
            patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4],
            # 深度 [3,6,40,3]，sr ratios 不变。
            # 总 Block 数 52，其中 stage3 占 40 个，是本文件计算最重的规格。
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 6, 40, 3], sr_ratios=[8, 4, 2, 1],
            # dropout 配置。
            # 随机深度最大值仍为 0.1，普通 dropout 保持 0。
            drop_rate=0.0, drop_path_rate=0.1)
