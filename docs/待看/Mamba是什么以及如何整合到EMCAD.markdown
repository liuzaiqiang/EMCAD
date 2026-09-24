# Mamba 是什么，以及如何整合到 EMCAD

## 一、先给结论

Mamba 是一种选择性状态空间模型（Selective State Space Model, SSM）。它不是普通卷积，也不是把 Transformer 的注意力矩阵换个名字。

它主要解决的问题是：

```text
当序列很长时，Self-Attention 的计算量和显存通常随序列长度平方增长，
而 Mamba 用线性扫描和输入相关的状态更新，在更低的复杂度下建模长距离依赖。
```

Mamba 可以接入 EMCAD，但当前仓库没有 Mamba 代码和依赖。当前有效结构是：

```text
单通道 CT
  -> 1->3 通道适配器
  -> PVTv2 编码器
  -> 四级特征 x1,x2,x3,x4
  -> EMCAD 解码器
  -> 四个分割头
```

最推荐的整合方式是：

> 保留 PVTv2 和 EMCAD，只在最深层低分辨率特征 `x4` 或解码器最深层 `d4` 插入一个 Mamba block。

不建议一开始就把整个 PVTv2 编码器替换成 Mamba 视觉骨干，因为那会同时改变预训练权重、四级特征接口、通道数、空间扫描方式和复杂度统计，调试成本远高于一个小模块。

## 二、Mamba 到底是什么

### 1. 从普通状态空间模型说起

状态空间模型用一个隐藏状态 `h(t)` 压缩过去的信息：

```text
h'(t) = A h(t) + B x(t)
y(t)  = C h(t) + D x(t)
```

直观理解：

- `x(t)` 是当前输入；
- `h(t)` 是到目前为止保留的历史摘要；
- `A` 决定历史信息如何衰减或传播；
- `B` 决定当前输入如何写入状态；
- `C` 决定从状态读出什么；
- `D` 是当前输入到输出的直接通路。

离散化以后，可以逐位置更新：

```text
h_t = A_bar h_{t-1} + B_bar x_t
y_t = C h_t + D x_t
```

这类模型的优点是不用显式构造所有位置之间的两两注意力矩阵。

### 2. Mamba 增加了“选择性”

传统 SSM 的参数通常固定。Mamba 让部分状态更新参数依赖当前输入：

```text
Δ_t = f_Δ(x_t)
B_t = f_B(x_t)
C_t = f_C(x_t)
```

因此模型可以根据内容决定：

- 当前信息要不要写入状态；
- 历史信息保留多久；
- 当前输出应该读取哪部分状态。

这就是“selective”的含义。它不是对所有位置使用同一套固定记忆规则。

### 3. 为什么叫 Mamba

Mamba 通常指“选择性 SSM + 硬件友好的扫描实现 + 门控/投影结构”的组合。核心不是某一个普通 `nn.Linear`，而是让状态更新能够高效地在长序列上运行。

可以把 Transformer 和 Mamba 做一个工程化对比：

| 方面 | Self-Attention | Mamba/Selective SSM |
|---|---|---|
| 位置交互 | 显式计算位置两两关系 | 通过状态扫描传播历史 |
| 典型序列复杂度 | 随长度近似二次增长 | 随长度近似线性增长 |
| 长序列显存 | 注意力矩阵可能很大 | 不保存完整两两矩阵 |
| 信息选择 | 注意力权重选择 | 输入相关状态写入/读取 |
| 视觉使用 | 直接适合 token 注意力 | 需要设计二维扫描或展平策略 |
| 预训练生态 | Transformer 视觉模型成熟 | 视觉 Mamba 依赖具体实现和权重 |

复杂度是“典型理论口径”，实际速度仍取决于实现、GPU、序列长度、kernel 和 batch size。不能只看 O(L) 就断言 Mamba 一定比 PVTv2 快。

## 三、Mamba 解决了什么问题

### 1. 长距离依赖的计算代价

图像分割中，远处区域可能帮助判断当前器官边界。例如一个局部区域很模糊，整幅 CT 中的器官形状和上下文可能有帮助。

PVTv2 已经通过空间降采样降低了注意力成本，但较长 token 序列仍然会带来计算和显存开销。Mamba 的目标是用状态传播建模更长范围的上下文，而不显式建立完整的 token-to-token 矩阵。

### 2. 固定记忆规则不足

医学图像中既有均匀器官内部，也有边界、伪影和复杂组织。所有位置使用同样的历史融合规则并不理想。

Mamba 的选择性状态更新允许模型对不同位置保留不同程度的上下文，但这只是机制上的可能性，是否改善 Dice/HD95 必须通过实验验证。

### 3. 不是专门解决边界问题的模块

Mamba 本身不会自动产生更准确的边界，也不会自动理解医学器官。它主要提供长程上下文和内容相关状态更新。

如果你的 baseline 主要问题是边界毛刺，加入 Mamba 可能没有直接帮助；如果问题是大范围结构不一致、远距离上下文不足，Mamba 才更有理由尝试。

## 四、当前 EMCAD 的接口为什么允许接 Mamba

当前 `lib/networks.py` 的 `EMCADNet` 负责组装：

```text
backbone -> [x1,x2,x3,x4] -> decoder(x4, [x3,x2,x1])
```

PVTv2-B2 的典型特征形状为：

```text
x1: [B,  64, H/4,  W/4]
x2: [B, 128, H/8,  W/8]
x3: [B, 320, H/16, W/16]
x4: [B, 512, H/32, W/32]
```

以 Synapse 的 `224x224` 输入为例：

```text
x1: [B,  64, 56, 56]
x2: [B, 128, 28, 28]
x3: [B, 320, 14, 14]
x4: [B, 512,  7,  7]
```

只要 Mamba block 满足：

```text
[B, C, H, W] -> [B, C, H, W]
```

就可以放在 `x4` 后面，EMCAD 解码器的输入通道和空间尺寸都不变。

## 五、三种整合方式及推荐等级

### 方案 A：在 x4 后插入 Mamba block，推荐

数据流：

```text
x4 [B,512,7,7]
  -> Mamba block
  -> x4' [B,512,7,7]
  -> 原始 EMCAD decoder
```

优点：

- 不改变四级特征接口；
- 不影响 `x1/x2/x3` skip；
- 7×7 token 数只有 49，显存和调试压力较小；
- PVTv2 预训练权重仍然可以加载；
- EMCAD 解码器可以保持不变；
- 消融只需要比较“无 Mamba”和“有 Mamba”。

缺点：

- 7×7 序列本身并不长，Mamba 的长序列优势未必充分发挥；
- 它可能更像一个全局上下文混合器，而不是显著的效率创新。

### 方案 B：在 x3 或 x2 后插入 Mamba，次选

例如 x3：

```text
x3 [B,320,14,14] -> Mamba -> x3' [B,320,14,14]
```

优点是 token 数更多，上下文建模更有机会影响中等尺度结构。

风险是：

- 计算量和显存明显增加；
- skip 特征被改变，可能影响 LGAG；
- 对 batch size=16 的 Synapse 训练更容易爆显存；
- 需要严格重新测量 FLOPs、显存和时间。

### 方案 C：用 Mamba 彻底替换 PVTv2，暂不推荐

数据流变成：

```text
Mamba/VMamba encoder -> 四级特征 -> EMCAD decoder
```

这不是简单替换一行 import。新的 encoder 必须提供：

```text
[x1,x2,x3,x4]
```

并且通道、空间层级和预训练权重都要与 EMCAD 对齐。通常需要：

- 2D/四方向扫描设计；
- patch embedding 和 stage 划分；
- 通道适配层；
- 新预训练权重或从头训练；
- 新的 Params/FLOPs 统计；
- 重新调 batch、学习率和训练轮数。

这更像一个新 backbone 论文，不适合你当前“快速验证一个点”的目标。

## 六、为什么不能直接把二维特征 reshape 后就宣布用了视觉 Mamba

Mamba 原生处理的是序列。二维特征 `[B,C,H,W]` 要变成 token 序列，例如：

```python
tokens = feature.flatten(2).transpose(1, 2)
# [B,C,H,W] -> [B,H*W,C]
```

处理后再还原：

```python
feature = tokens.transpose(1, 2).reshape(B, C, H, W)
```

但简单按行展平有两个问题：

1. 图像上下左右邻域在序列中不一定都相邻；
2. 扫描顺序会造成方向偏置。

因此视觉 Mamba 常需要：

- 横向和纵向扫描；
- 正向和反向扫描；
- 多方向结果融合；
- 或者专门的二维状态空间算子。

如果你使用的是现成视觉 Mamba 实现，论文必须写清楚扫描方向和特征重排方式。不要把普通的 `flatten -> 一次 Mamba -> reshape` 直接写成完整二维建模。

## 七、当前仓库接入 Mamba 的实际前置条件

当前 `requirements.txt` 中没有 `mamba-ssm`，也没有任何 `Mamba` 类。接入前要确认：

- Linux 服务器是否有可用 CUDA；
- PyTorch 版本是否与 Mamba 包兼容；
- 是否有编译工具链、CUDA toolkit、ninja 等；
- 当前 GPU 是否支持对应 kernel；
- 包能否在你的训练环境中正常 import 和 forward。

建议先在服务器做独立检查，不要直接把依赖塞进正式实验：

```bash
python - <<'PY'
import torch
print('torch:', torch.__version__)
print('cuda available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('gpu:', torch.cuda.get_device_name(0))
try:
    import mamba_ssm
    print('mamba_ssm: import ok')
except Exception as exc:
    print('mamba_ssm import failed:', repr(exc))
PY
```

如果安装过程要求编译 CUDA 扩展，先在独立环境测试。不要因为安装失败就把一个普通卷积模块命名成 Mamba。

## 八、推荐的最小实现接口

如果使用兼容 PyTorch 的 Mamba/视觉 Mamba block，建议让它提供如下接口：

```python
class Mamba2DAdapter(nn.Module):
    def __init__(self, channels, ...):
        super().__init__()
        # 建议在这里完成 LayerNorm、Mamba mixer、投影和残差结构。

    def forward(self, feature):
        # 输入和输出都保持 [B,C,H,W]。
        return feature
```

更完整的结构可以是：

```python
class Mamba2DAdapter(nn.Module):
    def __init__(self, channels, hidden_dim):
        super().__init__()
        self.norm = nn.LayerNorm(channels)
        self.in_proj = nn.Linear(channels, hidden_dim * 2)
        self.mixer = ...  # 使用已验证的 Mamba/视觉 Mamba 实现
        self.out_proj = nn.Linear(hidden_dim, channels)

    def forward(self, feature):
        b, c, h, w = feature.shape
        tokens = feature.flatten(2).transpose(1, 2)  # [B,HW,C]
        residual = tokens
        tokens = self.norm(tokens)
        tokens = self.in_proj(tokens)
        tokens = self.mixer(tokens)
        tokens = self.out_proj(tokens)
        tokens = tokens + residual
        return tokens.transpose(1, 2).reshape(b, c, h, w)
```

这里的 `self.mixer` 不能用省略号直接训练；必须替换成实际可用且经过单元测试的 Mamba 实现。对于正式论文，不建议临时手写一个“看起来像状态空间”的递推模块来冒充 Mamba。

## 九、在 `lib/networks.py` 中的接入位置

当前 `EMCADNet.__init__()` 会创建：

```python
self.backbone = pvt_v2_b2()
self.decoder = EMCAD(...)
```

推荐新增：

```python
self.mamba_x4 = Mamba2DAdapter(channels=channels[0], hidden_dim=256)
```

然后在 `forward()` 中找到：

```python
x1, x2, x3, x4 = self.backbone(x)
dec_outs = self.decoder(x4, [x3, x2, x1])
```

改成：

```python
x1, x2, x3, x4 = self.backbone(x)
x4 = self.mamba_x4(x4)
dec_outs = self.decoder(x4, [x3, x2, x1])
```

这样 EMCAD 的其余路径不变。

### 必须同时提供关闭开关

为了做公平消融，建议增加：

```python
use_mamba=False
```

并在构造时：

```python
self.mamba_x4 = (
    Mamba2DAdapter(channels=channels[0], hidden_dim=256)
    if use_mamba else nn.Identity()
)
```

这样：

```text
use_mamba=False -> 原始 EMCAD 路径
use_mamba=True  -> PVTv2 + x4 Mamba + EMCAD
```

不要通过删除代码临时切换，因为每次手工删除都可能改变参数统计或加载路径。

## 十、最小实验矩阵

### 第一组：结构有效性

```text
M0：原始 EMCAD
M1：PVTv2 + x4 Mamba + EMCAD
M2：PVTv2 + x4 普通 Transformer/卷积混合块 + EMCAD
```

M2 的作用是判断提升是否只是“多加了一个上下文模块”，而不是 Mamba 的选择性状态机制带来的。

### 第二组：位置消融

```text
P0：不加 Mamba
P1：只在 x4 加 Mamba
P2：只在 x3 加 Mamba
P3：x4+x3 都加 Mamba
```

如果 P3 只增加显存和时间却没有明显收益，论文应保留 P1。

### 第三组：效率和质量

每个模型至少记录：

- Dice；
- HD95；
- ASSD/ASD；
- IoU/Jaccard；
- 逐器官结果；
- Total/Trainable Params；
- MACs/FLOPs；
- 每 epoch 训练时间；
- 最终测试时间；
- 峰值显存。

Mamba 不是只要 Dice 上升就算成功。如果 Dice 上升 `0.05` 个百分点，但 FLOPs、显存和时间翻倍，这不是一个有说服力的 EMCAD 改进。

## 十一、Mamba 是否适合成为你的论文创新点

### 适合的情况

- baseline 在大范围结构一致性或远距离上下文上失败；
- 小器官和复杂形状的错误具有明显全局性；
- x4 Mamba 能在相近参数量下改善 Dice/HD95；
- 两个数据集方向一致；
- 能完成 Mamba 与普通上下文模块的对照；
- 你能稳定安装、训练和复现依赖。

### 不适合的情况

- 只是因为 Mamba 流行，所以机械加入；
- 只在一个 seed 上提升；
- 只报告总体 Dice，不报告边界和效率；
- Mamba 包安装和 CUDA 编译不稳定；
- 换了 Mamba encoder 后无法公平加载预训练权重；
- 训练时间和显存明显增加，却没有可解释收益。

## 十二、最实用的决策

如果你的目标是尽快验证，而不是重写整个模型，按这个顺序：

1. 保留 PVTv2-B2 和原始 EMCAD；
2. 在 `x4 [B,512,7,7]` 后插入一个保持 shape 不变的 Mamba block；
3. 提供 `use_mamba` 开关；
4. 先跑原始 EMCAD、Mamba、普通上下文块三组；
5. 记录 Dice、HD95、Params、FLOPs、时间、显存；
6. 只有 Mamba 在质量或效率上有稳定优势，才扩展到第二数据集和多 seed；
7. 不要一开始替换整个 PVTv2 编码器。

一句话总结：

> Mamba 解决的是长序列建模的计算和记忆问题；它可以接入 EMCAD，但对当前项目最合理的入口是低分辨率深层特征，而不是直接重写整个编码器。

