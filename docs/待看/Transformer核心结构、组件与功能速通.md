# Transformer 核心结构、组件与功能速通

> 适用对象：已经掌握张量、全连接层、Softmax、反向传播和基本深度学习训练流程，希望快速建立 Transformer 完整心智模型的人。
>
> 阅读目标：读完后，你应能看懂标准 Transformer 结构图，解释每个模块为什么存在，跟踪主要张量形状，并区分 Encoder、Decoder、BERT、GPT、ViT 等常见结构。

---

## 一、先用一句话理解 Transformer

Transformer 的核心工作可以压缩成一句话：

> **让序列中的每个位置根据当前任务，动态地从其他位置读取信息，然后对读到的信息做非线性变换。**

一个 Transformer Block 主要只做两件事：

```text
1. Attention：不同位置之间交换、聚合信息
2. FFN：每个位置独立地加工已经聚合的信息
```

再加上两个保证深层网络可训练的结构：

```text
3. Residual Connection：保留原信息，改善梯度传播
4. Layer Normalization：稳定各层输入分布和训练过程
```

所以最重要的心智模型不是“Transformer 很复杂”，而是：

```text
Transformer Block = 信息通信层 Attention + 信息加工层 FFN
                    + 残差连接 + LayerNorm
```

很多大型模型的本质，就是把这种 Block 堆叠很多次，并改变 Attention 能看见哪些位置。

---

## 二、一眼看懂整体数据流

先以文本为例。输入一句话，标准流程如下：

```text
原始文本
  -> Tokenizer 分词并转成 token ID
  -> Token Embedding 查表得到向量
  -> 加入位置信息
  -> N 个 Transformer Block
  -> 输出头 Output Head
  -> 分类结果、序列表示或下一个 token 概率
```

统一使用以下形状符号：

| 符号 | 含义 |
| --- | --- |
| `B` | Batch size |
| `L` | 序列长度，即 token 数量 |
| `d_model` | Transformer 主干的特征维度 |
| `h` | 注意力头数 |
| `d_head` | 每个头的维度，通常 `d_model / h` |
| `d_ff` | FFN 中间隐藏维度，常大于 `d_model` |
| `V` | 词表大小 |

主干张量通常保持：

```text
X: [B, L, d_model]
```

这点非常重要：一个标准 Block 输入是 `[B, L, d_model]`，输出通常还是 `[B, L, d_model]`。这样多个 Block 才能直接堆叠。

---

## 三、组件总表：先记住每个模块负责什么

| 模块 | 输入 -> 输出 | 核心功能 |
| --- | --- | --- |
| Tokenizer | 文本 -> token ID | 把原始文本离散化 |
| Token Embedding | `[B,L] -> [B,L,d_model]` | 把离散 ID 映射成连续向量 |
| Positional Information | `[B,L,d_model] -> 同形状` | 注入顺序和相对位置 |
| Q/K/V Projection | `X -> Q,K,V` | 为“查询、匹配、传递内容”生成不同表示 |
| Attention Score | `QK^T` | 计算位置之间的相关性 |
| Mask | score -> masked score | 限制哪些位置可以被看见 |
| Softmax | score -> 权重 | 把相关性归一化成注意力分布 |
| Weighted Sum | `A V` | 按权重汇总其他位置的信息 |
| Multi-Head Attention | 多组 Attention -> 合并 | 在不同表示子空间捕捉不同关系 |
| FFN/MLP | `[B,L,d_model] -> 同形状` | 对每个位置做独立非线性特征变换 |
| Residual | `x + sublayer(x)` | 保留原特征并改善梯度传播 |
| LayerNorm | 同形状 -> 同形状 | 稳定特征尺度和深层训练 |
| Output Head | 隐状态 -> 任务输出 | 转成分类、词表概率或其他预测 |

如果只记一条：

```text
Attention 混合 token 维度的信息；FFN 混合 channel/feature 维度的信息。
```

---

## 四、输入部分：Tokenizer、Embedding 和位置

### 4.1 Tokenizer：把原始文本转成离散 ID

Tokenizer 不属于 Transformer 神经网络主体，但决定模型实际看到的基本单位。

例如：

```text
"transformers are powerful"
  -> [token_1, token_2, token_3, token_4]
  -> [1842, 231, 962, 18]
```

现代模型通常使用子词或字节级 token，而不是简单按完整单词切分。Tokenizer 输出的 ID 本身没有大小关系，`1842` 并不比 `231` 更“高级”，只是词表中的索引。

### 4.2 Token Embedding：ID 查表得到向量

设词表大小为 `V`，Embedding 矩阵为：

```text
E: [V, d_model]
```

输入 token ID：

```text
input_ids: [B, L]
```

查表后得到：

```text
X_token: [B, L, d_model]
```

Embedding 本质上是可训练查表。训练过程中，语义或使用方式相近的 token 往往会形成有结构的向量表示。

### 4.3 为什么必须加入位置信息

纯 Self-Attention 本身不天然区分顺序。如果同时对输入位置做同样的置换，输出也会跟着置换。换句话说，只给词向量时，模型没有充分信息区分：

```text
“狗咬人” 和 “人咬狗”
```

因此必须注入位置。

常见方式包括：

- 绝对位置编码：给第 0、1、2... 个位置一个向量；
- 可学习位置向量：位置表示作为参数训练；
- 相对位置偏置：直接影响两个位置之间的注意力分数；
- RoPE：通过旋转 Q 和 K，把相对位置信息编码进点积；
- ALiBi：按位置距离给注意力分数加入线性偏置。

原始 Transformer 使用正弦、余弦绝对位置编码。现代大语言模型常使用 RoPE。

最简单的输入形式是：

```text
X = TokenEmbedding + PositionEmbedding
X: [B, L, d_model]
```

注意：位置编码不是告诉模型“这个词是什么意思”，而是告诉模型“这个表示处于什么位置，以及与其他位置相距多远”。

---

## 五、Self-Attention：Transformer 的核心

### 5.1 Q、K、V 到底是什么

给定输入：

```text
X: [B, L, d_model]
```

通过三个不同的线性投影：

```text
Q = X W_Q
K = X W_K
V = X W_V
```

可以这样理解：

| 名称 | 含义 | 问的问题 |
| --- | --- | --- |
| Query，Q | 当前 token 想寻找什么 | “我需要什么信息？” |
| Key，K | 每个 token 用什么特征供别人匹配 | “我能以什么条件被找到？” |
| Value，V | 每个 token 真正提供的内容 | “如果你关注我，我把什么信息给你？” |

关键点：Q、K、V 不是三份固定语义的数据，而是从同一个输入通过不同参数学习出的三种角色表示。

### 5.2 注意力公式

标准缩放点积注意力：

```math
Attention(Q,K,V)=softmax\left(\frac{QK^T}{\sqrt{d_{head}}}+M\right)V
```

逐步拆开：

### 第一步：计算相关性

```math
S = QK^T
```

对一个头而言：

```text
Q: [B, h, L_q, d_head]
K: [B, h, L_k, d_head]
K^T: [B, h, d_head, L_k]

S: [B, h, L_q, L_k]
```

`S[i,j]` 表示第 `i` 个查询位置对第 `j` 个键位置的匹配程度。

在 Self-Attention 中通常 `L_q = L_k = L`，因此每个头产生一个 `L × L` 注意力矩阵。

### 第二步：除以 `sqrt(d_head)`

```math
S_{scaled} = \frac{S}{\sqrt{d_{head}}}
```

维度增大时，点积的数值幅度通常会变大。若直接送入 Softmax，分布容易过度尖锐，梯度变小。缩放用于控制数值范围，使训练更稳定。

### 第三步：加入 Mask

```math
S_{masked} = S_{scaled} + M
```

不能被关注的位置加上一个极小值，逻辑上近似负无穷；Softmax 后权重接近 0。

### 第四步：Softmax 得到注意力权重

```math
A = softmax(S_{masked})
```

```text
A: [B, h, L_q, L_k]
```

Softmax 通常沿最后一个维度 `L_k` 进行，因此对于每个查询位置，它对所有可见键位置的权重和为 1。

### 第五步：对 V 加权求和

```math
O = AV
```

```text
A: [B, h, L_q, L_k]
V: [B, h, L_k, d_head]
O: [B, h, L_q, d_head]
```

这一步才是真正的信息读取：相关性由 Q 和 K 决定，被汇总的内容来自 V。

### 5.3 用一个 token 理解完整过程

假设句子是：

```text
“小明把书放在桌上，因为它很重。”
```

处理“它”时：

1. “它”的 Q 表示当前需要寻找指代对象；
2. “书”“桌”等 token 的 K 提供可匹配特征；
3. Q 与各 K 点积，得到匹配分数；
4. Softmax 后，“书”可能得到较高权重；
5. 使用这些权重加权汇总所有 V；
6. “它”的新表示中融入了“书”的上下文信息。

注意力不是一个人工规定的语法规则。模型通过任务损失，学习什么情况下应该关注哪些位置。

---

## 六、Multi-Head Attention：为什么要多头

### 6.1 单头的限制

单头 Attention 只在一组 Q/K/V 表示空间中建立关系。实际序列可能同时包含：

- 局部邻近关系；
- 长距离依赖；
- 指代关系；
- 句法关系；
- 实体关系；
- 位置或结构关系。

多头允许模型在多个子空间中并行学习不同的匹配方式。

### 6.2 形状变化

输入：

```text
X: [B, L, d_model]
```

线性投影后仍可先表示为：

```text
Q, K, V: [B, L, d_model]
```

拆成 `h` 个头：

```text
[B, L, d_model]
  -> [B, L, h, d_head]
  -> transpose
  -> [B, h, L, d_head]
```

每个头独立计算 Attention：

```text
head_i: [B, L, d_head]
```

合并所有头：

```text
[B, h, L, d_head]
  -> transpose
  -> [B, L, h, d_head]
  -> reshape
  -> [B, L, d_model]
```

最后再经过输出投影：

```math
MultiHead(X)=Concat(head_1,...,head_h)W_O
```

输出仍然是：

```text
[B, L, d_model]
```

### 6.3 多头不是把完整特征复制多份

当 `d_model = 768`、`h = 12` 时，通常：

```text
d_head = 768 / 12 = 64
```

即每个头处理 64 维子空间，12 个头合并后仍是 768 维。标准实现不是简单把总计算维度扩大 12 倍。

### 6.4 多头的正确理解

不要机械认为“某一个头一定学语法，另一个头一定学指代”。部分头确实会表现出可解释模式，但不同模型、层和训练过程并不保证固定分工。

更稳妥的理解是：

> 多头给模型提供多组独立的关系建模参数，使不同类型的依赖可以同时被表达。

---

## 七、Mask：决定模型允许看到什么

Mask 直接改变注意力的可见范围，因此也决定模型的使用方式。

### 7.1 Padding Mask

为了组成 Batch，不同长度的序列通常被补齐：

```text
[A, B, C, PAD, PAD]
[D, E, F, G, H]
```

PAD 只是占位，不应被其他 token 当作真实内容。Padding Mask 把这些位置的注意力分数屏蔽掉。

### 7.2 Causal Mask

自回归生成时，第 `t` 个 token 只能看到自己和过去，不能偷看未来：

```text
位置 0：可看 [0]
位置 1：可看 [0,1]
位置 2：可看 [0,1,2]
位置 3：可看 [0,1,2,3]
```

可见矩阵是下三角结构：

```text
1 0 0 0
1 1 0 0
1 1 1 0
1 1 1 1
```

GPT 的核心特征之一，就是 Decoder-only 主干中的 Self-Attention 使用 Causal Mask。

### 7.3 Mask 的本质

Mask 不是在 Attention 计算结束后删除结果，而是在 Softmax 前修改 score：

```text
原 score:       [2.1, 1.3, 0.8, 3.0]
屏蔽未来后:     [2.1, 1.3, -inf, -inf]
Softmax 权重:   [0.69, 0.31, 0, 0]
```

因此被屏蔽的位置不会参与加权求和。

---

## 八、FFN：每个 token 自己加工信息

Attention 完成 token 之间的信息交换后，FFN 对每个位置独立做非线性变换。

标准形式：

```math
FFN(x)=W_2\sigma(W_1x+b_1)+b_2
```

形状通常是：

```text
[B, L, d_model]
  -> Linear
[B, L, d_ff]
  -> Activation
[B, L, d_ff]
  -> Linear
[B, L, d_model]
```

例如：

```text
d_model = 768
d_ff = 3072
```

先升维，让模型在更大的特征空间中进行非线性组合，再降回主干维度。

### 8.1 为什么 Attention 后还需要 FFN

Attention 的强项是按相关性对其他 token 的 Value 做加权组合，本质上侧重位置间的信息路由。FFN 提供更强的逐位置非线性特征变换。

可以把两者分工记为：

```text
Attention：我应该从谁那里读取信息？
FFN：读到这些信息后，我应该怎样变换它？
```

### 8.2 FFN 参数往往很多

不要因为结构图里 FFN 只是一个小框，就认为它不重要。两层大矩阵通常占据 Transformer Block 中相当可观的参数量和计算量。

现代模型常使用 GELU、SiLU/Swish、SwiGLU 等激活或门控 FFN，而不只使用原始论文中的 ReLU。

---

## 九、Residual 和 LayerNorm：让深层堆叠可训练

### 9.1 Residual Connection

残差结构：

```math
y=x+Sublayer(x)
```

它有两个主要作用：

1. 保留原始表示，子层只需学习增量修正；
2. 提供更直接的梯度传播路径，使深层网络更容易优化。

因为子层输出要与输入相加，所以 Attention 和 FFN 最终都必须回到 `d_model` 维。

### 9.2 Layer Normalization

LayerNorm 通常在每个 token 内，沿特征维 `d_model` 做归一化：

```text
输入: [B, L, d_model]
对每个 [d_model] 向量独立归一化
输出: [B, L, d_model]
```

它不依赖整个 Batch 的统计量，因此比 BatchNorm 更自然地适合变长序列、小 Batch 和自回归推理。

### 9.3 Post-Norm 与 Pre-Norm

原始 Transformer 常写成 Post-Norm：

```text
y = LayerNorm(x + Sublayer(x))
```

现代深层 Transformer 更常见 Pre-Norm：

```text
y = x + Sublayer(LayerNorm(x))
```

一个 Pre-Norm Encoder Block 可以写成：

```text
x = x + SelfAttention(LayerNorm(x))
x = x + FFN(LayerNorm(x))
```

Pre-Norm 通常让深层网络的梯度传播更稳定。阅读代码时，一定先看 Norm 在子层之前还是之后，不要只看模块名称。

---

## 十、完整 Encoder Block

标准 Encoder Layer 包含：

```text
输入 X
  -> Multi-Head Self-Attention
  -> Residual + LayerNorm
  -> FFN
  -> Residual + LayerNorm
  -> 输出
```

在 Pre-Norm 写法中：

```python
x = x + self_attention(layer_norm_1(x), mask=padding_mask)
x = x + ffn(layer_norm_2(x))
```

Encoder Self-Attention 的 Q、K、V 都来自当前输入 `X`：

```text
Q = XW_Q
K = XW_K
V = XW_V
```

如果没有特殊 Mask，每个位置都能双向看到全部有效位置。因此 Encoder 擅长产生“理解型”上下文表示。

堆叠 `N` 层后：

```text
X_0 -> Encoder Layer 1 -> X_1
    -> Encoder Layer 2 -> X_2
    ...
    -> Encoder Layer N -> X_N
```

浅层可能更偏局部和表面模式，深层逐渐组合出更抽象、任务相关的表示，但这不是严格固定分工。

---

## 十一、完整 Decoder Block

原始 Encoder-Decoder Transformer 的 Decoder Layer 包含三个子层：

```text
目标序列输入
  -> Masked Self-Attention
  -> Residual + Norm
  -> Cross-Attention
  -> Residual + Norm
  -> FFN
  -> Residual + Norm
  -> 输出
```

### 11.1 Masked Self-Attention

Decoder 在生成第 `t` 个位置时不能读取未来目标 token，因此使用 Causal Mask。

Q、K、V 都来自 Decoder 当前隐藏状态：

```text
Q = X_decoder W_Q
K = X_decoder W_K
V = X_decoder W_V
```

### 11.2 Cross-Attention

Cross-Attention 让 Decoder 读取 Encoder 的输出：

```text
Q：来自 Decoder
K：来自 Encoder 输出
V：来自 Encoder 输出
```

即：

```text
Q = X_decoder W_Q
K = H_encoder W_K
V = H_encoder W_V
```

它回答的问题是：

> Decoder 当前要生成这个 token 时，应该从输入序列的哪些位置读取信息？

Cross-Attention 与 Self-Attention 使用相同公式，区别只是 Q/K/V 的来源不同。

---

## 十二、三种主流结构：Encoder-only、Decoder-only、Encoder-Decoder

| 结构 | 可见范围 | 典型任务 | 代表 |
| --- | --- | --- | --- |
| Encoder-only | 双向看完整输入 | 分类、检索、标注、表征学习 | BERT 类 |
| Decoder-only | 只看当前及历史 token | 自回归生成、对话、代码生成 | GPT 类 |
| Encoder-Decoder | Encoder 双向理解，Decoder 自回归生成并跨注意力读取输入 | 翻译、摘要、条件生成 | 原始 Transformer、T5 类 |

### 12.1 BERT 为什么适合理解任务

BERT 使用 Encoder-only。一个 token 可以同时利用左侧和右侧上下文，因此适合：

- 文本分类；
- 命名实体识别；
- 句子匹配；
- 检索编码；
- Masked Language Modeling。

它不是按从左到右的方式直接生成长文本。

### 12.2 GPT 为什么可以生成文本

GPT 使用 Decoder-only，但通常没有原始 Encoder-Decoder 结构中的 Cross-Attention。每层主要是：

```text
Causal Self-Attention + FFN
```

它训练的核心目标是：

```math
P(x_t | x_1,x_2,...,x_{t-1})
```

也就是根据过去 token 预测下一个 token。

### 12.3 Encoder-Decoder 为什么适合条件生成

Encoder 先充分理解输入，Decoder 一边读取已经生成的目标 token，一边通过 Cross-Attention 读取 Encoder 表示。这种结构天然适合“输入一个序列，输出另一个序列”。

---

## 十三、训练与推理：最容易混淆的一处

### 13.1 自回归训练不是每次只算一个 token

训练时已知完整目标序列，可以把它右移一位作为输入：

```text
Decoder 输入:  <BOS> 我 喜欢 深度
监督目标:       我    喜欢 深度 学习
```

利用 Causal Mask，所有位置可以并行计算，但每个位置仍看不到未来答案。这叫 Teacher Forcing。

因此：

- 训练时：整段序列并行计算；
- 生成时：未来 token 不存在，只能逐 token 生成。

### 13.2 生成过程

```text
输入 prompt
  -> 模型输出下一个 token 的 logits
  -> 通过贪心、采样、Top-k、Top-p 等选择 token
  -> 把新 token 接到序列末尾
  -> 再预测下一个 token
  -> 直到停止条件
```

### 13.3 Output Head 和 logits

Decoder 最后一层隐藏状态：

```text
H: [B, L, d_model]
```

通过语言模型头映射到词表：

```text
logits = H W_vocab
logits: [B, L, V]
```

对每个位置，`V` 个 logits 表示下一个 token 的未归一化分数。训练时通常使用交叉熵损失；推理时根据 logits 选择 token。

### 13.4 KV Cache 为什么能加速生成

生成第 `t` 个 token 时，过去 token 的 K 和 V 不需要每一步重新计算，可以缓存：

```text
过去位置的 K/V：直接复用
新位置的 Q/K/V：只计算一次
新 Q 与所有缓存 K 做注意力
```

KV Cache 显著减少重复计算，但会占用大量显存。长上下文推理中，KV Cache 往往是重要的显存消耗来源。

---

## 十四、位置编码再深入一步：为什么 RoPE 常见

绝对位置向量是把位置信息直接加到 token 表示中。RoPE 则对 Q 和 K 的不同维度对进行与位置有关的旋转，使点积自然包含相对位置关系。

你不必先记旋转矩阵推导，只要抓住三点：

1. RoPE 主要作用在 Q 和 K，不直接作用在 V；
2. 它让 Attention score 感知 token 之间的相对距离；
3. 长上下文扩展时，RoPE 的尺度和外推策略非常重要。

位置编码最终影响的是：

```text
当前位置应该如何匹配其他位置，而不只是当前位置拥有什么内容。
```

---

## 十五、Attention 的计算复杂度与长序列问题

标准 Self-Attention 需要构造：

```text
Attention Matrix: [B, h, L, L]
```

因此相对于序列长度 `L`，计算量和注意力矩阵显存通常呈二次增长：

```text
时间复杂度核心项：O(L^2 d_model)
注意力矩阵空间：O(L^2)
```

序列从 1,000 增加到 2,000，注意力矩阵元素数量约变为 4 倍。

常见优化方向：

- FlashAttention：减少显存读写和中间矩阵保存，精确计算标准 Attention；
- 局部/窗口注意力：只看邻域，减少可见连接；
- 稀疏注意力：只计算部分位置关系；
- 线性注意力：重写计算顺序或使用近似；
- 分块与滑动窗口；
- 状态空间模型或其他序列架构。

注意：FlashAttention 并不是改变模型数学目标的“近似注意力”，其主要优势来自更高效的 GPU 内存访问和分块计算。

---

## 十六、ViT：图像如何进入 Transformer

Transformer 不要求输入必须是文字，它只要求输入能表示成 token 序列。

Vision Transformer 的基本流程：

```text
图像 [B, C, H, W]
  -> 切成 P × P patch
  -> 每个 patch 展平并线性映射
  -> patch token [B, N, d_model]
  -> 加入位置编码
  -> Transformer Encoder
  -> 分类头或下游解码器
```

Patch 数量：

```math
N = \frac{H}{P}\times\frac{W}{P}
```

例如 `224 × 224` 图像，patch 大小 `16 × 16`：

```text
N = 14 × 14 = 196
```

图像中的 patch token 与文本 token 在进入 Transformer 后，张量角色基本一致。

### 16.1 CNN 与 ViT 的核心差别

CNN 天然具有局部连接和平移相关的归纳偏置；标准 ViT 更依赖数据学习位置之间的关系。ViT 能直接建模全局 patch 关系，但标准全局 Attention 在高分辨率图像上成本很高。

因此视觉 Transformer 常使用：

- 分层特征；
- Patch Merging/下采样；
- Window Attention；
- 空间缩减 Attention；
- CNN 与 Transformer 混合结构。

这也是 PVT、Swin Transformer 等结构适合检测和分割的原因：它们输出多尺度特征，而不是只保留单一分辨率序列。

### 16.2 在医学图像分割中的位置

典型结构是：

```text
图像
  -> CNN/Transformer Encoder 提取多尺度特征
  -> Decoder 逐级融合和上采样
  -> 分割概率图
```

此处 Transformer 主要负责全局或长距离关系建模，Decoder 负责恢复空间分辨率和边界细节。必须区分：Transformer 主干中的 token 序列 Decoder，与图像分割网络中负责上采样的 segmentation decoder，不是同一个概念。

---

## 十七、读 Transformer 源码时的形状追踪法

看到任意实现，按下面顺序检查。

### 1. 输入布局

先确认是：

```text
[B, L, C] 还是 [L, B, C]
```

视觉模型还可能是：

```text
[B, C, H, W]
```

然后在进入 Attention 前展平成 token。

### 2. QKV 是一次投影还是三次投影

常见高效写法：

```text
qkv = Linear(x)              # [B, L, 3*d_model]
qkv = reshape(...)           # [B, L, 3, h, d_head]
qkv = permute(...)           # [3, B, h, L, d_head]
q, k, v = qkv[0], qkv[1], qkv[2]
```

数学上仍等价于分别生成 Q、K、V。

### 3. Softmax 在哪个维度

注意力分数通常：

```text
[B, h, L_q, L_k]
```

Softmax 应沿 `L_k`，也就是最后一个维度进行，使每个 Query 对所有 Key 的权重和为 1。

### 4. 多头怎样恢复

检查：

```text
[B,h,L,d_head]
  -> transpose
[B,L,h,d_head]
  -> reshape
[B,L,d_model]
```

### 5. Norm 是 Pre-Norm 还是 Post-Norm

```python
# Pre-Norm
x = x + attn(norm(x))

# Post-Norm
x = norm(x + attn(x))
```

### 6. Mask 的形状如何广播

Mask 可能是：

```text
[B, L]
[L, L]
[B, 1, 1, L]
[B, 1, L, L]
```

不要只看变量名，要确认它最终如何广播到 `[B, h, L_q, L_k]`。

---

## 十八、最小伪代码：一个 Pre-Norm Transformer Block

```python
class TransformerBlock:
    def forward(self, x, mask=None):
        # token 之间交换信息
        x = x + self.attention(self.norm1(x), mask=mask)

        # 每个 token 独立加工信息
        x = x + self.ffn(self.norm2(x))
        return x
```

Multi-Head Self-Attention 的核心伪代码：

```python
def attention(x, mask=None):
    # x: [B, L, d_model]
    q, k, v = project_qkv(x)

    # [B, h, L, d_head]
    q = split_heads(q)
    k = split_heads(k)
    v = split_heads(v)

    # [B, h, L, L]
    scores = q @ k.transpose(-2, -1)
    scores = scores / sqrt(d_head)

    if mask is not None:
        scores = apply_mask(scores, mask)

    weights = softmax(scores, dim=-1)

    # [B, h, L, d_head]
    context = weights @ v

    # [B, L, d_model]
    context = merge_heads(context)
    return output_projection(context)
```

理解这段伪代码，就理解了绝大多数 Attention 实现的骨架。工程代码的复杂部分主要来自高效算子、缓存、并行、不同 Mask、量化和设备管理。

---

## 十九、常见误解，一次纠正

### 误解 1：Attention 权重就是模型解释

Attention 权重能展示信息路由模式，但不能自动等同于完整因果解释。模型输出还受到 Value、输出投影、FFN、残差和后续层共同影响。

### 误解 2：Q、K、V 分别对应三个不同 token

不是。Self-Attention 中，每个 token 都会产生自己的 Q、K、V。Q/K/V 是角色不同的投影，不是把 token 分成三组。

### 误解 3：Self-Attention 只计算相似度

QK 点积负责匹配，但最终输出还要用权重乘 V。只有相似度矩阵，没有完成信息聚合。

### 误解 4：多头就是做多次完全相同的 Attention

不同头有不同的投影参数，学习不同子空间中的关系。它们不是简单复制同一个结果。

### 误解 5：Decoder 一定有 Cross-Attention

原始 Encoder-Decoder Transformer 的 Decoder 有 Cross-Attention；GPT 的 Decoder-only Block 通常只有 Causal Self-Attention，没有 Encoder，也就没有标准 Cross-Attention。

### 误解 6：Transformer 不需要位置信息

内容相关性不能自动完整表达顺序。模型必须通过绝对、相对或旋转等方式获得位置关系。

### 误解 7：训练生成模型也必须逐 token 串行计算

训练时已知完整目标序列，可以借助 Causal Mask 并行计算所有位置；真正逐 token 串行的是自回归推理。

### 误解 8：Attention 解决一切，FFN 不重要

FFN 提供逐 token 的非线性特征变换，参数和计算占比通常都很高，是 Block 的核心组成部分。

---

## 二十、用五个问题检验你是否真正理解

### 问题 1：Self-Attention 中 Q、K、V 来自哪里？

都来自同一输入 `X`，但经过不同线性投影。

### 问题 2：Cross-Attention 中 Q、K、V 来自哪里？

Q 来自 Decoder，K 和 V 来自 Encoder 输出。

### 问题 3：Attention 矩阵为什么是 `L × L`？

因为序列中每个 Query 位置都要与每个 Key 位置计算匹配关系。

### 问题 4：Attention 与 FFN 分别混合什么？

Attention 主要混合不同 token/位置的信息；FFN 对每个 token 的特征维做非线性变换。

### 问题 5：BERT 和 GPT 最本质的结构差异是什么？

BERT 是双向 Encoder-only；GPT 是带因果可见约束的 Decoder-only，自回归预测下一个 token。

如果这五个问题能不看文档直接回答，Transformer 的主骨架已经掌握。

---

## 二十一、最后压缩成一张脑图

```text
输入
├── Tokenizer：文本 -> token ID
├── Embedding：ID -> d_model 向量
└── Position：注入顺序/距离
        |
        v
Transformer Block × N
├── Attention：位置之间交换信息
│   ├── Q：我要找什么
│   ├── K：我以什么特征被匹配
│   ├── V：我实际提供什么内容
│   ├── QK^T / sqrt(d_head)：匹配分数
│   ├── Mask：限制可见范围
│   ├── Softmax：变成权重
│   └── 权重 × V：聚合内容
├── Multi-Head：多组关系子空间
├── FFN：逐位置非线性加工
├── Residual：保留信息、传递梯度
└── LayerNorm：稳定深层训练
        |
        v
输出头
├── 分类头：类别 logits
├── LM Head：词表 logits
├── 检测/分割 Decoder：恢复空间输出
└── 其他任务头
```

三种结构只需这样区分：

```text
Encoder-only：双向理解输入                -> BERT
Decoder-only：因果地预测下一个 token       -> GPT
Encoder-Decoder：理解输入后条件生成输出     -> 原始 Transformer/T5
```

最终只保留一句最有用的话：

> **Attention 负责让 token 互相取信息，FFN 负责让每个 token 消化信息，残差与 LayerNorm 负责让这套过程能够稳定地堆叠很多层。**
