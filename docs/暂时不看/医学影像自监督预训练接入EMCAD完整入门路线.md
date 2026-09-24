# 医学影像自监督预训练接入 EMCAD：新手完整路线

## 0. 你要完成的事情

目标不是简单替换一个权重文件，而是完成这条实验链：

    PVTv2-B2 + ImageNet 参数
        -> 医学图像无标签自监督预训练
        -> 保存医学领域适配后的 PVTv2-B2 encoder
        -> 接回原始 EMCAD decoder
        -> 在分割数据集上微调
        -> 和 ImageNet 基线公平比较

核心研究问题是：

> 在 EMCAD 结构、分割损失和数据划分不变时，医学影像自监督预训练是否比 ImageNet 初始化更适合医学分割？

当前仓库默认使用 PVTv2-B2 和本地 ImageNet 权重 pretrained_pth/pvt/pvt_v2_b2.pth。仓库目前没有现成的医学自监督预训练脚本，因此需要新增独立的预训练流程，但第一阶段不要改坏原有分割流程。

---

## 1. 先区分三个阶段

### 1.1 ImageNet 预训练

PVTv2-B2 已经在 ImageNet 上学习过自然图像特征，得到一份与 PVTv2-B2 结构匹配的参数。

### 1.2 医学自监督预训练

不使用人工器官/病灶标签，而是从图像自身构造目标，例如：

- 同一图像的两个增强视图，特征应该相近；
- 遮挡图像的一部分，模型恢复被遮挡区域；
- 同一病例相邻切片，特征具有一定连续性。

### 1.3 分割微调

把医学自监督后的 encoder 接入原始 EMCAD decoder 和输出头，用分割标签计算 CE、Dice 等损失。

预训练阶段的 projection head 只服务自监督；微调阶段通常丢弃它。

---

## 2. 当前 EMCAD 的实际结构

### 2.1 训练入口

train_synapse.py 默认选择：

    encoder = pvt_v2_b2
    pretrain = True
    pretrained_dir = ./pretrained_pth/pvt/

实际建模时由 lib/networks.py 创建 EMCADNet。encoder 决定编码器结构，pretrain 决定是否加载已有参数。

### 2.2 PVTv2-B2 的接口

lib/pvtv2.py 中的 PVTv2-B2 结构大致是：

    四个阶段通道：64, 128, 320, 512
    四个阶段深度：3, 4, 6, 3
    返回四级特征：x1, x2, x3, x4

例如输入 352 x 352 时：

    x1: [B,  64, 88, 88]
    x2: [B, 128, 44, 44]
    x3: [B, 320, 22, 22]
    x4: [B, 512, 11, 11]

EMCAD decoder 需要这四级特征。因此自监督预训练不能随意改变 PVTv2-B2 的输出接口。

### 2.3 单通道输入

Synapse CT 和 ACDC MRI 通常是单通道。EMCAD 当前会用一个可学习的 1x1 卷积把 1 通道变成 3 通道，再送给原本按 RGB 设计的 PVTv2。

第一版自监督模型应保留这个输入适配器，保证预训练和微调的输入路径一致。

---

## 3. 数据集为什么必须分开处理

### 3.1 Synapse

- 模态：腹部 CT；
- 任务：背景加 8 个器官的多类分割；
- 当前仓库：训练读取二维 NPZ 切片，验证/测试按病例处理三维体。

Synapse 是第一优先级，因为当前训练入口最完整。

### 3.2 ACDC

- 模态：心脏 MRI；
- 任务：背景、右心室、心肌、左心室；
- 当前仓库：训练使用二维切片，验证按病例重组成三维体。

ACDC 和 Synapse 都是医学切片，但 CT 与 MRI 的强度分布不同，建议先独立预训练。

### 3.3 Polyp 五个数据集

当前 loader 支持的常见名称包括：

    ClinicDB
    ColonDB
    ETIS
    Kvasir-SEG
    BKAI

你写的 EKAI 很可能是 BKAI，实际以当前数据目录名称为准。

Polyp 数据是彩色内窥镜图像，和 CT/MRI 不同：

- 输入通常是 RGB；
- 任务是病灶/背景二分类；
- 颜色增强比 CT/MRI 更重要；
- 五个数据集可能存在同源视频或近重复图像。

第一版不要把 Synapse、ACDC、Polyp 全部混合成一个预训练池。推荐：

    Synapse SSL -> Synapse 微调
    ACDC SSL -> ACDC 微调
    Polyp SSL -> Polyp 微调

---

## 4. 总体执行顺序

### 阶段 A：先跑通基线

先确认：

    PVTv2-B2 + ImageNet + 原始 EMCAD

能稳定训练、验证和保存 best.pth。

必须保存：

- 配置文件；
- 完整命令；
- 环境版本；
- 训练日志；
- 验证指标；
- best.pth 和 last.pth。

如果 ImageNet 基线不稳定，后续不能判断医学 SSL 是否有效。

### 阶段 B：只在 Synapse 完成自监督闭环

先做三个实验：

    S0: PVTv2-B2 + ImageNet -> EMCAD 分割
    S1: PVTv2-B2 + 随机初始化 -> EMCAD 分割
    S2: ImageNet PVTv2-B2 -> Synapse SSL -> EMCAD 分割

S0 和 S2 的区别只有预训练阶段，S1 用来测量 ImageNet 初始化本身的收益。

### 阶段 C：再扩展到 ACDC

    A0: ImageNet PVTv2-B2 -> ACDC 分割
    A1: ACDC 无标签 SSL PVTv2-B2 -> ACDC 分割

### 阶段 D：最后扩展到 Polyp

先选 ClinicDB 或 Kvasir-SEG 做单数据集闭环，再做跨数据集泛化。

---

## 5. 第一步：建立数据清单

### 5.1 为什么没有标签也必须做清单

自监督也会发生数据泄漏。例如：

- 测试患者图像进入预训练；
- 同一患者不同切片跨越训练和验证；
- 同一内窥镜视频的相邻帧被分到训练和测试；
- 相同图像换文件名重复出现。

### 5.2 推荐目录

    data/
      ssl/
        synapse/
          images/
          manifest.csv
        ACDC/
          images/
          manifest.csv
        polyp/
          images/
          manifest.csv

    experiments/
      ssl/
        synapse_pvtb2_v1/
        acdc_pvtb2_v1/
        polyp_pvtb2_v1/

### 5.3 manifest.csv 最少字段

    sample_id
    case_id
    patient_id_or_video_id
    image_path
    dataset
    modality
    original_split
    ssl_split
    used_for_segmentation
    sha256

### 5.4 推荐隔离规则

预训练数据只能来自：

- 训练患者的无标签图像；
- 或明确允许使用的独立外部无标签数据。

验证和测试患者不要进入预训练。最好先采用：

    SSL 数据 = 训练患者图像
    分割训练 = 同一训练患者的图像和标签
    验证/测试 = 完全隔离

如果使用外部数据，记录来源、许可证、下载日期、处理方法和是否可能与测试集重复。

---

## 6. 第二步：制作自监督 Dataset

当前仓库的分割 Dataset 会返回 image 和 label。自监督 Dataset 应该只返回同一图像的两个增强版本：

    {
        "view1": augmented_image_1,
        "view2": augmented_image_2,
        "sample_id": sample_id
    }

训练时忽略分割标签。为了复现，Dataset 还应该记录：

- 原始图像路径；
- 归一化方法；
- 输出尺寸；
- 增强参数；
- 随机种子。

### 6.1 Synapse Dataset

从当前 Synapse NPZ 读取 image，忽略 label。使用与分割训练一致的 CT 预处理和尺寸缩放。

### 6.2 ACDC Dataset

从当前 ACDC NPZ 读取 img，忽略 label。不要在第一版中改动 ACDC 的病例级验证逻辑。

### 6.3 Polyp Dataset

只读取 images。mask 不参与自监督损失，但仍需使用固定的训练来源清单，不能偷偷读取测试图像。

---

## 7. 第三步：设计增强策略

### 7.1 CT/MRI 增强

推荐从轻到重：

- 小幅旋转；
- 小幅平移或裁剪；
- 适度强度扰动；
- 轻微模糊；
- 必须谨慎使用水平翻转。

如果左右方向具有医学含义，不要默认启用强水平翻转。

### 7.2 Polyp 增强

可以考虑：

- 水平翻转；
- 小幅旋转；
- 亮度和对比度变化；
- 轻微颜色抖动；
- 轻微模糊。

不要让增强把息肉边缘完全破坏。

### 7.3 不同模态不要强行共享增强

建议分别保存：

    utils/ssl_augmentations_ct.py
    utils/ssl_augmentations_mri.py
    utils/ssl_augmentations_polyp.py

第一版只实现一种自监督目标和一套经过检查的增强，避免变量太多。

---

## 8. 第四步：选择第一个自监督目标

第一次推荐双视图对比学习，而不是立即实现复杂 MAE。

数据流：

    原图 -> view1
    原图 -> view2
    view1 -> encoder -> projection head -> z1
    view2 -> encoder -> projection head -> z2
    z1、z2 -> contrastive loss

模型结构：

    输入适配器
        -> PVTv2-B2 encoder
        -> x4
        -> 全局平均池化
        -> projection head

x4 通常是 [B,512,h,w]，池化后为 [B,512]，projection head 可输出 [B,128]。

projection head 只用于自监督，微调 EMCAD 时不加载。

### 为什么不先做 MAE

MAE 需要额外处理 patch、mask、重建头和图像归一化，第一次实现容易同时出现多个问题。先用双视图对比学习完成端到端闭环，再增加 MAE 或跨切片一致性。

---

## 9. 第五步：建立自监督模型

当前仓库没有这个模型，需要新增独立文件，例如：

    pretrain_medical_ssl.py
    utils/dataset_ssl.py
    utils/ssl_losses.py
    utils/ssl_augmentations_ct.py

概念代码：

    class MedicalSSLModel(nn.Module):
        input_adapter: 1 -> 3
        encoder: pvt_v2_b2
        projector: 512 -> 512 -> 128

        forward(image):
            单通道时执行 input_adapter
            features = encoder(image)
            x4 = features[-1]
            pooled = global_average_pool(x4)
            z = projector(pooled)
            返回 normalize(z)

预训练时不接 EMCAD decoder，不接四个 segmentation heads，也不使用 CE/Dice 分割损失。

---

## 10. 第六步：实现对比损失

对每个 batch：

    z1 = model(view1)
    z2 = model(view2)

要求：

- z1、z2 先做 L2 normalize；
- 使用温度参数 temperature；
- batch 太小时负样本少，必要时使用梯度累积；
- 不要把同一病例相邻切片简单视为完全独立的负样本。

第一版只完成普通双视图对比学习。相邻切片一致性放到第二版，否则无法判断提升来自哪个因素。

---

## 11. 第七步：预训练配置

可用的起点：

    encoder: pvt_v2_b2
    input size: 224 或当前任务尺寸
    optimizer: AdamW
    learning rate: 1e-4 或 3e-4
    weight decay: 1e-4
    epochs: 100 起步
    temperature: 0.2
    seed: 固定并记录
    mixed precision: 显存允许时使用

这些只是起始值，不是最终结论。每个 epoch 记录：

- ssl_loss；
- learning rate；
- epoch_time；
- GPU 显存；
- 图像数量；
- checkpoint 路径。

先用很小的数据跑 1 个 epoch，确认代码无误后再正式训练。

---

## 12. 第八步：保存 checkpoint

预训练至少保存：

    ssl_last.pth
    ssl_best.pth

建议 checkpoint 记录：

    encoder
    input_adapter
    projector
    epoch
    ssl_loss
    config
    manifest_sha256

分割微调时：

- 加载 encoder；
- 视情况加载 input_adapter；
- 丢弃 projector；
- 不加载 segmentation heads；
- 不把自监督 checkpoint 当成完整 EMCAD 分割 checkpoint。

建议额外保存只含 encoder 的文件：

    pretrained_pth/medical_ssl/
      pvt_v2_b2_synapse_ssl_encoder.pth
      pvt_v2_b2_acdc_ssl_encoder.pth
      pvt_v2_b2_polyp_ssl_encoder.pth

不要覆盖原始 pretrained_pth/pvt/pvt_v2_b2.pth。

---

## 13. 第九步：接回 EMCAD

推荐给 train_synapse.py、train_acdc.py 和 train_polyp.py 增加明确参数：

    --pretrained_source imagenet|medical_ssl|none
    --encoder_pretrained_path PATH

逻辑：

    imagenet
        加载原始 PVTv2-B2 ImageNet 权重

    medical_ssl
        加载医学自监督后的 PVTv2-B2 encoder

    none
        不加载 encoder 预训练

加载时打印并记录：

- matched keys；
- missing keys；
- unexpected keys；
- shape mismatch；
- 实际 checkpoint 路径。

不要只看到程序没有报错就认为加载成功。

特别注意，当前 lib/networks.py 的过滤逻辑主要按参数名匹配。若 checkpoint 外层有 state_dict、encoder 或 module 前缀，需要先解包和清理键名。

---

## 14. 第十步：选择微调策略

至少记录一种固定策略，再公平比较。

### 策略 A：全部解冻

加载医学 encoder 后，encoder、decoder、heads 全部训练。

### 策略 B：先冻结后解冻

前若干 epoch 冻结 encoder，只训练 decoder/head，之后解冻 encoder。

### 策略 C：分层学习率

例如：

    encoder: 1e-5
    decoder/head: 1e-4

第一次推荐策略 B 或 C，但 ImageNet 基线和医学 SSL 模型必须使用完全相同的策略。

---

## 15. Synapse 具体实验顺序

### S0：ImageNet 基线

    PVTv2-B2 + ImageNet + EMCAD

### S1：随机初始化

    PVTv2-B2 + no_pretrain + EMCAD

### S2：医学 SSL 主实验

    ImageNet PVTv2-B2
        -> 只用 Synapse 训练患者无标签图像做 SSL
        -> 提取 encoder
        -> 接回原始 EMCAD
        -> 用同一分割配置微调

### S3：预训练数据量消融

    25%、50%、100% 的无标签图像

### S4：预训练目标消融

    只做双视图一致性
    双视图一致性 + 相邻切片一致性

一次只增加一个因素。

---

## 16. ACDC 具体实验顺序

    A0: ImageNet PVTv2-B2 -> ACDC
    A1: ACDC 无标签 SSL PVTv2-B2 -> ACDC
    A2: Synapse SSL encoder -> ACDC

A2 是跨模态探索，不应替代 A1 作为主结论。ACDC 仍然使用当前仓库的病例级三维验证和指标，不能为了方便只报告二维切片结果。

---

## 17. Polyp 五个数据集具体实验顺序

第一轮先选一个数据集：

    P0: ImageNet PVTv2-B2 -> ClinicDB
    P1: ClinicDB 无标签 SSL -> ClinicDB

然后做独立数据集：

    K0: ImageNet -> Kvasir-SEG
    K1: Kvasir-SEG SSL -> Kvasir-SEG

最后再做跨数据集泛化：

    ClinicDB 无标签 SSL
        -> 在 ColonDB、ETIS 或 BKAI 上测试

必须明确：

- SSL 数据来源；
- 分割训练来源；
- 测试来源；
- 是否使用测试图像做预训练；
- 是否存在同源视频或重复图像。

---

## 18. 公平性要求

所有对照必须固定：

- EMCAD decoder；
- 输入尺寸；
- 数据划分；
- 分割损失；
- supervision；
- batch size；
- epoch；
- 学习率策略；
- checkpoint 选择规则；
- 随机种子集合。

只改变 encoder 初始化来源。

最小实验表：

| 实验 | 结构 | encoder 初始化 |
|---|---|---|
| B0 | PVTv2-B2 + EMCAD | ImageNet |
| B1 | PVTv2-B2 + EMCAD | 随机 |
| B2 | PVTv2-B2 + EMCAD | 医学 SSL |
| B3 | PVTv2-B2 + EMCAD | ImageNet 后医学 SSL |

最值得比较的是 B0 与 B3。

---

## 19. 指标和证据

至少记录：

- mean Dice；
- 每类 Dice；
- HD95；
- ASD/ASSD；
- Jaccard；
- best epoch；
- 收敛速度；
- 训练时间；
- 显存；
- 参数量和 FLOPs；
- 多个随机种子的均值和标准差。

医学 SSL 可能带来的改善不只有 Dice，也可能是：

- 小器官/小病灶更好；
- 边界指标更好；
- 达到目标 Dice 更快；
- 训练更稳定；
- 不同 seed 波动更小。

一次运行提升很小，不能直接下结论。

---

## 20. 实验目录和记录

推荐：

    experiments/
      ssl/
        synapse_pvtb2_v1/
          config.json
          command.txt
          manifest.csv
          environment.txt
          pretrain.log
          ssl_best.pth
          ssl_last.pth
          notes.md
        finetune_synapse_medical_ssl_v1/
          config.json
          command.txt
          train.log
          best.pth
          last.pth
          metrics.csv
          per_case_metrics.csv
          notes.md

notes.md 至少记录：

    实验 ID：
    日期：
    数据来源：
    患者级划分：
    是否使用验证/测试图像：
    预训练任务：
    输入尺寸：
    增强策略：
    预训练 epoch：
    微调策略：
    最佳验证 Dice：
    HD95：
    失败情况：
    下一步：

---

## 21. 四周入门计划

### 第 1 周：基线和清单

1. 备份当前代码和原始 ImageNet 权重。
2. 跑通 Synapse ImageNet 基线。
3. 生成 Synapse manifest.csv。
4. 检查患者级 train/valid/test 隔离。
5. 保存命令、环境和日志。

### 第 2 周：最小 SSL 闭环

1. 新建 ssl Dataset。
2. 新建 CT 增强。
3. 新建对比损失。
4. 新建自监督模型。
5. 用极小数据跑 1 个 epoch。
6. 检查 loss、梯度和 checkpoint。

### 第 3 周：正式 Synapse SSL

1. 只使用训练患者的无标签图像。
2. 完成正式预训练。
3. 提取 encoder 参数。
4. 接回 EMCAD。
5. 用与基线相同的配置微调。

### 第 4 周：重复和消融

1. ImageNet；
2. random initialization；
3. medical SSL；
4. 至少 3 个随机种子；
5. 汇总 Dice、HD95、收敛曲线和失败运行。

之后再扩展到 ACDC 和 Polyp。

---

## 22. 最终建议

第一次不要同时做：

- 更换 Swin 或 ConvNeXt；
- 重写 EMCAD decoder；
- 混合 CT、MRI、内窥镜；
- 同时加入 MAE、对比学习、边界损失和跨切片损失。

最稳妥的创新路线是：

    固定 PVTv2-B2 结构
    固定原始 EMCAD decoder
    固定分割训练和数据划分
    只增加医学无标签自监督预训练
    先完成 Synapse 闭环
    再独立扩展到 ACDC 和 Polyp

当你拥有 ImageNet 基线、随机初始化基线、医学 SSL 模型、无泄漏数据清单、至少 3 个种子，以及 Dice + HD95 + 收敛曲线后，才有足够证据判断这项方法是否真正有效。

