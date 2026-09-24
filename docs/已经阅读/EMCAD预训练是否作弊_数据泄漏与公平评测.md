# EMCAD 使用预训练权重是否“作弊”：预训练、数据泄漏与公平评测

## 1. 先说结论

拿 ImageNet 预训练好的 PVTv2-B2 参数参与 EMCAD 训练，通常不算作弊，而是深度学习论文中非常常见的迁移学习做法。

真正需要警惕的不是“有没有预训练”，而是：

1. 预训练数据是否包含当前评测集的信息；
2. 是否把测试标签或测试图像用于模型训练；
3. 是否用测试集指标选择 best checkpoint；
4. 是否隐瞒了预训练数据、标签和处理流程；
5. 是否把不同数据划分下的结果直接与别人比较。

可以用一句话概括：

> 预训练本身不是作弊；让测试集信息提前影响模型，才是数据泄漏或不公平评测。

---

## 2. 为什么 ImageNet 预训练是被接受的

### 2.1 ImageNet 不是 Synapse 的测试答案

当前 EMCAD 使用的是：

    PVTv2-B2 的网络结构
    + ImageNet 上学到的编码器参数
    + Synapse/ACDC/Polyp 上的分割微调

ImageNet 训练的目标是自然图像分类，例如猫、汽车、飞机等类别。它没有看到 Synapse 的器官分割标签，也没有直接告诉模型 Synapse 测试病例的答案。

因此它提供的是通用视觉初始化，而不是当前测试集的正确结果。

### 2.2 预训练是公开协议的一部分

医学分割中常见的实验会明确写：

    encoder: PVTv2-B2
    initialization: ImageNet pretrained
    decoder: EMCAD
    fine-tuning: target segmentation dataset

只要所有方法在相同协议下比较，使用 ImageNet 预训练是公平的。很多论文甚至把：

    ImageNet 初始化
    随机初始化

作为单独的消融实验，证明预训练本身带来多少收益。

### 2.3 预训练和“恢复答案”不同

预训练学到的是参数的初始状态，不是把测试集结果写进模型。微调阶段仍然需要在目标训练集上学习器官、心脏结构或息肉的分割边界。

类比：

    ImageNet 预训练 = 学会一般的视觉基础
    目标分割训练 = 学会当前医学任务
    测试集 = 最后一次不能提前看的考试

---

## 3. 哪些情况才接近“作弊”

### 3.1 用目标测试集的分割标签训练

例如：

    用 Synapse train + Synapse test 的 image/label 做预训练
    再报告 Synapse test Dice

这是最明显的泄漏。模型已经通过测试标签间接看过答案，测试结果不能再解释为对未知病例的泛化。

### 3.2 用测试图像做有监督预训练

即使不直接使用器官标签，如果测试图像参与了分类、重建或其他有监督训练，模型也已经见过评测样本。此时至少不能再称为严格的独立测试。

### 3.3 用测试图像做无监督或自监督预训练

这是更容易被忽略的情况。

例如：

    用 Synapse test 患者的图像做 MAE/对比学习
    不使用 test label
    再在 Synapse test 上报告分割 Dice

虽然没有使用测试标签，但测试图像本身已经影响了 encoder 参数。这属于 transductive 或半监督式设置，不应冒充普通的 inductive 泛化结果。

正确做法是：

    主结果：预训练不使用测试图像
    额外实验：如果研究 transductive setting，明确标注并单独报告

### 3.4 用测试集选择超参数或 best checkpoint

当前项目尤其要注意：

    用测试指标挑选 epoch
    根据测试 Dice 修改学习率
    看到测试结果后反复改模块
    选择表现最好的数据集或运行作为唯一结果

模型选择应该使用 validation set。测试集只在方法和配置冻结后进行一次最终评估。

### 3.5 反复尝试后只报告最好的一次

例如跑 20 个随机种子、10 套增强和多个学习率，只报告最高 Dice，而不说明试过哪些设置。这不是传统意义上的权重泄漏，但属于选择性报告，会严重夸大结果。

---

## 4. “我拿市面上大部分数据集来预训练”是否可行

答案是：**可以研究，但不会自动让 Dice 很高，也不能自动保证公平。**

需要逐个检查数据集和目标任务的关系。

### 4.1 外部数据集、不同任务

例如：

    在公开自然图像上预训练
    在 Synapse 上做器官分割

这是标准迁移学习。ImageNet 就属于这一类。

### 4.2 外部医学数据集、相近模态

例如：

    在其他腹部 CT 无标签图像上预训练
    在 Synapse 上做分割

通常可以做，但必须披露：

    数据来源
    患者数量
    是否有标签
    预训练任务
    是否和 Synapse 病例重叠
    许可证和使用范围

如果外部数据和 Synapse 来自相同病例或同一数据发布包，就不能简单称为独立外部预训练。

### 4.3 同一数据集的训练图像

例如：

    用 Synapse train 图像做自监督
    再用 Synapse train 标签做分割微调
    在 Synapse validation/test 上评估

这通常是可以接受的，但要明确写成：

    in-domain pretraining on training split

并且不能让 validation/test 图像进入预训练。

### 4.4 同一数据集的验证/测试图像

这会改变评测协议。即使只做无标签预训练，也应单独说明：

    使用了测试图像进行 transductive pretraining

不能把结果和“不看测试图像”的普通监督训练结果直接并列为同一种指标。

### 4.5 使用其他数据集的分割标签

例如：

    用多个带器官标签的数据集预训练分割网络
    再在 Synapse 上评估

这可能是合法的多数据集迁移，但研究问题已经变成：

    多源有监督医学迁移

而不是简单的 ImageNet 对比。必须报告源数据集、标签类型、预训练类别和目标数据集，不能把它包装成“无额外标签的预训练”。

### 4.6 使用目标数据集同源数据

如果两个数据集来自同一批患者、同一视频或同一病例，只是名称不同，那么样本重复会造成严重风险。

Polyp 尤其要检查：

    同一内窥镜视频的相邻帧
    同一图像不同裁剪
    文件改名后的重复图像
    同源训练集和外部测试集

“数据集名字不同”不等于“患者和图像完全独立”。

---

## 5. 为什么预训练并不能保证 Dice 一定很高

### 5.1 预训练只改变起点

预训练参数决定优化从哪里开始，但最终结果仍受以下因素影响：

    目标数据量
    标签质量
    train/validation/test 划分
    模态差异
    输入尺寸
    decoder 结构
    loss
    学习率
    随机种子
    训练时间
    评估脚本

起点更好，不代表一定到达更高终点。

### 5.2 领域差异可能很大

ImageNet 学习的是 RGB 自然图像，Synapse 是 CT，ACDC 是 MRI，Polyp 是内窥镜图像。即使编码器学到了边缘和纹理，医学任务仍需要重新适应：

    灰度分布
    器官形态
    小目标边界
    医学噪声
    跨切片结构
    病灶与背景的细微差异

### 5.3 过度预训练可能损害迁移

如果预训练数据和目标数据差异很大，或者预训练任务迫使模型学习不适合分割的特征，微调效果可能没有提升，甚至变差。

因此必须实测：

    ImageNet
    random initialization
    medical pretraining

而不能根据“预训练数据更多”直接推断 Dice 更高。

---

## 6. 对你当前 EMCAD 项目的公平边界

### 6.1 当前 ImageNet 基线

当前项目默认：

    PVTv2-B2 结构
    + pretrained_pth/pvt/pvt_v2_b2.pth
    + EMCAD decoder
    + 目标数据集分割训练

这是合理基线。

### 6.2 推荐的医学自监督主实验

对 Synapse：

    只使用 Synapse 训练患者的 image
    忽略训练 label 做自监督
    不使用 validation/test 图像
    得到 medical SSL encoder
    接回原始 EMCAD decoder
    在相同分割训练配置下微调

对 ACDC：

    只使用 ACDC 训练病例图像做 SSL
    不能把 valid/test 病例图像混入主实验预训练

对 Polyp：

    先固定一个数据集做 SSL
    明确 video/case 去重
    测试跨数据集泛化时不能把测试图像提前用于 SSL

### 6.3 最小公平实验表

| 实验 | encoder 结构 | 初始化/预训练来源 | 是否看目标测试图像 |
|---|---|---|---|
| B0 | PVTv2-B2 | ImageNet | 否 |
| B1 | PVTv2-B2 | 随机初始化 | 否 |
| B2 | PVTv2-B2 | 目标训练集医学 SSL | 否 |
| B3 | PVTv2-B2 | 外部医学 SSL | 否，需检查重复 |
| B4 | PVTv2-B2 | 目标测试图像 SSL | 是，必须单独标 transductive |

主论文结果优先使用 B0、B1、B2、B3。B4 不应和它们混成普通测试结果。

---

## 7. 如何避免“预训练把指标刷高”

### 7.1 先冻结评测协议

在实验开始前写下：

    train 清单
    validation 清单
    test 清单
    预训练清单
    checkpoint 选择规则
    评价指标
    随机种子

之后不要因为看到测试结果再改这些规则。

### 7.2 预训练清单必须可审计

保存：

    ssl_manifest.csv
    finetune_train_manifest.csv
    validation_manifest.csv
    test_manifest.csv

每条记录至少包含：

    sample_id
    patient_id 或 video_id
    image_path
    dataset
    modality
    split
    used_for_ssl
    used_for_segmentation
    sha256

### 7.3 记录所有尝试

不要只保存成功运行。至少记录：

    实验 ID
    配置
    数据来源
    是否发生泄漏检查
    失败原因
    最佳验证结果
    最终测试结果
    是否被纳入论文表格

### 7.4 先验证再测试

正确顺序：

    训练集训练
        -> validation 选 best checkpoint
        -> 冻结代码和配置
        -> test 只评估一次
        -> 报告结果

如果当前脚本把名为 test_vol 的数据用于每个 epoch 的模型选择，应先确认它到底是验证列表还是官方测试列表。若是官方测试集，就应修正评估协议后再做论文实验。

---

## 8. 怎样在论文中诚实描述预训练

### ImageNet 基线

可以写：

    We initialize the PVTv2-B2 encoder with publicly available ImageNet-pretrained weights.
    The decoder and segmentation heads are trained on the target training split.

### 医学 SSL

可以写：

    We further adapt the ImageNet-initialized PVTv2-B2 encoder using self-supervised learning
    on unlabeled images from the training subjects only. Validation and test subjects are excluded
    from pretraining.

### 外部医学数据

需要补充：

    source dataset
    modality
    number of subjects/images
    pretraining objective
    overlap check
    license
    whether labels were used

### 不能这样写

不要把下面的设置写成普通独立测试：

    We pretrain on all images, including the target test set,
    and then evaluate on the same test set.

如果确实这样做，应该明确叫：

    transductive pretraining
    test-image adaptation
    unlabeled target-domain adaptation

并单独报告，不能和标准 inductive 结果混淆。

---

## 9. “作弊”和“创新”的区别

### 合规但创新性弱

    下载公开 ImageNet 权重
    下载公开医学分类权重
    修改路径后直接加载

这通常可以合规，但创新性有限，因为主要贡献是采用已有资源。

### 合规且可能有研究价值

    只用训练患者无标签图像
    设计医学结构相关的自监督任务
    保持 EMCAD decoder 不变
    与 ImageNet/random-init 严格对照
    证明 Dice、HD95、收敛速度或稳定性改善

这可以形成“医学领域自适应预训练”的研究方法。

### 不合规或结论失真

    用测试标签预训练
    用测试图像自监督却不披露
    用测试 Dice 选择 best epoch
    只报告多次尝试中的最高一次
    隐藏源数据集与目标数据集重叠

---

## 10. 给你的最终建议

不要把目标定成：

    “尽量收集很多数据集，让 Dice 变高”

应该定成：

    “在不使用目标验证/测试信息的前提下，
    验证医学领域自监督预训练是否比 ImageNet 初始化更适合 EMCAD。”

建议采用以下主线：

    1. 跑通 PVTv2-B2 + ImageNet 基线
    2. 跑 PVTv2-B2 随机初始化基线
    3. 只用 Synapse 训练患者图像做医学 SSL
    4. 只加载 SSL encoder，保持 EMCAD decoder 不变
    5. 使用相同分割训练设置微调
    6. 用 validation 选择 best checkpoint
    7. 冻结配置后只测试一次
    8. 使用至少 3 个随机种子
    9. 报告 Dice、HD95、每类结果和收敛曲线
    10. 再把流程独立扩展到 ACDC 和 Polyp

最重要的判断标准是：

> 你的指标高，是因为模型真正学到了可迁移的医学表征，还是因为模型提前看过了评测数据？

只要数据边界清楚、预训练来源公开、实验协议固定、结果完整披露，使用预训练就不是作弊，而是规范的科学实验。
