# 当前项目代码分析（2026-10-22）

仓库 `D:/files/Medical_Image_Segmentation_Projects/SLDGroup_EMCAD`，分支 `baseModel_2Modules_1022`。开始检查时 `git status --short` 为空；最近提交 `0494907` 声称恢复原始 EMCAD，但该提交信息自身也说明当时尚未实验核验。根目录 `networks.py` 是副本，训练/测试实际导入 `lib/networks.py`。

## 实际网络与形状

`EMCADNet` 用可选的 1→3 Conv-BN-ReLU 适配灰度输入；PVTv2 或 ResNet 输出 `x1,x2,x3,x4`。标准 PVTv2-B2、输入 224² 时分别为 `[B,64,56,56]`、`[B,128,28,28]`、`[B,320,14,14]`、`[B,512,7,7]`。B0 通道为 32/64/160/256；ResNet18/34 为 64/128/256/512；ResNet50/101/152 为 256/512/1024/2048。解码器按 `[x4;x3,x2,x1]` 接收，输出 `d4,d3,d2,d1`，标准 B2 通道和尺度为 512@1/32、320@1/16、128@1/8、64@1/4。四个 1×1 head 输出 K 类 logits，分别固定放大 32/16/8/4 倍，返回 `[p4,p3,p2,p1]`。

`lib/decoders.py` 中 MSCB 使用 1/3/5 多尺度深度卷积及残差；EUCB 做二倍上采样及通道转换；LGAG 门控同尺度 skip；CAB/SAB 分别进行通道/空间加权。没有独立命名的 MSRF 类，所谓 MSCAM 是 CAB→SAB→MSCB 的组合。d3/d2/d1 在 skip 相加后已具备不同感受野与语义层级，适合在分割头附近进行逐类比较，同时避免改动 MSCB/EUCB。

## 训练与数据

入口为 `train_synapse.py`、`train_acdc.py`、`train_polyp.py`，后者还被 `train_isic.py`、`train_busi.py` 复用；Cell 通过 `sh/start_train_cell.sh` 复用 Polyp。现有 12 个任务名：Synapse、ACDC、五个 Polyp（BKAI/ClinicDB/ColonDB/ETIS/Kvasir）、ISIC2017/2018、BUSI、DSB18、EM。不同族的路径、增强和二/多类监督各自保留。Synapse 与 ACDC 使用 CE+Dice 及多输出监督；Polyp/ISIC/BUSI/Cell 沿用二分类监督及其既有可选像素可靠性融合。优化器和调度器由各训练入口定义，没有统一训练入口。`lib/experiment_paths.py` 创建带 seed/时间戳的独立目录；训练保存 `config.json` 和 checkpoint。`sh/run_seed_queue.sh` 按用户显式指定的 seeds 顺序等待每个训练进程，现有模式不应改变。

## 工程约束

- 原解码器相邻层需要精确二倍空间关系；建议 H/W 为 32 的倍数。上层固定倍率插值也有此假设。
- PVT 预训练默认从 `pretrained_pth/pvt/<encoder>.pth` 读取；本地只见 `pvt_v2_b2.pth`。ResNet 预训练可能通过 model zoo 访问网络。
- Synapse 训练入口强制 CUDA；现有 sh 写有 Linux Conda 路径，Windows 本机不能直接验证完整训练。
- 本机默认 Python 3.12 无 torch；已找到 `sld_emcad` 环境的 PyTorch 1.11.0+cu113、CUDA 11.3、timm 0.6.12，CUDA 可用，但环境 NumPy 2.2.6 与 torch 的 NumPy 1.x ABI 不兼容，导入会报警。
- 既有 Polyp/ISIC/BUSI 启动器默认像素可靠性融合开启，因此论文四组比较需显式统一该开关，避免归因混淆。
