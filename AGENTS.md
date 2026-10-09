# 仓库工作约定

- 用户要求：后续模型、训练、损失和实验功能修改默认覆盖全部 12 个数据集，只有用户明确缩小范围时例外。
- Synapse 使用 `train_synapse.py` / `trainer.py`；ACDC 使用 `train_acdc.py` / `utils/acdc_utils.py`。
- ClinicDB、Kvasir、ColonDB、ETIS、BKAI、BUSI、ISIC2017、ISIC2018、DSB18、EM 复用 `train_polyp.py` / `utils/polyp_utils.py`；同时核对 Polyp、BUSI、ISIC、Cell 启动器及包装入口。
- 保留用户已有监督策略（当前实验使用 `mutation`）、CAA、数据划分、预处理、随机种子、基础损失、指标和 checkpoint 选择规则。新增模块不得要求用户更换基线监督策略。
- 核心开关用 `0/1`。关闭后保留原行为；权重只能由训练数据或训练历史计算。
- 在 `docs/` 更新中文机制说明和运行命令。检查、测试和训练 loss 不能证明 Dice 提升；完整配对实验前写“效果尚未验证”。
