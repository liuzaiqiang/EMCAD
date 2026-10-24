# 实现与验证记录（2026-10-22）

## 修改范围

新增 `research_emcad/class_prototype_query.py`、`class_conditional_scale_fusion.py`、测试、统一研究启动器和文档。最小改动 `lib/networks.py`：只在四个 head 建立后有条件实例化模块，并在 head logits 形成后、原始插值前改写 p1。`lib/decoders.py` 未修改。`train_synapse.py`、`train_acdc.py`、`train_polyp.py` 增加默认关闭的 CLI 参数；ISIC/BUSI 复用 Polyp 参数解析。`utils/acdc_utils.py`、`utils/polyp_utils.py` 传参给模型。五个原始训练启动器传递显式开关并记录日志，Cell 沿用 Polyp 启动器；`sh/run_seed_queue.sh` 未修改。`test_synapse.py`、`test_acdc.py`、`test_polyp.py` 增加重建参数，ISIC/BUSI 测试复用 Polyp；Synapse 的训练 config 恢复列表已包含两个开关。

## 已实际通过

- `python -m compileall` 覆盖新增模块、全部受影响的训练/测试入口：通过。
- `git diff --check`：通过。
- Git Bash `bash -n` 覆盖新脚本、五个原训练启动器和 seed 队列：通过。
- `sld_emcad` 环境运行 `python -m unittest discover research_emcad/tests -v`：4 个测试通过。覆盖独立模块多尺寸前向/反向、NaN/Inf、训练/推理模式、梯度、state_dict 保存恢复；四开关组合的全模型构造和 128² 输出；两开关关闭的 state_dict 不含新增参数且固定随机种子下输出逐元素相同；联合模式合成输入的 AdamW 一步与 checkpoint 严格恢复。此为合成张量的模型级 smoke test，不是数据训练。
- Synapse/ACDC 的 Python `--help` 显示两个参数。Polyp 入口因现有 NumPy ABI 问题在导入阶段失败，不能称为 CLI 实跑通过。

## 未完成及限制

没有执行真实数据训练、验证、12 数据集逐一推理或正式统计检验。Windows 本机的原始 sh 使用 Linux Conda 路径；现有 `sld_emcad` 为 PyTorch 1.11.0+cu113、NumPy 2.2.6，torch 会发出 NumPy 1.x ABI 警告，Polyp 训练入口因 `numpy.dtype size changed` 导入失败。需要在服务器原环境或修复依赖版本后运行完整链路；不在本任务中直接安装/更换用户环境。当前只有 B2 预训练文件已在本地核查，其余编码器权重需补充。

原始 checkpoint 的严格加载在**模型构造双开关关闭**时具备相同 state_dict 键集合，但尚未用某个既有完整 checkpoint 做逐值加载；历史模型代码的逐值输出对照也未做。测试入口新 checkpoint 的模型开关必须一致。`class_queries` 与原型组合均参与梯度；规模与时间需在目标 GPU 上实测。

## 学术风险复核

类别查询、原型聚合、尺度融合均有现有研究。模块 A 目前只是 EMCAD 三层特征上的类条件原型适配，初步创新性**较弱至一般**；主要工程风险为背景关注/类别坍缩，优先做类别原型相似度及每器官 Dice，若所有查询趋同或小器官退化则否定。模块 B 的每类三尺度选择有明确可证伪假设，但与已有跨尺度注意力和本仓库像素可靠性融合相近，初步创新性**证据不足**；优先做每类权重与器官面积关系、固定权重和等参数对照，若类别权重几乎相同或收益仅来自额外参数则否定。两模块是否互补完全未由性能实验证明。不能据当前代码和合成测试声称达到 CCF-B 投稿新颖性要求。
