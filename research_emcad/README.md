# EMCAD 类别条件尺度融合与低秩通道协方差交互

本目录包含已验证的类别条件尺度融合、可选的[低秩通道协方差交互](low_rank_covariance.md)、消融启动器与测试。两个开关默认关闭；双关时保留原始 EMCAD 四头输出及参数结构。启动命令见 [ablation_commands.md](ablation_commands.md)。

按原项目 `requirements.txt` 准备 PyTorch/CUDA、数据路径与 PVT 预训练权重。研究启动器经 `sh/run_seed_queue.sh` 顺序运行手动指定的 seed，输出保存到 `research_emcad/runs/<配置>/...`。`OUTPUT_DIR` 可覆盖输出根目录。

运行 `python -m unittest discover research_emcad/tests -v` 检查模型路径。测试模型时通过训练配置恢复两个开关；Synapse 可传 `--train_config`。基础模型使用两个默认值 0。
