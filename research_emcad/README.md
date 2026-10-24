# EMCAD 类别条件尺度融合

本目录包含可选的类别条件尺度融合模块、消融启动器与测试。模型默认关闭融合；关闭时保留原始 EMCAD 四头输出及参数结构。启动命令见 [ablation_commands.md](ablation_commands.md)。

按原项目 `requirements.txt` 准备 PyTorch/CUDA、数据路径与 PVT 预训练权重。研究启动器经 `sh/run_seed_queue.sh` 顺序运行手动指定的 seed，输出保存到 `research_emcad/runs/<配置>/...`。`OUTPUT_DIR` 可覆盖输出根目录。

运行 `python -m unittest discover research_emcad/tests -v` 检查模型路径。测试融合模型的 checkpoint 时，传入 `--use_class_conditional_scale_fusion 1`；Synapse 也可通过 `--train_config` 恢复该值。基础模型使用默认值 0。
