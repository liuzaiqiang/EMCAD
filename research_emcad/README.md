# EMCAD 类别原型与逐类尺度消融

先阅读 [代码分析](code_analysis.md)、[机制设计](innovation_design.md)、[实验矩阵](experiment_plan.md)、[相关工作](related_work.md)。本目录没有更改原始 EMCAD 解码器。两个开关默认 0，训练入口经 `sh/run_seed_queue.sh` 保留多 seed 顺序队列。启动命令见 [ablation_commands.md](ablation_commands.md)。

运行前按原项目 `requirements.txt` 准备 PyTorch/CUDA 与数据路径，确保 PVT 预训练文件存在。12 个任务由六类启动器覆盖，分别使用原有数据处理。研究脚本按配置写到 `research_emcad/runs/<配置>/...`；设置 `OUTPUT_DIR` 时它作为自定义输出根，脚本仍追加配置名。训练入口按自身的 dataset、seed、时间戳继续细分，并保存 config/checkpoint；队列与启动日志仍在项目 `logs/`。

测试：`python -m unittest discover research_emcad/tests -v`。测试新 checkpoint 时，测试入口的 `--use_class_prototype_query` 与 `--use_class_conditional_scale_fusion` 必须与训练配置一致；Synapse 可用 `--train_config` 读取训练配置。既有测试脚本默认两开关均为 0，因此旧 checkpoint 路径不变。
