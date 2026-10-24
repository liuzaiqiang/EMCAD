# 实现与验证记录

`lib/networks.py` 在四个 head 之后可选实例化 `ClassConditionalScaleFusion`，只在原始插值前改写 p1。`lib/decoders.py` 保持不变。训练与测试入口仅接收 `use_class_conditional_scale_fusion`，默认关闭。六类训练启动器与研究队列脚本传递同一开关。测试覆盖关闭路径的原始四头读出、开启路径对前三头的保持，以及梯度和 checkpoint 恢复。

真实数据上的涨点结果以服务器实验为准；本地合成张量测试只检查代码行为，不代表数据集指标。
