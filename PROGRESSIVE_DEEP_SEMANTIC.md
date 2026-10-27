# 深层语义逐级传递模块（PDS）

## 作用和接入点

EMCAD 的三个编码器跳连保留边缘与纹理，也可能带入外观相似的背景。PDS 从最深层解码特征 `d4` 用 `1×1` 卷积压缩到 32 通道，并随解码过程向浅层传递。在 `x3`、`x2`、`x1` 三次跳连处，压缩语义与当前跳连的投影共同生成空间门控，对原 LGAG 已筛选的跳连做小幅残差调节。`d3` 和 `d2` 融合后分别更新语义，再传到下一层。它不读取真值，也不增加额外损失或改变四个分割头。

实现位置：[lib/decoders.py](lib/decoders.py) 的 `ProgressiveDeepSemantic` 和 `EMCAD.forward`。门控系数经过 `tanh` 限幅，初始幅度约为 0.1。它与内容感知抗混叠上采样（CAA）位于不同路径：CAA 改 EUCB，PDS 改三个跳连。

## 开关和消融

所有正式训练入口均接受 `--use_progressive_deep_semantic 0|1`；对应 `.sh` 使用 `USE_PROGRESSIVE_DEEP_SEMANTIC=0|1`。原 CAA 开关保持 `USE_CONTENT_AWARE_ANTIALIAS=0|1`。Cell 的 DSB18/EM 入口复用 Polyp 启动器和训练代码；ISIC、BUSI 也复用二分类模型入口。测试入口从训练 checkpoint 同目录的 `config.json` 恢复 PDS 与 CAA 配置；旧配置缺少 PDS 字段时按 0 处理。

| CAA | PDS | 结构 |
|---:|---:|---|
| 0 | 0 | 原 EMCAD 解码路径；不创建 CAA/PDS 可训练参数 |
| 1 | 0 | 仅 CAA |
| 0 | 1 | 仅 PDS |
| 1 | 1 | CAA + PDS |

示例：`USE_CONTENT_AWARE_ANTIALIAS=0 USE_PROGRESSIVE_DEEP_SEMANTIC=1 bash sh/start_train_synapse.sh`。其他数据集的 `start_train_*.sh` 使用同名变量。为保证消融可比，四组应固定数据划分、编码器预训练权重、种子、训练日程及评估代码。模块是否提高 Dice 仍须以各数据集独立实验验证，代码检查无法保证涨点。
