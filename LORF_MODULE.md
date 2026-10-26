# 局部序关系残差增强模块（LORF）

EMCAD 在最浅层使用 LGAG 筛选编码器跳连，再与上采样的解码特征相加。LORF 放在 LGAG 之后、相加之前。它压缩跳连通道，分别比较上下左右邻域的响应，经 `tanh` 得到四组有符号的局部序关系，以 `1×1` 卷积投影回原通道并作为小幅残差加入跳连。投影零初始化，开启时初始前向仍等于原始跳连，随后由训练学习修正。

CAA 只处理三个 EUCB 上采样；LORF 只处理最高分辨率的编码器跳连。不修改既有四个分割头、损失和指标，也不读取类别原型。

| `USE_CONTENT_AWARE_ANTIALIAS` | `USE_LORF` | 模型 |
| --- | --- | --- |
| 0 | 0 | 基础 EMCAD；原始 EUCB，且不创建 LORF 参数 |
| 1 | 0 | 基础 EMCAD + CAA；不创建 LORF 参数 |
| 0 | 1 | 基础 EMCAD + LORF；原始 EUCB |
| 1 | 1 | 基础 EMCAD + CAA + LORF |

所有训练启动器的两个环境变量均只接受 `0/1`。例如：

```bash
USE_CONTENT_AWARE_ANTIALIAS=0 USE_LORF=1 bash sh/start_train_synapse.sh
```

Synapse、ACDC、BKAI、ClinicDB、ColonDB、ETIS、Kvasir、ISIC2017、ISIC2018、BUSI、DSB18、EM 共 12 个数据集分别通过 `start_train_synapse.sh`、`start_train_acdc.sh`、`start_train_polyp.sh`、`start_train_isic.sh`、`start_train_busi.sh`、`start_train_cell.sh` 接入。测试由 checkpoint 同目录的 `config.json` 恢复结构；旧配置缺少 `use_lorf` 时按 `0` 处理。

四组消融必须使用相同数据划分、随机种子、训练和评估协议。LORF 单独及联合 CAA 是否提升 Dice 尚未经过服务器实验验证。
