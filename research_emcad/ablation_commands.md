# 四组完整启动命令

在项目根目录运行；环境变量和数据根目录按目标 Linux 服务器配置。每组保留同一 seed 队列与同一数据设置。

```bash
USE_CLASS_PROTOTYPE_QUERY=0 USE_CLASS_CONDITIONAL_SCALE_FUSION=0 bash research_emcad/scripts/train.sh synapse 2222 3407 5678
USE_CLASS_PROTOTYPE_QUERY=1 USE_CLASS_CONDITIONAL_SCALE_FUSION=0 bash research_emcad/scripts/train.sh synapse 2222 3407 5678
USE_CLASS_PROTOTYPE_QUERY=0 USE_CLASS_CONDITIONAL_SCALE_FUSION=1 bash research_emcad/scripts/train.sh synapse 2222 3407 5678
USE_CLASS_PROTOTYPE_QUERY=1 USE_CLASS_CONDITIONAL_SCALE_FUSION=1 bash research_emcad/scripts/train.sh synapse 2222 3407 5678
```

其余任务更换第一个位置参数：`acdc`、`polyp`、`isic`、`busi`、`cell`。Polyp 设 `DATASET_NAME=BKAI|ClinicDB|ColonDB|ETIS|Kvasir`，ISIC 为 `ISIC2017|ISIC2018`，Cell 为 `DSB18|EM`。如 `DATASET_NAME=Kvasir USE_PIXEL_RELIABILITY_FUSION=0 USE_CLASS_PROTOTYPE_QUERY=1 USE_CLASS_CONDITIONAL_SCALE_FUSION=1 bash research_emcad/scripts/train.sh polyp 2222 3407 5678`。脚本会导出开关和隔离的 `OUTPUT_DIR`，原队列按 seed 顺序启动并等待。当前脚本的 Linux Conda 路径和各任务数据路径需要按服务器环境核对。
