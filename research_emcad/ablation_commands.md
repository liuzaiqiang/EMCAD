# 四组启动命令

在项目根目录运行，同任务的四组使用相同数据设置和 seed 队列：

```bash
USE_CLASS_CONDITIONAL_SCALE_FUSION=0 USE_SPATIAL_KERNEL_SELECTION=0 bash research_emcad/scripts/train.sh synapse 2222 3407 5678
USE_CLASS_CONDITIONAL_SCALE_FUSION=0 USE_SPATIAL_KERNEL_SELECTION=1 bash research_emcad/scripts/train.sh synapse 2222 3407 5678
USE_CLASS_CONDITIONAL_SCALE_FUSION=1 USE_SPATIAL_KERNEL_SELECTION=0 bash research_emcad/scripts/train.sh synapse 2222 3407 5678
USE_CLASS_CONDITIONAL_SCALE_FUSION=1 USE_SPATIAL_KERNEL_SELECTION=1 bash research_emcad/scripts/train.sh synapse 2222 3407 5678
```

其他任务将 `synapse` 改为 `acdc`、`polyp`、`isic`、`busi` 或 `cell`。Polyp 设置 `DATASET_NAME=BKAI|ClinicDB|ColonDB|ETIS|Kvasir`，ISIC 设置 `ISIC2017|ISIC2018`，Cell 设置 `DSB18|EM`。例如：

```bash
DATASET_NAME=Kvasir USE_CLASS_CONDITIONAL_SCALE_FUSION=1 USE_SPATIAL_KERNEL_SELECTION=1 bash research_emcad/scripts/train.sh polyp 2222 3407 5678
```

脚本按四组开关隔离 `OUTPUT_DIR`，并通过原 seed 队列顺序运行。各数据集的四组结果必须分别比较仅新模块对基础、双开对基础和双开对仅尺度融合的 Dice。Linux Conda 路径及数据路径需与服务器环境匹配。
