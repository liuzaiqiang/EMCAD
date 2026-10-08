"""统一构造训练实验输出目录。

本模块只负责文件系统路径命名，不参与模型构造、损失计算或训练流程。
所有训练入口都通过 :func:`make_experiment_dir` 保存日志、配置和 checkpoint，
从而让不同数据集的输出目录具有相同的参数层级。
"""

from datetime import datetime
from pathlib import Path
import re


def _safe_component(value):
    """把参数值转换成可以跨 Windows/Linux 使用的目录名片段。"""
    if isinstance(value, (list, tuple)):
        value = "-".join(str(item) for item in value)
    text = str(value)
    text = text.replace("[", "").replace("]", "")
    text = re.sub(r"[\\/:*?\"<>|\s,]+", "-", text)
    text = text.strip("-._")
    return text or "na"


def _value(args, name, default="na"):
    """读取可选参数；不同数据集缺少某字段时仍保持目录构造可用。"""
    return getattr(args, name, default)


def make_experiment_dir(args, family, output_root=None):
    """创建并返回统一格式的实验目录绝对路径。

    目录层级按稳定性从高到低排列：数据集、编码器、输入尺寸、batch、
    epoch、监督方式、学习率、模型/融合参数、随机种子、秒级时间戳。
    ``output_root`` 未传时使用 args.output_dir。若旧入口的 output_dir 已经
    以 family 命名（例如 ``model_pth/ACDC``），这里自动提升到其父目录，
    避免生成 ``ACDC/ACDC`` 的重复层级。
    """
    root = Path(output_root or _value(args, "output_dir", "./model_pth")).expanduser()
    if root.name.lower() == str(family).lower():
        root = root.parent

    dataset = _value(args, "dataset_name", _value(args, "dataset", family))
    values = [
        ("dataset", dataset),
        ("encoder", _value(args, "encoder")),
        ("img", _value(args, "img_size")),
        ("batch", _value(args, "batch_size")),
        ("epochs", _value(args, "max_epochs")),
        ("supervision", _value(args, "supervision")),
        ("lr", _value(args, "base_lr")),
        ("expansion", _value(args, "expansion_factor")),
        ("kernels", _value(args, "kernel_sizes")),
        ("lgag_ks", _value(args, "lgag_ks")),
        ("act", _value(args, "activation_mscb")),
        ("caa", _value(args, "caa_mode")),
        ("seed", _value(args, "seed")),
    ]
    parameter_parts = [
        Path(_safe_component(label) + "_" + _safe_component(value))
        for label, value in values
    ]

    # 秒级时间戳满足可读性；冲突时追加编号，避免覆盖同一秒内启动的实验。
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    base = root.joinpath(str(family), *parameter_parts, timestamp)
    candidate = base
    suffix = 1
    while candidate.exists():
        candidate = Path(str(base) + "_{:02d}".format(suffix))
        suffix += 1
    candidate.mkdir(parents=True, exist_ok=False)
    return str(candidate.resolve())
