# hashlib 重新计算当前 manifest.csv 内容哈希。
import hashlib
# json 读取训练 config.json 与数据 split_summary.json。
import json
# shutil.copy2 保存测试使用的数据划分证据。
import shutil
# sys.argv 识别用户是否显式给出结构/阈值选项。
import sys
# Path 解析检查点、配置、数据和输出路径。
from pathlib import Path

# BUSI 复用 test_polyp 的模型、推理、二分类指标和结果保存流程。
import test_polyp as _base
# BUSI 加载器额外检查 manifest 和类别分布。
from utils.dataloader_busi import (
    # 准备数据允许的扩展名。
    SUPPORTED_EXTENSIONS,
    # BUSI DataLoader。
    get_loader,
)


# 训练/测试必须共同使用的固定划分协议。
EXPECTED_PROTOCOL = (
    # 图像级分层80/10/10。
    "emcad_80_10_10_stratified_image_level"
)

# 固定划分总数。
EXPECTED_COUNTS = {
    # train。
    "train": 517,
    # val。
    "val": 65,
    # test。
    "test": 65,
}

# 固定良性/恶性分布。
EXPECTED_CLASS_COUNTS = {
    # 训练。
    "train": {
        # 良性。
        "benign": 349,
        # 恶性。
        "malignant": 168,
    },
    # 验证。
    "val": {
        # 良性。
        "benign": 44,
        # 恶性。
        "malignant": 21,
    },
    # 测试。
    "test": {
        # 良性。
        "benign": 44,
        # 恶性。
        "malignant": 21,
    },
}

# 会影响模型结构/输入的 config 字段与命令行开关映射。
ARCHITECTURE_OPTIONS = {
    # 编码器。
    "encoder": "--encoder",
    # MSDC 核尺寸。
    "kernel_sizes": "--kernel_sizes",
    # MSCB 扩张倍数。
    "expansion_factor": (
        # 对应命令行选项。
        "--expansion_factor"
    ),
    # LGAG 核。
    "lgag_ks": "--lgag_ks",
    # MSCB 激活。
    "activation_mscb": (
        # 命令行选项。
        "--activation_mscb"
    ),
    # MSDC 串行开关。
    "no_dw_parallel": (
        # 命令行选项。
        "--no_dw_parallel"
    ),
    # 拼接聚合开关。
    "concatenation": (
        # 命令行选项。
        "--concatenation"
    ),
    # 输入尺寸。
    "img_size": "--img_size",
    # 灰度模式。
    "grayscale": "--grayscale",
}

# 保存通用解析器。
_original_parse_args = (
    # test_polyp.parse_args。
    _base.parse_args
)
# 保存通用评测函数。
_original_evaluate_loader = (
    # test_polyp.evaluate_loader。
    _base.evaluate_loader
)


# 判断选项是否由用户显式提供。
def _option_was_given(name):
    # 任一参数命中返回True。
    return any(
        # 独立选项形式。
        argument == name
        # 或 name=value。
        or argument.startswith(
            # 构造前缀。
            name + "="
        )
        # 跳过脚本名。
        for argument in sys.argv[1:]
    )


# 流式计算文件 SHA-256。
def _sha256_file(path):
    # 初始化哈希。
    digest = hashlib.sha256()

    # 二进制打开。
    with Path(path).open("rb") as stream:
        # 每次读1MiB直到EOF。
        for block in iter(
            # 延迟读函数。
            lambda: stream.read(
                # 块大小。
                1024 * 1024
            ),
            # EOF 哨兵。
            b"",
        ):
            # 更新摘要。
            digest.update(block)

    # 返回十六进制摘要。
    return digest.hexdigest()


# 统一 JSON list 与 argparse tuple 便于结构参数比较。
def _normalized(value):
    # tuple 转 list。
    if isinstance(value, tuple):
        # 保持元素顺序和值。
        return list(value)

    # 其他类型原样返回。
    return value


# 在通用 Polyp 测试参数上恢复 BUSI 训练配置并验证数据版本。
def _parse_args():
    # 解析通用命令行参数。
    args = _original_parse_args()

    # 未显式指定数据根时使用 BUSI 目录。
    if not _option_was_given(
        # 检查选项。
        "--data_root"
    ):
        # 设置默认路径。
        args.data_root = (
            # 准备数据根。
            "../data/busi/target"
        )

    # 检查点转绝对 Path。
    checkpoint = Path(
        # 参数字符串。
        args.checkpoint
    # 规范路径。
    ).resolve()

    # 训练配置与检查点同目录。
    config_path = (
        # 检查点父目录。
        checkpoint.parent
        # 配置文件。
        / "config.json"
    )

    # 缺少 config 无法安全重建网络。
    if not config_path.is_file():
        # 报告路径。
        raise FileNotFoundError(
            # 错误正文第一段。
            "Checkpoint config is required "
            # 第二段。
            "for safe BUSI testing: "
            # 插入路径。
            "{}".format(config_path)
        )

    # 读取训练配置。
    with config_path.open(
        # 只读。
        "r",
        # UTF-8。
        encoding="utf-8",
    ) as stream:
        # JSON转字典。
        config = json.load(stream)

    # 读取保存的数据集名。
    config_dataset = config.get(
        # 字段名。
        "dataset_name"
    )

    # 用户未传 dataset_name 时继承检查点。
    if not _option_was_given(
        # 检查选项。
        "--dataset_name"
    ):
        # 设置参数。
        args.dataset_name = (
            # 保存值。
            config_dataset
        )

    # 检查点和请求必须都为 BUSI。
    if (
        # 请求值检查。
        args.dataset_name != "BUSI"
        # 或保存值检查。
        or config_dataset != "BUSI"
    ):
        # 报告两边值。
        raise RuntimeError(
            # 错误正文第一段。
            "Checkpoint and requested "
            # 第二段。
            "dataset must both be BUSI: "
            # 格式化保存/请求值。
            "checkpoint={} requested={}".format(
                # 保存值。
                config_dataset,
                # 请求值。
                args.dataset_name,
            )
        )

    # 逐个恢复所有模型结构字段。
    for (
        # config字段。
        field,
        # 命令行选项。
        option,
    ) in ARCHITECTURE_OPTIONS.items():
        # 缺字段无法确定 state_dict shape。
        if field not in config:
            # 报告字段。
            raise RuntimeError(
                # 错误正文。
                "Missing architecture field "
                # 插入字段名。
                "in config.json: {}".format(
                    # field。
                    field
                )
            )

        # 规范保存值。
        configured = _normalized(
            # config值。
            config[field]
        )
        # 规范当前命令行解析值。
        current = _normalized(
            # 动态取属性。
            getattr(args, field)
        )

        # 用户显式给冲突结构时拒绝运行。
        if (
            # 是否显式提供。
            _option_was_given(option)
            # 是否与训练不同。
            and current != configured
        ):
            # 报告冲突。
            raise RuntimeError(
                # 错误正文第一段。
                "{} conflicts with checkpoint "
                # 第二段。
                "config: requested={} "
                # 插入选项、请求值、保存值。
                "saved={}".format(
                    # 选项。
                    option,
                    # 请求值。
                    current,
                    # 保存值。
                    configured,
                )
            )

        # 强制应用训练保存值。
        setattr(
            # Namespace。
            args,
            # 属性名。
            field,
            # 值。
            config[field],
        )

    # BUSI 检查点不应是灰度输入。
    if args.grayscale:
        # 报告异常配置。
        raise RuntimeError(
            # 错误正文第一段。
            "BUSI checkpoint unexpectedly "
            # 第二段。
            "uses grayscale model input"
        )

    # 二分类概率阈值必须继承训练配置。
    saved_threshold = config.get(
        # 字段名。
        "threshold"
    )

    # 训练配置缺阈值时拒绝猜测默认值。
    if saved_threshold is None:
        # 报告缺失。
        raise RuntimeError(
            # 错误正文第一段。
            "Training config has no "
            # 字段名。
            "threshold"
        )

    # 用户显式阈值与保存值冲突时拒绝静默覆盖。
    if (
        # 检查选项是否提供。
        _option_was_given(
            # 阈值选项。
            "--threshold"
        )
        # 且浮点差异大于数值容差。
        and abs(
            # 当前阈值。
            float(args.threshold)
            # 减保存阈值。
            - float(saved_threshold)
        )
        # 绝对容差。
        > 1e-12
    ):
        # 报告冲突。
        raise RuntimeError(
            # 错误正文第一段。
            "--threshold conflicts with "
            # 第二段。
            "checkpoint config: requested={} "
            # 格式化值。
            "saved={}".format(
                # 请求值。
                args.threshold,
                # 保存值。
                saved_threshold,
            )
        )

    # 最终使用训练时记录的阈值。
    args.threshold = float(
        # 保存值转float。
        saved_threshold
    )

    # 当前准备数据 BUSI 根。
    dataset_root = (
        # 公共数据根绝对路径。
        Path(args.data_root).resolve()
        # BUSI 子目录。
        / "BUSI"
    )

    # 样本清单。
    manifest_path = (
        # 拼固定文件名。
        dataset_root / "manifest.csv"
    )
    # 划分摘要。
    summary_path = (
        # 数据根。
        dataset_root
        # 文件名。
        / "split_summary.json"
    )

    # 两份准备元数据必须存在。
    if (
        # 清单检查。
        not manifest_path.is_file()
        # 摘要检查。
        or not summary_path.is_file()
    ):
        # 报告路径。
        raise FileNotFoundError(
            # 错误正文第一段。
            "Prepared BUSI metadata is "
            # 第二段并换行。
            "missing:\n{}\n{}".format(
                # manifest。
                manifest_path,
                # summary。
                summary_path,
            )
        )

    # 读取当前划分摘要。
    with summary_path.open(
        # 只读。
        "r",
        # UTF-8。
        encoding="utf-8",
    ) as stream:
        # JSON转字典。
        summary = json.load(stream)

    # 摘要必须属于 BUSI。
    if (
        # 字段值。
        summary.get("dataset_name")
        # 与常量比较。
        != "BUSI"
    ):
        # 拒绝错误数据目录。
        raise RuntimeError(
            # 错误正文第一段。
            "Current split_summary.json "
            # 第二段。
            "is not BUSI"
        )

    # 当前划分协议必须与训练协议相同。
    if (
        # 当前协议。
        summary.get("protocol")
        # 预期协议。
        != EXPECTED_PROTOCOL
    ):
        # 报告不匹配。
        raise RuntimeError(
            # 错误正文第一段。
            "Current BUSI split protocol "
            # 第二段。
            "is not the training protocol"
        )

    # 当前总数必须是固定517/65/65。
    if (
        # 摘要计数。
        summary.get("counts")
        # 预期计数。
        != EXPECTED_COUNTS
    ):
        # 拒绝数据缺失/替换。
        raise RuntimeError(
            # 错误正文第一段。
            "Current BUSI split counts "
            # 第二段。
            "are invalid"
        )

    # 当前类别分布也必须一致。
    if (
        # 摘要分布。
        summary.get("class_counts")
        # 预期分布。
        != EXPECTED_CLASS_COUNTS
    ):
        # 报告无效。
        raise RuntimeError(
            # 错误正文第一段。
            "Current BUSI class counts "
            # 第二段。
            "are invalid"
        )

    # 重新计算当前 manifest.csv 文件哈希。
    current_manifest_sha256 = (
        # 流式哈希。
        _sha256_file(manifest_path)
    )

    # 当前文件内容必须与摘要记录一致。
    if (
        # 记录值。
        summary.get(
            # 字段名。
            "manifest_file_sha256"
        )
        # 与当前值比较。
        != current_manifest_sha256
    ):
        # 说明清单文件已改变。
        raise RuntimeError(
            # 错误正文第一段。
            "Current manifest.csv hash "
            # 第二段。
            "differs from "
            # 第三段。
            "split_summary.json"
        )

    # 训练 config 中的规范清单哈希必须与当前摘要一致。
    if (
        # 训练保存值。
        config.get(
            # 字段名。
            "split_manifest_sha256"
        )
        # 与当前数据值比较。
        != summary.get(
            # 当前字段名。
            "manifest_sha256"
        )
    ):
        # 防止在不同划分上测试。
        raise RuntimeError(
            # 错误正文第一段。
            "BUSI split manifest differs "
            # 第二段。
            "from the one used for training"
        )

    # 使用显式输出目录，或在检查点旁按split生成默认目录。
    output_dir = Path(
        # 用户值优先。
        args.output_dir
        # 否则构造默认。
        or (
            # 检查点运行目录。
            checkpoint.parent
            # 追加目录名。
            / "{}_BUSI_outputs".format(
                # val/test。
                args.split
            )
        )
    # 规范绝对路径。
    ).resolve()

    # 创建输出目录。
    output_dir.mkdir(
        # 递归。
        parents=True,
        # 可复用空/已有目录。
        exist_ok=True,
    )

    # 复制 manifest 作为评测证据。
    shutil.copy2(
        # 来源。
        manifest_path,
        # 目标目录。
        output_dir
        # 目标名。
        / "data_split_manifest.csv",
    )
    # 复制摘要。
    shutil.copy2(
        # 来源。
        summary_path,
        # 目标目录。
        output_dir
        # 目标名。
        / "data_split_summary.json",
    )

    # 输出实际检查点配置路径。
    print(
        # 格式化日志。
        "CHECKPOINT_CONFIG={}".format(
            # 路径。
            config_path
        )
    )
    # 输出当前规范划分哈希。
    print(
        # 格式化日志。
        "SPLIT_MANIFEST_SHA256={}".format(
            # 读取哈希。
            summary.get(
                # 字段名。
                "manifest_sha256"
            )
        )
    )

    # 返回已锁定配置的参数。
    return args


# 包装通用评测函数，替换显示名称。
def _evaluate_loader(
    # 位置参数。
    *args,
    # 关键字参数。
    **kwargs,
):
    # 读取描述。
    description = kwargs.get(
        # 字段名。
        "description"
    )

    # 字符串才替换。
    if isinstance(description, str):
        # 更新 kwargs。
        kwargs["description"] = (
            # Polyp -> BUSI。
            description.replace(
                # 原词。
                "Polyp",
                # 目标词。
                "BUSI",
            )
        )

    # 复用原推理/指标实现。
    return _original_evaluate_loader(
        # 位置参数。
        *args,
        # 关键字参数。
        **kwargs,
    )


# 替换通用参数解析器。
_base.parse_args = _parse_args
# 替换 BUSI DataLoader。
_base.get_loader = get_loader
# 替换允许扩展名集合。
_base.SUPPORTED_EXTENSIONS = (
    # BUSI 仅PNG。
    SUPPORTED_EXTENSIONS
)
# 替换显示包装器。
_base.evaluate_loader = (
    # 包装函数。
    _evaluate_loader
)


# 直接运行时启动通用测试主流程。
if __name__ == "__main__":
    # 执行 BUSI 测试。
    _base.main()
