# json 读取训练运行目录中的 config.json 和数据划分摘要。
import json
# shutil.copy2 把测试所用划分元数据复制到输出目录。
import shutil
# sys.argv 判断用户是否显式覆盖模型结构参数。
import sys
# Path 解析检查点、配置、数据清单和输出路径。
from pathlib import Path

# ISIC 测试复用 test_polyp 的模型构建、检查点加载、推理和指标逻辑。
import test_polyp as _base
# 导入 ISIC 文件扩展名约束和专用加载器。
from utils.dataloader_isic import (
    # 允许的输入图像后缀集合。
    SUPPORTED_EXTENSIONS,
    # ISIC DataLoader 构造器。
    get_loader,
)


# 允许测试的数据版本。
ALLOWED_DATASETS = {
    # ISIC 2017。
    "ISIC2017",
    # ISIC 2018。
    "ISIC2018",
}

# 数据版本到固定划分协议的映射。
EXPECTED_PROTOCOLS = {
    # 官方 ISIC2017 三划分。
    "ISIC2017": "official_train_val_test",
    # EMCAD 口径 ISIC2018 图像级 80/10/10。
    "ISIC2018": "emcad_80_10_10_image_level",
}

# 必须从训练 config.json 恢复并检查的模型结构字段及其命令行选项。
ARCHITECTURE_OPTIONS = {
    # 编码器类型。
    "encoder": "--encoder",
    # MSDC 多尺度核。
    "kernel_sizes": "--kernel_sizes",
    # MSCB 扩张倍数。
    "expansion_factor": "--expansion_factor",
    # LGAG 核大小。
    "lgag_ks": "--lgag_ks",
    # MSCB 激活函数。
    "activation_mscb": "--activation_mscb",
    # 并行/串行 MSDC 开关。
    "no_dw_parallel": "--no_dw_parallel",
    # 相加/拼接聚合开关。
    "concatenation": "--concatenation",
    # 模型输入尺寸。
    "img_size": "--img_size",
    # 灰度/RGB 模式。
    "grayscale": "--grayscale",
}

# 保存通用测试参数解析函数，避免补丁递归。
_original_parse_args = _base.parse_args
# 保存通用评测函数，只包装显示文案。
_original_evaluate_loader = (
    # test_polyp.evaluate_loader。
    _base.evaluate_loader
)


# 判断命令行是否显式包含某选项。
def _option_was_given(name):
    # 任一参数匹配即返回 True。
    return any(
        # 支持独立 --option。
        argument == name
        # 支持 --option=value。
        or argument.startswith(name + "=")
        # 跳过脚本名。
        for argument in sys.argv[1:]
    )


# 把 JSON 反序列化后可能的 list 与 argparse 可能的 tuple 统一为可比较形式。
def _normalized(value):
    # tuple 转 list。
    if isinstance(value, tuple):
        # 保持元素不变。
        return list(value)

    # 其他类型直接返回。
    return value


# 在通用 Polyp 测试参数上加载并锁定 ISIC 训练配置。
def _parse_args():
    # 解析所有通用参数。
    args = _original_parse_args()

    # 未指定数据根时使用 ISIC 目录。
    if not _option_was_given("--data_root"):
        # 设置默认路径。
        args.data_root = "../data/isic/target"

    # 把检查点转换成绝对 Path。
    checkpoint = Path(
        # argparse 字符串。
        args.checkpoint
    # resolve 规范路径。
    ).resolve()

    # 训练配置必须与 best.pth 等检查点位于同一目录。
    config_path = (
        # 检查点父目录。
        checkpoint.parent
        # 配置文件名。
        / "config.json"
    )

    # 没有配置就无法可靠重建相同网络结构。
    if not config_path.is_file():
        # 报告缺失路径。
        raise FileNotFoundError(
            # 错误正文第一段。
            "Checkpoint config is required "
            # 插入路径。
            "for safe ISIC testing: {}".format(
                # config 路径。
                config_path
            )
        )

    # 读取训练配置 JSON。
    with config_path.open(
        # 只读。
        "r",
        # UTF-8。
        encoding="utf-8",
    ) as stream:
        # 解析为字典。
        config = json.load(stream)

    # 取得检查点记录的数据集版本。
    config_dataset = config.get(
        # 字段名。
        "dataset_name"
    )

    # 用户未指定版本时自动继承检查点配置。
    if not _option_was_given(
        # 需要检查的选项。
        "--dataset_name"
    ):
        # 设置为保存值。
        args.dataset_name = config_dataset

    # 最终版本必须受支持。
    if (
        # 参数值。
        args.dataset_name
        # 集合成员检查。
        not in ALLOWED_DATASETS
    ):
        # 报告允许值。
        raise ValueError(
            # 错误正文第一段。
            "--dataset_name must be "
            # 错误正文第二段。
            "ISIC2017 or ISIC2018"
        )

    # 检查点数据版本必须与待测试版本一致。
    if config_dataset != args.dataset_name:
        # 阻止跨数据集错误加载。
        raise RuntimeError(
            # 错误正文。
            "Checkpoint/data mismatch: "
            # 插入保存值和请求值。
            "checkpoint={} requested={}".format(
                # 保存版本。
                config_dataset,
                # 请求版本。
                args.dataset_name,
            )
        )

    # 逐个锁定所有会影响 state_dict shape 或前向行为的结构字段。
    for (
        # config 字段名。
        field,
        # 对应命令行选项。
        option,
    ) in ARCHITECTURE_OPTIONS.items():
        # 配置缺字段意味着无法完整重建训练模型。
        if field not in config:
            # 报告具体字段。
            raise RuntimeError(
                # 错误正文。
                "Missing architecture field "
                # 插入字段名。
                "in config.json: {}".format(
                    # field。
                    field
                )
            )

        # 规范化保存值以比较 tuple/list。
        configured = _normalized(
            # 配置字段值。
            config[field]
        )
        # 规范化当前 argparse 值。
        current = _normalized(
            # 动态读取 Namespace 属性。
            getattr(args, field)
        )

        # 用户显式给了冲突值时拒绝静默覆盖。
        if (
            # 判断是否显式提供。
            _option_was_given(option)
            # 并且值不同。
            and current != configured
        ):
            # 报告选项、请求值和保存值。
            raise RuntimeError(
                # 错误正文。
                "{} conflicts with checkpoint "
                # 格式化三项。
                "config: requested={} saved={}".format(
                    # 命令行选项名。
                    option,
                    # 用户请求值。
                    current,
                    # 训练保存值。
                    configured,
                )
            )

        # 无冲突后强制使用训练保存值重建模型。
        setattr(
            # 目标 Namespace。
            args,
            # 属性名。
            field,
            # 原始配置值。
            config[field],
        )

    # ISIC 加载器不允许灰度，若训练配置异常则立即报错。
    if args.grayscale:
        # 不尝试自动转换检查点结构。
        raise RuntimeError(
            # 错误正文第一段。
            "ISIC checkpoint unexpectedly "
            # 错误正文第二段。
            "uses grayscale input"
        )

    # 具体数据版本目录。
    dataset_root = (
        # 数据根规范绝对路径。
        Path(args.data_root).resolve()
        # ISIC2017/2018 子目录。
        / args.dataset_name
    )

    # 逐样本划分清单。
    manifest_path = (
        # 数据版本目录。
        dataset_root
        # CSV 文件名。
        / "split_manifest.csv"
    )
    # 划分统计与协议摘要。
    summary_path = (
        # 数据版本目录。
        dataset_root
        # JSON 文件名。
        / "split_summary.json"
    )

    # 两份元数据缺一不可。
    if (
        # CSV 不存在。
        not manifest_path.is_file()
        # 或 JSON 不存在。
        or not summary_path.is_file()
    ):
        # 报告两条期望路径。
        raise FileNotFoundError(
            # 错误正文。
            "ISIC split metadata is missing:\n"
            # 插入路径。
            "{}\n{}".format(
                # manifest。
                manifest_path,
                # summary。
                summary_path,
            )
        )

    # 读取划分摘要。
    with summary_path.open(
        # 只读。
        "r",
        # UTF-8。
        encoding="utf-8",
    ) as stream:
        # JSON 转字典。
        summary = json.load(stream)

    # 摘要版本必须与检查点版本一致。
    if (
        # 元数据版本。
        summary.get("dataset_name")
        # 与最终参数比较。
        != args.dataset_name
    ):
        # 防止检查点与另一版本数据混用。
        raise RuntimeError(
            # 错误正文第一段。
            "Dataset split summary does "
            # 错误正文第二段。
            "not match checkpoint"
        )

    # 划分协议也必须匹配该版本预期。
    if (
        # 实际协议。
        summary.get("protocol")
        # 与映射中的预期比较。
        != EXPECTED_PROTOCOLS[
            # 数据集版本键。
            args.dataset_name
        ]
    ):
        # 阻止在不一致划分上报告不可比结果。
        raise RuntimeError(
            # 错误正文第一段。
            "Dataset split protocol does "
            # 错误正文第二段。
            "not match EMCAD setup"
        )

    # 使用显式 output_dir，或在检查点旁按 split+版本生成默认目录。
    output_dir = Path(
        # 优先使用用户参数。
        args.output_dir
        # 若为空则构造默认值。
        or (
            # 检查点运行目录。
            checkpoint.parent
            # 追加输出目录名。
            / "{}_{}_outputs".format(
                # val/test。
                args.split,
                # ISIC2017/2018。
                args.dataset_name,
            )
        )
    # 规范为绝对路径。
    ).resolve()

    # 创建输出目录。
    output_dir.mkdir(
        # 递归创建父目录。
        parents=True,
        # 已存在时复用。
        exist_ok=True,
    )

    # 复制逐样本划分清单到评测输出，绑定结果与测试样本。
    shutil.copy2(
        # 来源 CSV。
        manifest_path,
        # 输出内固定名称。
        output_dir / "data_split_manifest.csv",
    )
    # 复制划分摘要。
    shutil.copy2(
        # 来源 JSON。
        summary_path,
        # 输出内固定名称。
        output_dir / "data_split_summary.json",
    )

    # 在控制台记录实际加载的训练配置文件。
    print(
        # 格式化键值日志。
        "CHECKPOINT_CONFIG={}".format(
            # config 路径。
            config_path
        )
    )

    # 返回已由检查点配置锁定的参数。
    return args


# 包装通用评测函数，只修改进度/日志描述中的数据集名称。
def _evaluate_loader(*args, **kwargs):
    # 读取 description。
    description = kwargs.get("description")

    # 字符串才替换。
    if isinstance(description, str):
        # 更新描述。
        kwargs["description"] = (
            # 把通用 Polyp 文案替换为 ISIC。
            description.replace(
                # 原词。
                "Polyp",
                # 目标词。
                "ISIC",
            )
        )

    # 其余参数和评测逻辑完全复用 test_polyp。
    return _original_evaluate_loader(
        # 位置参数。
        *args,
        # 关键字参数。
        **kwargs,
    )


# 替换通用模块参数解析器为安全配置恢复版本。
_base.parse_args = _parse_args
# 替换为 ISIC 专用加载器。
_base.get_loader = get_loader
# 替换允许扩展名集合，供通用测试代码校验。
_base.SUPPORTED_EXTENSIONS = (
    # ISIC 图像后缀集合。
    SUPPORTED_EXTENSIONS
)
# 替换评测包装函数。
_base.evaluate_loader = _evaluate_loader


# 直接运行时调用通用测试主流程。
if __name__ == "__main__":
    # 启动 ISIC 检查点评测。
    _base.main()
