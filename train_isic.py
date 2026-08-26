# json 读取固定数据划分的 split_summary.json。
import json
# shutil.copy2 把数据划分清单复制到每次实验目录，保留元数据。
import shutil
# sys.argv 用于判断某个命令行选项是否由用户显式给出。
import sys
# datetime 为未指定 run_name 的实验生成时间戳。
from datetime import datetime
# Path 构造数据集、清单和运行目录。
from pathlib import Path

# ISIC 训练复用 train_polyp 的模型、损失、优化器和完整训练循环。
import train_polyp as _base
# 只替换为 ISIC 严格图像/掩膜配对加载器。
from utils.dataloader_isic import get_loader

# 允许的两个 ISIC 数据集版本。
ALLOWED_DATASETS = {
    # 使用官方 train/val/test 划分的 ISIC 2017。
    "ISIC2017",
    # 按 EMCAD 论文口径准备 80/10/10 的 ISIC 2018。
    "ISIC2018",
}

# 每个版本必须匹配的数据划分协议标识。
EXPECTED_PROTOCOLS = {
    # ISIC 2017 使用官方三划分。
    "ISIC2017": "official_train_val_test",
    # ISIC 2018 使用项目准备脚本记录的图像级 80/10/10。
    "ISIC2018": "emcad_80_10_10_image_level",
}

# 保存 train_polyp 原始解析函数，避免下面替换后递归调用自身。
_original_parse_args = _base.parse_args
# 保存原始验证函数，后面只替换日志中的数据集名称。
_original_evaluate_loader = (
    # train_polyp.evaluate_loader 实际实现不变。
    _base.evaluate_loader
)


# 判断 name 或 name=value 形式的选项是否出现在用户命令行中。
def _option_was_given(name):
    # any 在任一参数命中时返回 True。
    return any(
        # 支持 --option value 形式的独立选项名。
        argument == name
        # 也支持 --option=value 形式。
        or argument.startswith(name + "=")
        # 跳过 argv[0] 脚本名。
        for argument in sys.argv[1:]
    )


# 在通用 Polyp 参数解析结果上应用 ISIC 默认值和数据划分防护。
def _parse_args():
    # 先取得 train_polyp 定义的全部模型/训练参数。
    args = _original_parse_args()

    # 用户未显式指定数据根时使用 ISIC 准备目录。
    if not _option_was_given("--data_root"):
        # 相对路径从通常项目根工作目录解析。
        args.data_root = "../data/isic/target"

    # 用户未指定版本时默认训练 ISIC2018。
    if not _option_was_given("--dataset_name"):
        # 设置数据集名称。
        args.dataset_name = "ISIC2018"

    # 用户未指定输出根时单独放到 ISIC 模型目录。
    if not _option_was_given("--output_dir"):
        # 设置输出根目录。
        args.output_dir = "./model_pth/ISIC"

    # 拒绝其他名字，防止误把 Polyp 数据送入 ISIC 包装器。
    if args.dataset_name not in ALLOWED_DATASETS:
        # 报告允许值。
        raise ValueError(
            # 错误正文第一段。
            "--dataset_name must be "
            # 错误正文第二段。
            "ISIC2017 or ISIC2018"
        )

    # ISIC 路径要求 RGB 输入以匹配 ImageNet 预训练编码器。
    if args.grayscale:
        # 明确提示移除冲突参数。
        raise ValueError(
            # 错误正文第一段。
            "ISIC training requires RGB input; "
            # 错误正文第二段。
            "remove --grayscale"
        )

    # 未给出运行名时生成包含版本和秒级时间戳的唯一名称。
    if args.run_name is None:
        # 组合运行名。
        args.run_name = (
            # 格式模板。
            "train_ISIC_{}_{}".format(
                # 数据集版本。
                args.dataset_name,
                # 当前本地时间。
                datetime.now().strftime(
                    # 文件名安全时间格式。
                    "%Y-%m-%d_%H%M%S"
                ),
            )
        )

    # 准备后的具体版本根目录。
    dataset_root = (
        # 规范化公共 data_root。
            Path(args.data_root).resolve()
            # 追加 ISIC2017/ISIC2018。
            / args.dataset_name
    )

    # 逐样本划分清单路径。
    manifest_path = (
        # 数据版本根。
            dataset_root
            # 清单文件名。
            / "split_manifest.csv"
    )
    # 划分统计和协议元数据路径。
    summary_path = (
        # 数据版本根。
            dataset_root
            # 摘要文件名。
            / "split_summary.json"
    )

    # 两份元数据都必须存在，确保训练使用可追踪固定划分。
    if (
            # 检查 CSV。
            not manifest_path.is_file()
            # 或检查 JSON。
            or not summary_path.is_file()
    ):
        # 告知先运行准备脚本并列出缺失路径。
        raise FileNotFoundError(
            # 错误正文第一段。
            "ISIC split metadata is missing. "
            # 错误正文第二段及换行。
            "Run prepare_isic_splits.py first:\n"
            # 插入两条路径。
            "{}\n{}".format(
                # CSV 路径。
                manifest_path,
                # JSON 路径。
                summary_path,
            )
        )

    # 以 UTF-8 读取 JSON 摘要。
    with summary_path.open(
            # 只读文本模式。
            "r",
            # 编码。
            encoding="utf-8",
    ) as stream:
        # 反序列化为字典。
        summary = json.load(stream)

    # 摘要声明的数据集必须与命令行版本一致。
    if (
            # 读取元数据字段。
            summary.get("dataset_name")
            # 与请求版本比较。
            != args.dataset_name
    ):
        # 报告期望和实际。
        raise RuntimeError(
            # 错误正文。
            "split_summary dataset mismatch: "
            # 插入两个值。
            "expected {}, got {}".format(
                # 请求版本。
                args.dataset_name,
                # 摘要版本。
                summary.get("dataset_name"),
            )
        )

    # 查出该数据集版本要求的协议字符串。
    expected_protocol = EXPECTED_PROTOCOLS[
        # 用版本名索引映射。
        args.dataset_name
    ]

    # 实际准备协议必须严格匹配。
    if (
            # 摘要中的 protocol。
            summary.get("protocol")
            # 与预期比较。
            != expected_protocol
    ):
        # 阻止在不同划分协议上混合比较实验。
        raise RuntimeError(
            # 错误正文。
            "split protocol mismatch: "
            # 插入预期/实际。
            "expected {}, got {}".format(
                # 预期协议。
                expected_protocol,
                # 实际协议。
                summary.get("protocol"),
            )
        )

    # 与 train_polyp.main 最终写入位置一致地预先构造运行目录。
    run_dir = (
        # 输出根绝对路径。
            Path(args.output_dir).resolve()
            # 数据集版本子目录。
            / args.dataset_name
            # 当前运行名子目录。
            / args.run_name
    )

    # 创建运行目录及父目录；已存在时不报错。
    run_dir.mkdir(
        # 递归创建。
        parents=True,
        # 允许存在。
        exist_ok=True,
    )

    # 把实际划分 CSV 复制进实验目录，实现训练结果与样本名单绑定。
    shutil.copy2(
        # 来源清单。
        manifest_path,
        # 实验内固定名称。
        run_dir / "data_split_manifest.csv",
    )
    # 同样复制协议/统计摘要。
    shutil.copy2(
        # 来源摘要。
        summary_path,
        # 实验内固定名称。
        run_dir / "data_split_summary.json",
    )

    # 返回增强后的 Namespace 给通用训练主函数。
    return args


# 包装通用验证函数，仅把进度条/日志中的 Polyp 文案替换为 ISIC。
def _evaluate_loader(*args, **kwargs):
    # 读取可选 description 关键字。
    description = kwargs.get("description")

    # 只有字符串才执行替换。
    if isinstance(description, str):
        # 更新 kwargs 中的描述。
        kwargs["description"] = (
            # 保持其余文字不变。
            description.replace(
                # 原通用名称。
                "Polyp",
                # ISIC 名称。
                "ISIC",
            )
        )

    # 将全部位置/关键字参数原样传给原函数。
    return _original_evaluate_loader(
        # 解包位置参数。
        *args,
        # 解包关键字参数。
        **kwargs,
    )


# 运行前猴子补丁：替换通用模块的参数解析器。
_base.parse_args = _parse_args
# 替换为 ISIC 专用加载器。
_base.get_loader = get_loader
# 替换为仅改日志名称的验证包装器。
_base.evaluate_loader = _evaluate_loader

# 直接执行本文件时进入 train_polyp 的通用 main；被导入时只完成定义和补丁。
if __name__ == "__main__":
    # 启动完整 ISIC 训练流程。
    _base.main()
