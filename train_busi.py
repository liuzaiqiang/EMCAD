# hashlib 校验准备数据 manifest.csv 的文件内容。
import hashlib
# json 读取 split_summary.json。
import json
# shutil.copy2 把固定划分元数据复制到实验目录。
import shutil
# sys.argv 判断用户是否显式覆盖通用训练默认值。
import sys
# datetime 生成默认运行时间戳。
from datetime import datetime
# Path 构造 BUSI 数据和输出目录。
from pathlib import Path

# BUSI 复用 train_polyp 的 EMCAD 模型、二分类损失和训练循环。
import train_polyp as _base
# 替换为带 BUSI 清单/类别分布检查的 DataLoader。
from utils.dataloader_busi import (
    # BUSI get_loader。
    get_loader,
)

# prepare_busi_splits.py 写入摘要的预期划分协议。
EXPECTED_PROTOCOL = (
    # 图像级分层 80/10/10；当前元数据会显式记录不是患者级划分。
    "emcad_80_10_10_stratified_image_level"
)

# 三个划分的固定总样本数。
EXPECTED_COUNTS = {
    # 训练总数。
    "train": 517,
    # 验证总数。
    "val": 65,
    # 测试总数。
    "test": 65,
}

# 每个划分内良性/恶性样本数，用于严格复现实验划分。
EXPECTED_CLASS_COUNTS = {
    # 训练分布。
    "train": {
        # 良性。
        "benign": 349,
        # 恶性。
        "malignant": 168,
    },
    # 验证分布。
    "val": {
        # 良性。
        "benign": 44,
        # 恶性。
        "malignant": 21,
    },
    # 测试分布。
    "test": {
        # 良性。
        "benign": 44,
        # 恶性。
        "malignant": 21,
    },
}

# 保存通用参数解析器，供包装函数调用。
_original_parse_args = (
    # train_polyp.parse_args。
    _base.parse_args
)
# 保存通用验证函数，仅修改数据集显示名称。
_original_evaluate_loader = (
    # train_polyp.evaluate_loader。
    _base.evaluate_loader
)


# 检查 --name 或 --name=value 是否由用户显式提供。
def _option_was_given(name):
    # 任一 argv 参数命中即 True。
    return any(
        # 独立选项形式。
        argument == name
        # 或等号形式。
        or argument.startswith(
            # 组成 --name= 前缀。
            name + "="
        )
        # 跳过脚本名。
        for argument in sys.argv[1:]
    )


# 以1MiB块流式计算文件 SHA-256。
def _sha256_file(path):
    # 创建哈希状态。
    digest = hashlib.sha256()

    # 二进制只读打开。
    with Path(path).open("rb") as stream:
        # 反复读块直到空字节串。
        for block in iter(
                # 延迟读取函数。
                lambda: stream.read(
                    # 1MiB。
                    1024 * 1024
                ),
                # EOF 哨兵。
                b"",
        ):
            # 更新哈希。
            digest.update(block)

    # 返回十六进制摘要。
    return digest.hexdigest()


# 在通用 Polyp 参数上应用 BUSI 默认值与固定划分校验。
def _parse_args():
    # 取得全部通用模型/训练参数。
    args = _original_parse_args()

    # 未指定数据根时使用 BUSI 准备目录。
    if not _option_was_given(
            # 检查选项。
            "--data_root"
    ):
        # 设置默认值。
        args.data_root = (
            # 目标数据根。
            "../data/busi/target"
        )

    # 未指定名称时固定 BUSI。
    if not _option_was_given(
            # 检查选项。
            "--dataset_name"
    ):
        # 设置名称。
        args.dataset_name = "BUSI"

    # 未指定输出目录时使用 model_pth 根。
    if not _option_was_given(
            # 检查选项。
            "--output_dir"
    ):
        # 设置输出根。
        args.output_dir = (
            # train_polyp.main 会继续追加 BUSI/run_name。
            "./model_pth"
        )

    # EMCAD 论文 BUSI 实验输入固定为256x256；用户未覆盖时设256。
    if not _option_was_given(
            # 检查尺寸选项。
            "--img_size"
    ):
        # 设置固定尺寸。
        args.img_size = 256

    # 拒绝其他数据集名称。
    if args.dataset_name != "BUSI":
        # 报告要求。
        raise ValueError(
            # 错误正文。
            "--dataset_name must be BUSI"
        )

    # BUSI 以3通道形式送入 ImageNet 预训练 PVT。
    if args.grayscale:
        # 禁止单通道配置漂移。
        raise ValueError(
            # 错误正文分段。
            "BUSI uses 3-channel input "
            # 说明原因。
            "for the ImageNet-pretrained "
            # 编码器名称。
            "PVT encoder; remove "
            # 解决方法。
            "--grayscale"
        )

    # EMCAD specifies fixed 256x256
    # input for BUSI. Polyp/ISIC
    # multi-scale training is disabled.
    # 强制关闭 train_polyp 的 0.75/1/1.25 多尺度训练，保持256固定输入。
    args.no_multi_scale = True

    # 未传运行名时生成秒级时间戳名称。
    if args.run_name is None:
        # 拼运行名。
        args.run_name = (
            # BUSI 前缀。
            "train_BUSI_{}".format(
                # 当前时间字符串。
                datetime.now().strftime(
                    # 文件名安全格式。
                    "%Y-%m-%d_%H%M%S"
                )
            )
        )

    # 准备数据具体 BUSI 根目录。
    dataset_root = (
        # data_root 绝对路径。
            Path(args.data_root).resolve()
            # BUSI 子目录。
            / "BUSI"
    )

    # 样本清单路径。
    manifest_path = (
        # 拼 manifest.csv。
            dataset_root / "manifest.csv"
    )
    # 划分摘要路径。
    summary_path = (
        # 数据根。
            dataset_root
            # 摘要文件。
            / "split_summary.json"
    )

    # 两份准备元数据必须同时存在。
    if (
            # 清单检查。
            not manifest_path.is_file()
            # 摘要检查。
            or not summary_path.is_file()
    ):
        # 告知运行准备脚本并列出路径。
        raise FileNotFoundError(
            # 错误正文第一段。
            "Prepared BUSI metadata is "
            # 第二段。
            "missing. Run "
            # 第三段和换行。
            "prepare_busi_splits.py first:\n"
            # 插入路径。
            "{}\n{}".format(
                # manifest。
                manifest_path,
                # summary。
                summary_path,
            )
        )

    # 读取 UTF-8 JSON 摘要。
    with summary_path.open(
            # 只读。
            "r",
            # 编码。
            encoding="utf-8",
    ) as stream:
        # 转字典。
        summary = json.load(stream)

    # 摘要必须声明 BUSI。
    if (
            # 读取字段。
            summary.get("dataset_name")
            # 与常量比较。
            != "BUSI"
    ):
        # 拒绝拿其他数据集摘要训练。
        raise RuntimeError(
            # 错误正文第一段。
            "split_summary.json is not "
            # 错误正文第二段。
            "for BUSI"
        )

    # 划分协议必须与本代码预期一致。
    if (
            # 实际协议。
            summary.get("protocol")
            # 预期协议。
            != EXPECTED_PROTOCOL
    ):
        # 报告双方值。
        raise RuntimeError(
            # 错误正文。
            "BUSI split protocol mismatch: "
            # 插入预期/实际。
            "expected={} actual={}".format(
                # 预期。
                EXPECTED_PROTOCOL,
                # 实际。
                summary.get("protocol"),
            )
        )

    # 三个划分总数必须完全一致。
    if (
            # 摘要计数。
            summary.get("counts")
            # 与固定计数比较。
            != EXPECTED_COUNTS
    ):
        # 报告不一致。
        raise RuntimeError(
            # 错误正文。
            "BUSI split counts mismatch: "
            # 插入字典。
            "expected={} actual={}".format(
                # 预期计数。
                EXPECTED_COUNTS,
                # 实际计数。
                summary.get("counts"),
            )
        )

    # 每个划分的良性/恶性数量也必须一致。
    if (
            # 实际类别分布。
            summary.get("class_counts")
            # 与预期比较。
            != EXPECTED_CLASS_COUNTS
    ):
        # 报告分布不一致。
        raise RuntimeError(
            # 错误正文。
            "BUSI class counts mismatch: "
            # 插入预期/实际。
            "expected={} actual={}".format(
                # 预期。
                EXPECTED_CLASS_COUNTS,
                # 实际从摘要取值。
                summary.get(
                    # 字段名。
                    "class_counts"
                ),
            )
        )

    # 重新计算当前 manifest.csv 文件内容哈希。
    manifest_file_sha256 = (
        # 流式哈希函数。
        _sha256_file(manifest_path)
    )

    # 文件哈希必须与摘要生成时记录值一致。
    if (
            # 摘要字段。
            summary.get(
                # 文件内容哈希键。
                "manifest_file_sha256"
            )
            # 与当前计算值比较。
            != manifest_file_sha256
    ):
        # 表示元数据文件被替换或修改。
        raise RuntimeError(
            # 错误正文第一段。
            "manifest.csv hash differs "
            # 错误正文第二段。
            "from split_summary.json"
        )

    # 取得规范样本清单哈希；它通常基于样本字段而非CSV原始字节。
    manifest_sha256 = summary.get(
        # 字段名。
        "manifest_sha256"
    )

    # SHA-256 十六进制应为64字符字符串。
    if (
            # 类型检查。
            not isinstance(
                # 值。
                manifest_sha256,
                # 期望类型。
                str,
            )
            # 或长度检查。
            or len(manifest_sha256) != 64
    ):
        # 拒绝无效实验指纹。
        raise RuntimeError(
            # 错误正文第一段。
            "Invalid BUSI manifest "
            # 错误正文第二段。
            "SHA-256 in split_summary.json"
        )

    # 将划分协议附加到 args，train_polyp 会写入 config.json。
    args.split_protocol = (
        # 固定协议。
        EXPECTED_PROTOCOL
    )
    # 记录划分单位，例如 image。
    args.split_unit = summary.get(
        # 字段名。
        "split_unit"
    )
    # 记录规范清单哈希。
    args.split_manifest_sha256 = (
        # 64字符摘要。
        manifest_sha256
    )
    # 明确记录是否为患者级划分；本协议通常为 False。
    args.patient_level_split = bool(
        # 从摘要取值，缺失默认False。
        summary.get(
            # 字段名。
            "patient_level_split",
            # 默认值。
            False,
        )
    )
    # 记录重复样本处理策略。
    args.duplicate_policy = (
        # 从摘要读取。
        summary.get(
            # 字段名。
            "duplicate_policy"
        )
    )

    # 最终运行目录与通用训练主函数规则一致。
    run_dir = (
        # 输出根绝对路径。
            Path(args.output_dir).resolve()
            # BUSI 子目录。
            / args.dataset_name
            # 当前运行名。
            / args.run_name
    )

    # 已有非空目录意味着可能覆盖旧实验，直接拒绝。
    if (
            # 目录存在。
            run_dir.exists()
            # 且至少含一个条目。
            and any(run_dir.iterdir())
    ):
        # 报告目标目录。
        raise FileExistsError(
            # 错误正文第一段。
            "BUSI run directory is not "
            # 第二段。
            "empty; refusing to overwrite: "
            # 插入路径。
            "{}".format(run_dir)
        )

    # 创建新的运行目录。
    run_dir.mkdir(
        # 递归父目录。
        parents=True,
        # 空目录已存在时允许。
        exist_ok=True,
    )

    # 复制实际样本清单到运行目录。
    shutil.copy2(
        # 来源。
        manifest_path,
        # 目标目录表达式第一段。
        run_dir
        # 目标文件名。
        / "data_split_manifest.csv",
    )
    # 复制划分摘要到运行目录。
    shutil.copy2(
        # 来源。
        summary_path,
        # 目标目录。
        run_dir
        # 目标文件名。
        / "data_split_summary.json",
    )

    # 返回扩展参数给通用主函数。
    return args


# 包装通用验证函数，只把显示名称改成 BUSI。
def _evaluate_loader(
        # 任意位置参数。
        *args,
        # 任意关键字参数。
        **kwargs,
):
    # 读取描述。
    description = kwargs.get(
        # 字段名。
        "description"
    )

    # 字符串才替换。
    if isinstance(description, str):
        # 更新描述。
        kwargs["description"] = (
            # 保持其他文字不变。
            description.replace(
                # 原通用名。
                "Polyp",
                # BUSI 名。
                "BUSI",
            )
        )

    # 调用原验证实现。
    return _original_evaluate_loader(
        # 位置参数。
        *args,
        # 关键字参数。
        **kwargs,
    )


# 替换通用参数解析器。
_base.parse_args = _parse_args
# 替换为 BUSI 加载器。
_base.get_loader = get_loader
# 替换验证显示包装器。
_base.evaluate_loader = (
    # 包装函数。
    _evaluate_loader
)

# 直接执行时进入通用训练主流程。
if __name__ == "__main__":
    # 启动 BUSI 训练。
    _base.main()
