# csv 读取准备数据时生成的 manifest.csv。
import csv
# hashlib 计算 manifest 文件内容的 SHA-256。
import hashlib
# Counter 汇总 train/val/test 中 benign 与 malignant 的样本数量。
from collections import Counter
# Path 提供目录推导、文件检查和读取。
from pathlib import Path

# BUSI 复用通用二分类加载器，只在外层增加数据清单与类别分布校验。
from utils.dataloader_polyp import (
    # 使用私有别名避免与本文件对外 get_loader 同名。
    get_loader as _get_polyp_loader,
)

# 准备后的 BUSI 图像和掩膜只允许无损 PNG。
SUPPORTED_EXTENSIONS = {".png"}
# 本项目排除 normal 类，仅训练良性/恶性病灶分割。
VALID_CLASSES = {
    # 良性病灶。
    "benign",
    # 恶性病灶。
    "malignant",
}
# 合法数据划分。
VALID_SPLITS = {
    # 训练集。
    "train",
    # 验证集。
    "val",
    # 测试集。
    "test",
}

# 固定患者/样本划分后每个 split 的预期类别数量，用于防止误用数据版本。
EXPECTED_SPLIT_CLASS_COUNTS = {
    # 训练集总计 517。
    "train": {
        # 良性训练样本数。
        "benign": 349,
        # 恶性训练样本数。
        "malignant": 168,
    },
    # 验证集总计 65。
    "val": {
        # 良性验证样本数。
        "benign": 44,
        # 恶性验证样本数。
        "malignant": 21,
    },
    # 测试集总计 65。
    "test": {
        # 良性测试样本数。
        "benign": 44,
        # 恶性测试样本数。
        "malignant": 21,
    },
}


# 流式计算单个文件 SHA-256，避免一次把大文件读入内存。
def _sha256_file(path):
    # 创建新的 SHA-256 状态对象。
    digest = hashlib.sha256()

    # 以二进制只读方式打开文件。
    with Path(path).open("rb") as stream:
        # iter(callable, sentinel) 不断读取块，遇到空字节串停止。
        for block in iter(
                # 每次调用 lambda 从流读取下一块。
                lambda: stream.read(
                    # 块大小 1 MiB。
                    1024 * 1024
                ),
                # EOF 哨兵。
                b"",
        ):
            # 把当前块追加到哈希状态。
            digest.update(block)

    # 返回 64 位十六进制摘要字符串。
    return digest.hexdigest()


# 从规范 sample_id 前缀解析 benign 或 malignant 类别。
def _class_from_id(sample_id):
    # 取第一个下划线前的前缀并做不区分大小写规范化。
    class_name = (
        # 原始样本 ID。
        sample_id
        # 最多切分一次，取第0项。
        .split("_", 1)[0]
        # casefold 规范大小写。
        .casefold()
    )

    # 前缀必须属于本项目保留的两类。
    if class_name not in VALID_CLASSES:
        # 报告不符合命名约定的 ID。
        raise RuntimeError(
            # 错误正文第一段。
            "BUSI sample ID must start "
            # 错误正文第二段。
            "with benign_ or malignant_: "
            # 插入原 ID。
            "{}".format(sample_id)
        )

    # 返回规范类别名。
    return class_name


# 读取并严格验证数据根目录下的 manifest.csv。
def _read_manifest(dataset_root):
    # manifest 固定放在 dataset_root 根目录。
    path = (
        # 转 Path。
            Path(dataset_root)
            # 拼文件名。
            / "manifest.csv"
    )

    # 清单必须存在且为普通文件。
    if not path.is_file():
        # 报告缺失清单路径。
        raise FileNotFoundError(
            # 错误正文。
            "BUSI manifest not found: "
            # 插入路径。
            "{}".format(path)
        )

    # 保存 sample_id 到类别/划分元数据。
    rows = {}

    # newline="" 让 csv 模块自行处理换行；显式使用 UTF-8。
    with path.open(
            # 只读文本模式。
            "r",
            # 禁止通用换行预处理干扰 CSV。
            newline="",
            # 文件编码。
            encoding="utf-8",
    ) as stream:
        # 第一行作为字段名解析每行字典。
        reader = csv.DictReader(stream)

        # 清单必须具备的三列。
        required = {
            # 样本唯一 ID。
            "sample_id",
            # benign/malignant。
            "class_name",
            # train/val/test。
            "split",
        }

        # 字段名缺失或不包含全部必需列时拒绝读取。
        if (
                # 空文件可能使 fieldnames 为 None。
                reader.fieldnames is None
                # 或必需列不是实际列集合的子集。
                or not required.issubset(
            # CSV 实际表头。
            reader.fieldnames
        )
        ):
            # 报告排序后的必需列名。
            raise RuntimeError(
                # 错误正文。
                "BUSI manifest requires "
                # 插入列名列表。
                "columns: {}".format(
                    # 稳定排序显示。
                    sorted(required)
                )
            )

        # 逐行规范化并校验。
        for row in reader:
            # 样本 ID 转为大小写无关形式。
            sample_id = (
                # 读取列值。
                row["sample_id"]
                # 规范大小写。
                .casefold()
            )
            # 类别名规范化。
            class_name = (
                # 读取类别列。
                row["class_name"]
                # 规范大小写。
                .casefold()
            )
            # 划分名规范化。
            split = (
                # 读取划分列。
                row["split"]
                # 规范大小写。
                .casefold()
            )

            # sample_id 必须全表唯一。
            if sample_id in rows:
                # 报告重复 ID。
                raise RuntimeError(
                    # 错误正文。
                    "Duplicate BUSI manifest "
                    # 插入 ID。
                    "ID: {}".format(sample_id)
                )

            # 类别和划分都必须属于允许集合。
            if (
                    # 检查类别。
                    class_name
                    # 类别不合法。
                    not in VALID_CLASSES
                    # 或检查划分。
                    or split
                    # 划分不合法。
                    not in VALID_SPLITS
            ):
                # 报告完整原始行。
                raise RuntimeError(
                    # 错误正文。
                    "Invalid BUSI manifest "
                    # 插入 row 字典。
                    "row: {}".format(row)
                )

            # ID 前缀携带的类别必须与 class_name 列一致。
            if (
                    # 从 ID 解析类别。
                    _class_from_id(sample_id)
                    # 与清单列比较。
                    != class_name
            ):
                # 报告矛盾行。
                raise RuntimeError(
                    # 错误正文第一段。
                    "BUSI manifest class "
                    # 错误正文第二段。
                    "disagrees with ID: "
                    # 插入 row。
                    "{}".format(row)
                )

            # 注册该样本的类别和划分。
            rows[sample_id] = {
                # 规范类别名。
                "class_name": class_name,
                # 规范划分名。
                "split": split,
            }

    # 本项目排除 normal 后的 BUSI 总数固定为 647。
    if len(rows) != 647:
        # 数据行数不同意味着清单版本或准备过程不匹配。
        raise RuntimeError(
            # 错误正文第一段。
            "BUSI manifest must contain "
            # 插入实际行数。
            "647 rows, found {}".format(
                # 行数字典长度。
                len(rows)
            )
        )

    # 同时返回解析行和清单文件内容哈希。
    return rows, _sha256_file(path)


# 构造 BUSI DataLoader，并在通用加载器基础上执行清单和类别分布校验。
def get_loader(
        # 当前 split 的 images 目录。
        image_root,
        # 当前 split 的 masks 目录。
        gt_root,
        # 批大小。
        batchsize,
        # 模型输入尺寸。
        trainsize,
        # 是否打乱。
        shuffle=False,
        # worker 数量。
        num_workers=4,
        # 锁页内存。
        pin_memory=True,
        # 训练增强开关。
        augmentation=False,
        # train/val/test。
        split="train",
        # BUSI 必须为 RGB True。
        color_image=True,
        # DataLoader 随机种子。
        seed=2222,
):
    # 限定合法划分。
    if split not in VALID_SPLITS:
        # 报告允许值。
        raise ValueError(
            # 错误正文第一段。
            "split must be train, val, "
            # 错误正文第二段。
            "or test"
        )

    # 本项目把超声图像复制/读取为三通道以适配 ImageNet 预训练编码器。
    if not color_image:
        # 禁止调用方切换为单通道并造成预处理口径漂移。
        raise ValueError(
            # 错误正文第一段。
            "BUSI is supplied to the "
            # 错误正文第二段。
            "ImageNet-pretrained encoder "
            # 错误正文第三段。
            "as 3-channel input"
        )

    # 先由通用息肉加载器完成 stem 配对、图像读取、掩膜二值化和增强。
    loader = _get_polyp_loader(
        # 图像目录。
        image_root=image_root,
        # 掩膜目录。
        gt_root=gt_root,
        # 批大小。
        batchsize=batchsize,
        # 输入尺寸。
        trainsize=trainsize,
        # 打乱开关。
        shuffle=shuffle,
        # worker 数。
        num_workers=num_workers,
        # 锁页内存。
        pin_memory=pin_memory,
        # 增强开关。
        augmentation=augmentation,
        # 数据划分。
        split=split,
        # 强制 RGB。
        color_image=True,
        # 随机种子。
        seed=seed,
    )

    # 收集任何不是 PNG 的实际图像/掩膜路径。
    invalid = [
        # 输出字符串路径便于错误显示。
        str(path)
        # 遍历通用数据集中的 (stem,image_path,mask_path)。
        for (
            # 忽略 stem。
            _,
            # 输入图像 Path。
            image_path,
            # 掩膜 Path。
            mask_path,
        ) in loader.dataset.samples
        # 对图像和掩膜两条路径分别检查。
        for path in (
            # 图像路径。
            image_path,
            # 掩膜路径。
            mask_path,
        )
        # 扩展名必须严格为 .png，比较时忽略大小写。
        if path.suffix.lower()
           # 与要求值比较。
           != ".png"
    ]

    # 发现非法格式就终止。
    if invalid:
        # 最多展示前十条路径。
        raise RuntimeError(
            # 错误正文第一段。
            "Prepared BUSI files must "
            # 插入非法路径。
            "be PNG: {}".format(
                # 限制错误文本长度。
                invalid[:10]
            )
        )

    # 从 .../<split>/images 推导准备数据根目录。
    dataset_root = (
        # 图像根路径。
        Path(image_root)
        # 绝对规范化，消除 .. 等片段。
        .resolve()
        # images 的父目录是 split。
        .parent
        # split 的父目录是 dataset_root。
        .parent
    )

    # 读取清单内容和文件哈希。
    (
        # sample_id -> 元数据字典。
        manifest,
        # manifest.csv 内容 SHA-256。
        manifest_sha256,
    ) = _read_manifest(dataset_root)

    # 从完整清单筛出当前 split 的 ID->类别映射。
    expected = {
        # 当前样本 ID 对应类别。
        sample_id: row["class_name"]
        # 遍历清单项。
        for sample_id, row
        # in 独立换行保持原格式。
        in manifest.items()
        # 只保留当前划分。
        if row["split"] == split
    }

    # 通用加载器实际发现的 stem 集合。
    actual_ids = set(
        # stems 在通用 PolypDataset 中按排序保存。
        loader.dataset.stems
    )

    # 清单当前划分与磁盘实际文件必须完全一致。
    if set(expected) != actual_ids:
        # 报告双向差集。
        raise RuntimeError(
            # 错误正文第一段。
            "BUSI {} files disagree with "
            # 错误正文第二段。
            "manifest. missing={} "
            # 格式化 split、缺失、额外项。
            "unexpected={}".format(
                # 当前 split。
                split,
                # 清单有但磁盘没有，最多十项。
                sorted(
                    # 预期集合。
                    set(expected)
                    # 减实际集合。
                    - actual_ids
                )[:10],
                # 磁盘有但清单没有，最多十项。
                sorted(
                    # 实际集合。
                    actual_ids
                    # 减预期集合。
                    - set(expected)
                )[:10],
            )
        )

    # 按 expected 类别字段统计当前磁盘样本分布。
    class_counts = dict(
        # Counter 生成类别到数量映射，再转普通 dict。
        Counter(
            # 取得每个实际 ID 的类别。
            expected[sample_id]
            # 遍历实际 ID。
            for sample_id
            # 集合内容。
            in actual_ids
        )
    )

    # 类别数必须等于固定划分方案的预期值。
    if (
            # 实际计数。
            class_counts
            # 与当前 split 预期字典比较。
            != EXPECTED_SPLIT_CLASS_COUNTS[
        # 当前 split 键。
        split
    ]
    ):
        # 报告预期与实际。
        raise RuntimeError(
            # 错误正文第一段。
            "BUSI {} class counts must "
            # 插入三个值。
            "be {}, found {}".format(
                # split。
                split,
                # 当前划分预期计数。
                EXPECTED_SPLIT_CLASS_COUNTS[
                    # split 键。
                    split
                ],
                # 实际计数。
                class_counts,
            )
        )

    # 把校验后的类别分布附加到 dataset，供日志/实验记录读取。
    loader.dataset.class_counts = (
        # 实际类别计数。
        class_counts
    )
    # 附加 manifest 文件哈希，标识固定划分版本。
    loader.dataset.prepared_manifest_sha256 = (
        # SHA-256 字符串。
        manifest_sha256
    )
    # 统一暴露 sample_ids 别名。
    loader.dataset.sample_ids = (
        # 直接使用通用数据集 stems。
        loader.dataset.stems
    )

    # 返回已经通过全部完整性检查的加载器。
    return loader
