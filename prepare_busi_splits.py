# argparse 解析原始BUSI目录、输出目录和固定随机种子。
import argparse
# csv 写出逐样本manifest。
import csv
# hashlib 同时用于文件哈希、解码像素哈希和确定性分组排序。
import hashlib
# json 记录多个源mask路径和最终划分摘要。
import json
# os.replace 用于把完整临时目录原子发布为最终数据目录。
import os
# re 解析原图和 _mask/_mask_1 等文件命名规则。
import re
# shutil 复制原图，并在失败时清理临时目录。
import shutil
# tempfile 在目标父目录创建唯一的暂存目录。
import tempfile
# defaultdict 简化一个图像对应多个mask、一个像素哈希对应多个样本的收集。
from collections import defaultdict
# 以UTC记录数据制备时间。
from datetime import datetime, timezone
# pathlib.Path 提供跨平台路径处理。
from pathlib import Path

# OpenCV负责实际解码PNG、读取灰度mask和写出合并mask。
import cv2
# NumPy创建空mask并执行多个mask的逻辑OR合并。
import numpy as np

# EMCAD二分类BUSI实验只使用良性和恶性病灶，不把normal当作空mask样本。
CLASSES = ("benign", "malignant")
# 固定三份集合顺序，manifest和摘要均沿用此顺序。
SPLITS = ("train", "val", "test")

# 官方BUSI包中可用于病灶分割的类别计数：437良性、210恶性，共647张。
EXPECTED_SOURCE_COUNTS = {
    # 良性病灶数。
    "benign": 437,
    # 恶性病灶数。
    "malignant": 210,
}

# 按类别分层的80/10/10精确目标；总计train=517、val=65、test=65。
EXPECTED_SPLIT_CLASS_COUNTS = {
    # 训练集每类目标数。
    "train": {"benign": 349, "malignant": 168},
    # 验证集每类目标数。
    "val": {"benign": 44, "malignant": 21},
    # 测试集每类目标数。
    "test": {"benign": 44, "malignant": 21},
}

# 写入摘要的协议标识，明确这是分层图像级划分而非患者级划分。
PROTOCOL = "emcad_80_10_10_stratified_image_level"

# BUSI mask文件名形式：原图stem后接_mask，可选再接_数字表示同一图像的额外病灶mask。
MASK_PATTERN = re.compile(
    # 命名捕获base原图ID和可选index，例如 benign (1)_mask_2。
    r"^(?P<base>.+)_mask(?:_(?P<index>[0-9]+))?$",
    # 文件名大小写不作为配对差异。
    re.IGNORECASE,
)


# 流式计算文件原始字节SHA-256，用于manifest审计原图和生成mask文件。
def sha256_file(path):
    # 创建空SHA-256状态。
    digest = hashlib.sha256()

    # 以二进制方式打开，避免任何文本编码或换行转换。
    with Path(path).open("rb") as stream:
        # 每次读取1 MiB直到空字节串，避免大文件一次性进入内存。
        for block in iter(
                lambda: stream.read(1024 * 1024),
                b"",
        ):
            # 累积当前文件块。
            digest.update(block)

    # 返回64字符十六进制摘要。
    return digest.hexdigest()


# 对OpenCV解码后的像素计算SHA-256，用来识别“文件编码不同但像素完全相同”的重复图像。
def decoded_image_sha256(image):
    # 创建摘要对象。
    digest = hashlib.sha256()
    # 先加入数组形状，防止不同宽高布局恰好字节序列相同。
    digest.update(
        str(image.shape).encode("ascii")
    )
    # 再加入dtype，区分同一字节在不同数据类型下的语义。
    digest.update(
        str(image.dtype).encode("ascii")
    )
    # 最后加入连续像素字节。
    digest.update(image.tobytes())
    # 返回像素级重复组的键。
    return digest.hexdigest()


# 从用户给定路径中唯一定位真正包含 benign/malignant 子目录的 Dataset_BUSI_with_GT。
def resolve_source_root(requested):
    # 展开~并转换为绝对规范路径。
    requested = (
        Path(requested)
        .expanduser()
        .resolve()
    )

    # 顶层请求路径必须存在。
    if not requested.is_dir():
        # 报告解析后的完整路径。
        raise FileNotFoundError(
            "BUSI source directory not found: {}".format(
                requested
            )
        )

    # 候选1允许用户直接传数据根目录，候选2允许传它的父目录。
    candidates = [
        requested,
        requested / "Dataset_BUSI_with_GT",
    ]
    # 还递归查找嵌套的同名目录，以兼容解压后多套一层目录的情况。
    candidates.extend(
        requested.rglob("Dataset_BUSI_with_GT")
    )

    # 收集结构有效且去重后的候选。
    valid = []
    # seen避免requested/显式子目录/rglob重复指向同一路径。
    seen = set()

    # 逐候选验证。
    for candidate in candidates:
        # 规范化绝对路径，便于集合去重。
        candidate = candidate.resolve()

        # 已检查过同一路径则跳过。
        if candidate in seen:
            continue

        # 标记为已检查。
        seen.add(candidate)

        # 数据根必须同时包含benign和malignant目录。
        if all(
                (candidate / name).is_dir()
                for name in CLASSES
        ):
            # 结构有效时加入列表。
            valid.append(candidate)

    # 必须恰好找到一个数据集，多个候选会造成来源歧义。
    if len(valid) != 1:
        # 把全部有效候选列出来，便于用户修正source_root。
        raise RuntimeError(
            "Expected exactly one Dataset_BUSI_with_GT "
            "directory under {}, found: {}".format(
                requested,
                [str(path) for path in valid],
            )
        )

    # 返回唯一有效数据根。
    return valid[0]


# 统计某类别目录中“不是mask的PNG原图”数量；main用它记录normal类排除数量。
def count_original_images(class_dir):
    # 标准化路径对象。
    class_dir = Path(class_dir)

    # 目录不存在时按0张处理，normal不是本脚本的必需输入类别。
    if not class_dir.is_dir():
        return 0

    # 遍历一层目录并统计原图。
    return sum(
        # 每个满足条件的文件贡献1。
        1
        for path in class_dir.iterdir()
        if (
            # 必须是普通文件。
                path.is_file()
                # 必须是PNG。
                and path.suffix.lower() == ".png"
                # stem不能匹配_mask命名，否则它是标签文件。
                and MASK_PATTERN.fullmatch(
            path.stem
        )
                is None
        )
    )


# 扫描一个病灶类别，严格配对原图与一个或多个mask，并生成可审计样本记录。
def collect_class_samples(
        # BUSI数据根目录。
        dataset_root,
        # 当前只会传benign或malignant。
        class_name,
):
    # 拼出类别目录。
    class_dir = (
            Path(dataset_root) / class_name
    )

    # images映射规范stem到唯一原图。
    images = {}
    # masks映射同一规范stem到一个或多个mask路径。
    masks = defaultdict(list)

    # 按文件名排序扫描，保证错误和输出稳定。
    for path in sorted(class_dir.iterdir()):
        # 忽略子目录和非PNG文件。
        if (
                not path.is_file()
                or path.suffix.lower() != ".png"
        ):
            continue

        # 判断当前stem是否为_mask或_mask_数字标签。
        mask_match = MASK_PATTERN.fullmatch(
            path.stem
        )

        # 不匹配mask模式即视为原图。
        if mask_match is None:
            # casefold后的stem作为配对键。
            key = path.stem.casefold()

            # 同一规范ID只能有一个原图。
            if key in images:
                # 同时报告冲突文件。
                raise RuntimeError(
                    "Duplicate BUSI image ID in {}: "
                    "{} and {}".format(
                        class_dir,
                        images[key].name,
                        path.name,
                    )
                )

            # 保存绝对原图路径。
            images[key] = path.resolve()

        # 匹配mask模式时，使用捕获的base关联原图。
        else:
            # 去掉_mask后缀并casefold，得到对应原图键。
            key = (
                mask_match
                .group("base")
                .casefold()
            )
            # 同一原图可追加多个mask，后续会做逻辑OR。
            masks[key].append(path.resolve())

    # 原图ID集合和mask base集合必须完全一致。
    if set(images) != set(masks):
        # 分别列出缺mask原图和无原图孤立mask。
        raise RuntimeError(
            "BUSI image/mask IDs do not match "
            "in {}. missing_masks={} "
            "orphan_masks={}".format(
                class_dir,
                # 最多展示20个缺mask ID。
                sorted(
                    set(images) - set(masks)
                )[:20],
                # 最多展示20个孤立mask ID。
                sorted(
                    set(masks) - set(images)
                )[:20],
            )
        )

    # 最终样本记录列表。
    samples = []
    # 防止规范化后的目标sample_id重复。
    sample_ids = set()

    # 官方命名应为“类别名 (整数)”，例如 benign (12)。
    name_pattern = re.compile(
        # re.escape避免类别名中潜在正则字符产生歧义。
        r"^{}\s*\(([0-9]+)\)$".format(
            re.escape(class_name)
        ),
        # 大小写不敏感，但输出sample_id统一使用class_name参数。
        re.IGNORECASE,
    )

    # 按规范原图键生成样本。
    for key in sorted(images):
        # 当前原图绝对路径。
        image_path = images[key]

        # 验证官方文件名并提取编号。
        match = name_pattern.fullmatch(
            image_path.stem
        )

        # 非官方命名可能来自其他数据集或人为改名，拒绝静默混入。
        if match is None:
            # 错误信息给出期望格式。
            raise RuntimeError(
                "Unexpected BUSI image name: {}. "
                "Expected '{} (number).png'.".format(
                    image_path.name,
                    class_name,
                )
            )

        # 目标ID统一为 benign_0001/malignant_0001，便于排序并避免括号和空格。
        sample_id = "{}_{:04d}".format(
            class_name,
            int(match.group(1)),
        )

        # 归一化ID必须唯一。
        if sample_id.casefold() in sample_ids:
            # 报告重复ID。
            raise RuntimeError(
                "Duplicate normalized sample ID: "
                + sample_id
            )

        # 登记已使用ID。
        sample_ids.add(
            sample_id.casefold()
        )

        # 以彩色模式解码原图；像素哈希基于该统一BGR数组。
        image = cv2.imread(
            str(image_path),
            cv2.IMREAD_COLOR,
        )

        # OpenCV读取失败返回None。
        if image is None:
            # 立即报告损坏或不可读文件。
            raise RuntimeError(
                "Cannot read BUSI image: "
                + str(image_path)
            )

        # 同一原图的多个mask按文件名排序并冻结为tuple。
        mask_paths = tuple(
            sorted(
                masks[key],
                # 仅按文件名排序，保证_mask、_mask_1等顺序稳定。
                key=lambda path: path.name,
            )
        )

        # 逐个mask执行可读性和尺寸检查。
        for mask_path in mask_paths:
            # 以单通道灰度读取标签。
            mask = cv2.imread(
                str(mask_path),
                cv2.IMREAD_GRAYSCALE,
            )

            # 读取失败意味着标签损坏或路径异常。
            if mask is None:
                raise RuntimeError(
                    "Cannot read BUSI mask: "
                    + str(mask_path)
                )

            # mask高宽必须与原图前两维完全一致。
            if mask.shape != image.shape[:2]:
                # 报告样本ID、图像尺寸、mask尺寸和具体mask名。
                raise RuntimeError(
                    "Image/mask size mismatch for {}: "
                    "image={} mask={} ({})".format(
                        sample_id,
                        image.shape[:2],
                        mask.shape,
                        mask_path.name,
                    )
                )

        # 保存完成校验的样本元数据。
        samples.append(
            {
                # 规范目标ID。
                "sample_id": sample_id,
                # 良性或恶性类别，用于分层计数。
                "class_name": class_name,
                # 原图绝对路径。
                "image_source": image_path,
                # 一个或多个mask路径。
                "mask_sources": mask_paths,
                # 文件字节哈希用于来源审计。
                "image_file_sha256": (
                    sha256_file(image_path)
                ),
                # 解码像素哈希用于跨文件名、跨编码识别完全重复图像。
                "image_pixel_sha256": (
                    decoded_image_sha256(image)
                ),
            }
        )

    # 返回当前类别全部样本。
    return samples


# 将解码后像素完全相同的原图绑定为一个不可拆分的重复组。
# 这样后续划分时，同一幅图即使文件名或PNG编码不同，也不会跨训练、验证和测试集造成泄漏。
def build_duplicate_groups(samples):
    # defaultdict(list)允许首次遇到某个像素哈希时直接append，无需预先创建空列表。
    grouped = defaultdict(list)

    # 逐个读取前面已经完成图像、mask和哈希校验的样本记录。
    for sample in samples:
        # image_pixel_sha256由统一解码后的BGR像素数组计算，专门识别“内容完全相同”的图像。
        grouped[
            sample["image_pixel_sha256"]
            # 同一哈希下保存所有对应样本；哈希碰撞在SHA-256下可忽略，但这里的语义仍是按哈希分组。
        ].append(sample)

    # 将临时的“哈希到样本列表”映射转换成带类别统计的结构化组列表。
    groups = []

    # key是像素SHA-256，members是该像素内容对应的一个或多个BUSI样本。
    for key, members in grouped.items():
        # 每个组同时保存稳定顺序的样本、分组键和按类别计数，供精确分层划分使用。
        groups.append(
            {
                # 分组键也是后面稳定随机排序、集合筛选和重复组归属判断的唯一标识。
                "key": key,
                # 组内按规范sample_id排序，使清单输出不依赖文件系统遍历顺序。
                "samples": sorted(
                    members,
                    # sample_id形如benign_0001或malignant_0001，字典序与编号顺序一致。
                    key=lambda sample: (
                        sample["sample_id"]
                    ),
                ),
                # counts记录这个不可拆分组对每个类别目标贡献多少张图。
                "counts": {
                    # 对当前class_name统计布尔条件为True的成员数；Python中True按1参与sum。
                    class_name: sum(
                        member["class_name"]
                        == class_name
                        # 遍历组内所有像素相同样本，允许同一重复组中出现跨类别记录。
                        for member in members
                    )
                    # 对CLASSES中的良性、恶性分别生成计数项，键顺序也定义后续状态向量顺序。
                    for class_name in CLASSES
                },
            }
        )

    # 返回全部不可拆分重复组；单样本组同样保留，以统一后续算法。
    return groups


# 从不可拆分组中选择一批组，使各类别样本数精确等于target_counts。
# 算法先用seed、phase和像素哈希生成确定性伪随机顺序，再用二维子集和动态规划寻找精确解。
def select_exact_groups(
        # groups是候选重复组列表，组内样本绝不能拆开。
        groups,
        # target_counts是当前阶段需要达到的良性、恶性精确数量字典。
        target_counts,
        # seed控制可复现的组顺序，相同数据和种子会得到相同划分。
        seed,
        # phase区分val与test，使两个阶段即使种子相同也使用不同的排序盐值。
        phase,
):
    # 把类别数字典转换为固定顺序的元组，例如(benign目标数, malignant目标数)。
    target = tuple(
        target_counts[name]
        # CLASSES的顺序必须与组counts转成增量元组时完全一致。
        for name in CLASSES
    )

    # 为每个重复组计算稳定的伪随机排序分数；这里不调用random，避免实现和调用顺序影响结果。
    def score(group):
        # 将种子、划分阶段和像素哈希用制表符隔开，形成无歧义的排序输入字符串。
        value = "{}\t{}\t{}".format(
            seed,
            phase,
            group["key"],
            # SHA-256接收字节，因此显式采用UTF-8编码。
        ).encode("utf-8")

        # 十六进制SHA-256摘要可直接按字典序排序，效果等价于稳定的256位伪随机值。
        return hashlib.sha256(
            value
        ).hexdigest()

    # 按稳定哈希打散候选组，避免总是偏向原始文件名或类别编号较小的样本。
    ordered = sorted(
        groups,
        key=lambda group: (
            # 第一排序键由seed和phase控制。
            score(group),
            # 极小概率排序分数相同，再以组哈希打破平局，保证全序和完全复现。
            group["key"],
        ),
    )

    # 动态规划状态映射：键为当前(良性数, 恶性数)，值为达到该状态所选的组哈希元组。
    selections = {
        # 空集合首先只能达到(0, 0)，对应尚未选择任何重复组。
        (0, 0): (),
    }

    # 依次考虑稳定排序后的每个不可拆分组，执行0/1子集和状态扩展。
    for group in ordered:
        # 当前组会同时增加的各类别样本数，例如普通良性单样本组为(1, 0)。
        increment = tuple(
            group["counts"][name]
            # 继续严格沿用CLASSES顺序，确保向量两维含义不发生错位。
            for name in CLASSES
        )

        # 对当前状态表做快照，确保本轮新增状态不会再次使用同一个group。
        # 这是0/1动态规划的关键：每个重复组只能“不选”或“选一次”。
        previous_states = list(
            selections.items()
        )

        # 遍历加入当前组之前已经可达的每个类别计数组合及其选择路径。
        for (
                state,
                selected_keys,
        ) in previous_states:
            # 将当前组的良性、恶性贡献分别加到旧状态，得到候选新状态。
            next_state = (
                state[0] + increment[0],
                state[1] + increment[1],
            )

            # 超过任一类别目标的状态永远不可能回退到目标，因此立即剪枝。
            # 若状态已存在，则保留按稳定顺序最先发现的路径，使结果唯一且可复现。
            if (
                    next_state[0] > target[0]
                    or next_state[1] > target[1]
                    or next_state in selections
            ):
                continue

            # 登记第一次到达next_state的具体选组路径。
            selections[next_state] = (
                    selected_keys
                    # 元组拼接创建新路径，不会修改旧状态保存的选择结果。
                    + (group["key"],)
            )

    # 所有组处理完后仍没有目标状态，说明在“重复组不可拆分”约束下无法精确配额。
    if target not in selections:
        # 明确报错而不是退化为近似比例，防止实际实验划分悄悄偏离既定协议。
        raise RuntimeError(
            "Cannot create an exact {} split "
            "with class target {} while keeping "
            "decoded-pixel duplicate groups "
            "together. Verify that you downloaded "
            "the original 780-image BUSI "
            "package.".format(
                phase,
                target_counts,
            )
        )

    # 将目标状态保存的组哈希元组转为集合，便于后面用O(1)成员判断分配split。
    return set(selections[target])


# 构造train/val/test三个最终分区，并验证每个分区的良恶性数量完全符合协议。
def make_partitions(groups, seed):
    # 首先从全部组中精确选验证集；先固定验证集可保证它不会与随后选择的测试集重叠。
    val_keys = select_exact_groups(
        groups,
        # 验证集目标按类别给出，而不是只限制总数，因此保持类别比例可控。
        EXPECTED_SPLIT_CLASS_COUNTS[
            "val"
        ],
        seed,
        "val",
    )

    # 测试集只能从未进入验证集的重复组中选择。
    remaining = [
        group
        for group in groups
        # 整个组按像素哈希排除，确保任何完全重复图像不会跨val和后续分区。
        if group["key"] not in val_keys
    ]

    # 在剩余组中按test阶段的独立稳定顺序求精确测试集。
    test_keys = select_exact_groups(
        remaining,
        EXPECTED_SPLIT_CLASS_COUNTS[
            "test"
        ],
        seed,
        "test",
    )

    # 为三个固定分区创建样本容器；SPLITS同时规定后面落盘和清单的顺序。
    partitions = {
        split: []
        for split in SPLITS
    }

    # 按组而不是按单张图分配分区，这是阻止像素级重复样本泄漏的核心约束。
    for group in groups:
        # 被动态规划选中的验证组整体进入val。
        if group["key"] in val_keys:
            split = "val"
        # 未进val但被第二次动态规划选中的组整体进入test。
        elif group["key"] in test_keys:
            split = "test"
        # 两次都未选中的所有剩余组自然构成train。
        else:
            split = "train"

        # 展开当前重复组中的每个样本，但所有成员共享上面确定的同一个split。
        for sample in group["samples"]:
            # 浅复制样本字典，避免向原始samples记录原地加入派生字段。
            copied = dict(sample)
            # 记录该样本所属像素重复组大小；1表示没有发现完全相同的另一图像。
            copied[
                "duplicate_group_size"
            ] = len(group["samples"])

            # 将带重复组审计信息的样本放入目标分区。
            partitions[split].append(
                copied
            )

    # 对每个分区内部执行稳定排序，避免groups字典构造顺序影响最终输出。
    for split in SPLITS:
        partitions[split].sort(
            key=lambda sample: (
                sample["sample_id"]
            )
        )

    # 重新从真实分配结果统计每个split的良性和恶性数量，不能只信任选择算法的中间状态。
    actual = {
        split: {
            # 布尔求和统计当前分区中属于指定类别的样本数。
            class_name: sum(
                sample["class_name"]
                == class_name
                for sample
                in partitions[split]
            )
            # 为良性和恶性分别生成计数。
            for class_name in CLASSES
        }
        # 为train、val和test分别生成类别计数字典。
        for split in SPLITS
    }

    # 最终实测字典必须与协议常量逐项完全一致，包括训练集的余量计数。
    if (
            actual
            != EXPECTED_SPLIT_CLASS_COUNTS
    ):
        # 任何不一致都代表内部算法或常量发生错误，立即终止而不写出不可信数据集。
        raise RuntimeError(
            "Internal split-count error: "
            "expected={} actual={}".format(
                EXPECTED_SPLIT_CLASS_COUNTS,
                actual,
            )
        )

    # 返回已经排序、带重复组大小字段且完成精确计数复核的分区字典。
    return partitions


# 将同一BUSI原图可能对应的多个病灶mask合并成一张二值分割标签。
def merge_masks(sample):
    # 重新读取原图，仅用于取得并再次确认输出mask应有的空间尺寸。
    image = cv2.imread(
        str(sample["image_source"]),
        cv2.IMREAD_COLOR,
    )

    # 虽然采集阶段已经验证过，这里再次防御文件在准备过程中被删除、替换或损坏。
    if image is None:
        raise RuntimeError(
            "Cannot read BUSI image: {}".format(
                sample["image_source"]
            )
        )

    # 创建与原图高宽相同的全背景标签，uint8足以表示0和1。
    merged = np.zeros(
        image.shape[:2],
        dtype=np.uint8,
    )

    # 遍历这张原图的全部mask；BUSI中少数图像具有多个独立病灶mask文件。
    for mask_path in sample[
        "mask_sources"
    ]:
        # 灰度读取避免保留无意义的颜色通道，并与前面尺寸校验方式一致。
        mask = cv2.imread(
            str(mask_path),
            cv2.IMREAD_GRAYSCALE,
        )

        # 再次检查可读性，防止采集完成后源文件状态发生变化。
        if mask is None:
            raise RuntimeError(
                "Cannot read BUSI mask: "
                + str(mask_path)
            )

        # 每个源mask必须仍与当前合并画布尺寸一致，否则逐像素逻辑合并没有定义。
        if mask.shape != merged.shape:
            raise RuntimeError(
                "Mask shape changed during "
                "preparation: {}".format(
                    mask_path
                )
            )

        # mask>0把任意非零标注统一为前景1；逐元素maximum等价于二值逻辑OR。
        # 因而多个病灶区域都会保留在同一最终标签中，重叠区域仍然只是1。
        merged = np.maximum(
            merged,
            (mask > 0).astype(np.uint8),
        )

    # 合并后像素和为0意味着没有任何病灶前景，不符合良性/恶性病灶分割样本定义。
    if int(merged.sum()) == 0:
        raise RuntimeError(
            "Lesion mask is empty for {}".format(
                sample["sample_id"]
            )
        )

    # 将内部0/1标签转成PNG常用的0/255二值灰度，语义仍是背景/病灶前景。
    return merged * 255


# 把已验证分区写入EMCAD期望的目录结构，同时生成可追溯manifest和summary。
# 整个数据集先写到同父目录临时文件夹，全部成功后再一次性重命名为正式目标，避免半成品被误用。
def write_target(
        # 已定位并解析后的原始BUSI根目录，用于安全边界检查和审计记录。
        source_root,
        # 用户指定的目标数据集根目录，函数要求它在执行前完全不存在。
        output_root,
        # make_partitions返回的train/val/test样本字典。
        partitions,
        # 像素重复组列表，用于统计重复样本情况并写入摘要。
        groups,
        # 固定划分种子，写入summary以支持复现实验。
        seed,
        # 被排除的normal原图数量，仅作为来源完整性和筛选策略记录。
        normal_images,
):
    # 将目标参数标准化为绝对Path，后续父子路径判断不会受相对路径或“..”影响。
    output_root = (
        Path(output_root)
        # 展开路径中可能出现的用户主目录符号。
        .expanduser()
        # 解析为规范绝对路径；strict默认为False，因此目标尚不存在也可以解析。
        .resolve()
    )

    # 既检查普通路径存在，也单独检查可能指向不存在目标的符号链接。
    if (
            output_root.exists()
            or output_root.is_symlink()
    ):
        # 为保护既有实验数据，脚本不覆盖、不合并也不自动清理旧目标。
        raise FileExistsError(
            "Target already exists; it was "
            "not modified: {}. Use a new "
            "--output_root or archive the old "
            "target first.".format(
                output_root
            )
        )

    # 禁止把输出直接设为原始数据根目录，也禁止写进原始数据目录的任何子目录。
    if (
            output_root == source_root
            or source_root
            in output_root.parents
    ):
        # 该限制避免临时目录、整理后副本或同名文件污染不可变的原始BUSI数据。
        raise ValueError(
            "Output root must be outside "
            "the raw BUSI directory"
        )

    # 只创建目标的父目录；正式output_root仍保持不存在，留给最后的原子替换。
    output_root.parent.mkdir(
        # 缺失的多级父目录可一并创建。
        parents=True,
        # 父目录已经存在属于正常情况。
        exist_ok=True,
    )

    # 在正式目标的同一父目录创建唯一临时stage。
    # 同父目录通常意味着同一文件系统，使末尾os.replace可以用一次重命名完成发布。
    stage = Path(
        tempfile.mkdtemp(
            # 隐藏式前缀便于识别准备过程的临时目录。
            prefix=".BUSI.prepare.",
            # 明确放在output_root.parent，而不是系统临时盘。
            dir=str(output_root.parent),
        )
    )

    # try覆盖全部落盘和发布流程；任何异常都会进入except删除stage。
    try:
        # 每处理一个样本就积累一条manifest记录，最后统一排序和写CSV。
        rows = []

        # 按SPLITS规定的train、val、test顺序创建和填充目录。
        for split in SPLITS:
            # EMCAD的polyp/BUSI加载约定要求每个split下有images子目录。
            image_dir = (
                    stage / split / "images"
            )
            # 与images平行的masks子目录保存一一同名的分割标签。
            mask_dir = (
                    stage / split / "masks"
            )

            # 创建当前分区图像目录以及尚不存在的父目录。
            image_dir.mkdir(
                parents=True,
                exist_ok=True,
            )
            # 创建当前分区标签目录。
            mask_dir.mkdir(
                parents=True,
                exist_ok=True,
            )

            # partitions内已经按sample_id排序，这里逐样本物化文件并生成审计行。
            for sample in partitions[split]:
                # 所有目标原图统一使用规范sample_id和.png扩展名，去除原始文件名中的空格与括号。
                image_target = image_dir / (
                        sample["sample_id"]
                        + ".png"
                )
                # mask与原图保持完全相同的文件名，加载器可按stem直接配对。
                mask_target = mask_dir / (
                        sample["sample_id"]
                        + ".png"
                )

                # 原图不重新编码，直接复制文件字节；copy2还尽量保留源文件元数据。
                shutil.copy2(
                    sample["image_source"],
                    image_target,
                )

                # 将该图像可能对应的多个病灶mask逻辑OR成单一二值标签。
                merged_mask = merge_masks(
                    sample
                )

                # OpenCV用目标.png后缀选择PNG编码器并返回写入是否成功。
                if not cv2.imwrite(
                        str(mask_target),
                        merged_mask,
                ):
                    # 写标签失败时立即中断；except会清理整个临时stage，不留下部分数据集。
                    raise RuntimeError(
                        "Failed to save merged "
                        "mask: "
                        + str(mask_target)
                    )

                # 记录一个样本从原始来源到整理后目标的完整映射和完整性字段。
                rows.append(
                    {
                        # 样本属于train、val或test中的哪一部分。
                        "split": split,
                        # 规范唯一ID，也是目标图像和mask的共同stem。
                        "sample_id": (
                            sample["sample_id"]
                        ),
                        # 原始诊断类别benign或malignant；分割训练仍把两者都视为病灶前景任务。
                        "class_name": (
                            sample["class_name"]
                        ),
                        # 目标图像保存为相对于数据集stage根目录的可迁移路径。
                        "image": str(
                            image_target.relative_to(
                                stage
                            )
                        ),
                        # 目标合并mask同样记录相对路径，避免manifest绑定某台机器的目标绝对路径。
                        "mask": str(
                            mask_target.relative_to(
                                stage
                            )
                        ),
                        # 原始图像绝对路径保留用于来源审计和问题回溯。
                        "source_image": str(
                            sample["image_source"]
                        ),
                        # 一个样本可能有多个源mask，因此把路径列表序列化为一个合法JSON字符串写入CSV单元格。
                        "source_masks": (
                            json.dumps(
                                # 将Path逐一转为字符串，JSON编码器不能直接序列化Path对象。
                                [
                                    str(path)
                                    for path
                                    in sample[
                                    "mask_sources"
                                ]
                                ],
                                # 源路径预期主要为ASCII；显式True也保证CSV字段编码形式稳定。
                                ensure_ascii=True,
                            )
                        ),
                        # 单独保存源mask数量，便于快速统计多病灶或多标注样本，无需反解析JSON。
                        "source_mask_count": len(
                            sample["mask_sources"]
                        ),
                        # 原始图像文件字节SHA-256，可核验复制来源是否一致。
                        "image_file_sha256": (
                            sample[
                                "image_file_sha256"
                            ]
                        ),
                        # 原图解码像素SHA-256用于识别不同文件编码下内容完全相同的图像。
                        "image_pixel_sha256": (
                            sample[
                                "image_pixel_sha256"
                            ]
                        ),
                        # 对实际写出的合并mask文件计算SHA-256，覆盖合并逻辑与PNG编码后的最终产物。
                        "merged_mask_sha256": (
                            sha256_file(
                                mask_target
                            )
                        ),
                        # 该图像所属像素重复组的成员数，用于清单级数据泄漏审计。
                        "duplicate_group_size": (
                            sample[
                                "duplicate_group_size"
                            ]
                        ),
                    }
                )

        # 建立split到序号的映射，显式固定CSV中train、val、test的先后次序。
        order = {
            # enumerate同时给出SPLITS中的位置index和名称split。
            split: index
            for index, split
            in enumerate(SPLITS)
        }

        # 先按split协议顺序、再按sample_id排序，使manifest不依赖处理过程的偶然顺序。
        rows.sort(
            key=lambda row: (
                order[row["split"]],
                row["sample_id"],
            )
        )

        # manifest.csv位于数据集根目录，集中描述所有样本而不是分散到各split。
        manifest_path = (
                stage / "manifest.csv"
        )

        # 所有row具有相同插入顺序的键，首行键列表直接定义CSV列顺序。
        # 前面的固定计数协议保证rows不为空，因此访问rows[0]是安全的。
        fieldnames = list(
            rows[0].keys()
        )

        # newline=""是csv模块推荐写法，可避免Windows下出现额外空行。
        with manifest_path.open(
                "w",
                newline="",
                # UTF-8可无损保存路径或元数据中的非ASCII字符。
                encoding="utf-8",
        ) as stream:
            # DictWriter根据fieldnames把每个字典字段按固定列序输出。
            writer = csv.DictWriter(
                stream,
                fieldnames=fieldnames,
            )
            # 先输出一次列名表头。
            writer.writeheader()
            # 再按已经稳定排序的rows批量输出全部样本。
            writer.writerows(rows)

        # 创建“逻辑清单摘要”哈希器；它只纳入实验关键字段，不依赖CSV引号或换行编码细节。
        logical_digest = (
            hashlib.sha256()
        )

        # 按稳定排序后的清单行依次更新摘要，顺序变化也会导致最终摘要变化。
        for row in rows:
            logical_digest.update(
                (
                    # 制表符分隔字段、换行分隔样本，形成明确的逻辑记录边界。
                    "{split}\t{sample_id}\t"
                    "{class_name}\t"
                    "{image_file_sha256}\t"
                    "{image_pixel_sha256}\t"
                    "{merged_mask_sha256}\n"
                )
                # 只使用格式串中列出的关键字段，其他审计列变化不会改变逻辑划分摘要。
                .format(**row)
                # 哈希算法处理字节，因此统一编码为UTF-8。
                .encode("utf-8")
            )

        # 汇总每个分区的总图像数量，例如协议预期train=517、val=65、test=65。
        counts = {
            split: len(partitions[split])
            for split in SPLITS
        }

        # 独立汇总每个split内部的良性/恶性数量，便于核对分层划分是否符合常量。
        class_counts = {
            split: {
                # 对类别相等条件做布尔求和。
                class_name: sum(
                    sample["class_name"]
                    == class_name
                    for sample
                    in partitions[split]
                )
                # 为CLASSES中的每个诊断类别生成一项。
                for class_name in CLASSES
            }
            # 为三个数据分区分别生成嵌套统计。
            for split in SPLITS
        }

        # 只保留成员数大于1的组作为“确实发现重复”的组；单成员组不计入重复统计。
        duplicate_groups = [
            group
            for group in groups
            if len(group["samples"]) > 1
        ]

        # 进一步找出同一像素内容却被放在不同诊断类别名下的跨类别重复组。
        cross_class_groups = [
            group
            for group in duplicate_groups
            # 统计该组中计数大于0的类别有几个；超过1说明同时含良性和恶性记录。
            if sum(
                group["counts"][name] > 0
                for name in CLASSES
            )
               > 1
        ]

        # 将三个分区重新展平为单一列表，便于计算与split无关的总体统计。
        all_samples = [
            sample
            for split in SPLITS
            for sample
            in partitions[split]
        ]

        # summary汇总协议、数量、重复策略、校验摘要和数据局限，供论文实验记录与复现使用。
        summary = {
            # 公开数据集名称。
            "dataset_name": "BUSI",
            # 本脚本采用的固定划分协议标识，定义在文件顶部常量中。
            "protocol": PROTOCOL,
            # 公开包缺少患者ID，因此可用的最小划分单位只能是单张原图。
            "split_unit": "image",
            # 明确标记这不是患者级划分，避免使用者把image-level误写成patient-level。
            "patient_level_split": False,
            # 保存实际使用的整数种子。
            "seed": int(seed),
            # 各split总数。
            "counts": counts,
            # 各split的良性/恶性分层数量。
            "class_counts": class_counts,
            # 三个split之和应为647张良性与恶性病灶图像。
            "total": sum(
                counts.values()
            ),
            # 原始公开包中良性和恶性应有数量，用于说明来源检查基线。
            "source_class_counts": (
                EXPECTED_SOURCE_COUNTS
            ),
            # normal类别不参与本二值病灶分割子集，但记录实际排除数量。
            "normal_images_excluded": (
                int(normal_images)
            ),
            # 统计源mask数大于1的样本，即发生逻辑OR合并的图像数量。
            "multi_mask_images": sum(
                len(
                    sample["mask_sources"]
                )
                > 1
                for sample in all_samples
            ),
            # 解码像素完全一致且成员数大于1的组数。
            "exact_pixel_duplicate_groups": (
                len(duplicate_groups)
            ),
            # 所有重复组包含的图像总数；与组数区分，可反映重复影响范围。
            "images_in_exact_pixel_duplicate_groups": (
                sum(
                    len(group["samples"])
                    for group
                    in duplicate_groups
                )
            ),
            # 跨良性/恶性标签的完全像素重复组数量，是值得单独审计的数据异常指标。
            "cross_class_exact_pixel_duplicate_groups": (
                len(cross_class_groups)
            ),
            # 用自然语言把脚本实际执行的完全重复样本策略写进产物，防止实验说明遗漏。
            "duplicate_policy": (
                "All decoded-pixel-identical "
                "images are retained but forced "
                "into the same split."
            ),
            # 明确披露没有执行感知哈希或相似度模型的近重复检测，避免夸大泄漏控制范围。
            "near_duplicate_policy": (
                "Near-duplicate grouping was not "
                "possible from the official "
                "metadata and was not performed."
            ),
            # 基于稳定关键字段计算的逻辑manifest摘要，适合比较两次划分内容是否相同。
            "manifest_sha256": (
                logical_digest.hexdigest()
            ),
            # 整个CSV文件的字节级摘要；任何列、引号或换行变化都会反映出来。
            "manifest_file_sha256": (
                sha256_file(manifest_path)
            ),
            # 保存原始BUSI根目录，便于追溯本次准备使用的具体来源位置。
            "source_root": str(
                source_root
            ),
            # 使用带UTC时区的ISO 8601时间记录本次数据产物创建时刻。
            "created_utc": datetime.now(
                timezone.utc
            ).isoformat(),
            # notes集中说明筛选范围、mask处理、划分局限和测试集用途。
            "notes": [
                # normal图像没有病灶前景，不纳入当前良恶性病灶分割数据子集。
                (
                    "Only benign and malignant "
                    "lesion images are used."
                ),
                # 同一图像的多个mask以逻辑OR合并，而不是只取第一个标注。
                (
                    "All masks belonging to one "
                    "image are merged with "
                    "logical OR."
                ),
                # 缺少patient ID意味着无法确认同一患者的多张非完全重复图像是否跨split。
                (
                    "The public BUSI package has "
                    "no patient IDs, so this is "
                    "not a patient-level split."
                ),
                # 验证集用于训练过程中的模型选择，测试集保持隔离直到最终评估。
                (
                    "Train and validation are "
                    "used during training; test "
                    "remains isolated for final "
                    "evaluation."
                ),
            ],
        }

        # 把结构化摘要写入数据集根目录，ensure_ascii=False使中文或非ASCII路径保持可读。
        with (
                stage / "split_summary.json"
        ).open(
            "w",
            encoding="utf-8",
        ) as stream:
            # JSON保存嵌套计数和策略说明，比CSV更适合数据集级元信息。
            json.dump(
                summary,
                stream,
                # 不把非ASCII字符转成\u序列。
                ensure_ascii=False,
                # 两空格缩进便于人工阅读和版本比较。
                indent=2,
            )

        # 只有图像、mask、manifest和summary全部成功后，才把完整stage发布到正式目标路径。
        # stage与output_root同父目录且目标预先保证不存在，因此该重命名避免暴露中间半成品。
        os.replace(
            stage,
            output_root,
        )

    # 捕获准备过程中的任意异常，包括读取、复制、编码、哈希和最终发布错误。
    except Exception:
        # 递归删除只属于本次运行的唯一临时stage；正式目标未发布时不会被触碰。
        shutil.rmtree(
            stage,
            # 清理失败不覆盖原始异常，便于看到真正的数据准备错误原因。
            ignore_errors=True,
        )
        # 原样重新抛出异常及其回溯，调用者可据此定位失败步骤。
        raise

    # 发布成功后打印已规范化的原始数据根目录。
    print(
        "SOURCE_ROOT={}".format(
            source_root
        )
    )
    # 打印最终目标目录，便于命令行日志直接记录数据集位置。
    print(
        "TARGET_ROOT={}".format(
            output_root
        )
    )
    # 输出固定协议的三个split总数，作为最醒目的快速验收信息。
    print(
        "COUNTS=train:517 val:65 test:65"
    )
    # 输出完整分层计数字典，可核对每个split的良性和恶性组成。
    print(
        "CLASS_COUNTS={}".format(
            EXPECTED_SPLIT_CLASS_COUNTS
        )
    )
    # 告知样本级审计清单路径。
    print(
        "MANIFEST={}".format(
            output_root / "manifest.csv"
        )
    )
    # 告知数据集级摘要路径。
    print(
        "SUMMARY={}".format(
            output_root
            / "split_summary.json"
        )
    )


# 定义命令行接口；该函数只声明参数并返回解析结果，不执行数据读写。
def parse_args():
    # ArgumentParser生成--help内容，并负责缺少必填参数或类型错误时的标准报错。
    parser = argparse.ArgumentParser(
        description=(
            # 说明实际使用的是647张良性/恶性病灶图，而不是包含normal的780张完整公开包。
            "Prepare the 647-image BUSI "
            "lesion subset for EMCAD using "
            # 协议目标为按图像划分的固定、可审计80/10/10近似比例，精确数由顶部常量规定。
            "a fixed, auditable 80/10/10 "
            "image-level split."
        )
    )

    # 原始BUSI目录必须由用户显式提供，避免脚本猜错同名数据集或路径。
    parser.add_argument(
        "--source_root",
        required=True,
    )
    # 输出根目录可覆盖；默认值对应仓库外层data/busi/target/BUSI布局。
    parser.add_argument(
        "--output_root",
        default=(
            "../data/busi/target/BUSI"
        ),
    )
    # seed必须解析为整数，用于重复组的确定性伪随机排序。
    parser.add_argument(
        "--seed",
        type=int,
        # 默认2222固定常规运行结果；改变种子会改变可行组中的具体样本选择。
        default=2222,
    )

    # 一次性解析当前进程命令行并返回Namespace对象。
    return parser.parse_args()


# 串联“解析参数、发现并验证源数据、分组防泄漏、精确划分、原子写出”的完整流程。
def main():
    # 获取source_root、output_root和seed三个命令行参数。
    args = parse_args()

    # 自动兼容公开包可能多套一层Dataset_BUSI_with_GT目录的情况，并返回规范绝对根路径。
    source_root = resolve_source_root(
        args.source_root
    )

    # 汇总良性与恶性样本；normal类别有意不加入这个病灶分割训练列表。
    samples = []
    # 单独记录每个来源类别实际采集到的样本数，用于严格核对官方数据规模。
    source_counts = {}

    # CLASSES固定为benign和malignant，按同一套规则分别配对原图及其一个或多个mask。
    for class_name in CLASSES:
        # collect_class_samples同时完成命名、配对、可读性、尺寸与哈希校验。
        class_samples = (
            collect_class_samples(
                source_root,
                class_name,
            )
        )

        # 把当前类别样本加入后续统一重复检测和划分池。
        samples.extend(class_samples)
        # 保存该类别实测数量，而不是从目录文件数直接推断。
        source_counts[class_name] = len(
            class_samples
        )

    # 必须准确得到437张良性和210张恶性原图；任何偏差都拒绝继续。
    if (
            source_counts
            != EXPECTED_SOURCE_COUNTS
    ):
        # 报错特别区分原始780图BUSI与论文误写数字、BUSI_WHU等不同数据来源。
        raise RuntimeError(
            "Wrong BUSI lesion counts. "
            "Expected {} but found {}. "
            "Do not use the paper's erroneous "
            "487-benign figure, BUSI_WHU, or "
            "another breast-ultrasound "
            "dataset.".format(
                EXPECTED_SOURCE_COUNTS,
                source_counts,
            )
        )

    # 单独清点normal目录中的原始图像，只写入排除统计，不加入病灶mask训练任务。
    normal_images = (
        count_original_images(
            source_root / "normal"
        )
    )

    # 按统一解码像素SHA-256把完全相同原图绑定为不可拆分组。
    groups = build_duplicate_groups(
        samples
    )

    # 在重复组约束下生成精确分层的train/val/test分区。
    partitions = make_partitions(
        groups,
        args.seed,
    )

    # 将最终分区安全物化到EMCAD可直接读取的目录，并生成manifest与summary。
    write_target(
        # 使用关键字参数明确每个对象含义，避免位置参数顺序误传。
        source_root=source_root,
        output_root=args.output_root,
        partitions=partitions,
        groups=groups,
        seed=args.seed,
        normal_images=normal_images,
    )


# 只有直接执行本脚本时才运行main；作为模块导入时可复用函数而不会自动处理数据。
if __name__ == "__main__":
    # 启动完整数据准备入口。
    main()
