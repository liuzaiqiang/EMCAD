# argparse 构建 isic2017/isic2018 两套命令行子命令。
import argparse
# csv 用于写出每个样本所属 split 及目标相对路径的清单。
import csv
# hashlib 用 SHA-256 固定排序并校验文件内容。
import hashlib
# json 写出可审计的划分摘要。
import json
# os 提供 samefile、硬链接、软链接和原子替换。
import os
# shutil 在 copy 模式下复制文件并保留元数据。
import shutil
# 时间戳以 UTC 写入摘要，便于记录数据制备时间。
from datetime import datetime, timezone
# pathlib.Path 统一处理 Windows/Linux 路径和文件操作。
from pathlib import Path


# ISIC 原始图像允许 jpg/jpeg 两种扩展名。
IMAGE_EXTENSIONS = {".jpg", ".jpeg"}
# 分割标签只接受 png，避免把普通图像误当mask。
MASK_EXTENSIONS = {".png"}
# 检查旧文件时需要识别图像和mask的全部合法扩展名。
ALL_EXTENSIONS = IMAGE_EXTENSIONS | MASK_EXTENSIONS
# 官方mask文件名通常在原图ID后增加 _segmentation。
MASK_SUFFIX = "_segmentation"
# manifest排序时固定 train、val、test 顺序，而不是依赖字符串字典序。
SPLIT_ORDER = {"train": 0, "val": 1, "test": 2}


# 把图像路径和mask路径归一化成同一个可比较ID。
def canonical_id(path, is_mask):
# stem 去掉目录和扩展名，例如 ISIC_0000000.jpg -> ISIC_0000000。
    stem = Path(path).stem
# mask需要再去掉官方 _segmentation 后缀；casefold实现稳健的不区分大小写比较。
    if is_mask and stem.casefold().endswith(MASK_SUFFIX):
# 只删除末尾后缀，保留真正样本ID。
        stem = stem[:-len(MASK_SUFFIX)]
# casefold 比 lower 更适合做跨大小写规范化键。
    return stem.casefold()


# 扫描单个目录并建立 canonical_id -> 绝对路径的唯一索引。
def index_directory(root, is_mask):
# 转换为绝对规范路径，避免相对工作目录变化影响结果。
    root = Path(root).resolve()
# 数据目录必须存在且是目录。
    if not root.is_dir():
# 尽早失败，并把实际解析后的路径写进错误信息。
        raise FileNotFoundError("Directory not found: {}".format(root))

# 图像目录和mask目录使用不同扩展名白名单。
    extensions = MASK_EXTENSIONS if is_mask else IMAGE_EXTENSIONS
# 用字典检测同一规范ID是否对应多个文件。
    indexed = {}

# 排序后遍历，使错误顺序和最终结果在不同机器上保持稳定。
    for path in sorted(root.iterdir()):
# 忽略子目录及不属于当前类型的扩展名。
        if not path.is_file() or path.suffix.lower() not in extensions:
# 继续检查下一个目录项。
            continue

# 为图像保留stem，为mask去掉_segmentation，再统一casefold。
        key = canonical_id(path, is_mask=is_mask)

# 空ID说明文件命名不合法，不能安全配对。
        if not key:
# 立即报告具体文件。
            raise RuntimeError("Empty ISIC ID for: {}".format(path))

# 一个目录内每个规范ID必须唯一，否则无法确定应使用哪个文件。
        if key in indexed:
# 同时列出冲突的两个文件名，便于人工修复数据。
            raise RuntimeError(
                "Duplicate canonical ID in {}: {} and {}".format(
# 冲突所在目录。
                    root,
# 已登记文件。
                    indexed[key].name,
# 新遇到的重复文件。
                    path.name,
                )
            )

# 保存绝对路径，后续物化过程不依赖当前工作目录。
        indexed[key] = path.resolve()

# 空目录或没有合法扩展名通常意味着传错数据路径。
    if not indexed:
# 拒绝生成一个空划分。
        raise RuntimeError("No supported files found in: {}".format(root))

# 返回唯一ID索引。
    return indexed


# 分别索引图像和mask，并进行严格一一配对。
def collect_pairs(image_root, mask_root):
# is_mask=False 时只接受jpg/jpeg，规范ID不删除后缀词。
    images = index_directory(image_root, is_mask=False)
# is_mask=True 时只接受png，并删除_segmentation。
    masks = index_directory(mask_root, is_mask=True)

# 图像ID集合。
    image_ids = set(images)
# mask ID集合。
    mask_ids = set(masks)

# 两集合必须完全一致，既不能缺mask，也不能存在孤立mask。
    if image_ids != mask_ids:
# 分别报告最多20个缺失ID，避免超大错误消息。
        raise RuntimeError(
            "Image/mask IDs do not match. "
            "missing_masks={} missing_images={}".format(
# 有图像但没有mask。
                sorted(image_ids - mask_ids)[:20],
# 有mask但没有图像。
                sorted(mask_ids - image_ids)[:20],
            )
        )

# 按规范ID排序，生成后续划分函数统一使用的样本字典列表。
    return [
        {
# key 用于去重、泄漏检查和确定性排序。
            "key": key,
# sample_id 保留原始图像stem，作为目标文件名。
            "sample_id": images[key].stem,
# 原图绝对路径。
            "image_source": images[key],
# 对应mask绝对路径。
            "mask_source": masks[key],
        }
# 排序保证同一输入得到相同列表顺序。
        for key in sorted(image_ids)
    ]


# 以流式方式计算文件SHA-256，避免一次性把大文件读入内存。
def sha256_file(path):
# 创建空SHA-256状态。
    digest = hashlib.sha256()

# 二进制读取，哈希必须基于原始字节而非文本解码结果。
    with Path(path).open("rb") as stream:
# 每次读取1 MiB，读到b""时停止。
        for block in iter(lambda: stream.read(1024 * 1024), b""):
# 把当前块加入摘要。
            digest.update(block)

# 返回64字符十六进制摘要。
    return digest.hexdigest()


# 判断目标文件是否已经与源文件相同，以支持脚本安全重复运行。
def destination_matches(source, destination):
# 标准化为Path对象。
    source = Path(source)
# 目标也标准化为Path。
    destination = Path(destination)

# 首先尝试判断两路径是否指向同一个底层文件，硬链接/同路径可快速命中。
    try:
# samefile会比较文件系统标识，而不仅是路径字符串。
        if os.path.samefile(source, destination):
# 指向同一文件则无需继续计算哈希。
            return True
# 目标不存在或文件系统不支持samefile时，退回后续内容比较。
    except (FileNotFoundError, OSError):
# 忽略该快速检查失败。
        pass

# 文件大小不同必然内容不同，可避免不必要的哈希计算。
    if source.stat().st_size != destination.stat().st_size:
# 明确返回不匹配。
        return False

# 大小相同后再比较完整SHA-256，降低误判风险。
    return sha256_file(source) == sha256_file(destination)


# 按 mode 把源文件物化到目标目录：硬链接、软链接或真实复制。
def materialize(source, destination, mode):
# 源路径转成绝对路径，软链接也会记录这个绝对目标。
    source = Path(source).resolve()
# 目标路径保持调用方指定的根目录结构。
    destination = Path(destination)
# 确保父目录存在；重复运行时不会报错。
    destination.parent.mkdir(parents=True, exist_ok=True)

# 已存在文件或符号链接时绝不盲目覆盖。
    if destination.exists() or destination.is_symlink():
# 若已存在且内容/底层文件完全相同，则把本次操作视为已完成。
        if destination.exists() and destination_matches(source, destination):
# 幂等返回。
            return

# 不同内容意味着可能混入旧实验数据，必须人工处理。
        raise FileExistsError(
            "Destination exists with different content: {}".format(
# 报告冲突目标。
                destination
            )
        )

# hardlink不复制文件数据，要求源和目标位于支持硬链接的同一文件系统。
    if mode == "hardlink":
# 捕获跨盘或权限错误并给出明确建议。
        try:
# 创建硬链接；源文件和目标共享同一底层数据。
            os.link(str(source), str(destination))
# 转换底层OSError为更易理解的RuntimeError。
        except OSError as error:
# 提示跨文件系统时改用copy。
            raise RuntimeError(
                "Hardlink failed for {}. Use --mode copy if source and "
                "target are on different filesystems. Error: {}".format(
# 失败源路径。
                    source,
# 原始系统错误。
                    error,
                )
            ) from error

# symlink只创建路径引用，移动或删除原始数据后链接可能失效。
    elif mode == "symlink":
# 创建指向绝对源路径的符号链接。
        os.symlink(str(source), str(destination))

# copy生成独立文件副本，占用额外空间但不依赖原目录继续存在。
    elif mode == "copy":
# copy2尽量保留源文件时间戳等元数据。
        shutil.copy2(source, destination)

# 拒绝任何未支持模式，避免静默采用错误行为。
    else:
# 错误信息回显非法mode。
        raise ValueError("Unknown materialization mode: {}".format(mode))


# 使用 seed 和样本ID的SHA-256得分生成跨机器稳定的伪随机顺序。
def deterministic_order(samples, seed):
# 局部评分函数不依赖Python random实现或输入原始顺序。
    def score(sample):
# seed与规范ID通过制表符连接，避免简单字符串拼接歧义。
        payload = "{}\t{}".format(seed, sample["key"]).encode("utf-8")
# 十六进制哈希可直接用于字符串排序。
        return hashlib.sha256(payload).hexdigest()

# 返回新列表，不原地改变调用方samples。
    return sorted(
# 被排序的样本集合。
        samples,
# 首先按哈希得分，极端碰撞时再按key稳定打破平局。
        key=lambda sample: (
            score(sample),
            sample["key"],
        ),
    )


# 检查同一规范ID绝不会同时出现在多个split，直接防止样本级泄漏。
def assert_disjoint(partitions):
# seen记录 key 首次出现在哪个split。
    seen = {}

# 遍历train/val/test及其样本。
    for split, samples in partitions.items():
# 检查当前split中的每个样本。
        for sample in samples:
# 规范ID是跨目录比较依据。
            key = sample["key"]

# 已出现说明划分集合不互斥。
            if key in seen:
# 报告发生冲突的两个split。
                raise RuntimeError(
                    "Split leakage: {} is in both {} and {}".format(
# 冲突样本ID。
                        key,
# 首次出现split。
                        seen[key],
# 当前再次出现split。
                        split,
                    )
                )

# 首次出现时登记归属。
            seen[key] = split


# 计算一个split预期生成的图像文件名集合和mask文件名集合。
def expected_names(samples):
# 原图保留jpg/jpeg扩展名。
    image_names = {
# sample_id加原始小写扩展名构成目标文件名。
        "{}{}".format(
            sample["sample_id"],
            sample["image_source"].suffix.lower(),
        )
# 集合可用于检测目标目录中的多余旧文件。
        for sample in samples
    }

# 所有目标mask统一命名为 sample_id.png，不再保留_segmentation后缀。
    mask_names = {
        "{}.png".format(sample["sample_id"])
        for sample in samples
    }

# 返回两套白名单。
    return image_names, mask_names


# 如果目标split目录含本次划分不期望的合法图像文件，则拒绝继续。
def reject_extra_files(root, expected, label):
# 标准化目录路径。
    root = Path(root)

# 尚未创建的目录自然不存在旧文件。
    if not root.exists():
# 无需检查。
        return

# 收集目录中所有受支持扩展名的普通文件名。
    actual = {
        path.name
        for path in root.iterdir()
# 忽略子目录和无关扩展名。
        if path.is_file() and path.suffix.lower() in ALL_EXTENSIONS
    }

# 实际集合减预期集合得到残留文件。
    extra = sorted(actual - expected)

# 任何残留都可能污染划分或复用旧实验样本。
    if extra:
# 最多展示前20个文件名。
        raise RuntimeError(
            "Unexpected stale files in {} {}: {}".format(
# 人类可读标签，如train images。
                label,
# 实际目录。
                root,
# 残留样例。
                extra[:20],
            )
        )


# 把已经确定的partitions写成EMCAD loader需要的目录，并生成manifest和summary。
def write_dataset(
# 输出根目录包含train/val/test三级子目录。
    output_root,
# dataset_name写入摘要，例如ISIC2017。
    dataset_name,
# protocol记录采用官方划分还是固定80/10/10。
    protocol,
# partitions是split -> 样本列表映射。
    partitions,
# mode控制硬链接、软链接或复制。
    mode,
# seed仅对重新划分的数据集有值；官方2017划分传None。
    seed,
# sources记录所有原始目录，保证可追溯。
    sources,
):
# 输出路径转成绝对路径。
    output_root = Path(output_root).resolve()
# 创建根目录；该脚本允许安全重复运行，但会拒绝不同内容或多余文件。
    output_root.mkdir(parents=True, exist_ok=True)

# 在写任何样本前验证三份集合互斥。
    assert_disjoint(partitions)

# 保存各split的(images_dir,masks_dir)，供第二轮物化使用。
    directories = {}

# 第一轮只计算预期文件并检查现有目录是否干净。
    for split in ("train", "val", "test"):
# 当前split样本。
        samples = partitions[split]
# 生成图像和mask目标文件名白名单。
        image_names, mask_names = expected_names(samples)

# EMCAD图像目录约定。
        image_dir = output_root / split / "images"
# EMCAD mask目录约定。
        mask_dir = output_root / split / "masks"

# 检查图像目录没有不属于当前划分的旧文件。
        reject_extra_files(
            image_dir,
            image_names,
            "{} images".format(split),
        )
# 同样检查mask目录。
        reject_extra_files(
            mask_dir,
            mask_names,
            "{} masks".format(split),
        )

# 记录目录路径；materialize会负责创建父目录。
        directories[split] = (image_dir, mask_dir)

# 收集manifest每一行。
    all_rows = []

# 第二轮真正物化文件。
    for split in ("train", "val", "test"):
# 取出当前split的目标目录。
        image_dir, mask_dir = directories[split]

# 逐样本写入成对文件。
        for sample in partitions[split]:
# 原图保留原扩展名，sample_id保持官方命名。
            image_name = "{}{}".format(
                sample["sample_id"],
                sample["image_source"].suffix.lower(),
            )
# mask统一使用sample_id.png。
            mask_name = "{}.png".format(sample["sample_id"])

# 目标图像完整路径。
            image_target = image_dir / image_name
# 目标mask完整路径。
            mask_target = mask_dir / mask_name

# 按指定模式物化原图。
            materialize(
                sample["image_source"],
                image_target,
                mode,
            )
# mask使用完全相同模式。
            materialize(
                sample["mask_source"],
                mask_target,
                mode,
            )

# 写入相对路径，保证移动整个数据根目录后manifest仍可解释。
            all_rows.append(
                {
# 样本所属划分。
                    "split": split,
# 官方样本ID。
                    "sample_id": sample["sample_id"],
# 图像相对output_root路径。
                    "image": str(
                        image_target.relative_to(output_root)
                    ),
# mask相对路径。
                    "mask": str(
                        mask_target.relative_to(output_root)
                    ),
                }
            )

# 固定manifest行序，便于版本比较和哈希复现。
    all_rows.sort(
# 先train/val/test，再按不区分大小写的sample_id。
        key=lambda row: (
            SPLIT_ORDER[row["split"]],
            row["sample_id"].casefold(),
        )
    )

# 最终manifest路径。
    manifest_path = output_root / "split_manifest.csv"
# 先写临时文件，再原子替换，防止中途中断留下半个CSV。
    manifest_tmp = output_root / "split_manifest.csv.tmp"

# newline=""是csv模块推荐写法，避免Windows产生空行。
    with manifest_tmp.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as stream:
# 固定列顺序。
        writer = csv.DictWriter(
            stream,
            fieldnames=[
                "split",
                "sample_id",
                "image",
                "mask",
            ],
        )
# 写表头。
        writer.writeheader()
# 一次写入所有排序后的样本行。
        writer.writerows(all_rows)

# 用原子替换发布完整manifest。
    os.replace(manifest_tmp, manifest_path)

# 逻辑manifest摘要只依赖字段内容，不依赖CSV换行风格或写入元数据。
    manifest_digest = hashlib.sha256()

# 逐行按固定制表符协议加入哈希。
    for row in all_rows:
# 字段次序固定为split、sample_id、image、mask。
        manifest_digest.update(
            (
                "{split}\t{sample_id}\t"
                "{image}\t{mask}\n"
            ).format(**row).encode("utf-8")
        )

# 统计每个split样本数。
    counts = {
        split: len(partitions[split])
        for split in ("train", "val", "test")
    }

# 摘要记录复现实验划分所需的关键信息。
    summary = {
# 数据集版本名。
        "dataset_name": dataset_name,
# 划分协议。
        "protocol": protocol,
# 当前只能确认按图像ID互斥，不能宣称患者级互斥。
        "split_unit": "image",
# 2018为整数seed，2017官方划分为None。
        "seed": seed,
# 各split计数。
        "counts": counts,
# 总配对数。
        "total": sum(counts.values()),
# 逻辑清单哈希用于判断两次运行划分是否一致。
        "manifest_sha256": manifest_digest.hexdigest(),
# 记录物化方式，解释磁盘占用和源目录依赖。
        "materialization_mode": mode,
# 保存全部原始目录的绝对路径。
        "source_directories": {
            key: str(Path(value).resolve())
            for key, value in sources.items()
        },
# 生成时刻使用带时区UTC格式。
        "created_utc": datetime.now(
            timezone.utc
        ).isoformat(),
# 明确样本ID互斥不等于患者/病灶互斥，避免研究结论越界。
        "notes": (
            "Canonical IDs are disjoint across splits. "
            "This does not prove patient-level or "
            "lesion-level separation."
        ),
    }

# 最终JSON路径。
    summary_path = output_root / "split_summary.json"
# 同样先写临时文件以保证原子发布。
    summary_tmp = output_root / "split_summary.json.tmp"

# UTF-8写入，允许摘要内容包含非ASCII字符。
    with summary_tmp.open(
        "w",
        encoding="utf-8",
    ) as stream:
# 以2空格缩进输出便于人工审阅。
        json.dump(
            summary,
            stream,
            ensure_ascii=False,
            indent=2,
        )

# 原子替换最终summary。
    os.replace(summary_tmp, summary_path)

# 控制台输出供shell脚本或人工快速确认结果。
    print("DATASET={}".format(dataset_name))
# 输出协议名。
    print("PROTOCOL={}".format(protocol))
# 输出三份计数。
    print(
        "COUNTS=train:{} val:{} test:{}".format(
            counts["train"],
            counts["val"],
            counts["test"],
        )
    )
# 输出manifest绝对位置。
    print("MANIFEST={}".format(manifest_path))
# 输出summary绝对位置。
    print("SUMMARY={}".format(summary_path))


# ISIC2017已有官方train/val/test目录，本函数只验证并原样保留官方划分。
def prepare_2017(args):
# 分别严格配对三套官方目录。
    partitions = {
# 官方训练集。
        "train": collect_pairs(
            args.train_images,
            args.train_masks,
        ),
# 官方验证集。
        "val": collect_pairs(
            args.val_images,
            args.val_masks,
        ),
# 官方测试集。
        "test": collect_pairs(
            args.test_images,
            args.test_masks,
        ),
    }

# ISIC2017官方样本数，用于阻止误传目录或漏下载。
    expected = {
        "train": 2000,
        "val": 150,
        "test": 600,
    }

# 计算实际配对数。
    actual = {
        split: len(samples)
        for split, samples in partitions.items()
    }

# 默认严格要求官方计数；只有显式开关才能进行有记录的非标准研究。
    if (
        not args.allow_nonstandard_count
        and actual != expected
    ):
# 报告期望和实际计数。
        raise RuntimeError(
            "ISIC2017 official counts must be {} "
            "but found {}. Use "
            "--allow_nonstandard_count only for a "
            "documented nonstandard study.".format(
                expected,
                actual,
            )
        )

# 写出统一目录、manifest和summary。
    write_dataset(
# 用户指定目标根目录。
        output_root=args.output_root,
# 摘要数据集名。
        dataset_name="ISIC2017",
# 明确没有重新随机划分。
        protocol="official_train_val_test",
# 官方三份集合。
        partitions=partitions,
# 文件物化方式。
        mode=args.mode,
# 官方划分不依赖随机种子。
        seed=None,
# 保存六个原始目录位置。
        sources={
            "train_images": args.train_images,
            "train_masks": args.train_masks,
            "val_images": args.val_images,
            "val_masks": args.val_masks,
            "test_images": args.test_images,
            "test_masks": args.test_masks,
        },
    )


# ISIC2018只有一个标注池，本函数按固定哈希顺序切成80/10/10。
def prepare_2018(args):
# 收集2594对图像和mask。
    samples = collect_pairs(
        args.images,
        args.masks,
    )

# 默认要求EMCAD使用的完整2594样本池。
    if (
        not args.allow_nonstandard_count
        and len(samples) != 2594
    ):
# 非标准数量必须显式确认，避免无意中训练不完整数据。
        raise RuntimeError(
            "ISIC2018 EMCAD pool must contain "
            "2594 paired images, found {}. Use "
            "--allow_nonstandard_count only for a "
            "documented nonstandard study.".format(
                len(samples)
            )
        )

# 用seed+规范ID哈希产生可复现顺序，而不是依赖文件系统枚举顺序。
    ordered = deterministic_order(
        samples,
        args.seed,
    )

# 训练集取向下取整的80%。
    train_count = int(len(ordered) * 0.80)
# 验证集取向下取整的10%，剩余全部进入测试集。
    val_count = int(len(ordered) * 0.10)

# 按连续切片生成互不重叠的三份集合。
    partitions = {
# 前80%。
        "train": ordered[:train_count],
# 接下来的10%。
        "val": ordered[
            train_count:train_count + val_count
        ],
# 剩余约10%，吸收整数取整余数。
        "test": ordered[
            train_count + val_count:
        ],
    }

# 写出统一目录和审计文件。
    write_dataset(
        output_root=args.output_root,
        dataset_name="ISIC2018",
# 协议名明确这是图像级划分，不是患者级划分。
        protocol="emcad_80_10_10_image_level",
        partitions=partitions,
        mode=args.mode,
# 保存实际seed以便完全复现。
        seed=args.seed,
# 记录原始图像和mask目录。
        sources={
            "images": args.images,
            "masks": args.masks,
        },
    )


# 为2017和2018子命令添加相同的输出与安全参数。
def add_common_arguments(parser):
# 目标数据根目录必须显式给出。
    parser.add_argument(
        "--output_root",
        required=True,
    )
# 选择文件物化方式。
    parser.add_argument(
        "--mode",
# 限制为脚本已实现的三种模式。
        choices=[
            "hardlink",
            "symlink",
            "copy",
        ],
# 硬链接默认不额外占用文件数据空间。
        default="hardlink",
    )
# 允许显式绕过官方样本数检查，但不绕过ID配对和泄漏检查。
    parser.add_argument(
        "--allow_nonstandard_count",
# 出现该开关时值为True。
        action="store_true",
    )


# 构建完整CLI并把子命令绑定到对应处理函数。
def parse_args():
# 顶层帮助说明强调固定和可审计。
    parser = argparse.ArgumentParser(
        description=(
            "Prepare fixed, auditable ISIC "
            "splits for EMCAD"
        )
    )

# 两种数据集版本参数不同，因此使用必选子命令。
    subparsers = parser.add_subparsers(
        dest="command",
# Python会要求用户必须选择isic2017或isic2018。
        required=True,
    )

# ISIC2017子命令保留官方划分。
    parser_2017 = subparsers.add_parser(
        "isic2017",
        help="Preserve the official 2000/150/600 split",
    )
# 官方训练图像目录。
    parser_2017.add_argument(
        "--train_images",
        required=True,
    )
# 官方训练mask目录。
    parser_2017.add_argument(
        "--train_masks",
        required=True,
    )
# 官方验证图像目录。
    parser_2017.add_argument(
        "--val_images",
        required=True,
    )
# 官方验证mask目录。
    parser_2017.add_argument(
        "--val_masks",
        required=True,
    )
# 官方测试图像目录。
    parser_2017.add_argument(
        "--test_images",
        required=True,
    )
# 官方测试mask目录。
    parser_2017.add_argument(
        "--test_masks",
        required=True,
    )
# 加入output_root、mode和非标准计数开关。
    add_common_arguments(parser_2017)
# 解析后通过args.function统一调度到prepare_2017。
    parser_2017.set_defaults(
        function=prepare_2017
    )

# ISIC2018子命令从单一池重新划分。
    parser_2018 = subparsers.add_parser(
        "isic2018",
        help=(
            "Split the 2594 labeled images "
            "using fixed 80/10/10"
        ),
    )
# 2018完整图像目录。
    parser_2018.add_argument(
        "--images",
        required=True,
    )
# 2018完整mask目录。
    parser_2018.add_argument(
        "--masks",
        required=True,
    )
# 固定排序seed，默认与项目其他实验一致为2222。
    parser_2018.add_argument(
        "--seed",
        type=int,
        default=2222,
    )
# 添加公共参数。
    add_common_arguments(parser_2018)
# 绑定prepare_2018。
    parser_2018.set_defaults(
        function=prepare_2018
    )

# 返回解析后的Namespace，其中function字段是可调用对象。
    return parser.parse_args()


# 程序主入口只负责解析参数并分派到版本专属函数。
def main():
# 读取CLI参数。
    args = parse_args()
# 调用prepare_2017(args)或prepare_2018(args)。
    args.function(args)


# 作为脚本运行时执行main；被import时不会自动制备数据。
if __name__ == "__main__":
# 启动命令行流程。
    main()
