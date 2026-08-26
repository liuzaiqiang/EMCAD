# argparse 定义 Polyp 验证/测试命令行参数。
import argparse
# json 保存机器可读的评估摘要与实际配置。
import json
# logging 同时记录到 test.log 和控制台。
import logging
# math.isfinite 把 NaN/Inf 转成 JSON 可接受的 null。
import math
# os 负责路径拼接、绝对路径解析和目录创建。
import os
# sys.stdout 用作日志流处理器目标。
import sys
# Path 用于按扩展名扫描指定数据划分的图像 stem。
from pathlib import Path

# torch 仅用于版本/CUDA 环境记录；模型推理封装在 utils.polyp_utils 中。
import torch

# 数据加载器同时暴露支持扩展名集合，供划分泄漏检查复用同一文件口径。
from utils.dataloader_polyp import (
    # 合法图像/掩膜后缀集合。
    SUPPORTED_EXTENSIONS,
    # 构建严格图像-掩膜配对的评估 DataLoader。
    get_loader,
)
# Polyp 评估公共能力：模型构建、检查点、推理、指标和 CSV。
from utils.polyp_utils import (
    # 按命令行的 EMCAD/编码器参数创建模型。
    build_model,
    # 逐图恢复原尺寸并计算 Dice、IoU、HD95 等指标。
    evaluate_loader,
    # 兼容不同 checkpoint 字典结构并加载权重。
    load_checkpoint,
    # 把 auto/cpu/cuda 字符串解析成 torch.device。
    resolve_device,
    # 固定 Python、NumPy、PyTorch 和 CUDA 随机源。
    seed_everything,
    # 将逐图、均值和标准差写成统一字段 CSV。
    write_metrics_csv,
)


# 解析评估所需的数据、模型、运行环境和输出选项。
def parse_args():
    # 创建命令行解析器并给出用途说明。
    parser = argparse.ArgumentParser(
        # 该文字显示在 --help 顶部。
        description="Evaluate EMCAD on one Polyp split"
    )

    # 必填：要评估的模型检查点。
    parser.add_argument(
        # 选项名。
        "--checkpoint",
        # 未传该参数时 argparse 直接报错。
        required=True,
    )
    # 已准备 Polyp 数据集的总根目录。
    parser.add_argument(
        # 选项名。
        "--data_root",
        # 目录下预期包含 ClinicDB、Kvasir 等数据集子目录。
        default="../data/polyp/target",
    )
    # 当前要评估的数据集子目录名。
    parser.add_argument(
        # 选项名。
        "--dataset_name",
        # 默认与训练入口一致使用 ClinicDB。
        default="ClinicDB",
    )
    # 选择验证集或最终测试集。
    parser.add_argument(
        # 选项名。
        "--split",
        # 限制为 val/test，避免误用训练集做正式评估。
        choices=["val", "test"],
        # 默认进行最终测试。
        default="test",
    )
    # 二值掩膜、概率图、日志和 JSON 的输出目录。
    parser.add_argument(
        # 选项名。
        "--output_dir",
        # None 时根据 checkpoint 所在目录自动生成。
        default=None,
    )
    # 指标 CSV 的可选显式路径。
    parser.add_argument(
        # 选项名。
        "--output_csv",
        # None 时写到 output_dir/test_metrics.csv。
        default=None,
    )

    # 层次化编码器名称；必须与训练检查点架构匹配。
    parser.add_argument(
        # 选项名。
        "--encoder",
        # 论文标准模型默认 PVTv2-B2。
        default="pvt_v2_b2",
    )
    # MSDC 多尺度深度卷积核列表。
    parser.add_argument(
        # 选项名。
        "--kernel_sizes",
        # 每个命令行值转为 int。
        type=int,
        # 接受一个或多个核尺寸。
        nargs="+",
        # 论文主文 PDF 第6页 Sec.4.1 的默认并行核 [1,3,5]。
        default=[1, 3, 5],
    )
    # MSCB 内部通道扩张倍数。
    parser.add_argument(
        # 选项名。
        "--expansion_factor",
        # 整数类型。
        type=int,
        # 论文默认 e=2，方法公式见主文 PDF 第4页 Eq.(4)。
        default=2,
    )
    # LGAG 两路分组卷积核尺寸。
    parser.add_argument(
        # 选项名。
        "--lgag_ks",
        # 整数类型。
        type=int,
        # 默认 3x3，对应主文 PDF 第3页 Sec.3.1.1。
        default=3,
    )
    # MSCB 使用的激活名称。
    parser.add_argument(
        # 选项名。
        "--activation_mscb",
        # 论文实现采用 ReLU6。
        default="relu6",
    )
    # 出现该标志时关闭论文默认的并行 MSDC 路径。
    parser.add_argument(
        # 选项名。
        "--no_dw_parallel",
        # 布尔开关，未出现时为 False。
        action="store_true",
    )
    # 出现时用通道拼接替代多尺度分支相加。
    parser.add_argument(
        # 选项名。
        "--concatenation",
        # 布尔开关；拼接是工程消融选项，不是论文默认 Eq.(5) 求和路径。
        action="store_true",
    )
    # PVT 本地预训练权重目录；测试构建模型时 pretrain=False，不会实际加载这里的权重。
    parser.add_argument(
        # 选项名。
        "--pretrained_dir",
        # 默认目录。
        default="./pretrained_pth/pvt/",
    )

    # 推理输入的统一正方形边长。
    parser.add_argument(
        # 选项名。
        "--img_size",
        # 整数像素。
        type=int,
        # Polyp 论文实现使用 352x352。
        default=352,
    )
    # 单次前向的图像数量。
    parser.add_argument(
        # 选项名。
        "--inference_batch_size",
        # 整数。
        type=int,
        # 默认 1，降低不同原图尺寸恢复时的内存压力。
        default=1,
    )
    # DataLoader 子进程数量。
    parser.add_argument(
        # 选项名。
        "--num_workers",
        # 整数。
        type=int,
        # Windows 兼容默认 0，即主进程读取。
        default=0,
    )
    # 把归一化概率图转成二值掩膜的阈值。
    parser.add_argument(
        # 选项名。
        "--threshold",
        # 浮点数。
        type=float,
        # 默认 0.5；evaluate_loader 在逐图 min-max 后应用该值。
        default=0.5,
    )
    # 随机种子，主要保证数据加载等可重复设置。
    parser.add_argument(
        # 选项名。
        "--seed",
        # 整数。
        type=int,
        # 与训练默认种子一致。
        default=2222,
    )
    # 是否要求 cuDNN 确定性行为。
    parser.add_argument(
        # 选项名。
        "--deterministic",
        # 使用 0/1 整数，main 中转为 bool。
        type=int,
        # 限制合法值。
        choices=[0, 1],
        # 默认开启。
        default=1,
    )
    # 推理设备选择。
    parser.add_argument(
        # 选项名。
        "--device",
        # auto 优先 CUDA，不可用时回退 CPU。
        default="auto",
    )
    # 只评估前 N 张图的调试上限。
    parser.add_argument(
        # 选项名。
        "--max_cases",
        # 整数。
        type=int,
        # 0 表示不限制。
        default=0,
    )
    # 以灰度而非 RGB 方式读取输入图像。
    parser.add_argument(
        # 选项名。
        "--grayscale",
        # 布尔开关。
        action="store_true",
    )
    # 除二值掩膜外同时保存 8 位概率可视化图。
    parser.add_argument(
        # 选项名。
        "--save_probabilities",
        # 布尔开关。
        action="store_true",
    )
    # 仅计算指标，不写逐图预测文件。
    parser.add_argument(
        # 选项名。
        "--no_save_predictions",
        # 布尔开关。
        action="store_true",
    )

    # 返回 Namespace；后续 build_model/get_loader 直接读取其中字段。
    return parser.parse_args()


# 扫描某一 split/images 目录，返回大小写归一化后的文件 stem 集合。
def _split_stems(dataset_root, split):
    # 按 <dataset>/<split>/images 构造路径。
    image_root = Path(dataset_root) / split / "images"

    # 划分目录不存在时返回空集，使其他不存在的划分不阻断当前评估。
    if not image_root.is_dir():
        # 空集合与 selected_stems 求交后自然无重叠。
        return set()

    # 集合推导式用于后续 train/val/test 泄漏检查。
    return {
        # casefold 比 lower 更适合做不区分大小写比较。
        path.stem.casefold()
        # 只遍历当前 images 目录第一层，不递归子目录。
        for path in image_root.iterdir()
        # 同时要求普通文件且后缀属于加载器支持集合。
        if (
            # 排除目录和其他特殊条目。
                path.is_file()
                # 文件扩展名统一转小写后匹配。
                and path.suffix.lower()
                # 与实际数据加载器共享同一合法后缀口径。
                in SUPPORTED_EXTENSIONS
        )
    }


# 递归把报告中的 NaN/Inf 替换为 None，保证 json.dump 输出标准 JSON null。
def _json_safe(value):
    # 字典需要逐键递归处理嵌套值。
    if isinstance(value, dict):
        # 保持原键不变，只清洗对应 item。
        return {
            # 对每个值递归调用本函数。
            key: _json_safe(item)
            # 遍历字典项。
            for key, item in value.items()
        }

    # 列表同样逐元素递归清洗。
    if isinstance(value, list):
        # 列表推导保持顺序。
        return [_json_safe(item) for item in value]

    # JSON 标准不定义 NaN 和正负无穷；表面距离在空掩膜时可能产生这些值。
    if isinstance(value, float) and not math.isfinite(value):
        # Python None 会被 json.dump 序列化为 null。
        return None

    # 字符串、整数、有限浮点和布尔值原样返回。
    return value


# 程序主入口：校验数据与划分，构建模型，评估并保存完整报告。
def main():
    # 读取命令行参数。
    args = parse_args()

    # 在任何模型构建前先确认 checkpoint 路径指向现有普通文件。
    if not os.path.isfile(args.checkpoint):
        # 给出包含用户实际输入路径的明确错误。
        raise FileNotFoundError(
            # format 把路径插入消息。
            "Checkpoint not found: {}".format(
                # 原始命令行值保留相对/绝对写法，便于定位输入错误。
                args.checkpoint
            )
        )

    # 二值阈值必须严格位于 0 和 1 之间。
    if not 0.0 < args.threshold < 1.0:
        # 阻止无意义阈值进入评估过程。
        raise ValueError(
            # 错误信息说明合法区间。
            "--threshold must be between 0 and 1"
        )

    # 数据集根形如 <data_root>/<dataset_name>。
    dataset_root = os.path.join(
        # 已准备数据总根。
        args.data_root,
        # ClinicDB/Kvasir 等具体数据集名。
        args.dataset_name,
    )
    # 当前评估划分根形如 <dataset_root>/test 或 val。
    split_root = os.path.join(
        # 具体数据集根。
        dataset_root,
        # 用户选择的 val/test。
        args.split,
    )

    # 当前划分必须同时具有 images 与 masks 两个目录。
    required = [
        # 输入图像目录。
        os.path.join(split_root, "images"),
        # 真值掩膜目录。
        os.path.join(split_root, "masks"),
    ]

    # 收集所有缺失目录，一次性报告而不是逐个失败。
    missing = [
        # 保留缺失路径字符串。
        path
        # 遍历两个必需目录。
        for path in required
        # 只记录不是现有目录的项。
        if not os.path.isdir(path)
    ]

    # 任一目录缺失时停止评估。
    if missing:
        # 多行列出全部缺失路径。
        raise FileNotFoundError(
            # 固定标题行。
            "Missing Polyp evaluation directories:\n"
            # 每个路径独占一行。
            + "\n".join(missing)
        )

    # 获取当前划分所有合法图像的标准化 stem。
    selected_stems = _split_stems(
        # 具体数据集根。
        dataset_root,
        # val 或 test。
        args.split,
    )

    # 与 train、val、test 逐一比较，防止同名图像跨划分泄漏。
    for other_split in ("train", "val", "test"):
        # 当前划分不需要与自身比较。
        if other_split == args.split:
            # 跳到下一个划分。
            continue

        # 集合交集得到两划分共有的 stem。
        overlap = (
            # 当前评估划分集合。
                selected_stems
                # 与另一个划分集合求交。
                & _split_stems(
            # 同一数据集根。
            dataset_root,
            # train/val/test 中的另一个名称。
            other_split,
        )
        )

        # 发现任意同名样本就拒绝继续，避免泄漏污染指标。
        if overlap:
            # RuntimeError 中报告划分名和前十个冲突 stem。
            raise RuntimeError(
                # format 填充当前/另一划分和冲突示例。
                "{} and {} overlap: {}".format(
                    # 当前划分。
                    args.split,
                    # 冲突划分。
                    other_split,
                    # 排序后截取前十项，使错误消息稳定且不过长。
                    sorted(overlap)[:10],
                )
            )

    # 固定随机源和 cuDNN 确定性设置。
    seed_everything(
        # 用户种子。
        args.seed,
        # 0/1 转为布尔值。
        bool(args.deterministic),
    )

    # 解析最终推理设备。
    device = resolve_device(args.device)

    # 创建不打乱顺序、不开启增强的评估 DataLoader。
    loader = get_loader(
        # 图像目录。
        image_root=os.path.join(
            # 当前划分根。
            split_root,
            # 固定子目录名。
            "images",
        ),
        # 掩膜目录。
        gt_root=os.path.join(
            # 当前划分根。
            split_root,
            # 固定子目录名。
            "masks",
        ),
        # 推理 batch 大小。
        batchsize=args.inference_batch_size,
        # 网络统一输入边长。
        trainsize=args.img_size,
        # 测试必须保持确定顺序，便于 CSV 与文件名对应。
        shuffle=False,
        # DataLoader worker 数。
        num_workers=args.num_workers,
        # CUDA 时锁页内存可加速主机到 GPU 传输。
        pin_memory=device.type == "cuda",
        # 评估不能使用随机训练增强。
        augmentation=False,
        # 传入 val/test 名称，加载器据此选择返回字段。
        split=args.split,
        # 默认读取 RGB；--grayscale 时改为单通道。
        color_image=not args.grayscale,
        # 传给 worker 初始化和数据顺序控制。
        seed=args.seed,
    )

    # 将检查点统一为绝对路径，写入报告后不依赖运行时工作目录。
    checkpoint = os.path.abspath(args.checkpoint)
    # 取检查点父目录作为默认输出位置的基准。
    checkpoint_dir = os.path.dirname(checkpoint)

    # 解析最终输出目录。
    output_dir = os.path.abspath(
        # 优先使用命令行显式路径；否则自动拼接划分和数据集名。
        args.output_dir
        # Python or 在左值为 None/空字符串时采用右侧默认值。
        or os.path.join(
            # 检查点目录。
            checkpoint_dir,
            # 目录名如 test_ClinicDB_outputs。
            "{}_{}_outputs".format(
                # val/test。
                args.split,
                # 数据集名。
                args.dataset_name,
            ),
        )
    )

    # 解析最终指标 CSV 路径。
    output_csv = os.path.abspath(
        # 优先使用显式 --output_csv。
        args.output_csv
        # 未指定时放到输出目录内。
        or os.path.join(
            # 评估输出目录。
            output_dir,
            # 固定 CSV 文件名。
            "test_metrics.csv",
        )
    )

    # 创建输出目录；已存在时不报错。
    os.makedirs(output_dir, exist_ok=True)

    # 初始化根日志配置。
    logging.basicConfig(
        # 文件日志写入 output_dir/test.log。
        filename=os.path.join(
            # 输出目录。
            output_dir,
            # 日志文件名。
            "test.log",
        ),
        # 记录 INFO 及以上级别。
        level=logging.INFO,
        # 每行包含时间到毫秒和消息正文。
        format="[%(asctime)s.%(msecs)03d] %(message)s",
        # 时分秒格式。
        datefmt="%H:%M:%S",
        # 强制替换此前可能由导入模块建立的处理器。
        force=True,
    )
    # 再添加标准输出处理器，使终端实时看到同样日志。
    logging.getLogger().addHandler(
        # 日志写到当前进程 stdout。
        logging.StreamHandler(sys.stdout)
    )

    # 记录完整 Namespace，便于复现实验命令。
    logging.info("args=%s", args)
    # 记录最终设备。
    logging.info("device=%s", device)
    # 记录严格配对后实际可评估的图像数量。
    logging.info(
        # 日志格式。
        "images=%d",
        # Dataset 长度。
        len(loader.dataset),
    )

    # 按与训练一致的架构参数创建 EMCADNet。
    model = build_model(
        # build_model 从 args 读取 encoder、kernel_sizes、LGAG 等字段。
        args,
        # 测试只加载完整 checkpoint，不另行加载编码器预训练权重。
        pretrain=False,
    )

    # 把检查点参数载入刚创建的同构模型。
    load_checkpoint(
        # 目标模型。
        model,
        # 已绝对化的检查点路径。
        checkpoint,
    )

    # 将参数迁移到目标设备并切换 eval，关闭 Dropout 随机性并冻结 BN 运行统计更新。
    model.to(device).eval()

    # 对整个 loader 做逐图评估，并可选保存掩膜/概率图。
    rows, mean_row, std_row = evaluate_loader(
        # 已加载权重的模型。
        model=model,
        # 评估 DataLoader。
        loader=loader,
        # CPU 或 CUDA。
        device=device,
        # 二值化阈值；evaluate_loader 会先逐图 min-max 归一化概率。
        threshold=args.threshold,
        # 0 表示全部图像，否则只处理前 N 张。
        max_cases=args.max_cases,
        # --no_save_predictions 时传 None，阻止逐图文件写入。
        output_dir=(
            # 条件表达式真分支。
            None
            # 用户要求不保存预测时使用 None。
            if args.no_save_predictions
            # 否则写到评估输出目录。
            else output_dir
        ),
        # 是否额外写概率可视化图。
        save_probabilities=args.save_probabilities,
        # 计算 HD95/ASSD 表面距离；其单位是原图像素。
        compute_surface=True,
        # tqdm 进度条标题。
        description="Polyp {}".format(args.split),
    )

    # 保存逐图 rows、宏平均 mean_row 和总体标准差 std_row。
    write_metrics_csv(
        # CSV 目标路径。
        output_csv,
        # 每张图一行。
        rows,
        # 汇总均值行。
        mean_row,
        # 汇总标准差行。
        std_row,
    )

    # 组装包含数据、配置、指标口径和运行环境的完整摘要。
    report = {
        # 数据集名。
        "dataset_name": args.dataset_name,
        # 实际评估 val/test。
        "split": args.split,
        # 绝对检查点路径。
        "checkpoint": checkpoint,
        # 绝对输出目录。
        "output_dir": output_dir,
        # 绝对 CSV 路径。
        "output_csv": output_csv,
        # rows 长度反映 max_cases 限制后真正处理的图像数。
        "evaluated_images": len(rows),
        # Dataset 中全部严格配对样本数。
        "dataset_images": len(loader.dataset),
        # 加载器基于样本清单计算的哈希，用于确认不同实验使用同一病例集合。
        "manifest_sha256": loader.dataset.manifest_sha256,
        # 实际二值阈值。
        "threshold": args.threshold,
        # 按图像宏平均后的指标。
        "macro_mean": mean_row,
        # 按图像计算的总体标准差。
        "macro_std": std_row,
        # 明确报告中的统计与空掩膜处理规则，避免只看数值却误解定义。
        "metric_policy": {
            # 聚合不是把所有像素拼在一起算，而是先逐图再宏平均。
            "aggregation": (
                # population std 表示分母使用 N，而不是无偏样本标准差 N-1。
                "per-image macro mean and population std"
            ),
            # Dice/IoU 基于固定阈值后的二值预测与二值真值。
            "dice_and_iou": (
                # sigmoid 之后还会由 evaluate_loader 做逐图 min-max；该字符串概括既有报告口径。
                "binary masks at fixed sigmoid threshold"
            ),
            # 未使用物理 spacing，表面距离以像素计。
            "hd95_and_assd_unit": "pixels",
            # 仅一侧为空时距离没有有限定义。
            "surface_one_empty": (
                # 用 NaN 表示距离，并把 defined 标志设为 0。
                "NaN and surface_distance_defined=0"
            ),
            # 双方都为空按完全一致，距离记 0。
            "surface_both_empty": "0 pixels",
        },
        # 保存所有原始命令行参数。
        "args": vars(args),
        # 记录 PyTorch 版本，便于复现数值/API 环境。
        "torch_version": torch.__version__,
        # 记录当前进程是否能看到 CUDA。
        "cuda_available": torch.cuda.is_available(),
        # 记录 PyTorch 构建时使用的 CUDA 运行时版本；CPU 构建时通常为 None。
        "cuda_version": torch.version.cuda,
    }

    # 写出完整测试摘要 JSON。
    with open(
            # 文件固定放在 output_dir。
            os.path.join(
                # 输出目录。
                output_dir,
                # 摘要文件名。
                "test_summary.json",
            ),
            # 覆盖写入文本。
            "w",
            # UTF-8 保留数据集名等非 ASCII 字符。
            encoding="utf-8",
    ) as stream:
        # 序列化清洗后的报告。
        json.dump(
            # 递归把 NaN/Inf 替换成 None。
            _json_safe(report),
            # 输出文件对象。
            stream,
            # 中文不转义为 \uXXXX。
            ensure_ascii=False,
            # 两空格缩进便于人工阅读与版本比较。
            indent=2,
        )

    # 另写 test_config.json，突出本次运行参数和解析后的关键绝对路径。
    with open(
            # 配置文件路径。
            os.path.join(
                # 输出目录。
                output_dir,
                # 配置文件名。
                "test_config.json",
            ),
            # 覆盖写入。
            "w",
            # UTF-8 编码。
            encoding="utf-8",
    ) as stream:
        # 序列化配置字典。
        json.dump(
            # 同样清理 JSON 非有限浮点。
            _json_safe(
                # 构造配置快照。
                {
                    # 展开 Namespace 的全部字段。
                    **vars(args),
                    # 用绝对路径覆盖原始 checkpoint 值。
                    "checkpoint": checkpoint,
                    # 记录绝对数据总根。
                    "data_root": os.path.abspath(
                        # 原始数据根参数。
                        args.data_root
                    ),
                    # 记录绝对输出目录。
                    "output_dir": output_dir,
                    # 记录绝对 CSV 路径。
                    "output_csv": output_csv,
                }
            ),
            # 文件对象。
            stream,
            # 保留 Unicode。
            ensure_ascii=False,
            # 两空格缩进。
            indent=2,
        )

    # 在终端打印固定列标题。
    print("metric          MEAN          STD")

    # 按统一顺序打印所有区域、分类和表面距离指标。
    for name in (
            # Dice 重合度。
            "dice",
            # IoU/Jaccard。
            "iou",
            # 前景召回率。
            "sensitivity",
            # 背景真负率。
            "specificity",
            # 前景精确率。
            "precision",
            # 全像素准确率。
            "accuracy",
            # 95 百分位 Hausdorff 距离。
            "hd95",
            # 平均对称表面距离。
            "assd",
    ):
        # 每个指标一行，名称左对齐，均值/标准差保留六位小数。
        print(
            # format 控制列宽和精度。
            "{:<12} {:>12.6f} {:>12.6f}".format(
                # 指标名。
                name,
                # 宏平均。
                mean_row[name],
                # 总体标准差。
                std_row[name],
            )
        )

    # 打印表面距离有定义的病例数与总评估病例数。
    print(
        # 文本格式 SURFACE_VALID=x/y。
        "SURFACE_VALID={}/{}".format(
            # summarize_rows 汇总的有效数量。
            mean_row["surface_distance_defined"],
            # 实际评估图像总数。
            len(rows),
        )
    )
    # 打印 CSV 绝对路径，方便脚本或人工定位。
    print("CSV=" + output_csv)
    # 打印所有输出文件所在目录。
    print("OUTPUT_DIR=" + output_dir)


# 只有直接执行本文件时进入 main；作为模块导入时只定义函数。
if __name__ == "__main__":
    # 启动完整评估流程。
    main()
