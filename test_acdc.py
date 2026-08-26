# ============================== 初学者阅读总览 ==============================
# 本文件加载 ACDC checkpoint，对完整三维心脏 MRI 逐切片推理并汇总病例级指标。
# 类别约定：0=背景，1=RV（右心室），2=MYO（心肌），3=LV（左心室）。
# 数据流：[1,D,H,W] 病例 -> 去 batch 维得到 [D,H,W] -> predict_volume 按深度分批 ->
# 最后一级 [B,4,H,W] logits -> argmax -> [D,H,W] 类别图 -> 每个前景类计算
# Dice/HD95/Jaccard/ASD -> 病例宏平均 -> 全测试集均值 -> CSV/可选 NPZ/NIfTI。
# 论文对应：模型结构见第 3.2 节与图 2；ACDC 结果见第 4.2.3 节；
# 训练/输入设置见第 4.1 节；数据集与指标定义见补充材料第 7.1、7.2 节。
# 路径检查、CSV 格式、NPZ/NIfTI 导出及 max_cases 属于工程实现。
# ========================================================================

# argparse 定义 checkpoint、数据路径、模型结构和输出选项。
import argparse
# csv 把逐病例和总体指标写成结构化表格。
import csv
# logging 同时记录参数与逐病例指标。
import logging
# os 拼接并创建 checkpoint、日志、CSV、预测目录。
import os
# sys.stdout 用于把日志同步显示在终端。
import sys

# NumPy 用于保存压缩预测、聚合跨病例均值。
import numpy as np
# torch 构建模型、选择设备并运行推理。
import torch
# DataLoader 每次提供一个完整病例。
from torch.utils.data import DataLoader
# tqdm 显示病例级测试进度。
from tqdm import tqdm

# ACDC 专用工具封装模型构建、checkpoint 兼容、体推理、指标和 NIfTI 保存。
from utils.acdc_utils import (
    # 三个前景类名称，顺序对应标签 1、2、3。
    ACDC_CLASS_NAMES,
    # 总类别数 4，包含背景。
    ACDC_NUM_CLASSES,
    # 指标名固定为 dice、hd95、jaccard、asd。
    METRIC_NAMES,
    # 根据测试参数重建 EMCADNet。
    build_model,
    # 读取 checkpoint，并处理 state_dict/module. 前缀兼容。
    load_checkpoint,
    # 对三个前景类的指标再做宏平均。
    mean_metrics,
    # 把 [D,H,W] 体数据分批送入二维网络并拼回预测体。
    predict_volume,
    # 保存原图、预测和标签三份 NIfTI。
    save_nifti_triplet,
    # 固定随机源及 cuDNN 行为。
    seed_everything,
    # 对每个前景类别计算四项三维指标。
    volume_metrics,
)
# ACDCVolumeDataset 的 test 分支读取完整 [D,H,W] NPZ 病例。
from utils.dataset_ACDC import ACDCVolumeDataset


# 定义并解析测试参数；返回 argparse.Namespace。
def parse_args():
    # --help 中显示脚本用途。
    parser = argparse.ArgumentParser(description="Test EMCAD on ACDC")
    # 必填模型权重路径；通常使用训练验证集选择出的 best.pth。
    parser.add_argument("--checkpoint", required=True)
    # ACDC 根目录，测试时要求存在 test/。
    parser.add_argument("--root_path", default="./data/ACDC")
    # 划分清单目录，测试时要求 test.txt。
    parser.add_argument("--list_dir", default="./data/ACDC/lists/lists_ACDC")
    # 预测/日志目录；None 时放在 checkpoint 同级 predictions/。
    parser.add_argument("--output_dir", default=None)
    # 指标 CSV 路径；None 时放在 checkpoint 同级 test_metrics.csv。
    parser.add_argument("--output_csv", default=None)

    # 编码器架构必须与 checkpoint 训练时一致。
    parser.add_argument("--encoder", default="pvt_v2_b2")
    # MSDC 多尺度卷积核必须与训练结构一致。
    parser.add_argument("--kernel_sizes", type=int, nargs="+", default=[1, 3, 5])
    # MSCB 通道扩张倍数。
    parser.add_argument("--expansion_factor", type=int, default=2)
    # LGAG 卷积核大小。
    parser.add_argument("--lgag_ks", type=int, default=3)
    # MSCB 激活函数。
    parser.add_argument("--activation_mscb", default="relu6")
    # 出现该旗标表示关闭并行深度卷积。
    parser.add_argument("--no_dw_parallel", action="store_true")
    # 出现该旗标表示用 concat 聚合多尺度特征；默认 add。
    parser.add_argument("--concatenation", action="store_true")
    # 预训练目录仍是 build_model 所需属性，但测试构造时 pretrain=False，不会加载它。
    parser.add_argument("--pretrained_dir", default="./pretrained_pth/pvt/")

    # 每张切片进入模型前缩放到的正方形尺寸，须与训练设置相符。
    parser.add_argument("--img_size", type=int, default=224)
    # 一个前向 batch 中包含的切片数，不是病例数。
    parser.add_argument("--inference_batch_size", type=int, default=8)
    # DataLoader 读取完整病例的 worker 数。
    parser.add_argument("--num_workers", type=int, default=1)
    # z 方向体素间距，参与 HD95/ASD 的物理距离换算和 NIfTI 元数据。
    parser.add_argument("--z_spacing", type=float, default=10.0)
    # 测试随机种子；推理虽无增强，仍用于复现运行环境。
    parser.add_argument("--seed", type=int, default=2222)
    # auto 自动选择 CUDA/CPU，也可显式传 cpu、cuda、cuda:1。
    parser.add_argument("--device", default="auto")
    # 正数时只测前 N 个病例用于 smoke test；0 表示全部。
    parser.add_argument("--max_cases", type=int, default=0)
    # 出现后保存每例原图/预测/标签 NIfTI。
    parser.add_argument("--save_nii", action="store_true")
    # 出现后把离散预测类别图保存为压缩 NPZ。
    parser.add_argument("--save_npz", action="store_true")
    # 实际解析命令行。
    return parser.parse_args()


# 把设备字符串解析为 torch.device。
def resolve_device(requested):
    # auto 优先使用 CUDA，不可用时回退 CPU。
    if requested == "auto":
        # 返回实际设备对象。
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 显式设备字符串直接交给 PyTorch 解析。
    return torch.device(requested)


# 将 rows 写入 CSV。输入 rows 的最后一行通常是 case_name='MEAN' 的总体汇总。
def write_csv(path, rows):
    # 第一列总是病例名。
    fieldnames = ["case_name"]
    # 依次为 RV、MYO、LV 构造四项指标列。
    for class_name in ACDC_CLASS_NAMES:
        # extend 接收生成器，产生 RV_dice、RV_hd95...等列名。
        fieldnames.extend(
            # 单个列名由类别名和指标名拼接。
            "{}_{}".format(class_name, metric_name)
            # 遍历固定四项指标。
            for metric_name in METRIC_NAMES
            # extend 调用结束。
        )
    # 最后追加 mean_dice、mean_hd95、mean_jaccard、mean_asd。
    fieldnames.extend("mean_" + name for name in METRIC_NAMES)
    # 写模式覆盖同名旧 CSV；newline='' 避免 Windows 空白行。
    with open(path, "w", newline="", encoding="utf-8") as stream:
        # DictWriter 按 fieldnames 固定列顺序。
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        # 写表头。
        writer.writeheader()
        # 一次写入所有病例行和总体行。
        writer.writerows(rows)


# 测试总控：检查输入、加载模型、逐病例推理、聚合指标并保存结果。
def main():
    # 读取命令行配置。
    args = parse_args()
    # 列出测试前必须存在的 checkpoint、测试目录和划分清单。
    required = [
        # 模型权重。
        args.checkpoint,
        # 完整三维测试病例目录。
        os.path.join(args.root_path, "test"),
        # 测试病例文件名清单。
        os.path.join(args.list_dir, "test.txt"),
        # 必需路径列表结束。
    ]
    # 收集不存在的路径。
    missing = [path for path in required if not os.path.exists(path)]
    # 任一缺失则提前停止。
    if missing:
        # 每行打印一个缺失路径，便于定位。
        raise FileNotFoundError("Missing ACDC paths:\n" + "\n".join(missing))

    # 测试固定为确定性模式，保证同环境重复评估稳定。
    seed_everything(args.seed, deterministic=True)
    # 解析实际设备。
    device = resolve_device(args.device)
    # 测试直接加载完整 checkpoint，因此构造模型时无需先加载 ImageNet 预训练权重。
    model = build_model(args, pretrain=False)
    # 严格把 checkpoint 参数装入重建的网络；结构参数不一致会报形状/键错误。
    load_checkpoint(model, args.checkpoint)
    # 移到设备并立即切换 eval；链式调用返回 model 自身。
    model.to(device).eval()

    # checkpoint_dir 是权重文件所在实验目录。
    checkpoint_dir = os.path.dirname(os.path.abspath(args.checkpoint))
    # 未指定 output_dir 时在实验目录下创建 predictions。
    output_dir = args.output_dir or os.path.join(checkpoint_dir, "predictions")
    # 未指定 output_csv 时在实验目录根部写 test_metrics.csv。
    output_csv = args.output_csv or os.path.join(checkpoint_dir, "test_metrics.csv")
    # 创建预测/日志目录。
    os.makedirs(output_dir, exist_ok=True)
    # 创建 CSV 父目录；abspath 保证即使只给文件名也能得到有效目录。
    os.makedirs(os.path.dirname(os.path.abspath(output_csv)), exist_ok=True)

    # 配置测试日志，文件位于 output_dir/test.log。
    logging.basicConfig(
        # 日志文件路径。
        filename=os.path.join(output_dir, "test.log"),
        # INFO 级别。
        level=logging.INFO,
        # 毫秒级时间格式。
        format="[%(asctime)s.%(msecs)03d] %(message)s",
        # 时间显示为时分秒。
        datefmt="%H:%M:%S",
        # logging 配置结束。
    )
    # 同步输出到终端。
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    # 保存全部测试参数。
    logging.info("args=%s", args)

    # 测试数据集逐项返回 image/label=[D,H,W] 和 case_name。
    dataset = ACDCVolumeDataset(args.root_path, args.list_dir, split="test")
    # DataLoader 每次一个病例，禁止打乱以保持 test.txt/CSV 顺序。
    loader = DataLoader(
        # 数据集对象。
        dataset,
        # 病例 batch 固定 1，因为不同病例深度 D 可能不同。
        batch_size=1,
        # 保持确定顺序。
        shuffle=False,
        # 并行读取病例数。
        num_workers=args.num_workers,
        # DataLoader 构造结束。
    )

    # rows 收集逐病例指标字典，最后再附加总体 MEAN 行。
    rows = []
    # 为每个前景类别、每个指标维护跨病例数值列表，供最终逐类均值使用。
    class_accumulator = {
        # 内层形如 {'dice':[],'hd95':[],'jaccard':[],'asd':[]}。
        class_index: {name: [] for name in METRIC_NAMES}
        # 类别索引只遍历 1..3，明确排除背景 0。
        for class_index in range(1, ACDC_NUM_CLASSES)
        # 字典推导结束。
    }
    # medpy 距离指标使用 (z,y,x) 间距；平面内固定 1，z 取命令行参数。
    voxelspacing = (float(args.z_spacing), 1.0, 1.0)

    # 逐病例测试；case_index 从 0 开始，tqdm 显示进度。
    for case_index, sampled in enumerate(tqdm(loader, desc="ACDC test")):
        # 正数 max_cases 用于只处理前 N 例。
        if args.max_cases and case_index >= args.max_cases:
            # 达到上限后停止。
            break
        # 去掉 batch_size=1 的外层并转 NumPy，得到 [D,H,W] float 图像。
        image = sampled["image"][0].numpy()
        # 标签同样得到 [D,H,W]，值域 0..3。
        label = sampled["label"][0].numpy()
        # 取单个病例名字符串。
        case_name = sampled["case_name"][0]
        # 按深度把切片分批送入二维 EMCAD。
        prediction = predict_volume(
            # 已加载权重并 eval 的模型。
            model,
            # 三维图像。
            image,
            # 运行设备。
            device=device,
            # 网络输入切片尺寸。
            img_size=args.img_size,
            # 每次推理的切片数。
            batch_size=args.inference_batch_size,
            # 输出 [D,H,W] int16 类别编号。
        )
        # 对 RV/MYO/LV 分别计算 Dice、HD95、Jaccard、ASD。
        per_class = volume_metrics(
            # 预测类别体。
            prediction,
            # 真值类别体。
            label,
            # 总类别数 4；函数内部 range(1,4) 跳过背景。
            num_classes=ACDC_NUM_CLASSES,
            # 距离指标采用真实/配置体素间距。
            voxelspacing=voxelspacing,
            # 指标调用结束。
        )
        # 对三个前景类逐指标做宏平均，返回四项指标字典。
        means = mean_metrics(per_class)
        # 当前 CSV 行先写病例名。
        row = {"case_name": case_name}
        # enumerate(...,start=1) 保证 RV/MYO/LV 分别映射标签 1/2/3。
        for class_index, class_name in enumerate(ACDC_CLASS_NAMES, start=1):
            # 遍历四项指标。
            for metric_name in METRIC_NAMES:
                # 取当前类别当前指标标量。
                value = per_class[class_index][metric_name]
                # 写入当前病例行，例如 RV_dice。
                row["{}_{}".format(class_name, metric_name)] = value
                # 同时追加到跨病例逐类累计列表。
                class_accumulator[class_index][metric_name].append(value)
        # 把病例宏平均四项指标写到 mean_* 列。
        for metric_name in METRIC_NAMES:
            # 例如 mean_dice。
            row["mean_" + metric_name] = means[metric_name]
        # 保存当前病例完整行。
        rows.append(row)
        # 记录当前病例四项宏平均。
        logging.info(
            # 日志模板。
            "case=%s dice=%.6f hd95=%.6f jaccard=%.6f asd=%.6f",
            # 病例名。
            case_name,
            # 三类平均 Dice。
            means["dice"],
            # 三类平均 HD95。
            means["hd95"],
            # 三类平均 Jaccard。
            means["jaccard"],
            # 三类平均 ASD。
            means["asd"],
            # 日志调用结束。
        )

        # 可选保存离散预测类别体为压缩 NPZ。
        if args.save_npz:
            # np.savez_compressed 创建 <case>_prediction.npz，键名 prediction。
            np.savez_compressed(
                # 输出路径。
                os.path.join(output_dir, case_name + "_prediction.npz"),
                # 保存 [D,H,W] 预测数组。
                prediction=prediction,
                # 保存调用结束。
            )
        # 可选保存便于医学影像软件查看的 NIfTI 三件套。
        if args.save_nii:
            # 工具函数写 image、prediction、label 三个 .nii.gz。
            save_nifti_triplet(
                # 原图。
                image,
                # 预测。
                prediction,
                # 真值。
                label,
                # 输出目录。
                output_dir,
                # 病例名前缀。
                case_name,
                # z 方向间距。
                args.z_spacing,
                # 保存调用结束。
            )

    # 数据为空或 max_cases 导致零例时，不允许生成 NaN 汇总。
    if not rows:
        # 显式报错。
        raise RuntimeError("ACDC test produced no cases")

    # 创建总体汇总行，以 MEAN 区别于真实病例。
    summary = {"case_name": "MEAN"}
    # 分别汇总 RV/MYO/LV。
    for class_index, class_name in enumerate(ACDC_CLASS_NAMES, start=1):
        # 遍历四项指标。
        for metric_name in METRIC_NAMES:
            # 对当前类别在所有已测试病例上的数值取均值。
            summary["{}_{}".format(class_name, metric_name)] = float(
                # np.mean 输入前面累计的列表。
                np.mean(class_accumulator[class_index][metric_name])
                # float 转成普通 Python 标量，便于 CSV 序列化。
            )
    # 总体 mean_* 列按“各病例已先跨类别平均”的结果再跨病例平均。
    for metric_name in METRIC_NAMES:
        # 写入汇总字典。
        summary["mean_" + metric_name] = float(
            # 从每个病例 row 取对应 mean 指标。
            np.mean([row["mean_" + metric_name] for row in rows])
            # 转换结束。
        )
    # 把汇总行追加在 CSV 最后一行。
    rows.append(summary)
    # 写出 CSV。
    write_csv(output_csv, rows)

    # 打印终端表头。
    print("class       Dice       HD95       Jaccard       ASD")
    # 逐类打印 RV/MYO/LV 的跨病例均值。
    for class_name in ACDC_CLASS_NAMES:
        # 格式化一行，类别左对齐、数值固定 6 位小数。
        print(
            # format 模板。
            "{:<8} {:>10.6f} {:>10.6f} {:>13.6f} {:>10.6f}".format(
                # 类别名。
                class_name,
                # Dice。
                summary[class_name + "_dice"],
                # HD95。
                summary[class_name + "_hd95"],
                # Jaccard。
                summary[class_name + "_jaccard"],
                # ASD。
                summary[class_name + "_asd"],
                # format 结束。
            )
            # print 结束。
        )
    # 打印跨类别、跨病例总体 MEAN 行。
    print(
        # 同一列宽模板。
        "{:<8} {:>10.6f} {:>10.6f} {:>13.6f} {:>10.6f}".format(
            # 行名。
            "MEAN",
            # 总体 Dice。
            summary["mean_dice"],
            # 总体 HD95。
            summary["mean_hd95"],
            # 总体 Jaccard。
            summary["mean_jaccard"],
            # 总体 ASD。
            summary["mean_asd"],
            # format 结束。
        )
        # print 结束。
    )
    # 打印 CSV 绝对路径，便于定位结果。
    print("CSV=" + os.path.abspath(output_csv))
    # 打印预测/日志目录绝对路径。
    print("OUTPUT_DIR=" + os.path.abspath(output_dir))


# 仅直接运行脚本时执行测试。
if __name__ == "__main__":
    # 进入测试总控。
    main()
