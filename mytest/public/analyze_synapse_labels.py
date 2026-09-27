"""只按整数标签统计 Synapse 数据，不给标签猜测器官名称。"""

import argparse
import csv
from pathlib import Path

import h5py
import numpy as np


def count_labels(label_array):
    """统计数组中每个整数标签的体素数，返回 {标签整数: 体素数}。"""
    labels = np.asarray(label_array)
    if labels.size == 0:
        return {}
    if not np.isfinite(labels).all():
        raise ValueError("标签数组包含 NaN 或无穷大")
    labels = labels.astype(np.int64, copy=False).ravel()
    if (labels < 0).any():
        raise ValueError(f"标签包含负数，最小值为 {labels.min()}")
    values, counts = np.unique(labels, return_counts=True)
    return {int(value): int(count) for value, count in zip(values, counts)}


def make_row(source, counts):
    """把一个文件或整个数据集的标签计数转换成一行 CSV。"""
    total = sum(counts.values())
    foreground_total = sum(n for label, n in counts.items() if label != 0)
    row = {
        "source": source,
        "total_voxels": total,
        "foreground_voxels": foreground_total,
    }
    # 只使用 label_数字，不写入任何器官名称。
    for label in sorted(counts):
        count = counts[label]
        row[f"label_{label}_voxels"] = count
        row[f"label_{label}_pct_all"] = 100.0 * count / total if total else 0.0
        row[f"label_{label}_pct_foreground"] = (
            100.0 * count / foreground_total
            if label != 0 and foreground_total
            else 0.0
        )
    return row


def merge_counts(total_counts, current_counts):
    """把当前文件计数累加到总体计数。"""
    for label, count in current_counts.items():
        total_counts[label] = total_counts.get(label, 0) + count


def print_counts(source, counts):
    """在终端打印标签编号、体素数和百分比。"""
    total = sum(counts.values())
    foreground_total = sum(n for label, n in counts.items() if label != 0)
    print(f"\n===== {source} =====")
    print(f"total_voxels      = {total:,}")
    print(f"foreground_voxels = {foreground_total:,}")
    print("label_id | voxels        | pct_all    | pct_foreground")
    print("---------+---------------+------------+---------------")
    for label in sorted(counts):
        count = counts[label]
        pct_all = 100.0 * count / total if total else 0.0
        pct_foreground = (
            100.0 * count / foreground_total
            if label != 0 and foreground_total
            else 0.0
        )
        print(
            f"{label:8d} | {count:13,d} | "
            f"{pct_all:9.4f}% | {pct_foreground:12.4f}%"
        )


def write_csv(path, rows):
    """把统计结果写入 CSV；没有结果时不创建文件。"""
    if not rows:
        return
    # 统一收集所有列，避免不同文件出现标签集合不同时丢列。
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def analyze_one_npz(path):
    """读取并统计一个 NPZ 文件，同时显示其中的键和数组形状。"""
    with np.load(path, allow_pickle=False) as data:
        print(f"\n文件：{path}")
        print(f"keys：{data.files}")
        for key in data.files:
            print(f"{key}: shape={data[key].shape}, dtype={data[key].dtype}")
        if "label" not in data.files:
            raise KeyError(f"{path} 不包含 label 键")
        labels = data["label"]
        counts = count_labels(labels)
    print_counts(path.name, counts)
    return counts


def analyze_train(directory, output_dir, print_files):
    """统计 train_npz 目录下所有 NPZ 文件。"""
    files = sorted(directory.glob("*.npz"))
    if not files:
        print(f"没有找到 NPZ 文件：{directory}")
        return
    total_counts = {}
    rows = []
    for index, path in enumerate(files, start=1):
        with np.load(path, allow_pickle=False) as data:
            if "label" not in data.files:
                raise KeyError(f"{path} 不包含 label 键")
            labels = data["label"]
            counts = count_labels(labels)
            row = make_row(path.name, counts)
            row["label_shape"] = str(labels.shape)
            row["image_shape"] = str(data["image"].shape) if "image" in data.files else ""
        merge_counts(total_counts, counts)
        rows.append(row)
        if print_files:
            print(f"[{index}/{len(files)}] {path.name} label_shape={labels.shape}")
            print_counts(path.name, counts)
    write_csv(output_dir / "synapse_label_stats_train_npz_per_file.csv", rows)
    write_csv(
        output_dir / "synapse_label_stats_train_npz_aggregate.csv",
        [make_row("ALL_TRAIN_NPZ", total_counts)],
    )
    print_counts("ALL_TRAIN_NPZ", total_counts)


def analyze_test(directory, output_dir, print_files):
    """统计 test_vol_h5 目录下所有 H5 病例。"""
    files = sorted(directory.glob("*.h5"))
    if not files:
        print(f"没有找到 H5 文件：{directory}")
        return
    total_counts = {}
    rows = []
    for index, path in enumerate(files, start=1):
        with h5py.File(path, "r") as file:
            if "label" not in file:
                raise KeyError(f"{path} 不包含 label 数据集")
            labels = file["label"][:]
            counts = count_labels(labels)
            row = make_row(path.name, counts)
            row["label_shape"] = str(labels.shape)
            row["image_shape"] = str(file["image"].shape) if "image" in file else ""
        merge_counts(total_counts, counts)
        rows.append(row)
        if print_files:
            print(f"[{index}/{len(files)}] {path.name} label_shape={labels.shape}")
            print_counts(path.name, counts)
    write_csv(output_dir / "synapse_label_stats_test_h5_per_file.csv", rows)
    write_csv(
        output_dir / "synapse_label_stats_test_h5_aggregate.csv",
        [make_row("ALL_TEST_H5", total_counts)],
    )
    print_counts("ALL_TEST_H5", total_counts)


def main():
    """解析命令行参数，并按用户选择执行一种或多种统计。"""
    parser = argparse.ArgumentParser(description="只统计 Synapse 整数标签，不猜测器官名称")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(r"D:\files\Medical_Image_Segmentation_Projects\data\Synapse"),
        help="Synapse 根目录",
    )
    parser.add_argument(
        "--npz",
        type=Path,
        help="只分析一个指定的 .npz 文件",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="CSV 输出目录，默认是 --root",
    )
    parser.add_argument(
        "--no-print-files",
        action="store_true",
        help="分析训练/测试集合时不逐文件打印",
    )
    args = parser.parse_args()
    root = args.root.resolve()
    output_dir = (args.output_dir or root).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # 指定 --npz 时，只读取一个文件，不扫描整个数据集。
    if args.npz:
        analyze_one_npz(args.npz.resolve())
        return

    # 未指定 --npz 时，分别扫描训练 NPZ 和测试 H5。
    analyze_train(root / "train_npz", output_dir, not args.no_print_files)
    analyze_test(root / "test_vol_h5", output_dir, not args.no_print_files)
    print(f"\nCSV 输出目录：{output_dir}")


if __name__ == "__main__":
    main()
