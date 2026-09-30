# -*- coding: utf-8 -*-  # 声明本文件使用 UTF-8，保证下面的中文注释在 Windows 和 Linux 上都能正常读取。
"""将 EMCAD/Synapse 测试集的三维 .npy.h5 病例拆成训练格式的二维 .npz 切片。""" 

import argparse  # 解析输入目录、输出目录和病例数量等命令行参数。
from pathlib import Path  # 使用跨平台路径对象处理 Windows 和 Linux 路径。

import h5py  # 按 EMCAD 当前约定读取 .npy.h5 文件中的 image 和 label 数据集。
import numpy as np  # 用于切片、形状检查以及以 NPZ 格式保存二维数组。


H5_SUFFIX = ".npy.h5"  # EMCAD 的三维测试病例文件使用的完整后缀，而不是 Path.stem 的单一后缀。
IMAGE_KEY = "image"  # EMCAD 的 H5 和训练 NPZ 都使用 image 作为图像数组键名。
LABEL_KEY = "label"  # EMCAD 的 H5 和训练 NPZ 都使用 label 作为分割标签数组键名。


def parse_args():  # 定义命令行参数解析函数，使脚本可以直接复用而不依赖硬编码路径。
    parser = argparse.ArgumentParser(  # 创建一个带有清晰帮助信息的参数解析器。
        description="按 EMCAD 的 Synapse 训练切片协议，把 3D .npy.h5 拆成 2D .npz。"  # 向用户解释脚本用途。
    )  # 结束参数解析器的创建。
    parser.add_argument(  # 注册三维 H5 病例所在目录参数。
        "--input-dir",  # 参数名；目录中应直接放置 12 个 .npy.h5 病例文件。
        type=Path,  # 自动把命令行字符串转换成 Path 对象。
        default=Path("data/Synapse/test_vol_h5"),  # 默认采用仓库中 EMCAD 常见的测试体目录。
        help="输入目录，默认：data/Synapse/test_vol_h5。",  # 在 --help 中说明默认输入位置。
    )  # 完成输入目录参数定义。
    parser.add_argument(  # 注册二维 NPZ 输出目录参数。
        "--output-dir",  # 参数名；脚本会在这里写入每一张二维切片。
        type=Path,  # 自动把命令行字符串转换成 Path 对象。
        default=Path("data/Synapse/test_npz_from_h5"),  # 使用独立输出目录，避免覆盖原始 train_npz。
        help="二维 NPZ 输出目录，默认：data/Synapse/test_npz_from_h5。",  # 说明默认输出位置。
    )  # 完成输出目录参数定义。
    parser.add_argument(  # 注册切片清单文件名参数。
        "--list-name",  # 参数名；清单内容与 EMCAD 的 train.txt 格式一致。
        type=str,  # 清单文件名以字符串形式接收。
        default="test_slices.txt",  # 默认在输出目录中生成 test_slices.txt。
        help="输出切片清单文件名，默认：test_slices.txt。",  # 说明清单的默认文件名。
    )  # 完成切片清单参数定义。
    parser.add_argument(  # 注册病例数量校验参数。
        "--expected-cases",  # 参数名；Synapse 官方测试集默认应有 12 个病例。
        type=int,  # 允许用户传入整数病例数。
        default=12,  # 默认严格检查 12 个 H5 文件，防止误指向错误目录。
        help="期望的 H5 病例数量，默认：12；传 0 可关闭数量检查。",  # 说明如何关闭数量检查。
    )  # 完成病例数量参数定义。
    return parser.parse_args()  # 解析当前命令行并返回参数命名空间。


def case_name_from_h5_path(h5_path: Path) -> str:  # 从 H5 文件名得到训练切片命名所需的病例基名。
    file_name = h5_path.name  # 只取文件名，避免把目录分隔符写入输出样本名。
    if not file_name.endswith(H5_SUFFIX):  # 检查文件是否确实使用 EMCAD 的 .npy.h5 后缀。
        raise ValueError(f"文件不是 EMCAD .npy.h5 格式：{h5_path}")  # 对不符合协议的输入立即报错。
    case_name = file_name[: -len(H5_SUFFIX)]  # 去掉完整的 .npy.h5 后缀，得到例如 case0001。
    if case_name.startswith("img"):  # 兼容少数仍使用原始 imgXXXX 命名的 H5 文件。
        case_name = "case" + case_name[3:]  # 按 EMCAD 预处理脚本的规则把 imgXXXX 改成 caseXXXX。
    if not case_name:  # 防止出现只有后缀而没有病例名的非法文件。
        raise ValueError(f"无法从文件名解析病例名：{h5_path}")  # 给出明确错误而不是生成空前缀文件。
    return case_name  # 返回不带扩展名的病例名。


def read_volume(h5_path: Path):  # 读取一个病例并验证 EMCAD 切片所要求的三维数组协议。
    with h5py.File(h5_path, mode="r") as h5_file:  # 使用上下文管理器，确保 H5 文件句柄始终关闭。
        missing_keys = [  # 收集 H5 中缺失的必需数据集名称。
            key for key in (IMAGE_KEY, LABEL_KEY) if key not in h5_file  # 逐个检查 image 和 label 是否存在。
        ]  # 结束缺失键列表的构造。
        if missing_keys:  # 如果任意必需键不存在，则当前文件不能按 EMCAD 协议处理。
            missing_text = ", ".join(missing_keys)  # 把缺失键转成便于阅读的字符串。
            raise KeyError(f"{h5_path} 缺少 H5 数据集：{missing_text}")  # 报告缺失键和对应文件。
        image = h5_file[IMAGE_KEY][:]  # 一次性读出已预处理的三维 CT 数组，保持 H5 中的原始数值和类型。
        label = h5_file[LABEL_KEY][:]  # 一次性读出与 CT 对齐的三维标签数组，保持原始标签值和类型。
    if image.ndim != 3 or label.ndim != 3:  # EMCAD 当前 Synapse 流程要求 H5 数组为 [D,H,W] 三维数组。
        raise ValueError(  # 构造带有实际维度信息的错误，避免悄悄沿错误轴切片。
            f"{h5_path} 的 image/label 必须都是 [D,H,W] 三维数组，实际为 "  # 说明协议要求。
            f"image.ndim={image.ndim}, label.ndim={label.ndim}"  # 报告实际维度。
        )  # 结束维度错误信息。
    if image.shape != label.shape:  # 图像和标签必须逐体素对齐，才能使用相同的切片索引。
        raise ValueError(  # 构造带有两者形状的错误信息。
            f"{h5_path} 的 image 与 label 形状不一致："  # 指出发生了空间对齐问题。
            f"image.shape={image.shape}, label.shape={label.shape}"  # 给出实际形状便于定位数据问题。
        )  # 结束形状错误信息。
    return image, label  # 返回已经验证过的 [D,H,W] 图像和标签数组。


def split_one_volume(h5_path: Path, output_dir: Path, list_file):  # 按 EMCAD 的逐深度切片规则处理一个病例。
    case_name = case_name_from_h5_path(h5_path)  # 解析病例名，确保输出名与 EMCAD 的 caseXXXX 约定一致。
    image, label = read_volume(h5_path)  # 读取并验证整个病例的 image/label 三维数组。
    depth = image.shape[0]  # EMCAD 沿已经整理好的第 0 维 D 逐张取出二维切片。
    for slice_index in range(depth):  # 从第 0 张切片到最后一张，保持体数据原始深度顺序。
        image_slice = image[slice_index, :, :]  # 完全复现 EMCAD：直接取 image[s_idx, :, :]，不重新采样。
        label_slice = label[slice_index, :, :]  # 完全复现 EMCAD：取同一深度的 label[s_idx, :, :] 保持对齐。
        slice_name = f"{case_name}_slice{slice_index:03d}"  # 使用 EMCAD 的三位补零命名，例如 case0001_slice007。
        npz_path = output_dir / f"{slice_name}.npz"  # 为当前切片构造训练 NPZ 的完整输出路径。
        np.savez(npz_path, image=image_slice, label=label_slice)  # 使用非压缩 np.savez，键名和训练 NPZ 完全一致。
        list_file.write(slice_name + "\n")  # 写入不带 .npz 后缀的样本名，匹配 Synapse_dataset 的列表协议。
    return depth, case_name, image.shape  # 返回统计信息，供主函数打印并用于人工核对。


def main():  # 定义脚本主流程，便于命令行调用和后续导入复用。
    args = parse_args()  # 读取用户提供的输入、输出和校验参数。
    input_dir = args.input_dir.expanduser().resolve()  # 将输入目录转换为绝对路径，避免工作目录造成歧义。
    output_dir = args.output_dir.expanduser().resolve()  # 将输出目录转换为绝对路径，日志中可直接复制使用。
    if not input_dir.is_dir():  # 在扫描文件前确认输入目录确实存在。
        raise FileNotFoundError(f"输入目录不存在：{input_dir}")  # 用明确路径报告输入错误。
    h5_paths = sorted(input_dir.glob(f"*{H5_SUFFIX}"))  # 按文件名排序，保证病例处理顺序确定且可复现。
    if not h5_paths:  # 没有找到任何 H5 时继续执行没有意义。
        raise FileNotFoundError(f"输入目录中没有 {H5_SUFFIX} 文件：{input_dir}")  # 报告实际扫描目录。
    if args.expected_cases > 0 and len(h5_paths) != args.expected_cases:  # 默认验证 Synapse 测试集的 12 病例数量。
        raise RuntimeError(  # 数量不符时停止，避免把错误目录误当成测试集处理。
            f"期望找到 {args.expected_cases} 个 H5 病例，但实际找到 {len(h5_paths)} 个：{input_dir}"  # 提供可操作的数量诊断。
        )  # 结束病例数量错误信息。
    output_dir.mkdir(parents=True, exist_ok=True)  # 创建输出目录及其父目录，但不删除已有文件。
    list_path = output_dir / args.list_name  # 将切片清单放在同一个输出目录，便于整体迁移和复现实验。
    total_slices = 0  # 初始化所有病例的二维切片计数器。
    processed_cases = []  # 保存每个病例的名称、形状和切片数用于最终汇总。
    with list_path.open(mode="w", encoding="utf-8", newline="\n") as list_file:  # 用 UTF-8 和 Unix 换行写出稳定的清单格式。
        for h5_path in h5_paths:  # 按排序后的顺序逐个处理 12 个三维病例。
            depth, case_name, shape = split_one_volume(h5_path, output_dir, list_file)  # 生成当前病例的全部二维 NPZ。
            total_slices += depth  # 把当前病例深度累加到总切片数量。
            processed_cases.append((case_name, shape, depth))  # 保存当前病例统计，最终逐病例打印。
            print(f"[OK] {h5_path.name} -> {case_name}: shape={shape}, slices={depth}")  # 立即报告当前病例，便于定位中断位置。
    print(f"[DONE] cases={len(processed_cases)}, slices={total_slices}")  # 打印整个转换任务的总计。
    print(f"[DONE] npz_dir={output_dir}")  # 打印二维 NPZ 的绝对输出目录。
    print(f"[DONE] list_file={list_path}")  # 打印切片清单的绝对路径。


if __name__ == "__main__":  # 仅在直接执行本文件时启动转换，作为模块导入时不产生副作用。
    main()  # 调用主流程完成 H5 到二维 NPZ 的转换。
