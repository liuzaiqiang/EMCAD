# argparse 把数据路径做成命令行参数，便于在不同机器上复用检查脚本。
import argparse
# os 用于判断文件是否存在以及构造跨平台路径。
import os
# re 用于解析验证集切片文件名中的病例、心动周期和切片序号。
import re
# defaultdict(int) 自动把首次出现的验证病例分组计数初始化为 0。
from collections import defaultdict

# NumPy 负责读取 NPZ、检查数组维数/形状并枚举标签类别值。
import numpy as np


# ACDC 验证切片命名规则：case 编号 + slice + ED/ES 心动相位 + 层号，可带 .npz 后缀。
VALID_PATTERN = re.compile(
    # 三个捕获组依次是病例编号、舒张末期/收缩末期标记、切片编号。
    r"^(case_?\d+)_slice(ED|ES)_(\d+)(?:\.npz)?$",
    # IGNORECASE 允许文件名中的 case、ED、ES 使用不同大小写。
    re.IGNORECASE,
)


# 读取 train.txt、valid.txt 或 test.txt，返回去除空白后的样本名列表。
def read_list(list_dir, split):
    # 按“划分名.txt”约定构造列表文件路径。
    path = os.path.join(list_dir, split + ".txt")
    # 列表缺失意味着数据准备不完整，因此立即报出确切路径。
    if not os.path.isfile(path):
        # 使用标准 FileNotFoundError 便于调用方区分路径问题和数据内容问题。
        raise FileNotFoundError(path)
    # 显式使用 UTF-8 读取文本列表并在代码块结束时自动关闭文件。
    with open(path, "r", encoding="utf-8") as stream:
        # strip() 去掉换行及首尾空格，同时过滤空行。
        return [line.strip() for line in stream if line.strip()]


# 将列表中的样本名解析为真实 NPZ 路径，兼容列表项带或不带 .npz 后缀。
def resolve(directory, name):
    # 第一候选路径严格使用列表中的原始名称。
    paths = [os.path.join(directory, name)]
    # 若列表项没有 .npz，则再加入一个自动补后缀的候选路径。
    if not name.lower().endswith(".npz"):
        # 追加候选而不覆盖第一项，保留对无后缀真实文件的兼容性。
        paths.append(os.path.join(directory, name + ".npz"))
    # 依次检查所有候选路径。
    for path in paths:
        # 找到第一个存在的普通文件就返回。
        if os.path.isfile(path):
            # 返回解析后的实际路径，后续由 np.load 打开。
            return path
    # 所有候选均不存在时，把候选路径合并到异常信息中，便于定位命名问题。
    raise FileNotFoundError(" or ".join(paths))


# 主检查流程：验证列表、文件结构、数组维数、标签范围以及验证集病例分组。
def main():
    # 创建命令行解析器；不改数据，只执行只读一致性检查。
    parser = argparse.ArgumentParser()
    # root_path 下应包含 train、valid、test 三个数据子目录。
    parser.add_argument("--root_path", default="./data/ACDC")
    # list_dir 下应包含 train.txt、valid.txt、test.txt。
    parser.add_argument("--list_dir", default="./data/ACDC/lists/lists_ACDC")
    # 解析用户传入参数，未提供时使用上面的项目默认值。
    args = parser.parse_args()

    # 汇总所有划分中出现过的类别编号，最终应只包含 ACDC 的 0、1、2、3。
    labels_seen = set()
    # 按“病例 + ED/ES 相位”累计验证切片数，用分组数量推导验证体数量。
    valid_groups = defaultdict(int)
    # 保存各划分列表项数量，供最终报告输出。
    counts = {}

    # 依次核验训练切片、验证切片和测试体数据。
    for split in ("train", "valid", "test"):
        # 读取当前划分的样本名称列表。
        names = read_list(args.list_dir, split)
        # 记录当前划分总样本数；训练/验证是切片数，测试是体数据数。
        counts[split] = len(names)
        # 对列表中的每个样本执行文件内容检查。
        for name in names:
            # 当前划分目录与样本名组合，并兼容可选 .npz 后缀。
            path = resolve(os.path.join(args.root_path, split), name)
            # allow_pickle=False 禁止反序列化任意 Python 对象，只读取普通 NumPy 数组。
            with np.load(path, allow_pickle=False) as data:
                # 项目 ACDC 加载器要求每个 NPZ 同时包含 img 和 label 两个键。
                if "img" not in data or "label" not in data:
                    # 缺少任一键时报告具体文件，防止训练阶段才出现难定位错误。
                    raise KeyError(path + " lacks img/label")
                # 取出图像数组。
                image = data["img"]
                # 取出与图像对应的分割标签数组。
                label = data["label"]
            # 图像和标签必须逐像素/逐体素对齐，因此形状必须完全一致。
            if image.shape != label.shape:
                # 形状不匹配时停止检查并报告文件。
                raise ValueError(path + " image/label shape mismatch")
            # 测试样本按完整三维体保存；训练和验证样本按二维切片保存。
            expected_ndim = 3 if split == "test" else 2
            # 检查当前数组维数是否符合该划分的数据加载约定。
            if image.ndim != expected_ndim:
                # 异常信息同时显示期望维数和实际 shape。
                raise ValueError(
                    # format 将路径、期望维数及实际形状写入错误文本。
                    "{} expected {}D but got {}".format(path, expected_ndim, image.shape)
                )
            # np.unique 找到当前标签中全部类别值，并转成 Python int 合并到全局集合。
            labels_seen.update(int(value) for value in np.unique(label))
            # 只有验证切片需要从文件名恢复病例/相位分组。
            if split == "valid":
                # fullmatch 要求整个基本文件名都符合 VALID_PATTERN，而不是只匹配局部。
                match = VALID_PATTERN.fullmatch(os.path.basename(name))
                # 不符合约定的名称无法重组为验证体，因此立即报错。
                if match is None:
                    # 报告无法分组的原始列表项。
                    raise ValueError("Cannot group valid filename: " + name)
                # 用病例编号和大写 ED/ES 组成同一心动时相体数据的分组键。
                key = "{}_{}".format(match.group(1), match.group(2).upper())
                # 累加该病例/相位下已发现的二维切片数量。
                valid_groups[key] += 1

    # ACDC 四类分割的合法标签集合为背景 0 加三个心脏结构 1、2、3。
    if not labels_seen.issubset({0, 1, 2, 3}):
        # 出现越界类别时列出全部实际标签值。
        raise ValueError("Unexpected ACDC labels: {}".format(sorted(labels_seen)))

    # 所有检查通过后输出稳定的成功标志，便于 Shell 脚本或人工确认。
    print("ACDC_DATA_OK")
    # 训练数据按二维样本统计。
    print("train_slices={}".format(counts["train"]))
    # 验证数据也按二维切片统计。
    print("valid_slices={}".format(counts["valid"]))
    # 不同“病例 + ED/ES”键的数量即验证三维体数量。
    print("valid_volumes={}".format(len(valid_groups)))
    # 测试列表每一项对应一个完整体数据文件。
    print("test_volumes={}".format(counts["test"]))
    # 排序输出实际观察到的类别编号。
    print("labels={}".format(sorted(labels_seen)))


# Python 入口保护：直接运行时执行 main，作为模块导入时只暴露函数和常量。
if __name__ == "__main__":
    # 启动只读数据完整性检查。
    main()
