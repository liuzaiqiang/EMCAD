# os 用于读取划分列表、检查样本文件和拼接数据路径。
import os
# random 用于在训练增强中按概率选择旋转/翻转分支。
import random
# re 解析验证切片名中的病例、ED/ES 相位和层号。
import re
# defaultdict(list) 用于把同一病例同一相位的验证切片归组为三维体。
from collections import defaultdict

# NumPy 负责 NPZ 读取、数组类型、旋转翻转和切片堆叠。
import numpy as np
# PyTorch 把 NumPy 图像与标签转换成 DataLoader 可批处理的张量。
import torch
# ndimage.rotate 实现小角度同步旋转增强。
from scipy import ndimage
# zoom 使用连续/离散两种插值阶数统一样本空间尺寸。
from scipy.ndimage import zoom
# Dataset 是 ACDC 二维训练集和三维评测集的 PyTorch 基类。
from torch.utils.data import Dataset


# 同步执行 90 度倍数旋转和随机轴翻转，保持图像与标签配准。
def random_rot_flip(image, label):
    # k∈{0,1,2,3} 分别对应 0/90/180/270 度旋转。
    k = np.random.randint(0, 4)
    # 图像在二维平面旋转，不发生插值。
    image = np.rot90(image, k)
    # 标签使用相同旋转参数。
    label = np.rot90(label, k)
    # 随机选高轴或宽轴翻转。
    axis = np.random.randint(0, 2)
    # copy() 消除 np.flip 产生的负步长，便于 torch.from_numpy 接收。
    return np.flip(image, axis=axis).copy(), np.flip(label, axis=axis).copy()


# 同步执行 [-20,20) 度的小角度旋转。
def random_rotate(image, label):
    # 采样整数旋转角度。
    angle = np.random.randint(-20, 20)
    # 连续 CT 图像使用三次插值，reshape=False 保持输出尺寸。
    image = ndimage.rotate(image, angle, order=3, reshape=False)
    # 离散标签使用最近邻插值，防止生成不存在的类别编号。
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    # 返回同步旋转后的样本对。
    return image, label


# ACDC 二维训练切片的随机增强、缩放和张量化变换。
class RandomGenerator:
    # output_size 通常为 [args.img_size, args.img_size]。
    def __init__(self, output_size):
        # 保存网络期望的高和宽。
        self.output_size = output_size

    # sample 是包含 image、label NumPy 数组的字典。
    def __call__(self, sample):
        # 同时取出 CT 切片和对应分割标签。
        image, label = sample["image"], sample["label"]
        # 只采样一次 choice，使三个分支概率明确为 0.5、0.25、0.25。
        choice = random.random()
        # choice>0.5：执行旋转90度倍数加翻转，概率 50%。
        if choice > 0.5:
            # 图像和标签同步增强。
            image, label = random_rot_flip(image, label)
        # 0.25<choice<=0.5：执行小角度旋转，概率 25%。
        elif choice > 0.25:
            # 图像三次插值、标签最近邻插值。
            image, label = random_rotate(image, label)

        # 读取当前二维切片高、宽。
        x, y = image.shape
        # 仅在尺寸不等于目标尺寸时重采样。
        if [x, y] != list(self.output_size):
            # 缩放连续 CT 图像。
            image = zoom(
                # 输入二维数组。
                image,
                # 两个轴各自的缩放比例。
                (self.output_size[0] / x, self.output_size[1] / y),
                # order=3 为三次样条插值。
                order=3,
            )
            # 缩放离散标签。
            label = zoom(
                # 输入类别编号数组。
                label,
                # 使用与图像完全相同的几何缩放比例。
                (self.output_size[0] / x, self.output_size[1] / y),
                # order=0 最近邻保证类别值不变。
                order=0,
            )

        # 返回 PyTorch 字典；DataLoader 将在外层增加 batch 维。
        return {
            # 图像转 float32，并增加单通道维成 [1,H,W]。
            "image": torch.from_numpy(image.astype(np.float32)).unsqueeze(0),
            # 标签直接转 int64 [H,W]，供 CrossEntropyLoss 使用。
            "label": torch.from_numpy(label.astype(np.int64)),
        }


# 读取指定 ACDC 划分列表并拒绝缺失或空列表。
def _read_list(list_dir, split):
    # 列表文件约定为 <list_dir>/<split>.txt。
    list_path = os.path.join(list_dir, split + ".txt")
    # 文件不存在时立即报出准确路径。
    if not os.path.isfile(list_path):
        # 使用 FileNotFoundError 区分路径问题。
        raise FileNotFoundError("Missing ACDC list: " + list_path)
    # 显式用 UTF-8 读取，退出 with 后自动关闭文件。
    with open(list_path, "r", encoding="utf-8") as stream:
        # 去除首尾空白并过滤空行。
        names = [line.strip() for line in stream if line.strip()]
    # 空列表会导致训练/评测静默运行零次，故提前拒绝。
    if not names:
        # 报告具体空文件。
        raise RuntimeError("Empty ACDC list: " + list_path)
    # 返回稳定保留文件顺序的样本名列表。
    return names


# 把列表项解析为真实 NPZ 文件，兼容名称带或不带 .npz。
def _resolve_npz(directory, name):
    # 第一候选严格使用原始列表项。
    candidates = [os.path.join(directory, name)]
    # 原始项没有扩展名时添加 .npz 候选。
    if not name.lower().endswith(".npz"):
        # 保留第一候选，不覆盖可能存在的无后缀文件。
        candidates.append(os.path.join(directory, name + ".npz"))
    # 按候选顺序寻找第一个真实文件。
    for path in candidates:
        # 只接受普通文件，不接受同名目录。
        if os.path.isfile(path):
            # 返回实际路径。
            return path
    # 均不存在时列出全部尝试路径。
    raise FileNotFoundError("Missing ACDC sample: " + " or ".join(candidates))


# 安全读取一个 ACDC NPZ，并检查键、类型和图像标签形状。
def _load_npz(path):
    # 禁用 pickle，避免读取 NPZ 时反序列化任意 Python 对象。
    with np.load(path, allow_pickle=False) as data:
        # 本项目 ACDC 约定键名固定为 img 和 label。
        if "img" not in data or "label" not in data:
            # 缺少任一键都无法形成监督样本。
            raise KeyError(path + " must contain 'img' and 'label'")
        # CT 统一为 float32，减少显存并匹配模型权重类型。
        image = data["img"].astype(np.float32)
        # 标签统一为 int64，匹配类别索引损失。
        label = data["label"].astype(np.int64)
    # 图像和标签必须逐像素/体素对齐。
    if image.shape != label.shape:
        # 形状不同则报告双方实际 shape。
        raise ValueError(
            # 格式化错误正文。
            "Image/label shape mismatch in {}: {} vs {}".format(
                # 插入文件路径、图像 shape、标签 shape。
                path, image.shape, label.shape
            )
        )
    # 返回经过类型和形状校验的数组。
    return image, label


# 仅用于 ACDC train/valid 二维切片的数据集。
class ACDCdataset(Dataset):
    """Two-dimensional slice dataset used only for ACDC training."""

    # transform 通常只在训练集传入 RandomGenerator。
    def __init__(self, base_dir, list_dir, split="train", transform=None):
        # 该类不负责测试三维体；测试由 ACDCVolumeDataset 读取。
        if split not in {"train", "valid"}:
            # 阻止把三维测试数据误送进二维训练接口。
            raise ValueError("ACDCdataset supports only train/valid slices")
        # 保存 ACDC 数据根目录。
        self.base_dir = base_dir
        # 保存当前划分。
        self.split = split
        # 保存可选同步增强变换。
        self.transform = transform
        # 读取划分列表并验证非空。
        self.sample_list = _read_list(list_dir, split)

    # 返回二维切片数量。
    def __len__(self):
        # 每个列表项对应一个 NPZ 样本。
        return len(self.sample_list)

    # 根据 index 读取一个训练/验证切片。
    def __getitem__(self, index):
        # 取得列表中的样本名称。
        name = self.sample_list[index]
        # 在 <base_dir>/<split> 下解析真实文件路径。
        path = _resolve_npz(os.path.join(self.base_dir, self.split), name)
        # 加载并校验图像标签。
        image, label = _load_npz(path)
        # 此类严格要求二维切片。
        if image.ndim != 2:
            # 报告误传入的体数据或带通道数据形状。
            raise ValueError("Training slice must be 2D: {} -> {}".format(path, image.shape))
        # 先以 NumPy 字典封装，供同步 transform 操作。
        sample = {"image": image, "label": label}
        # 训练时若提供变换，则执行增强、缩放和张量化。
        if self.transform is not None:
            # transform 必须同时处理 image、label。
            sample = self.transform(sample)
        # 验证等无 transform 情况也要转换为张量。
        else:
            # 构造与训练输出布局一致的字典。
            sample = {
                # 单通道图像 [1,H,W]。
                "image": torch.from_numpy(image).unsqueeze(0),
                # 整型标签 [H,W]。
                "label": torch.from_numpy(label),
            }
        # 去掉扩展名和目录，只保留稳定 case_name 供日志使用。
        sample["case_name"] = os.path.splitext(os.path.basename(name))[0]
        # 返回单个样本。
        return sample


# ACDC 病例级评测数据集：验证集重组二维切片，测试集读取现成三维 NPZ。
class ACDCVolumeDataset(Dataset):
    """ACDC volume dataset: regroup valid slices and read test volumes."""

    # 预编译验证切片命名规则。
    valid_pattern = re.compile(
        # 捕获病例名、ED/ES 相位和整数层号，并兼容可选 .npz。
        r"^(case_?\d+)_slice(ED|ES)_(\d+)(?:\.npz)?$",
        # 文件名大小写不敏感。
        re.IGNORECASE,
    )

    # split 只能是 valid 或 test。
    def __init__(self, base_dir, list_dir, split):
        # 限定病例级数据用途。
        if split not in {"valid", "test"}:
            # 训练切片应使用 ACDCdataset。
            raise ValueError("split must be 'valid' or 'test'")
        # 保存数据根目录。
        self.base_dir = base_dir
        # 保存评测划分。
        self.split = split
        # 读取该划分全部文件名。
        names = _read_list(list_dir, split)

        # 验证列表按二维切片列出，需要恢复病例体。
        if split == "valid":
            # key 是 case+ED/ES，value 是 (层号,文件名) 列表。
            grouped = defaultdict(list)
            # 遍历验证切片。
            for name in names:
                # 去掉可能的父目录，仅匹配基本文件名。
                base_name = os.path.basename(name)
                # 要求整个名称符合模式。
                match = self.valid_pattern.fullmatch(base_name)
                # 无法解析就无法安全重组体数据。
                if match is None:
                    # 报告具体非法名称。
                    raise ValueError(
                        # 拼接错误信息。
                        "Cannot group ACDC valid slice filename: " + base_name
                    )
                # 生成例如 case_019_volume_ED 的病例/相位标识。
                case_name = "{}_volume_{}".format(
                    # 第一组是病例，第二组统一转大写 ED/ES。
                    match.group(1), match.group(2).upper()
                )
                # 保存整数层号，避免字符串排序把 slice10 排在 slice2 前。
                grouped[case_name].append((int(match.group(3)), name))
            # 转成稳定排序的 (case_name, ordered_names) 索引表。
            self.samples = [
                # indexed 先按整数层号排序，再只保留文件名。
                (case_name, [name for _, name in sorted(indexed)])
                # 病例键也排序，保证每次评测顺序一致。
                for case_name, indexed in sorted(grouped.items())
            ]
        # 测试列表中每项本身就是一个三维 NPZ。
        else:
            # 每个样本保留统一的名称列表结构以复用 __getitem__。
            self.samples = [
                # case_name 去扩展名，第二项是仅含当前文件的列表。
                (os.path.splitext(os.path.basename(name))[0], [name])
                # 遍历 test.txt 的既定顺序。
                for name in names
            ]

    # 返回验证/测试三维病例数量。
    def __len__(self):
        # 验证是病例相位组数，测试是列表文件行数。
        return len(self.samples)

    # 读取并返回一个完整 [D,H,W] 病例。
    def __getitem__(self, index):
        # 取得病例名以及组成该病例的一个或多个文件名。
        case_name, names = self.samples[index]
        # 当前划分实际文件目录。
        split_dir = os.path.join(self.base_dir, self.split)

        # 验证数据由多个二维 NPZ 重组。
        if self.split == "valid":
            # 分别缓存图像和标签切片。
            images, labels = [], []
            # names 已按整数层号排序。
            for name in names:
                # 解析、加载并检查当前二维切片。
                image, label = _load_npz(_resolve_npz(split_dir, name))
                # 阻止三维文件混入验证切片列表。
                if image.ndim != 2:
                    # 报告错误项名称。
                    raise ValueError("Valid item is not a 2D slice: " + name)
                # 按深度顺序加入图像列表。
                images.append(image)
                # 标签使用相同顺序加入。
                labels.append(label)
            # 在新轴0堆叠成 [D,H,W] CT 体。
            image = np.stack(images, axis=0)
            # 同样堆叠成 [D,H,W] 标签体。
            label = np.stack(labels, axis=0)
        # 测试样本直接读取一个完整三维文件。
        else:
            # names 只有一个元素。
            image, label = _load_npz(_resolve_npz(split_dir, names[0]))
            # 测试接口严格要求 [D,H,W]。
            if image.ndim != 3:
                # 报告文件名和实际 shape。
                raise ValueError(
                    # 格式化详细错误。
                    "Test item must be a 3D volume: {} -> {}".format(
                        # 插入文件名和图像形状。
                        names[0], image.shape
                    )
                )

        # 转成 DataLoader 可批处理的返回字典。
        return {
            # ascontiguousarray 确保堆叠/切片后的内存布局可被 torch.from_numpy 接收。
            "image": torch.from_numpy(np.ascontiguousarray(image)),
            # 标签同样转为连续内存张量。
            "label": torch.from_numpy(np.ascontiguousarray(label)),
            # 病例标识用于逐病例日志和结果文件名。
            "case_name": case_name,
        }
