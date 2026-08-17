#!/usr/bin/env python
# -*- coding:utf-8 -*-

# os 负责根据列表项构造 ACDC NPZ 文件路径。
import os
# random 用于训练阶段随机选择几何增强。
import random
# re 在当前活动代码中未调用；下方被注释的 volume 版本曾使用它解析切片名。
import re  # 用于解析 case_019_sliceED_10.npz 这类文件名。2026.8.5 19:38新增
# NumPy 用于 NPZ 读取、旋转翻转和数据类型转换。
import numpy as np
# PyTorch 把图像与标签数组转换为训练张量。
import torch
# ndimage.rotate 提供任意小角度旋转。
from scipy import ndimage
# zoom 将图像和标签缩放到网络输入尺寸。
from scipy.ndimage.interpolation import zoom
# 继承 Dataset 后可被 PyTorch DataLoader 索引。
from torch.utils.data import Dataset


# 同步执行 90 度倍数旋转与随机方向翻转。
def random_rot_flip(image, label):
    # 随机选择 0/90/180/270 度。
    k = np.random.randint(0, 4)
    # 旋转图像。
    image = np.rot90(image, k)
    # 以相同 k 旋转标签，保持像素对应。
    label = np.rot90(label, k)
    # 随机选择高度轴或宽度轴。
    axis = np.random.randint(0, 2)
    # 翻转图像并复制，消除负步长数组视图。
    image = np.flip(image, axis=axis).copy()
    # 标签沿同一轴翻转并复制。
    label = np.flip(label, axis=axis).copy()
    # 返回同步增强后的数组。
    return image, label


# 同步执行 [-20,20) 度随机旋转。
def random_rotate(image, label):
    # 采样整数角度。
    angle = np.random.randint(-20, 20)
    # 原版本对图像使用最近邻插值并保持原尺寸。
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    # 标签同样使用最近邻插值，保证类别值合法。
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    # 返回旋转后的样本对。
    return image, label


# SLDGroup 早期版本的 ACDC 训练样本变换器。
class RandomGenerator(object):
    # output_size 是目标 [H,W]。
    def __init__(self, output_size):
        # 保存目标尺寸。
        self.output_size = output_size

    # 接收包含 image、label 的 NumPy 字典。
    def __call__(self, sample):
        # 同时取出图像和标签。
        image, label = sample['image'], sample['label']

        # 首次随机数大于 0.5 时执行离散旋转与翻转。
        if random.random() > 0.5:
            # 同步增强图像和标签。
            image, label = random_rot_flip(image, label)
        # 第一分支未命中时再次采样，大于 0.5 则执行小角度旋转。
        elif random.random() > 0.5:
            # 同步旋转。
            image, label = random_rotate(image, label)
        # 读取当前二维高宽。
        x, y = image.shape
        # 与目标尺寸不同才缩放。
        if x != self.output_size[0] or y != self.output_size[1]:
            # 连续 CT 用三次插值；行尾疑问是原代码注释，当前仅解释实际行为。
            image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=3)  # why not 3?
            # 离散标签用最近邻插值。
            label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        # 图像转 float32 并增加单通道维 [1,H,W]。
        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        # 标签先转 float32 张量，下一行再转 long。
        label = torch.from_numpy(label.astype(np.float32))
        # 组装训练字典；long 标签供交叉熵使用。
        sample = {'image': image, 'label': label.long()}
        # 返回变换后的样本。
        return sample


# 早期 ACDC 数据集适配器；train/valid 读二维切片，其余划分按单个 NPZ 读取。
class ACDCdataset(Dataset):
    # base_dir 为数据根，list_dir 存划分列表，transform 仅训练时使用。
    def __init__(self, base_dir, list_dir, split, transform=None):
        # 保存可选同步增强。
        self.transform = transform  # using transform in torch!
        # 保存划分名称。
        self.split = split
        # 读取 <list_dir>/<split>.txt 的全部行；原实现未显式关闭句柄，保持不变。
        self.sample_list = open(os.path.join(list_dir, self.split+'.txt')).readlines()
        # 保存数据根目录。
        self.data_dir = base_dir

    # 返回列表文件行数。
    def __len__(self):
        # 每一行被视为一个可索引样本。
        return len(self.sample_list)

    # 按索引加载一个 NPZ 样本。
    def __getitem__(self, idx):
        # train/valid 文件存放于对应划分子目录。
        if self.split == "train" or self.split == "valid":
            # 去除列表行末换行符。
            slice_name = self.sample_list[idx].strip('\n')
            # 构造 <data_dir>/<split>/<slice_name> 路径。
            data_path = os.path.join(self.data_dir, self.split, slice_name)
            # 读取 NPZ 容器。
            data = np.load(data_path)
            # ACDC 键名约定为 img 和 label。
            image, label = data['img'], data['label']
        # 其他划分直接在 data_dir 下查找列表项。
        else:
            # 去除换行得到 volume 文件名。
            vol_name = self.sample_list[idx].strip('\n')
            # 使用字符串格式构造路径，保持原实现。
            filepath = self.data_dir + "/{}".format(vol_name)
            # 加载完整 NPZ。
            data = np.load(filepath)
            # 读取图像与标签数组。
            image, label = data['img'], data['label']

        # 先以 NumPy 字典封装。
        sample = {'image': image, 'label': label}
        # 只有训练划分执行随机增强；验证/测试保持确定性。
        if self.transform and self.split == "train":
            # 同步变换图像和标签。
            sample = self.transform(sample)
        # 附加原始列表项作为病例/切片名称。
        sample['case_name'] = self.sample_list[idx].strip('\n')
        # 返回样本字典。
        return sample


# class ACDCVolumeDataset(Dataset):  # 定义按完整三维体读取验证集和测试集的数据集。
#     """把 valid 二维切片重组成 volume；test 直接读取三维 NPZ。"""  # 说明类的用途。

#     _VALID_PATTERN = re.compile(  # 预编译验证集切片文件名匹配规则。
#         r"^(case_\d+)_slice(ED|ES)_(\d+)\.npz$"  # 提取病例、ED/ES 和切片编号。
#     )

#     def __init__(self, base_dir, list_dir, split):  # 接收 ACDC 根目录、列表目录和数据划分。
#         if split not in {"valid", "test"}:  # 该类只允许读取验证 volume 或测试 volume。
#             raise ValueError("split must be 'valid' or 'test'")  # 拒绝错误的数据划分。

#         self.base_dir = base_dir  # 保存 ./data/ACDC 根目录。
#         self.split = split  # 保存当前划分名称。
#         list_path = os.path.join(list_dir, split + ".txt")  # 构造 valid.txt 或 test.txt 路径。

#         if not os.path.isfile(list_path):  # 检查列表文件是否存在。
#             raise FileNotFoundError(list_path)  # 列表不存在时立即报告准确路径。

#         with open(list_path, "r", encoding="utf-8") as stream:  # 使用 UTF-8 打开列表文件。
#             file_names = [line.strip() for line in stream if line.strip()]  # 删除空行和换行符。

#         if split == "valid":  # 验证集当前保存的是二维切片。
#             grouped_files = {}  # 建立 volume 名称到切片列表的映射。

#             for file_name in file_names:  # 遍历 valid.txt 中的全部切片文件。
#                 match = self._VALID_PATTERN.fullmatch(file_name)  # 解析当前文件名。

#                 if match is None:  # 文件名不符合预期格式时不能继续。
#                     raise ValueError("Invalid ACDC valid filename: " + file_name)  # 报告错误文件名。

#                 patient_name = match.group(1)  # 取得 case_019 形式的病例名。
#                 cardiac_phase = match.group(2)  # 取得 ED 或 ES。
#                 slice_index = int(match.group(3))  # 将切片编号转换成整数。
#                 case_name = f"{patient_name}_volume_{cardiac_phase}"  # 生成 volume 名称。

#                 grouped_files.setdefault(case_name, []).append(  # 将切片加入对应 volume。
#                     (slice_index, file_name)  # 同时保存整数切片编号，防止 slice10 排在 slice2 前。
#                 )

#             self.samples = []  # 保存最终的验证 volume 索引。

#             for case_name, indexed_files in sorted(grouped_files.items()):  # 按病例名稳定排序。
#                 indexed_files.sort(key=lambda item: item[0])  # 按整数切片编号排序。
#                 ordered_files = [item[1] for item in indexed_files]  # 只保留排序后的文件名。
#                 self.samples.append((case_name, ordered_files))  # 保存一个完整验证 volume。
#         else:  # 测试集中的每个 NPZ 本身已经是三维 volume。
#             self.samples = [  # 直接建立测试 volume 索引。
#                 (os.path.splitext(file_name)[0], [file_name])  # 去掉扩展名作为 case_name。
#                 for file_name in file_names  # 遍历 test.txt。
#             ]

#         if not self.samples:  # 防止空列表导致训练后期才出错。
#             raise RuntimeError(f"No ACDC {split} samples were found")  # 立即报告空数据集。

#     def __len__(self):  # 返回验证或测试 volume 数量。
#         return len(self.samples)  # valid 应为 20，test 应为 40。

#     def __getitem__(self, index):  # 读取一个完整 ED 或 ES volume。
#         case_name, file_names = self.samples[index]  # 取得 volume 名称和所属文件。

#         if self.split == "valid":  # 验证集需要把二维切片堆叠成三维体。
#             image_slices = []  # 保存图像切片。
#             label_slices = []  # 保存标签切片。

#             for file_name in file_names:  # 按数字切片顺序逐个读取。
#                 file_path = os.path.join(  # 构造验证切片路径。
#                     self.base_dir, "valid", file_name  # 路径为 data/ACDC/valid/xxx.npz。
#                 )

#                 if not os.path.isfile(file_path):  # 检查切片文件。
#                     raise FileNotFoundError(file_path)  # 精确报告缺失路径。

#                 with np.load(file_path, allow_pickle=False) as data:  # 安全打开 NPZ。
#                     image_slices.append(data["img"].astype(np.float32))  # 图像统一为 float32。
#                     label_slices.append(data["label"].astype(np.int64))  # 标签统一为 int64。

#             image = np.stack(image_slices, axis=0)  # 得到 [D,H,W] 验证图像。
#             label = np.stack(label_slices, axis=0)  # 得到 [D,H,W] 验证标签。
#         else:  # 测试文件已经是完整三维体。
#             file_path = os.path.join(  # 构造测试 volume 路径。
#                 self.base_dir, "test", file_names[0]  # 路径为 data/ACDC/test/xxx.npz。
#             )

#             if not os.path.isfile(file_path):  # 检查测试文件。
#                 raise FileNotFoundError(file_path)  # 精确报告缺失路径。

#             with np.load(file_path, allow_pickle=False) as data:  # 安全打开三维 NPZ。
#                 image = data["img"].astype(np.float32)  # 图像统一为 float32。
#                 label = data["label"].astype(np.int64)  # 标签统一为 int64。

#         if image.ndim != 3 or label.ndim != 3:  # 检查 volume 必须为 [D,H,W]。
#             raise ValueError(  # 报告实际错误形状。
#                 f"{case_name}: image={image.shape}, label={label.shape}"
#             )

#         if image.shape != label.shape:  # 图像和标签必须逐体素对应。
#             raise ValueError(  # 报告形状不匹配。
#                 f"{case_name}: image and label shapes do not match"
#             )

#         return {  # 返回 DataLoader 能自动拼接的字典。
#             "image": torch.from_numpy(np.ascontiguousarray(image)),  # float32 [D,H,W]。
#             "label": torch.from_numpy(np.ascontiguousarray(label)),  # int64 [D,H,W]。
#             "case_name": case_name,  # 返回不带扩展名的 volume 名称。
#         }
