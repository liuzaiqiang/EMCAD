# os 负责把数据根目录、列表目录和样本文件名组合成实际路径。
import os
# Python random 用于在 RandomGenerator 中选择不同的数据增强分支。
import random
# h5py 读取测试集按完整病例保存的 .npy.h5 文件。
import h5py
# NumPy 负责训练 NPZ 读取以及旋转、翻转、数组类型转换。
import numpy as np
# PyTorch 把 NumPy 图像和标签转换为训练张量。
import torch
# OpenCV 当前仅出现在下面保留的注释示例中，实际运行路径没有调用。
import cv2
# scipy.ndimage 提供任意角度旋转。
from scipy import ndimage
# zoom 用不同插值阶数同步缩放连续 CT 图像和离散分割标签。
from scipy.ndimage.interpolation import zoom
# 继承 Dataset 后，该类可交给 PyTorch DataLoader 按索引并行取样。
from torch.utils.data import Dataset

"""
    该函数接收一张图像（image）和对应的标签（label），对它们同步进行随机的旋转和翻转操作，最后返回处理后的图像和标签。
    此代码假设标签也是类似图像的数组形式（如掩膜 Mask）。如果标签是边界框坐标或分类标签，则需要完全不同的处理方式。
"""
# 同步执行 90 度倍数旋转与镜像，保证 CT 和标签仍逐像素对齐。
def random_rot_flip(image, label):
    # 从 0、1、2、3 中随机选 k，对应旋转 0、90、180、270 度。
    k = np.random.randint(0, 4)
    # 只旋转二维空间轴，图像数值不经过插值。
    image = np.rot90(image, k)
    # 标签使用完全相同的 k，避免增强后监督错位。
    label = np.rot90(label, k)
    # 随机选择轴 0 或轴 1，分别对应上下或左右翻转。
    axis = np.random.randint(0, 2)
    # np.flip 可能返回负步长视图；copy() 生成 PyTorch 可安全接收的连续正步长数组。
    image = np.flip(image, axis=axis).copy()
    # 标签沿同一轴翻转并复制。
    label = np.flip(label, axis=axis).copy()
    # 返回同步变换后的图像与标签二元组。
    return image, label


# 同步执行 [-20, 20) 度的随机小角度旋转。
def random_rotate(image, label):
    # 以整数角度采样，最小 -20 度、最大 19 度。
    angle = np.random.randint(-20, 20)
    # order=0 使用最近邻插值，reshape=False 保持原尺寸；这是原代码对 CT 图像的选择。
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    # 标签必须同步旋转；最近邻插值保证不会凭空产生新的类别编号。
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    # 返回增强后的配对样本。
    return image, label


# 训练样本变换器：随机增强、缩放到网络输入尺寸并转换成张量。
class RandomGenerator(object):
    # output_size 通常由 train_synapse.py 的 args.img_size 构造为 [H, W]。
    def __init__(self, output_size):
        # 保存目标空间尺寸，后续每个样本都统一到该尺寸。
        self.output_size = output_size

    # sample 是至少包含 image、label 两个键的字典。
    def __call__(self, sample):
        # 同时取出输入切片和像素级标签。
        image, label = sample['image'], sample['label']
        # 第一次随机数大于 0.5 时，选择离散旋转加翻转增强。
        if random.random() > 0.5:
            # 图像和标签由同一函数同步处理。
            image, label = random_rot_flip(image, label)
        # 仅在第一分支未命中时重新采样随机数；命中则执行小角度旋转。
        elif random.random() > 0.5:
            # 同步旋转图像和标签。
            image, label = random_rotate(image, label)
        # 读取增强后二维切片的高和宽。
        x, y = image.shape
        # 只在当前尺寸与模型目标输入不一致时进行重采样。
        if x != self.output_size[0] or y != self.output_size[1]:
            """
                order=3 表示使用三次样条插值。
                原因：图像通常是连续的信号，包含丰富的灰度/颜色信息。三次样条插值能够产生更平滑、更高质量的缩放结果，保留更多的细节，视觉效果更好。
                为什么不是 0 或 1：order=0 是最近邻插值，会导致图像出现明显的锯齿（马赛克）；order=1 是双线性插值，比最近邻平滑，但不如三次插值精细。
             """
            # 连续 CT 图像使用三次样条插值，缩放因子分别是目标高/原高、目标宽/原宽。
            image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=3)
            """
            原因：在分割任务中，label（标签）通常是整数掩码，代表不同的类别（例如：0=背景，1=肝脏，2=肾脏）。
            如果对标签使用 order=3（三次插值），插值过程会产生非整数的浮点数（例如 0.5, 1.2 等）。这会导致标签中出现了不存在的“中间类别”，破坏了标签的语义含义。
            必须使用 order=0：最近邻插值选择最近的像素值，能保证缩放后的标签依然是整数，不会产生新的类别。
            """
            # 离散标签使用最近邻插值，保持像素值仍是合法整数类别。
            label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)

        # CT 数组转 float32 张量，并在最前面增加单通道维，得到 [1, H, W]。
        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        # 标签先转 float32 张量；下一步会再转为交叉熵要求的 long。
        label = torch.from_numpy(label.astype(np.float32))

        # 重新组装样本字典；label.long() 把类别编号转成 int64。
        sample = {'image': image, 'label': label.long()}
        # 返回 DataLoader 可自动堆叠的张量字典。
        return sample


# Synapse 数据集适配器：训练读取二维 NPZ，验证/测试读取完整三维 HDF5。
class Synapse_dataset(Dataset):
    # base_dir 是实际数据目录，list_dir 保存划分列表，split 决定读取格式。
    def __init__(self, base_dir, list_dir, split, nclass=9, transform=None):
        # 保存可选的训练变换；测试体通常不传入随机变换。 # using transform in torch!
        self.transform = transform
        # 保存 train、test_vol 等划分名。
        self.split = split
        # 一次性读取对应列表的全部行；每行是一个切片名或病例名。
        self.sample_list = open(os.path.join(list_dir, self.split + '.txt')).readlines()
        # 保存 NPZ/HDF5 文件所在根目录。
        self.data_dir = base_dir
        # 保存类别总数；当前活动代码不在此类内部重新映射标签。
        self.nclass = nclass

    # DataLoader 用它确定一个 epoch 中可索引的样本总数。
    def __len__(self):
        # 列表文件的非过滤行数就是数据集长度。
        return len(self.sample_list)

    # 根据 idx 返回一条样本；训练返回二维切片，测试返回完整三维体。
    def __getitem__(self, idx):
        # train 分支对应 preprocess_synapse_data.py 生成的逐切片 NPZ。
        if self.split == "train":
            # 去掉列表行末换行，得到不带扩展名的切片基名。
            slice_name = self.sample_list[idx].strip('\n')
            # 训练文件约定为 base_dir/<slice_name>.npz。
            data_path = os.path.join(self.data_dir, slice_name + '.npz')
            # 加载一个压缩/未压缩 NPZ 容器。
            data = np.load(data_path)
            # 读取预处理阶段使用的 image 和 label 两个键。
            image, label = data['image'], data['label']
            # print(image.shape)
            # image = np.reshape(image, (512, 512))
            # image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            # label = np.reshape(label, (512, 512))

        # 非 train 分支按完整体数据格式读取，用于验证或测试。
        else:
            # 列表项是病例名，例如 case0001。
            vol_name = self.sample_list[idx].strip('\n')
            # 测试预处理约定文件后缀为 .npy.h5。
            filepath = self.data_dir + "/{}.npy.h5".format(vol_name)
            # 打开 HDF5 病例文件；原代码依赖对象生命周期关闭文件，不在此处改变该行为。
            data = h5py.File(filepath)
            # [:] 把 image、label 数据集完整读入 NumPy 内存，通常形状为 [D, H, W]。
            image, label = data['image'][:], data['label'][:]
            # image = np.reshape(image, (image.shape[2], 512, 512))
            # label = np.reshape(label, (label.shape[2], 512, 512))
            # label[label==5]= 0
            # label[label==9]= 0
            # label[label==10]= 0
            # label[label==12]= 0
            # label[label==13]= 0
            # label[label==11]= 5

        # if self.nclass == 9:
        #     label[label==5]= 0
        #     label[label==9]= 0
        #     label[label==10]= 0
        #     label[label==12]= 0
        #     label[label==13]= 0
        #     label[label==11]= 5

        # 用统一键名封装二维或三维 NumPy 数组。
        sample = {'image': image, 'label': label}
        # 训练时若传入 RandomGenerator，则在返回前执行增强、缩放和张量化。
        if self.transform:
            # 变换必须同时接收并返回包含 image、label 的字典。
            sample = self.transform(sample)
        # 把列表中的原始名称附加到样本，供测试日志、结果文件命名和病例级统计使用。
        sample['case_name'] = self.sample_list[idx].strip('\n')
        # 返回单个样本；DataLoader 会在外层增加 batch 维。
        return sample
