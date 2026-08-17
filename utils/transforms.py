# Python 标准库的伪随机数生成器；本文件用它决定是否翻转以及采样高斯模糊强度。
import random

# NumPy 负责在 PIL 图像与数组之间转换，并进行通道顺序操作。
import numpy as np
# scikit-image 的 gaussian 在数组空间执行高斯平滑；它只改变输入图像，不改变分割标签。
from skimage.filters import gaussian
# PyTorch 仅用于把离散掩膜转换成训练所需的 LongTensor。
import torch
# PIL.Image 提供几何变换和图像封装；ImageFilter 在当前文件中虽被导入但没有被调用。
from PIL import Image, ImageFilter


# 随机垂直翻转增强：以 0.5 概率沿图像上下方向翻转。
class RandomVerticalFlip(object):
    # 让实例可像函数一样被 torchvision.transforms.Compose 调用；img 是一张 PIL 图像。
    def __call__(self, img):
        # random.random() 取 [0, 1) 均匀随机数，小于 0.5 时执行翻转。
        if random.random() < 0.5:
            # FLIP_TOP_BOTTOM 只交换上下像素位置，不改变图像尺寸和通道数。
            return img.transpose(Image.FLIP_TOP_BOTTOM)
        # 未命中增强概率时原样返回，避免额外复制图像。
        return img


# 反标准化工具：把经过 (x-mean)/std 的张量恢复到近似原始数值范围，常用于可视化。
class DeNormalize(object):
    # mean、std 应与此前 Normalize 使用的逐通道统计量顺序一致。
    def __init__(self, mean, std):
        # 保存每个通道的均值，供 __call__ 逐通道恢复。
        self.mean = mean
        # 保存每个通道的标准差。
        self.std = std

    # tensor 的首维应为通道维 C；该方法会原地修改传入张量。
    def __call__(self, tensor):
        # 同时遍历通道张量 t、对应均值 m 和标准差 s。
        for t, m, s in zip(tensor, self.mean, self.std):
            # x_norm * std + mean；下划线版本 mul_、add_ 表示原地运算，不创建新张量。
            t.mul_(s).add_(m)
        # 返回同一个、已经完成反标准化的张量对象。
        return tensor


# 掩膜转换器：把 PIL/数组形式的类别标签变成 PyTorch 整型标签张量。
class MaskToTensor(object):
    # img 是分割掩膜；这里不做归一化，因为像素值就是类别编号。
    def __call__(self, img):
        # 先固定为 int32 NumPy 数组，再转张量并升为 long，满足交叉熵对标签 dtype=torch.int64 的要求。
        return torch.from_numpy(np.array(img, dtype=np.int32)).long()


# 任意尺寸缩放工具；名称中的 Free 表示不强制保持输入宽高比。
class FreeScale(object):
    # size 按项目约定以 (height, width) 传入；默认用双线性插值处理连续图像。
    def __init__(self, size, interpolation=Image.BILINEAR):
        # PIL.resize 接受 (width, height)，因此这里反转调用方传入的 (height, width)。
        self.size = tuple(reversed(size))  # size: (h, w)
        # 保存插值策略；用于离散掩膜时调用方应显式传入最近邻插值，避免生成非法类别值。
        self.interpolation = interpolation

    # 对单张 PIL 图像执行确定性的尺寸变换。
    def __call__(self, img):
        # 输出空间尺寸固定为 self.size，通道数保持不变。
        return img.resize(self.size, self.interpolation)


# 通道翻转工具：典型用途是在 RGB 与 BGR 排列之间互换。
class FlipChannels(object):
    # 输入是 PIL 图像，输出仍封装为 PIL 图像。
    def __call__(self, img):
        # 前两个冒号保留高、宽顺序，::-1 将最后的颜色通道顺序完全反转。
        img = np.array(img)[:, :, ::-1]
        # 转为 uint8 后重新构造 PIL 图像，保证后续 PIL/torchvision 变换可继续使用。
        return Image.fromarray(img.astype(np.uint8))


# 随机高斯模糊增强：模拟失焦或成像平滑，提高模型对局部噪声变化的稳健性。
class RandomGaussianBlur(object):
    # img 是 PIL 图像；每次调用都会重新随机采样模糊尺度。
    def __call__(self, img):
        # sigma 均匀分布在 [0.15, 1.30)，数值越大，模糊越强。
        sigma = 0.15 + random.random() * 1.15
        # 转 NumPy 后做高斯滤波；multichannel=True 表示颜色通道不是需要互相平滑的空间轴。
        blurred_img = gaussian(np.array(img), sigma=sigma, multichannel=True)
        # skimage 默认把整数图像转换到 [0, 1] 浮点范围，因此乘 255 恢复 8 位图像量纲。
        blurred_img *= 255
        # 截取为 uint8 并转回 PIL，供后续数据增强流水线继续处理。
        return Image.fromarray(blurred_img.astype(np.uint8))
