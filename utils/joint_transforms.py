# math 提供平方根和向上取整，用于随机面积裁剪与滑窗步数计算。
import math
# numbers.Number 用于同时识别 int、float 等数值型尺寸参数。
import numbers
# random 为裁剪位置、翻转、旋转角度和缩放比例提供随机数。
import random

# PIL.Image 执行裁剪、缩放、旋转和翻转；ImageOps 用于统一补边。
from PIL import Image, ImageOps
# NumPy 用于滑窗裁剪阶段的数组切片和 padding。
import numpy as np


# 把多个“图像与mask联合变换”按顺序串联，作用类似 torchvision.transforms.Compose。
class Compose(object):
    # transforms 是可调用对象列表，每个对象都必须接收并返回 (img, mask)。
    def __init__(self, transforms):
        # 保存变换执行顺序；数据经过前一个变换后再进入下一个。
        self.transforms = transforms

    # img 和 mask 必须执行完全相同的几何变换，否则监督像素会错位。
    def __call__(self, img, mask):
        # PIL 的 size 顺序是 (width,height)；进入流水线前先确认图像与标签尺寸一致。
        assert img.size == mask.size
        # 逐个执行联合变换。
        for t in self.transforms:
            # 每一步都同时更新 img 和 mask。
            img, mask = t(img, mask)
        # 返回完成全部变换后的配对结果。
        return img, mask


# 在图像与mask的同一位置执行随机定尺寸裁剪。
class RandomCrop(object):
    # size 可传单个数字表示正方形，也可传 (height,width)；padding 可先扩展边界。
    def __init__(self, size, padding=0):
        # 单个数字统一转换为二元尺寸。
        if isinstance(size, numbers.Number):
            # 本类内部约定 self.size=(target_height,target_width)。
            self.size = (int(size), int(size))
        # 已是二元尺寸时直接保存。
        else:
            # 调用方负责保证顺序为 (height,width)。
            self.size = size
        # 保存四周补零像素数。
        self.padding = padding

    # 对一对 PIL 图像执行随机裁剪。
    def __call__(self, img, mask):
        # padding>0 时先给图像和mask增加相同宽度的黑边。
        if self.padding > 0:
            # 图像边界填充值为0。
            img = ImageOps.expand(img, border=self.padding, fill=0)
            # mask边界也填0，通常代表背景类别。
            mask = ImageOps.expand(mask, border=self.padding, fill=0)

        # 补边后再次保证二者尺寸相同。
        assert img.size == mask.size
        # PIL 返回 (width,height)。
        w, h = img.size
        # self.size 保存 (target_height,target_width)。
        th, tw = self.size
        # 已经等于目标尺寸时不做任何重新采样。
        if w == tw and h == th:
            # 保持原像素和值不变。
            return img, mask
        # 任一边小于目标裁剪尺寸时无法正常随机取窗口，退化为直接缩放。
        if w < tw or h < th:
            # 图像用双线性插值保持视觉连续；mask用最近邻，避免产生不存在的小数类别标签。
            return img.resize((tw, th), Image.BILINEAR), mask.resize((tw, th), Image.NEAREST)

        # 左上角横坐标在所有合法窗口起点中均匀采样。
        x1 = random.randint(0, w - tw)
        # 左上角纵坐标同理。
        y1 = random.randint(0, h - th)
        # 用完全相同的边界框裁剪图像和mask，返回尺寸均为 (tw,th)。
        return img.crop((x1, y1, x1 + tw, y1 + th)), mask.crop((x1, y1, x1 + tw, y1 + th))


# 从图像中心裁出固定尺寸区域，常用于确定性的验证/测试预处理。
class CenterCrop(object):
    # size 接受单个正方形边长或 (height,width)。
    def __init__(self, size):
        # 数值尺寸转换成二元组。
        if isinstance(size, numbers.Number):
            # 目标为正方形。
            self.size = (int(size), int(size))
        # 二元尺寸直接保存。
        else:
            # 保持调用方提供的目标高宽。
            self.size = size

    # 对图像和mask使用同一中心窗口。
    def __call__(self, img, mask):
        # 防止输入配对在进入本变换前已经错位。
        assert img.size == mask.size
        # 读取原始宽高。
        w, h = img.size
        # 读取目标高宽。
        th, tw = self.size
        # 计算水平居中的左边界，round后转整数。
        x1 = int(round((w - tw) / 2.))
        # 计算垂直居中的上边界。
        y1 = int(round((h - th) / 2.))
        # 同框裁剪，保证标签边界与图像内容对应。
        return img.crop((x1, y1, x1 + tw, y1 + th)), mask.crop((x1, y1, x1 + tw, y1 + th))


# 以50%概率执行水平镜像增强。
class RandomHorizontallyFlip(object):
    # 无构造参数，每次调用独立采样是否翻转。
    def __call__(self, img, mask):
        # random.random() 位于 [0,1)，小于0.5即执行翻转。
        if random.random() < 0.5:
            # 图像和mask同时左右翻转，避免监督错位。
            return img.transpose(Image.FLIP_LEFT_RIGHT), mask.transpose(Image.FLIP_LEFT_RIGHT)
        # 另一半概率原样返回。
        return img, mask


# 强制缩放到指定 (height,width)，不保持原始宽高比。
class FreeScale(object):
    # size 的外部约定是 (height,width)。
    def __init__(self, size):
        # PIL.resize 需要 (width,height)，因此反转元组顺序。
        self.size = tuple(reversed(size))  # size: (h, w)

    # 同时缩放图像和标签。
    def __call__(self, img, mask):
        # 输入尺寸必须一致。
        assert img.size == mask.size
        # 连续图像使用双线性，离散mask使用最近邻。
        return img.resize(self.size, Image.BILINEAR), mask.resize(self.size, Image.NEAREST)


# 等比例缩放，使原图较长边变为给定 size，不裁剪也不拉伸比例。
class Scale(object):
    # size 是目标长边长度。
    def __init__(self, size):
        # 保存标量目标长度。
        self.size = size

    # 对图像和mask执行相同等比例缩放。
    def __call__(self, img, mask):
        # 确认配对尺寸一致。
        assert img.size == mask.size
        # 读取宽高。
        w, h = img.size
        # 如果当前较长边已经等于目标值，可直接返回以避免重复插值。
        if (w >= h and w == self.size) or (h >= w and h == self.size):
            # 保留原像素。
            return img, mask
        # 横向图像以宽为长边。
        if w > h:
            # 新宽固定为目标长度。
            ow = self.size
            # 新高按原宽高比计算。
            oh = int(self.size * h / w)
            # 图像双线性、mask最近邻。
            return img.resize((ow, oh), Image.BILINEAR), mask.resize((ow, oh), Image.NEAREST)
        # 纵向或正方形图像以高为长边。
        else:
            # 新高固定为目标长度。
            oh = self.size
            # 新宽按比例计算。
            ow = int(self.size * w / h)
            # 使用相同目标尺寸缩放图像和mask。
            return img.resize((ow, oh), Image.BILINEAR), mask.resize((ow, oh), Image.NEAREST)


# 随机选择面积和宽高比裁剪，再统一缩放为 size x size，类似 torchvision RandomResizedCrop。
class RandomSizedCrop(object):
    # size 是最终正方形边长。
    def __init__(self, size):
        # 保存输出尺寸。
        self.size = size

    # 输入输出均为一对 PIL 图像。
    def __call__(self, img, mask):
        # 保证初始对齐。
        assert img.size == mask.size
        # 最多尝试10次寻找能完整落在原图内部的随机窗口。
        for attempt in range(10):
            # 原图总像素面积。
            area = img.size[0] * img.size[1]
            # 随机目标裁剪面积占原图45%到100%。
            target_area = random.uniform(0.45, 1.0) * area
            # 随机宽高比范围0.5到2。
            aspect_ratio = random.uniform(0.5, 2)

            # 根据面积和宽高比反解窗口宽度。
            w = int(round(math.sqrt(target_area * aspect_ratio)))
            # 反解窗口高度，使 w*h 约等于 target_area。
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            # 以50%概率交换宽高，减少采样方向偏好。
            if random.random() < 0.5:
                # 交换候选窗口的宽和高。
                w, h = h, w

            # 只有窗口能完整放入原图时才采用。
            if w <= img.size[0] and h <= img.size[1]:
                # 随机采样合法水平起点。
                x1 = random.randint(0, img.size[0] - w)
                # 随机采样合法垂直起点。
                y1 = random.randint(0, img.size[1] - h)

                # 图像按候选框裁剪。
                img = img.crop((x1, y1, x1 + w, y1 + h))
                # mask使用完全相同的候选框。
                mask = mask.crop((x1, y1, x1 + w, y1 + h))
                # 防御性确认PIL裁剪结果尺寸正确。
                assert (img.size == (w, h))

                # 裁剪后统一成正方形；再次强调mask只能使用最近邻插值。
                return img.resize((self.size, self.size), Image.BILINEAR), mask.resize((self.size, self.size),
                                                                                       Image.NEAREST)

        # Fallback
        # 十次都找不到合法窗口时，退化为“等比例缩放长边 + 中心裁剪”。
        scale = Scale(self.size)
        # 中心裁剪负责得到最终 size x size。
        crop = CenterCrop(self.size)
        # * 解包 Scale 返回的 (img,mask)，再传给 CenterCrop。
        return crop(*scale(img, mask))


# 在 [-degree,+degree] 范围内随机旋转图像和mask。
class RandomRotate(object):
    # degree 是最大绝对旋转角度。
    def __init__(self, degree):
        # 保存角度范围。
        self.degree = degree

    # 每次调用采样一个连续角度。
    def __call__(self, img, mask):
        # 将 [0,1) 线性映射到 [-degree,+degree)。
        rotate_degree = random.random() * 2 * self.degree - self.degree
        # 图像双线性旋转；mask最近邻旋转，避免类别值被混合。
        return img.rotate(rotate_degree, Image.BILINEAR), mask.rotate(rotate_degree, Image.NEAREST)


# 先随机改变宽高，再等比例缩放和随机裁剪到固定尺寸。
class RandomSized(object):
    # size 是最终裁剪尺寸。
    def __init__(self, size):
        # 保存目标尺寸。
        self.size = size
        # 复用前面定义的等比例 Scale。
        self.scale = Scale(self.size)
        # 复用固定尺寸 RandomCrop。
        self.crop = RandomCrop(self.size)

    # 联合随机缩放与裁剪。
    def __call__(self, img, mask):
        # 保证输入图像和mask对齐。
        assert img.size == mask.size

        # 宽度独立乘0.5到2之间的随机比例。
        w = int(random.uniform(0.5, 2) * img.size[0])
        # 高度也独立随机缩放，因此可改变宽高比。
        h = int(random.uniform(0.5, 2) * img.size[1])

        # 用同一目标宽高重采样图像和mask。
        img, mask = img.resize((w, h), Image.BILINEAR), mask.resize((w, h), Image.NEAREST)

        # 先把长边缩放到目标值，再随机裁出目标正方形。
        return self.crop(*self.scale(img, mask))


# 旧版滑窗实现；保留用于兼容历史代码，其中 xrange 是 Python 2 风格遗留名称。
class SlidingCropOld(object):
    # crop_size 是方形窗口边长，stride_rate 控制重叠比例，ignore_label 用于mask补边。
    def __init__(self, crop_size, stride_rate, ignore_label):
        # 保存裁剪边长。
        self.crop_size = crop_size
        # 实际步幅约为 crop_size*stride_rate。
        self.stride_rate = stride_rate
        # mask补边不能默认当作有效类别，因此使用指定忽略标签。
        self.ignore_label = ignore_label

    # NumPy层面的右侧和底部补边辅助函数。
    def _pad(self, img, mask):
        # NumPy图像形状为 (height,width,channels)。
        h, w = img.shape[: 2]
        # 高度不足 crop_size 时只在底部补齐。
        pad_h = max(self.crop_size - h, 0)
        # 宽度不足时只在右侧补齐。
        pad_w = max(self.crop_size - w, 0)
        # 图像补0，第三个通道维不补。
        img = np.pad(img, ((0, pad_h), (0, pad_w), (0, 0)), 'constant')
        # mask用 ignore_label 补边，防止补出的区域参与有效监督。
        mask = np.pad(mask, ((0, pad_h), (0, pad_w)), 'constant', constant_values=self.ignore_label)
        # 返回至少为 crop_size x crop_size 的数组。
        return img, mask

    # 把大图切成可重叠窗口；旧版返回值格式与新版不同。
    def __call__(self, img, mask):
        # PIL输入必须同尺寸。
        assert img.size == mask.size

        # PIL尺寸顺序为宽、高。
        w, h = img.size
        # 判断是否至少有一边超过窗口。
        long_size = max(h, w)

        # 转为NumPy以便使用二维切片。
        img = np.array(img)
        # mask同步转换。
        mask = np.array(mask)

        # 大图需要多窗口遍历。
        if long_size > self.crop_size:
            # 向上取整得到整数步幅，stride_rate<1时窗口互相重叠。
            stride = int(math.ceil(self.crop_size * self.stride_rate))
            # 计算垂直方向需要的窗口数，最后一个窗口允许经_pad补齐。
            h_step_num = int(math.ceil((h - self.crop_size) / float(stride))) + 1
            # 计算水平方向窗口数。
            w_step_num = int(math.ceil((w - self.crop_size) / float(stride))) + 1
            # 分别收集图像窗口和mask窗口。
            img_sublist, mask_sublist = [], []
            # 工程遗留：xrange 在 Python 3 中未定义，当前项目主路径不应依赖该旧类。
            for yy in xrange(h_step_num):
                # 横向遍历窗口。
                for xx in xrange(w_step_num):
                    # 计算窗口左上角坐标。
                    sy, sx = yy * stride, xx * stride
                    # 计算理论右下边界。
                    ey, ex = sy + self.crop_size, sx + self.crop_size
                    # 裁剪图像窗口。
                    img_sub = img[sy: ey, sx: ex, :]
                    # 裁剪同坐标mask窗口。
                    mask_sub = mask[sy: ey, sx: ex]
                    # 边缘不足窗口尺寸时补齐。
                    img_sub, mask_sub = self._pad(img_sub, mask_sub)
                    # 转回RGB PIL图像。
                    img_sublist.append(Image.fromarray(img_sub.astype(np.uint8)).convert('RGB'))
                    # mask转回调色板模式P，保留离散标签语义。
                    mask_sublist.append(Image.fromarray(mask_sub.astype(np.uint8)).convert('P'))
            # 返回所有窗口列表。
            return img_sublist, mask_sublist
        # 小图只需补齐成单窗口。
        else:
            # 补到crop_size。
            img, mask = self._pad(img, mask)
            # 图像转回RGB。
            img = Image.fromarray(img.astype(np.uint8)).convert('RGB')
            # mask转回P模式。
            mask = Image.fromarray(mask.astype(np.uint8)).convert('P')
            # 旧版小图分支直接返回单个对象，而非列表。
            return img, mask


# Python 3 可用的滑窗实现，同时返回每个窗口在原图中的坐标信息以便结果拼接。
class SlidingCrop(object):
    # 参数语义与旧版一致。
    def __init__(self, crop_size, stride_rate, ignore_label):
        # 保存窗口边长。
        self.crop_size = crop_size
        # 保存相对步幅。
        self.stride_rate = stride_rate
        # 保存mask补边忽略标签。
        self.ignore_label = ignore_label

    # 补边并额外返回补边前实际高宽，供恢复预测时裁掉无效区域。
    def _pad(self, img, mask):
        # 记录当前实际高宽。
        h, w = img.shape[: 2]
        # 计算底部补边。
        pad_h = max(self.crop_size - h, 0)
        # 计算右侧补边。
        pad_w = max(self.crop_size - w, 0)
        # 图像用0补齐。
        img = np.pad(img, ((0, pad_h), (0, pad_w), (0, 0)), 'constant')
        # mask用ignore_label补齐。
        mask = np.pad(mask, ((0, pad_h), (0, pad_w)), 'constant', constant_values=self.ignore_label)
        # 同时返回原始 h、w，而不是补边后的尺寸。
        return img, mask, h, w

    # 输出始终是图像窗口列表、mask窗口列表和坐标信息列表。
    def __call__(self, img, mask):
        # 输入配对尺寸检查。
        assert img.size == mask.size

        # 读取PIL宽高。
        w, h = img.size
        # 较长边用于判断是否需要多窗口。
        long_size = max(h, w)

        # 转为NumPy数组。
        img = np.array(img)
        # mask同步转换。
        mask = np.array(mask)

        # 至少一边大于窗口时执行滑窗。
        if long_size > self.crop_size:
            # 根据相对步幅计算像素步幅。
            stride = int(math.ceil(self.crop_size * self.stride_rate))
            # 垂直窗口数量。
            h_step_num = int(math.ceil((h - self.crop_size) / float(stride))) + 1
            # 水平窗口数量。
            w_step_num = int(math.ceil((w - self.crop_size) / float(stride))) + 1
            # 三个列表一一对应：第i个图像、第i个mask及其坐标元数据。
            img_slices, mask_slices, slices_info = [], [], []
            # 逐行遍历窗口。
            for yy in range(h_step_num):
                # 逐列遍历窗口。
                for xx in range(w_step_num):
                    # 窗口左上角。
                    sy, sx = yy * stride, xx * stride
                    # 窗口理论右下边界。
                    ey, ex = sy + self.crop_size, sx + self.crop_size
                    # 截取图像区域。
                    img_sub = img[sy: ey, sx: ex, :]
                    # 截取同坐标mask区域。
                    mask_sub = mask[sy: ey, sx: ex]
                    # 边缘窗口补齐，并记录补齐前sub_h、sub_w。
                    img_sub, mask_sub, sub_h, sub_w = self._pad(img_sub, mask_sub)
                    # 追加RGB PIL图像窗口。
                    img_slices.append(Image.fromarray(img_sub.astype(np.uint8)).convert('RGB'))
                    # 追加P模式mask窗口。
                    mask_slices.append(Image.fromarray(mask_sub.astype(np.uint8)).convert('P'))
                    # 记录原图坐标和该窗口真实有效高宽。
                    slices_info.append([sy, ey, sx, ex, sub_h, sub_w])
            # 返回所有窗口及其恢复信息。
            return img_slices, mask_slices, slices_info
        # 小图补齐成一个窗口，并仍统一包装成长度1的列表。
        else:
            # 补齐并获取真实有效尺寸。
            img, mask, sub_h, sub_w = self._pad(img, mask)
            # 转回RGB PIL图像。
            img = Image.fromarray(img.astype(np.uint8)).convert('RGB')
            # 转回P模式mask。
            mask = Image.fromarray(mask.astype(np.uint8)).convert('P')
            # 坐标从(0,0)开始，sub_h/sub_w标记补边前有效区域。
            return [img], [mask], [[0, sub_h, 0, sub_w, sub_h, sub_w]]
