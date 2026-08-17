# os 枚举目录并构造跨平台完整路径。
import os
# OpenCV 读取图像/掩膜并转换 BGR、RGB、灰度颜色空间。
import cv2
# NumPy 在当前活动代码中未直接调用；保留原依赖。
import numpy as np
# PyTorch 用于掩膜阈值后的 long 类型和通道维操作。
import torch
# data 提供 Dataset 和 DataLoader。
import torch.utils.data as data
# PIL.Image 用于读取测试掩膜的原始宽高。
from PIL import Image
# Albumentations 负责对图像与 mask 同步执行几何增强和归一化。
import albumentations as A
# ToTensorV2 把 Albumentations 的 NumPy 结果转换为 PyTorch 张量。
from albumentations.pytorch import ToTensorV2

# 使用 Albumentations 的统一息肉二分类数据集。
class PolypDataset(data.Dataset):
    """
    Unified adaptive dataloader for polyp segmentation.
    Uses Albumentations and strictly handles binary mask conversion.
    """
    # split 决定训练返回二元组还是评测返回尺寸/名称；color_image 控制 RGB 或灰度输入。
    def __init__(self, image_root, gt_root, trainsize, augmentation, split='train', color_image=True):
        # 保存目标输入边长。
        self.trainsize = trainsize
        # 保存是否按彩色三通道读取。
        self.color_image = color_image
        # 保存布尔增强开关。
        self.augmentation = augmentation
        # 保存 train/val/test 等模式字符串。
        self.split = split
        
        # Load and sort file paths
        # 项目接受的图像与掩膜文件扩展名集合，比较时统一转小写。
        exts = ('.jpg', '.png', '.jpeg', '.tif')
        # 枚举输入目录、过滤扩展名、拼完整路径并排序。
        self.images = sorted([os.path.join(image_root, f) for f in os.listdir(image_root) if f.lower().endswith(exts)])
        # 对掩膜目录执行相同处理，依赖排序后的同索引文件配对。
        self.gts = sorted([os.path.join(gt_root, f) for f in os.listdir(gt_root) if f.lower().endswith(exts)])
        
        # 过滤不存在的同索引路径对。
        self.filter_files()
        # 缓存最终样本数。
        self.size = len(self.images)

        # Transformation Setup
        # 彩色输入采用 ImageNet RGB 均值，灰度输入采用单通道 0.5。
        mean = [0.485, 0.456, 0.406] if color_image else [0.5]
        # 彩色使用 ImageNet 标准差；灰度沿用原实现的单值 0.229。
        std = [0.229, 0.224, 0.225] if color_image else [0.229]

        # 只有训练划分且 augmentation 为真时启用随机几何增强。
        if self.split == 'train' and self.augmentation:
            # Albumentations Compose 会为 image 和 mask 复用同一随机几何参数。
            self.transform = A.Compose([
                # 50% 概率在 ±90 度范围旋转；mask 使用库为标签选择的插值策略。
                A.Rotate(limit=90, p=0.5),
                # 50% 概率垂直翻转。
                A.VerticalFlip(p=0.5),
                # 50% 概率水平翻转。
                A.HorizontalFlip(p=0.5),
                # 把图像和掩膜同步缩放到固定正方形。
                A.Resize(height=self.trainsize, width=self.trainsize),
                # 对图像做 mean/std 标准化；mask 不做该归一化。
                A.Normalize(mean=mean, std=std),
                # 将图像和 mask 转成 PyTorch 张量。
                ToTensorV2()
            ])
        # 验证/测试或关闭增强时只做确定性预处理。
        else:
            # 同步缩放、图像归一化和张量化。
            self.transform = A.Compose([
                # 固定空间尺寸。
                A.Resize(height=self.trainsize, width=self.trainsize),
                # 标准化输入图像。
                A.Normalize(mean=mean, std=std),
                # 转张量。
                ToTensorV2()
            ])

    # 按排序后的索引过滤任何一侧文件不存在的配对。
    def filter_files(self):
        # 有效输入路径缓存。
        valid_images, valid_gts = [], []
        # zip 会按较短列表截止；原实现未额外检查两个目录数量相等。
        for img_p, gt_p in zip(self.images, self.gts):
            # 只有图像和掩膜路径都存在时保留。
            if os.path.exists(img_p) and os.path.exists(gt_p):
                # 记录图像路径。
                valid_images.append(img_p)
                # 记录对应掩膜路径。
                valid_gts.append(gt_p)
        # 同时替换两个列表，保持相同长度和索引对应。
        self.images, self.gts = valid_images, valid_gts

    # 读取并返回第 index 个样本。
    def __getitem__(self, index):
        # 1. Load Image
        # cv2.imread 默认返回 BGR 三通道数组。
        image = cv2.imread(self.images[index])
        # 彩色模式转 RGB；灰度模式通过 BGR2GRAY 转成二维数组。
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB if self.color_image else cv2.COLOR_BGR2GRAY)
        
        # 2. Load Mask
        # 掩膜始终以单通道 0..255 灰度读取。
        mask_np = cv2.imread(self.gts[index], cv2.IMREAD_GRAYSCALE)

        # 3. Apply Transformations
        # Compose 同时接收 image、mask，确保随机旋转/翻转完全同步。
        augmented = self.transform(image=image, mask=mask_np)
        # 取标准化后的图像张量。
        image = augmented['image']
        # 取几何变换后的掩膜张量。
        mask = augmented['mask']

        # 4. Adaptive Binary Mask Logic
        # Thinking carefully: This handles 0/255 with noise and 0/1/2/3 labels.
        # 查看变换后掩膜最大灰度，用于区分 0/255 掩膜和小整数类别掩膜。
        max_val = mask.max()
        # 最大值超过 127 时按常见 8 位二值掩膜处理。
        if max_val > 127.0:
            # Treats everything above 20 as foreground to catch 255 but ignore noise
            # 大于 20 的像素视为前景，并转为 int64 0/1。
            mask = (mask > 20).long()
        # 否则把任何正整数类别合并为二分类前景。
        else:
            # Treats all integer labels (1, 2, 3...) as foreground
            # >=1 变成前景 1，其余背景 0。
            mask = (mask >= 1).long()

        # ToTensorV2 对二维 mask 通常不自动增加通道维。
        if len(mask.shape) == 2:
            # 增加通道维，统一为 [1,H,W]。
            mask = mask.unsqueeze(0)

        # 5. Return Logic
        # 训练循环只需要标准化图像和二值掩膜。
        if self.split == 'train':
            # 返回二元组供训练脚本解包。
            return image, mask
        # 评测还需要原始掩膜尺寸和稳定输出文件名。
        else:
            # 使用 PIL 打开原掩膜以读取未经 Resize 的原始尺寸。
            with Image.open(self.gts[index]) as img:
                # PIL.size 顺序为 (width,height)。
                original_shape = img.size
            # 从输入路径提取基本文件名。
            name = os.path.basename(self.images[index])
            # JPG 输入的预测输出统一保存为 PNG 名称。
            if name.lower().endswith('.jpg'):
                # 只从最后一个点处分割，保留文件主体中的其他点。
                name = name.rsplit('.', 1)[0] + '.png'
            # 返回图像、缩放后二值 mask、原尺寸和结果名。
            return image, mask, original_shape, name

    # 返回过滤后的样本数量。
    def __len__(self):
        # 构造时已缓存 self.size。
        return self.size

# 根据配置创建 PolypDataset 和 PyTorch DataLoader。
def get_loader(image_root, gt_root, batchsize, trainsize, shuffle=False, num_workers=4, pin_memory=True, augmentation=False, split='train', color_image=True):
    # 实例化数据集并选择相应变换分支。
    dataset = PolypDataset(image_root, gt_root, trainsize, augmentation, split, color_image)
    # 直接返回数据加载器；批处理会自动堆叠图像和固定尺寸 mask。
    return data.DataLoader(dataset=dataset, batch_size=batchsize, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory)