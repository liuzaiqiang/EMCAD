# os 用于枚举图像/掩膜目录并构造文件列表。
import os
# PIL.Image 负责读取 RGB 图像、灰度掩膜和尺寸检查。
from PIL import Image
# torch.utils.data 提供 Dataset 与 DataLoader。
import torch.utils.data as data
# torchvision.transforms 组织旋转、翻转、缩放、张量化和标准化。
import torchvision.transforms as transforms
# NumPy 随机数用于生成同步图像/掩膜增强种子。
import numpy as np
# Python random 接收该种子并驱动 torchvision 的随机变换。
import random
# torch.manual_seed 同步由 PyTorch 随机源驱动的 torchvision 操作。
import torch


# 早期息肉二分类数据集：按排序后的文件列表配对图像和掩膜。
class PolypDataset(data.Dataset):
    """
    dataloader for polyp segmentation tasks
    """
    # image_root/gt_root 需以路径分隔符结尾，因为原代码用字符串加法拼文件名。
    def __init__(self, image_root, gt_root, trainsize, augmentations):
        # 保存网络训练输入边长。
        self.trainsize = trainsize
        # 保存增强开关；原实现比较字符串 'True'，而非布尔 True。
        self.augmentations = augmentations
        # 启动时输出收到的增强配置，便于排查参数类型。
        print(self.augmentations)
        # 收集 JPG/PNG 输入图像完整路径。
        self.images = [image_root + f for f in os.listdir(image_root) if f.endswith('.jpg') or f.endswith('.png')]
        # 收集 PNG/JPG 掩膜完整路径。
        self.gts = [gt_root + f for f in os.listdir(gt_root) if f.endswith('.png') or f.endswith('.jpg')]
        # 按路径字典序排序，依赖图像和掩膜同名实现配对。
        self.images = sorted(self.images)
        # 掩膜使用相同排序规则。
        self.gts = sorted(self.gts)
        # 删除尺寸不一致的图像/掩膜对。
        self.filter_files()
        # 缓存过滤后的样本数。
        self.size = len(self.images)
        # 只有 augmentations 字符串严格等于 'True' 时启用随机增强。
        if self.augmentations == 'True':
            # 打印实际启用的增强类别。
            print('Using RandomRotation, RandomFlip')
            # 输入图像变换流水线；随机变换稍后通过种子与掩膜同步。
            self.img_transform = transforms.Compose([
                # 随机旋转角度范围由 torchvision 对数值 90 的解释决定；保持原参数。
                transforms.RandomRotation(90, resample=False, expand=False, center=None, fill=None),
                # 50% 概率上下翻转。
                transforms.RandomVerticalFlip(p=0.5),
                # 50% 概率左右翻转。
                transforms.RandomHorizontalFlip(p=0.5),
                # 缩放到固定正方形训练尺寸。
                transforms.Resize((self.trainsize, self.trainsize)),
                # PIL 图像转 [C,H,W]、[0,1] 浮点张量。
                transforms.ToTensor(),
                # 使用 ImageNet RGB 均值和标准差标准化，以匹配预训练编码器输入分布。
                transforms.Normalize([0.485, 0.456, 0.406],
                                     # 三个通道对应标准差。
                                     [0.229, 0.224, 0.225])])
            # 掩膜使用独立 Compose，但会在 __getitem__ 中重置相同随机种子。
            self.gt_transform = transforms.Compose([
                # 与图像配置相同的随机旋转。
                transforms.RandomRotation(90, resample=False, expand=False, center=None, fill=None),
                # 与图像相同概率的垂直翻转。
                transforms.RandomVerticalFlip(p=0.5),
                # 与图像相同概率的水平翻转。
                transforms.RandomHorizontalFlip(p=0.5),
                # 把掩膜缩放到训练尺寸；原代码未显式指定最近邻插值。
                transforms.Resize((self.trainsize, self.trainsize)),
                # 灰度掩膜转 [1,H,W] 浮点张量。
                transforms.ToTensor()])
            
        # 不启用随机增强时只做确定性缩放、张量化和图像标准化。
        else:
            # 输出当前关闭增强。
            print('no augmentation')
            # 图像确定性变换。
            self.img_transform = transforms.Compose([
                # 固定输入尺寸。
                transforms.Resize((self.trainsize, self.trainsize)),
                # 转张量。
                transforms.ToTensor(),
                # ImageNet 标准化。
                transforms.Normalize([0.485, 0.456, 0.406],
                                     # RGB 标准差。
                                     [0.229, 0.224, 0.225])])
            
            # 掩膜只缩放并转张量。
            self.gt_transform = transforms.Compose([
                # 固定掩膜空间尺寸。
                transforms.Resize((self.trainsize, self.trainsize)),
                # 转 [1,H,W] 浮点张量。
                transforms.ToTensor()])
            

    # 读取、同步增强并返回第 index 个图像/掩膜对。
    def __getitem__(self, index):
        
        # RGB 读取输入图像。
        image = self.rgb_loader(self.images[index])
        # 单通道灰度读取二值掩膜。
        gt = self.binary_loader(self.gts[index])
        
        # 为当前样本生成一个足够大的整数种子。
        seed = np.random.randint(2147483647) # make a seed with numpy generator 
        # 在图像变换前固定 Python 随机状态。
        random.seed(seed) # apply this seed to img tranfsorms
        # 同时固定 PyTorch 随机状态，兼容 torchvision 0.7 的随机实现。
        torch.manual_seed(seed) # needed for torchvision 0.7
        # 已配置图像流水线时执行它。
        if self.img_transform is not None:
            # 得到标准化输入张量。
            image = self.img_transform(image)
            
        # 在掩膜变换前恢复完全相同的 Python 随机状态。
        random.seed(seed) # apply this seed to img tranfsorms
        # 恢复同一 PyTorch 随机状态，使旋转和翻转参数与图像一致。
        torch.manual_seed(seed) # needed for torchvision 0.7
        # 已配置掩膜变换时执行。
        if self.gt_transform is not None:
            # 得到与图像几何对齐的掩膜张量。
            gt = self.gt_transform(gt)
        # 二分类训练代码按二元组解包。
        return image, gt

    # 过滤图像/掩膜数量不一致和空间尺寸不一致的配对。
    def filter_files(self):
        # 排序后的两个列表必须先具有相同长度。
        assert len(self.images) == len(self.gts)
        # 保存有效图像路径。
        images = []
        # 保存对应有效掩膜路径。
        gts = []
        # 按相同索引遍历已排序路径。
        for img_path, gt_path in zip(self.images, self.gts):
            # 读取图像头信息。
            img = Image.open(img_path)
            # 读取掩膜头信息。
            gt = Image.open(gt_path)
            # 只有宽高完全一致才保留该对。
            if img.size == gt.size:
                # 添加输入图像路径。
                images.append(img_path)
                # 添加同索引掩膜路径。
                gts.append(gt_path)
        # 用过滤结果替换原图像列表。
        self.images = images
        # 用过滤结果替换原掩膜列表。
        self.gts = gts

    # 以二进制模式打开文件并强制转换成三通道 RGB PIL 图像。
    def rgb_loader(self, path):
        # with 确保底层文件句柄在返回后关闭。
        with open(path, 'rb') as f:
            # PIL 延迟读取文件内容。
            img = Image.open(f)
            # convert('RGB') 统一灰度/RGBA 等输入为三通道。
            return img.convert('RGB')

    # 以二进制模式打开掩膜并转换为单通道灰度。
    def binary_loader(self, path):
        # 自动关闭文件句柄。
        with open(path, 'rb') as f:
            # 打开掩膜图像。
            img = Image.open(f)
            # return img.convert('1')
            # L 模式保留 0..255 灰度；ToTensor 后通常映射到 0..1。
            return img.convert('L')

    # 旧的极坐标变换实验入口；polar_transformations 未在本文件导入，主训练路径不调用该方法。
    def convert2polar(self, img, gt):
    
	    # 根据掩膜计算极坐标中心。
    	center = polar_transformations.centroid(gt)
	    # 把输入图像转换到该中心定义的极坐标。
    	img = polar_transformations.to_polar(img, center)
	    # 标签使用完全相同中心转换。
    	gt = polar_transformations.to_polar(gt, center)
    	
	    # 返回转换后的图像和掩膜。
    	return img, gt
            #center_max_shift = 0.05 * LesionDataset.height
            #center = np.array(center)
            #center = (
               #center[0] + np.random.uniform(-center_max_shift, center_max_shift),
               #center[1] + np.random.uniform(-center_max_shift, center_max_shift))
    ## to PyTorch expected format
    #input = input.transpose(2, 0, 1)
    #label = np.expand_dims(label, axis=-1)
    #label = label.transpose(2, 0, 1)

    #input_tensor = torch.from_numpy(input)
    
    # 仅在图像任一边短于 trainsize 时等比例下限式放大到至少训练尺寸。
    def resize(self, img, gt):
        # 输入和掩膜必须原始尺寸一致。
        assert img.size == gt.size
        # PIL.size 顺序为宽、高。
        w, h = img.size
        # 任一边过小时触发放大。
        if h < self.trainsize or w < self.trainsize:
            # 高至少为 trainsize。
            h = max(h, self.trainsize)
            # 宽至少为 trainsize。
            w = max(w, self.trainsize)
            # 连续图像用双线性，离散掩膜用最近邻，避免掩膜灰度混合。
            return img.resize((w, h), Image.BILINEAR), gt.resize((w, h), Image.NEAREST)
        # 已足够大时不做处理。
        else:
            # 返回原对象。
            return img, gt

    # 返回过滤后的训练样本数。
    def __len__(self):
        # self.size 在构造函数中缓存。
        return self.size


# 构造训练 DataLoader；augmentation 原样传入 PolypDataset。
def get_loader(image_root, gt_root, batchsize, trainsize, shuffle=False, num_workers=4, pin_memory=True, augmentation=False): #shuffle=True

    # 实例化数据集并完成文件过滤和变换配置。
    dataset = PolypDataset(image_root, gt_root, trainsize, augmentation)
    # 按调用方批大小、打乱、worker 和锁页内存配置创建加载器。
    data_loader = data.DataLoader(dataset=dataset,
                                  # 每批样本数。
                                  batch_size=batchsize,
                                  # 是否打乱样本顺序。
                                  shuffle=shuffle,
                                  # 后台数据进程数。
                                  num_workers=num_workers,
                                  # CUDA 训练时锁页内存可加快主机到显卡传输。
                                  pin_memory=pin_memory)
    # 返回可迭代 DataLoader。
    return data_loader


# 旧式测试数据迭代器：内部维护 index，每次 load_data 返回一例。
class test_dataset:
    # image_root、gt_root 分别指向测试图像和掩膜目录。
    def __init__(self, image_root, gt_root, testsize):
        # 保存模型测试输入边长。
        self.testsize = testsize
        # 收集 JPG/PNG 输入图像并拼接完整路径。
        self.images = [image_root + f for f in os.listdir(image_root) if f.endswith('.jpg') or f.endswith('.png')]
        # 收集 TIFF/PNG 掩膜；原表达式中的 '.png' or ... 实际先求值为 '.png'，保持原逻辑。
        self.gts = [gt_root + f for f in os.listdir(gt_root) if f.endswith('.tif') or f.endswith('.png' or f.endswith('.jpg'))]
        # 图像按路径排序。
        self.images = sorted(self.images)
        # 掩膜按路径排序，依赖同名顺序配对。
        self.gts = sorted(self.gts)
        # 输入图像确定性预处理流水线。
        self.transform = transforms.Compose([
            # 缩放到测试尺寸。
            transforms.Resize((self.testsize, self.testsize)),
            # 转 [C,H,W] 浮点张量。
            transforms.ToTensor(),
            # 使用 ImageNet RGB 统计标准化。
            transforms.Normalize([0.485, 0.456, 0.406],
                                 # RGB 标准差。
                                 [0.229, 0.224, 0.225])])
        # 测试真值只转张量，不缩放，便于按原始尺寸评测/保存。
        self.gt_transform = transforms.ToTensor()
        # 缓存测试样本数量。
        self.size = len(self.images)
        # 初始化手动迭代索引。
        self.index = 0

    # 读取当前 index 样本并把索引推进一位。
    def load_data(self):
        # 读取 RGB 输入。
        image = self.rgb_loader(self.images[self.index])
        # 预处理后增加 batch 维，得到 [1,3,H,W]。
        image = self.transform(image).unsqueeze(0)
        # 读取原始尺寸灰度真值。
        gt = self.binary_loader(self.gts[self.index])
        # 用正斜杠拆路径取得文件名；这是原代码的跨平台假设。
        name = self.images[self.index].split('/')[-1]
        # 统一把 JPG 结果名改为 PNG 后缀。
        if name.endswith('.jpg'):
            # 去掉 .jpg 再追加 .png。
            name = name.split('.jpg')[0] + '.png'
        # 为下一次 load_data 推进索引。
        self.index += 1
        # 返回模型输入、原尺寸真值 PIL 图像和结果文件名。
        return image, gt, name

    # RGB 文件读取辅助方法。
    def rgb_loader(self, path):
        # 二进制打开并自动关闭句柄。
        with open(path, 'rb') as f:
            # 读取 PIL 图像。
            img = Image.open(f)
            # 统一成 RGB。
            return img.convert('RGB')

    # 灰度掩膜读取辅助方法。
    def binary_loader(self, path):
        # 二进制打开并自动关闭句柄。
        with open(path, 'rb') as f:
            # 读取 PIL 掩膜。
            img = Image.open(f)
            # 转单通道 L 模式。
            return img.convert('L')
