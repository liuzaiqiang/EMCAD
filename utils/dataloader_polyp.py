# hashlib 生成当前图像/掩膜配对清单的 SHA-256 指纹，便于记录实验数据版本。
import hashlib
# random 在每个 DataLoader worker 中接收独立且可复现的种子。
import random
# pathlib.Path 提供目录检查、扩展名、文件名和跨平台路径拼接。
from pathlib import Path

# Albumentations 对图像与掩膜同步执行几何增强和输入归一化。
import albumentations as A
# OpenCV 按彩色、灰度或原始通道方式读取不同数据集文件。
import cv2
# NumPy 负责掩膜二值化、颜色通道聚合和 worker 随机种子。
import numpy as np
# PyTorch 负责张量、随机生成器和评测掩膜封装。
import torch
# data 提供 Dataset 与 DataLoader。
import torch.utils.data as data
# ToTensorV2 把 Albumentations 输出转换为 PyTorch CHW 张量。
from albumentations.pytorch import ToTensorV2

# 本加载器接受的图像/掩膜扩展名；比较时统一使用小写。
SUPPORTED_EXTENSIONS = {
    # 常见 JPEG 扩展名。
    ".jpg",
    # JPEG 的长扩展名。
    ".jpeg",
    # PNG 常用于无损掩膜。
    ".png",
    # TIFF 两种常见扩展名。
    ".tif",
    # TIFF 长扩展名。
    ".tiff",
    # BMP 兼容项。
    ".bmp",
}


# 扫描一个目录，并以不区分大小写的文件主体 stem 建立唯一索引。
def _index_by_stem(root):
    # 统一转换为 Path 对象。
    root = Path(root)
    # 数据根必须是已存在目录。
    if not root.is_dir():
        # 报告实际缺失目录。
        raise FileNotFoundError("Directory not found: {}".format(root))

    # 字典结构为 canonical_stem -> Path。
    indexed = {}
    # 按路径名称稳定排序遍历目录第一层。
    for path in sorted(root.iterdir()):
        # 只接受普通文件且扩展名属于支持集合。
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
            # casefold 比 lower 更适合不区分大小写的规范键。
            key = path.stem.casefold()
            # 同一目录中不同扩展名但同 stem 会造成图像/掩膜配对歧义。
            if key in indexed:
                # 报告冲突目录和两个实际文件名。
                raise RuntimeError(
                    # 构造详细错误信息。
                    "Duplicate file stem in {}: {} and {}".format(
                        # 填入根目录、已有项、新项。
                        root, indexed[key].name, path.name
                    )
                )
            # 注册该唯一 stem。
            indexed[key] = path

    # 没有任何支持文件通常说明路径或数据准备错误。
    if not indexed:
        # 立即终止，避免生成长度为 0 的 DataLoader。
        raise RuntimeError("No supported image files found in: {}".format(root))

    # 返回完整索引。
    return indexed


# DataLoader 子进程初始化函数，使 Python/NumPy 增强随机性既独立又可复现。
def _seed_worker(worker_id):
    # PyTorch 已根据主 generator 和 worker_id 派生 initial_seed；截到 NumPy 接受的 32 位范围。
    worker_seed = torch.initial_seed() % (2 ** 32)
    # 固定当前 worker 的 Python random。
    random.seed(worker_seed)
    # 固定当前 worker 的 NumPy random。
    np.random.seed(worker_seed)


# 评测专用 collate：只堆叠固定尺寸图像，保留不同原始尺寸的掩膜列表。
def polyp_eval_collate(batch):
    # 每个 item[0] 已缩放为相同 CHW，因此可沿 batch 维堆叠。
    images = torch.stack([item[0] for item in batch], dim=0)
    # 原始分辨率 mask 可能尺寸不同，保留为 Python 列表而不 stack。
    masks = [item[1] for item in batch]
    # 每项原始 [H,W] 尺寸张量可以堆叠为 [B,2]。
    original_sizes = torch.stack([item[2] for item in batch], dim=0)
    # 输出文件名保留字符串列表。
    names = [item[3] for item in batch]
    # 返回测试脚本预期的四元组。
    return images, masks, original_sizes, names


# 通用二分类息肉数据集，支持 ClinicDB/Kvasir/ColonDB/ETIS/BKAI 等掩膜形式。
class PolypDataset(data.Dataset):
    # image_root 与 gt_root 通过文件 stem 精确配对，不依赖目录枚举顺序。
    def __init__(
            # 当前数据集实例。
            self,
            # 输入图像目录。
            image_root,
            # 掩膜目录。
            gt_root,
            # 模型输入正方形边长。
            trainsize,
            # 是否启用训练随机增强。
            augmentation=False,
            # train、val 或 test。
            split="train",
            # True 读取 RGB，False 读取单通道灰度。
            color_image=True,
    ):
        # 限定合法划分，避免拼写错误静默进入评测分支。
        if split not in {"train", "val", "test"}:
            # 报告允许值。
            raise ValueError("split must be train, val, or test")

        # 统一为 int，供 Resize 使用。
        self.trainsize = int(trainsize)
        # 统一为布尔值。
        self.augmentation = bool(augmentation)
        # 保存划分模式。
        self.split = split
        # 保存颜色输入模式。
        self.color_image = bool(color_image)

        # 建立图像 stem 索引。
        images = _index_by_stem(image_root)
        # 建立掩膜 stem 索引。
        masks = _index_by_stem(gt_root)

        # 图像 ID 集合。
        image_keys = set(images)
        # 掩膜 ID 集合。
        mask_keys = set(masks)

        # 两边 stem 必须完全一一对应，防止排序错配或漏标。
        if image_keys != mask_keys:
            # 最多列出十个缺失掩膜的图像 ID。
            missing_masks = sorted(image_keys - mask_keys)[:10]
            # 最多列出十个缺失图像的掩膜 ID。
            missing_images = sorted(mask_keys - image_keys)[:10]
            # 抛出带两类差集的错误。
            raise RuntimeError(
                # 错误正文分两段拼接。
                "Image/mask stems do not match. "
                # 格式化缺失列表。
                "missing_masks={} missing_images={}".format(
                    # 插入两个方向的差集。
                    missing_masks, missing_images
                )
            )

        # 建立稳定排序的 (stem,image_path,mask_path) 样本表。
        self.samples = [
            # 同一 key 分别索引图像和掩膜路径。
            (key, images[key], masks[key])
            # 排序保证跨运行样本基序一致。
            for key in sorted(image_keys)
        ]
        # 只提取 stem 元组，供 BUSI manifest 一致性检查等外层逻辑使用。
        self.stems = tuple(key for key, _, _ in self.samples)

        # 初始化配对清单哈希。
        digest = hashlib.sha256()
        # 按稳定样本顺序更新哈希。
        for key, image_path, mask_path in self.samples:
            # 每个样本写入 stem 和两个文件名；这里不读取文件内容。
            digest.update(
                # 用制表符和换行构造无歧义记录。
                "{}\t{}\t{}\n".format(
                    # 插入规范 ID、图像名、掩膜名。
                    key, image_path.name, mask_path.name
                    # 统一编码为 UTF-8 字节后更新 SHA-256。
                ).encode("utf-8")
            )
        # 保存十六进制清单指纹，供实验记录数据版本。
        self.manifest_sha256 = digest.hexdigest()

        # 根据输入通道数选择归一化均值。
        mean = (
            # RGB 使用 ImageNet 预训练编码器统计量。
            (0.485, 0.456, 0.406)
            # 三通道模式条件。
            if self.color_image
            # 灰度模式使用单通道中心值 0.5。
            else (0.5,)
        )
        # 根据输入通道数选择归一化标准差。
        std = (
            # RGB ImageNet 标准差。
            (0.229, 0.224, 0.225)
            # 三通道模式条件。
            if self.color_image
            # 灰度模式沿用项目单通道标准差。
            else (0.229,)
        )

        # 逐步构造 Albumentations 变换列表。
        transforms = []

        # 只有训练集且明确启用增强时添加随机几何操作。
        if self.split == "train" and self.augmentation:
            # extend 保持后续 Resize/Normalize 位于随机变换之后。
            transforms.extend(
                # 随机变换列表。
                [
                    # 50% 概率在 ±90 度范围旋转图像和 mask。
                    A.Rotate(limit=90, p=0.5),
                    # 50% 概率上下翻转。
                    A.VerticalFlip(p=0.5),
                    # 50% 概率左右翻转。
                    A.HorizontalFlip(p=0.5),
                ]
            )

        # 所有划分都执行固定尺寸、归一化和张量化。
        transforms.extend(
            # 确定性尾部变换列表。
            [
                # 同步把图像和训练 mask 缩放到 trainsize。
                A.Resize(
                    # 目标高度。
                    height=self.trainsize,
                    # 目标宽度。
                    width=self.trainsize,
                ),
                # 只标准化图像通道，mask 保留类别值。
                A.Normalize(mean=mean, std=std),
                # 图像转 CHW，mask 转张量。
                ToTensorV2(),
            ]
        )

        # Compose 负责在调用时对 image/mask 同步采样随机参数。
        self.transform = A.Compose(transforms)

    # 返回图像/掩膜配对数量。
    def __len__(self):
        # self.samples 已在构造时完整校验配对。
        return len(self.samples)

    # 读取、二值化并预处理第 index 个样本。
    def __getitem__(self, index):
        # 解包规范 stem、输入路径和掩膜路径。
        key, image_path, mask_path = self.samples[index]

        # 根据 color_image 选择 OpenCV 读取标志。
        image_flag = (
            # 彩色模式读取 BGR 三通道。
            cv2.IMREAD_COLOR
            # 颜色模式条件。
            if self.color_image
            # 灰度模式读取二维数组。
            else cv2.IMREAD_GRAYSCALE
        )

        # image = cv2.imread(str(image_path), image_flag)
        # mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

        # if image is None:
        #     raise RuntimeError("Cannot read image: {}".format(image_path))
        # if mask is None:
        #     raise RuntimeError("Cannot read mask: {}".format(mask_path))

        # if self.color_image:
        #     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # if image.shape[:2] != mask.shape[:2]:
        #     raise RuntimeError(
        #         "Image/mask size mismatch for {}: image={} mask={}".format(
        #             key, image.shape[:2], mask.shape[:2]
        #         )
        #     )

        # if int(mask.max()) > 10:
        #     mask = (mask >= 128).astype(np.uint8)
        # else:
        #     mask = (mask > 0).astype(np.uint8)

        # 按选择的颜色模式读取输入图像。
        image = cv2.imread(
            # OpenCV 接受字符串路径。
            str(image_path),
            # 彩色或灰度标志。
            image_flag,
        )
        # 使用 IMREAD_UNCHANGED 保留掩膜原始灰度或颜色通道，兼容 BKAI 彩色标注。
        mask_raw = cv2.imread(
            # 掩膜文件路径。
            str(mask_path),
            # 不强制转灰度。
            cv2.IMREAD_UNCHANGED,
        )

        # OpenCV 读取失败时返回 None。
        if image is None:
            # 报告具体无法读取的输入路径。
            raise RuntimeError(
                # 格式化错误文本。
                "Cannot read image: {}".format(image_path)
            )

        # 掩膜读取失败同样立即终止。
        if mask_raw is None:
            # 报告具体掩膜路径。
            raise RuntimeError(
                # 格式化错误文本。
                "Cannot read mask: {}".format(mask_path)
            )

        # OpenCV 彩色读取顺序是 BGR，而 ImageNet 归一化与预训练编码器期望 RGB。
        if self.color_image:
            # 把 BGR 通道交换成 RGB。
            image = cv2.cvtColor(
                # 输入三通道数组。
                image,
                # BGR -> RGB 转换码。
                cv2.COLOR_BGR2RGB,
            )

        # 二维原始掩膜是 ClinicDB/Kvasir/ColonDB/ETIS 等常见灰度格式。
        if mask_raw.ndim == 2:
            # ClinicDB, Kvasir, ColonDB, ETIS:
            # preserve the original grayscale-mask behavior.
            # 直接保留二维灰度数组引用，随后按值域二值化。
            mask = mask_raw

            # 最大灰度大于 10 时视为 0/255 一类的 8 位掩膜。
            if int(mask.max()) > 10:
                # 使用 128 阈值获得 0/1 前景。
                mask = (
                    # 阈值比较产生布尔数组。
                        mask >= 128
                    # 转 uint8 便于 Albumentations 和 torch 转换。
                ).astype(np.uint8)
            # 最大值较小时按 0/1 或小整数标签处理。
            else:
                # 任何正值都合并为二分类前景。
                mask = (
                    # 大于 0 的位置为 True。
                        mask > 0
                    # 转 0/1 uint8。
                ).astype(np.uint8)

        # 三维且至少三通道时按彩色掩膜处理。
        elif (
                # 首先要求 HWC 三维布局。
                mask_raw.ndim == 3
                # 并至少包含 BGR 三个颜色通道。
                and mask_raw.shape[2] >= 3
        ):
            # BKAI uses red and green foreground labels.
            # Merge all foreground colors into one binary mask.
            # 对前三个颜色通道逐像素取最大值，把红/绿等任一非零标注合并为信号强度。
            mask_signal = np.max(
                # 忽略可能的 alpha 等额外通道。
                mask_raw[:, :, :3],
                # 在颜色通道轴归约，输出 [H,W]。
                axis=2,
            )

            # 强度不小于 128 的任何颜色标记视为前景。
            mask = (
                # 阈值产生布尔图。
                    mask_signal >= 128
                # 转 uint8 0/1。
            ).astype(np.uint8)

            # 彩色标注若二值化后完全为空，通常表示颜色规则或数据异常。
            if not np.any(mask):
                # 立即报告具体文件，避免空标签静默进入训练。
                raise RuntimeError(
                    # 错误正文跨行拼接。
                    "Empty color mask after binarization: "
                    # 插入掩膜路径。
                    "{}".format(mask_path)
                )

        # 其他维数/通道布局不在当前数据约定内。
        else:
            # 报告文件及实际 shape。
            raise RuntimeError(
                # 构造详细错误信息。
                "Unsupported mask shape for {}: {}".format(
                    # 插入路径。
                    mask_path,
                    # 插入原始掩膜形状。
                    mask_raw.shape,
                )
            )

        # 图像与二值化掩膜必须在增强前就逐像素对齐。
        if image.shape[:2] != mask.shape[:2]:
            # 形状不一致时拒绝自动拉伸掩盖数据问题。
            raise RuntimeError(
                # 错误正文。
                "Image/mask size mismatch for {}: "
                # 格式化 ID 和双方空间 shape。
                "image={} mask={}".format(
                    # 样本规范键。
                    key,
                    # 图像 H,W。
                    image.shape[:2],
                    # 掩膜 H,W。
                    mask.shape[:2],
                )
            )

        # 训练分支对图像和 mask 同步执行完整 Compose。
        if self.split == "train":
            # 几何增强、Resize、图像Normalize和张量化。
            transformed = self.transform(
                # 输入图像。
                image=image,
                # 同步输入掩膜。
                mask=mask,
            )
            # 图像显式转 float，形状通常 [C,trainsize,trainsize]。
            image_tensor = transformed["image"].float()
            # 掩膜显式转 float 以供 BCE/IoU 损失使用。
            mask_tensor = transformed["mask"].float()

            # ToTensorV2 对二维 mask 可能返回 [H,W]。
            if mask_tensor.ndim == 2:
                # 增加单通道维成 [1,H,W]。
                mask_tensor = mask_tensor.unsqueeze(0)

            # 训练脚本按 image,gt 二元组解包。
            return image_tensor, mask_tensor

        # 验证/测试只对图像执行 Resize/Normalize，不缩放原始 GT。
        transformed = self.transform(image=image)
        # 取模型输入张量。
        image_tensor = transformed["image"].float()

        # Validation/test 保留原始分辨率 GT，避免先缩放后再放大。
        # 原始二值 mask 增加通道维 [1,H,W] 并转 float。
        mask_tensor = torch.from_numpy(mask).unsqueeze(0).float()
        # 保存原始 H,W，测试脚本把预测 logits 插值回该尺寸。
        original_size = torch.tensor(
            # NumPy mask.shape 顺序即 H,W。
            mask.shape,
            # 尺寸索引用 int64。
            dtype=torch.int64,
        )
        # 所有预测输出统一使用 PNG 后缀，主体来自原图文件 stem。
        output_name = "{}.png".format(Path(image_path).stem)

        # 返回可由 polyp_eval_collate 组织的四元组。
        return (
            # 固定尺寸模型输入。
            image_tensor,
            # 原始尺寸真值。
            mask_tensor,
            # [H,W] 尺寸。
            original_size,
            # 结果文件名。
            output_name,
        )


# 构造带确定性随机种子的通用息肉 DataLoader。
def get_loader(
        # 输入图像目录。
        image_root,
        # 掩膜目录。
        gt_root,
        # 每批样本数。
        batchsize,
        # 模型输入尺寸。
        trainsize,
        # 是否打乱样本。
        shuffle=False,
        # 数据加载子进程数。
        num_workers=4,
        # 是否启用锁页内存。
        pin_memory=True,
        # 训练随机增强开关。
        augmentation=False,
        # train/val/test。
        split="train",
        # RGB 或灰度输入。
        color_image=True,
        # 主随机种子。
        seed=2222,
):
    # 创建经过严格 stem 配对的数据集。
    dataset = PolypDataset(
        # 传入图像根目录。
        image_root=image_root,
        # 传入掩膜根目录。
        gt_root=gt_root,
        # 传入目标尺寸。
        trainsize=trainsize,
        # 传入增强开关。
        augmentation=augmentation,
        # 传入数据划分。
        split=split,
        # 传入通道模式。
        color_image=color_image,
    )

    # 创建独立 PyTorch Generator，控制 DataLoader 打乱和 worker 初始种子。
    generator = torch.Generator()
    # 固定生成器种子。
    generator.manual_seed(int(seed))

    # 返回配置好的 PyTorch DataLoader。
    return data.DataLoader(
        # 数据集实例。
        dataset=dataset,
        # 批大小转 int。
        batch_size=int(batchsize),
        # 打乱开关转 bool。
        shuffle=bool(shuffle),
        # worker 数转 int。
        num_workers=int(num_workers),
        # 锁页内存开关。
        pin_memory=bool(pin_memory),
        # 每个 worker 启动时同步 Python/NumPy 种子。
        worker_init_fn=_seed_worker,
        # 主生成器控制可复现顺序。
        generator=generator,
        # 训练样本固定尺寸可用默认 collate；评测需保留不同尺寸 mask。
        collate_fn=(
            # 训练时使用 PyTorch 默认堆叠。
            None
            # 根据 split 选择分支。
            if split == "train"
            # 验证/测试使用自定义四元组整理函数。
            else polyp_eval_collate
        ),
        # 有 worker 时保持进程跨 epoch 存活，0 worker 时必须为 False。
        persistent_workers=int(num_workers) > 0,
    )
