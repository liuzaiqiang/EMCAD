# hashlib 为当前 ISIC 样本清单生成可复现实验指纹。
import hashlib
# random 在 DataLoader worker 中接收派生种子。
import random
# Path 用于目录扫描、扩展名、stem、文件大小和输出名称。
from pathlib import Path

# Albumentations 同步处理皮肤镜图像与病灶掩膜。
import albumentations as A
# OpenCV 读取 JPEG 图像和 PNG 灰度掩膜。
import cv2
# NumPy 负责阈值二值化和 worker 随机种子。
import numpy as np
# PyTorch 负责张量、DataLoader 随机生成器和原始尺寸封装。
import torch
# data 提供 Dataset 与 DataLoader。
import torch.utils.data as data
# ToTensorV2 把增强后的 HWC/二维数组转为 PyTorch 张量。
from albumentations.pytorch import ToTensorV2


# ISIC 原始输入图像允许 JPG/JPEG。
IMAGE_EXTENSIONS = {".jpg", ".jpeg"}
# ISIC 分割真值使用 PNG。
MASK_EXTENSIONS = {".png"}
# 对外暴露的支持扩展名沿用图像集合。
SUPPORTED_EXTENSIONS = IMAGE_EXTENSIONS
# 官方掩膜名通常在图像 ID 后追加 _segmentation。
MASK_SUFFIX = "_segmentation"


# 把图像或掩膜路径转换成相同的规范样本 ID。
def _canonical_id(path, is_mask):
    # Path.stem 去掉最后一个扩展名。
    stem = Path(path).stem

    # 掩膜才需要去除 _segmentation 后缀。
    if (
        # 当前目录索引的是 mask。
        is_mask
        # 不区分大小写检查官方后缀。
        and stem.casefold().endswith(MASK_SUFFIX)
    ):
        # 从原 stem 末尾切除后缀长度，保留图像 ID。
        stem = stem[:-len(MASK_SUFFIX)]

    # casefold 后返回，避免大小写差异破坏配对。
    return stem.casefold()


# 扫描图像或掩膜目录并建立 canonical_id -> Path 唯一索引。
def _index_directory(root, is_mask):
    # 转成 Path 对象。
    root = Path(root)

    # 数据根必须是目录。
    if not root.is_dir():
        # 报告具体路径。
        raise FileNotFoundError(
            # 格式化错误正文。
            "Directory not found: {}".format(root)
        )

    # 根据当前索引的是 mask 还是 image 选择扩展名白名单。
    extensions = (
        # 掩膜只接受 PNG。
        MASK_EXTENSIONS
        # 条件分支。
        if is_mask
        # 图像只接受 JPG/JPEG。
        else IMAGE_EXTENSIONS
    )

    # 保存规范 ID 到实际文件路径。
    indexed = {}

    # 稳定排序遍历目录第一层。
    for path in sorted(root.iterdir()):
        # 跳过子目录和不支持扩展名。
        if (
            # 必须是普通文件。
            not path.is_file()
            # 或扩展名不在当前白名单时跳过。
            or path.suffix.lower() not in extensions
        ):
            # 继续下一个目录项。
            continue

        # 为图像/掩膜计算可配对的统一 ID。
        sample_id = _canonical_id(
            # 当前文件路径。
            path,
            # 告知是否需要去掩膜后缀。
            is_mask=is_mask,
        )

        # 空 ID 表示命名无有效主体。
        if not sample_id:
            # 报告具体异常文件。
            raise RuntimeError(
                # 格式化错误。
                "Empty sample ID for: {}".format(path)
            )

        # 同一规范 ID 只能对应一个文件。
        if sample_id in indexed:
            # 报告大小写/后缀归一化后发生冲突的两个文件。
            raise RuntimeError(
                # 错误正文。
                "Duplicate canonical ISIC ID in {}: "
                # 填入根目录和文件名。
                "{} and {}".format(
                    # 根目录。
                    root,
                    # 已注册文件名。
                    indexed[sample_id].name,
                    # 新冲突文件名。
                    path.name,
                )
            )

        # 注册唯一 ID。
        indexed[sample_id] = path

    # 目录中没有任何合法文件时提前失败。
    if not indexed:
        # 防止训练生成空 DataLoader。
        raise RuntimeError(
            # 错误正文。
            "No supported files found in: {}".format(
                # 插入根目录。
                root
            )
        )

    # 返回索引字典。
    return indexed


# 初始化 DataLoader 子进程的 Python 和 NumPy 随机状态。
def _seed_worker(worker_id):
    # PyTorch 根据主 Generator/worker_id 生成 64 位种子，再映射到 32 位。
    worker_seed = (
        # initial_seed 在每个 worker 中不同。
        torch.initial_seed() % (2 ** 32)
    )
    # 固定 Python random。
    random.seed(worker_seed)
    # 固定 NumPy random。
    np.random.seed(worker_seed)


# ISIC 验证/测试整理函数：图像可堆叠，原尺寸 mask 保持列表。
def isic_eval_collate(batch):
    # 固定 trainsize 的图像张量沿 batch 维堆叠。
    images = torch.stack(
        # 取每个样本第0项图像。
        [item[0] for item in batch],
        # 新维度放在最前面。
        dim=0,
    )
    # 不同原始分辨率 mask 不可直接 stack，保留列表。
    masks = [item[1] for item in batch]
    # [H,W] 尺寸张量可堆叠成 [B,2]。
    original_sizes = torch.stack(
        # 取每项第2项尺寸。
        [item[2] for item in batch],
        # batch 维。
        dim=0,
    )
    # 输出 PNG 名称字符串列表。
    names = [item[3] for item in batch]

    # 返回测试脚本预期四元组。
    return (
        # [B,3,trainsize,trainsize]。
        images,
        # 长度B的原尺寸掩膜列表。
        masks,
        # [B,2] 尺寸。
        original_sizes,
        # 长度B名称列表。
        names,
    )


# ISIC 皮肤病灶二分类数据集。
class ISICDataset(data.Dataset):
    # 构造函数参数逐行展开，便于命令行加载器传递。
    def __init__(
        # 当前实例。
        self,
        # JPG/JPEG 图像目录。
        image_root,
        # *_segmentation.png 掩膜目录。
        gt_root,
        # 模型输入边长。
        trainsize,
        # 训练几何增强开关。
        augmentation=False,
        # train/val/test。
        split="train",
        # ISIC 只支持 True；保留参数与通用加载器接口一致。
        color_image=True,
    ):
        # 限定合法划分。
        if split not in {"train", "val", "test"}:
            # 防止拼写错误落入评测分支。
            raise ValueError(
                # 报告合法值。
                "split must be train, val, or test"
            )

        # 预训练编码器路径要求 RGB 输入，拒绝灰度模式。
        if not color_image:
            # 明确说明不支持原因。
            raise ValueError(
                # 错误正文第一段。
                "ISIC must be loaded as RGB; "
                # 错误正文第二段。
                "grayscale is unsupported"
            )

        # 统一目标尺寸为 int。
        self.trainsize = int(trainsize)
        # 统一增强配置为 bool。
        self.augmentation = bool(augmentation)
        # 保存划分。
        self.split = split
        # 通过上面的校验后固定为 RGB 模式。
        self.color_image = True

        # 扫描并索引输入图像。
        images = _index_directory(
            # 图像目录。
            image_root,
            # 不去除 _segmentation 后缀。
            is_mask=False,
        )
        # 扫描并索引掩膜。
        masks = _index_directory(
            # 掩膜目录。
            gt_root,
            # 去除 _segmentation 以与图像 ID 配对。
            is_mask=True,
        )

        # 图像规范 ID 集合。
        image_ids = set(images)
        # 掩膜规范 ID 集合。
        mask_ids = set(masks)

        # 要求集合完全相同，防止静默错配。
        if image_ids != mask_ids:
            # 最多展示20个缺失掩膜 ID。
            missing_masks = sorted(
                # 图像有而掩膜无。
                image_ids - mask_ids
            )[:20]
            # 最多展示20个缺失图像 ID。
            missing_images = sorted(
                # 掩膜有而图像无。
                mask_ids - image_ids
            )[:20]

            # 抛出配对错误。
            raise RuntimeError(
                # 错误正文第一段。
                "ISIC image/mask IDs do not match. "
                # 插入两个方向的差集。
                "missing_masks={} missing_images={}".format(
                    # 缺失掩膜列表。
                    missing_masks,
                    # 缺失图像列表。
                    missing_images,
                )
            )

        # 建立稳定排序样本三元组。
        self.samples = [
            # 保存 ID、图像 Path、掩膜 Path。
            (
                # 规范样本 ID。
                sample_id,
                # 对应输入图像。
                images[sample_id],
                # 对应分割掩膜。
                masks[sample_id],
            )
            # 按 ID 排序保证跨运行顺序稳定。
            for sample_id in sorted(image_ids)
        ]

        # 提取全部样本 ID 元组。
        self.stems = tuple(
            # 每个三元组的第一项。
            sample_id
            # 遍历样本表并忽略两条路径。
            for sample_id, _, _ in self.samples
        )
        # 对外统一别名，便于实验记录代码读取。
        self.sample_ids = self.stems

        # 初始化样本清单 SHA-256。
        digest = hashlib.sha256()

        # 清单包含 ID、文件名和字节大小，可检测文件替换/截断。
        for (
            # 规范 ID。
            sample_id,
            # 图像路径。
            image_path,
            # 掩膜路径。
            mask_path,
        ) in self.samples:
            # 把当前记录编码后更新哈希。
            digest.update(
                # 使用制表符分隔五个字段，换行分隔样本。
                (
                    # 字段顺序：ID、图像名、图像字节、掩膜名、掩膜字节。
                    "{}\t{}\t{}\t{}\t{}\n"
                ).format(
                    # 规范 ID。
                    sample_id,
                    # 图像文件名。
                    image_path.name,
                    # 图像字节数。
                    image_path.stat().st_size,
                    # 掩膜文件名。
                    mask_path.name,
                    # 掩膜字节数。
                    mask_path.stat().st_size,
                # 转 UTF-8 字节。
                ).encode("utf-8")
            )

        # 保存最终清单指纹。
        self.manifest_sha256 = digest.hexdigest()

        # 构造 Albumentations 变换列表。
        transforms = []

        # 只有训练且启用 augmentation 时添加随机几何变换。
        if (
            # 当前划分为训练。
            self.split == "train"
            # 且增强开关为真。
            and self.augmentation
        ):
            # 追加三类同步几何增强。
            transforms.extend(
                # 随机变换列表。
                [
                    # 50% 概率在 ±90 度范围旋转。
                    A.Rotate(
                        # 最大角度绝对值。
                        limit=90,
                        # 应用概率。
                        p=0.5,
                    ),
                    # 50% 垂直翻转。
                    A.VerticalFlip(p=0.5),
                    # 50% 水平翻转。
                    A.HorizontalFlip(p=0.5),
                ]
            )

        # 所有划分追加固定尺寸、ImageNet 标准化和张量化。
        transforms.extend(
            # 确定性尾部变换列表。
            [
                # 同步 Resize 图像和训练 mask。
                A.Resize(
                    # 目标高。
                    height=self.trainsize,
                    # 目标宽。
                    width=self.trainsize,
                ),
                # 对 RGB 图像应用 ImageNet 均值/标准差。
                A.Normalize(
                    # RGB 均值元组。
                    mean=(
                        # R 均值。
                        0.485,
                        # G 均值。
                        0.456,
                        # B 均值。
                        0.406,
                    ),
                    # RGB 标准差元组。
                    std=(
                        # R 标准差。
                        0.229,
                        # G 标准差。
                        0.224,
                        # B 标准差。
                        0.225,
                    ),
                ),
                # 转 PyTorch 张量。
                ToTensorV2(),
            ]
        )

        # Compose 统一执行并同步 image/mask 几何随机参数。
        self.transform = A.Compose(transforms)

    # 返回严格配对后的 ISIC 样本数。
    def __len__(self):
        # 每个 self.samples 元素是一对图像和掩膜。
        return len(self.samples)

    # 读取并预处理第 index 个皮肤病灶样本。
    def __getitem__(self, index):
        # 解包规范 ID、图像 Path 和掩膜 Path。
        (
            # 规范样本 ID。
            sample_id,
            # 输入图像路径。
            image_path,
            # 分割真值路径。
            mask_path,
        ) = self.samples[index]

        # 以彩色 BGR 模式读取皮肤镜图像。
        image = cv2.imread(
            # Path 转字符串。
            str(image_path),
            # 强制三通道颜色读取。
            cv2.IMREAD_COLOR,
        )
        # 以单通道灰度模式读取 PNG 掩膜。
        mask = cv2.imread(
            # 掩膜路径字符串。
            str(mask_path),
            # 强制二维灰度读取。
            cv2.IMREAD_GRAYSCALE,
        )

        # OpenCV 读取失败返回 None。
        if image is None:
            # 报告具体输入路径。
            raise RuntimeError(
                # 格式化错误正文。
                "Cannot read ISIC image: {}".format(
                    # 插入图像路径。
                    image_path
                )
            )

        # 掩膜读取失败也立即终止。
        if mask is None:
            # 报告具体掩膜路径。
            raise RuntimeError(
                # 格式化错误正文。
                "Cannot read ISIC mask: {}".format(
                    # 插入掩膜路径。
                    mask_path
                )
            )

        # OpenCV 默认 BGR，转换为 ImageNet 预训练编码器期望的 RGB 顺序。
        image = cv2.cvtColor(
            # 输入 BGR 图像。
            image,
            # BGR 到 RGB 转换码。
            cv2.COLOR_BGR2RGB,
        )

        # 增强前先验证图像和掩膜空间尺寸一致。
        if image.shape[:2] != mask.shape[:2]:
            # 不自动修正原始错配，避免掩盖数据准备问题。
            raise RuntimeError(
                # 错误正文第一段。
                "Image/mask size mismatch for {}: "
                # 插入 ID 与双方 H,W。
                "image={} mask={}".format(
                    # 样本 ID。
                    sample_id,
                    # 图像高宽。
                    image.shape[:2],
                    # 掩膜高宽。
                    mask.shape[:2],
                )
            )

        # 官方掩膜通常为 0/255；阈值 128 转成 uint8 0/1。
        mask = (mask >= 128).astype(np.uint8)

        # 训练分支需要同步增强并把图像、mask 都缩放到固定尺寸。
        if self.split == "train":
            # Compose 同时接收 image 与 mask。
            transformed = self.transform(
                # RGB 图像。
                image=image,
                # 二值掩膜。
                mask=mask,
            )

            # 取标准化后的 CHW 图像并确保 float。
            image_tensor = transformed[
                # Albumentations 输出键。
                "image"
            ].float()
            # 取同步变换后的 mask 并转 float 供 BCE/IoU 损失使用。
            mask_tensor = transformed[
                # Albumentations 输出键。
                "mask"
            ].float()

            # 二维 mask 需要增加单通道维。
            if mask_tensor.ndim == 2:
                # [H,W] -> [1,H,W]。
                mask_tensor = mask_tensor.unsqueeze(0)

            # 训练脚本按二元组解包。
            return image_tensor, mask_tensor

        # 验证/测试只变换图像，真值保持原始分辨率。
        transformed = self.transform(
            # 不传 mask，避免其被 Resize。
            image=image
        )

        # 模型输入为固定尺寸标准化 CHW float 张量。
        image_tensor = transformed[
            # 取图像键。
            "image"
        ].float()

        # 原始二值 mask 从 NumPy 转张量并增加通道维。
        mask_tensor = (
            # 零拷贝包装 NumPy 数组。
            torch.from_numpy(mask)
            # [H,W] -> [1,H,W]。
            .unsqueeze(0)
            # 转 float。
            .float()
        )

        # 保存原始 H,W 以便评测时把预测插值回真实尺寸。
        original_size = torch.tensor(
            # mask.shape 顺序即高、宽。
            mask.shape,
            # 尺寸索引使用 int64。
            dtype=torch.int64,
        )

        # 预测结果名使用原图 stem 并统一 PNG 后缀。
        output_name = "{}.png".format(
            # Path.stem 去掉 JPG/JPEG 后缀。
            Path(image_path).stem
        )

        # 返回自定义 eval collate 所需四元组。
        return (
            # 固定尺寸模型输入。
            image_tensor,
            # 原始分辨率真值。
            mask_tensor,
            # [H,W] 尺寸。
            original_size,
            # PNG 输出名。
            output_name,
        )


# 构造具有确定性 worker 种子的 ISIC DataLoader。
def get_loader(
    # 图像目录。
    image_root,
    # 掩膜目录。
    gt_root,
    # 批大小。
    batchsize,
    # 模型输入尺寸。
    trainsize,
    # 是否打乱。
    shuffle=False,
    # worker 数量。
    num_workers=4,
    # 锁页内存开关。
    pin_memory=True,
    # 训练增强开关。
    augmentation=False,
    # train/val/test。
    split="train",
    # 必须为 True。
    color_image=True,
    # DataLoader 主随机种子。
    seed=2222,
):
    # 创建严格图像/掩膜 ID 配对的数据集。
    dataset = ISICDataset(
        # 图像目录。
        image_root=image_root,
        # 掩膜目录。
        gt_root=gt_root,
        # 目标输入尺寸。
        trainsize=trainsize,
        # 增强开关。
        augmentation=augmentation,
        # 数据划分。
        split=split,
        # RGB 模式。
        color_image=color_image,
    )

    # 独立生成器控制 shuffle 和 worker 初始种子。
    generator = torch.Generator()
    # 固定主种子。
    generator.manual_seed(int(seed))

    # 返回配置好的 DataLoader。
    return data.DataLoader(
        # 数据集实例。
        dataset=dataset,
        # 批大小。
        batch_size=int(batchsize),
        # 打乱开关。
        shuffle=bool(shuffle),
        # worker 数。
        num_workers=int(num_workers),
        # 锁页内存。
        pin_memory=bool(pin_memory),
        # 每个 worker 同步 Python/NumPy 随机源。
        worker_init_fn=_seed_worker,
        # 主 PyTorch 生成器。
        generator=generator,
        # 训练使用默认堆叠；评测保留不同尺寸真值。
        collate_fn=(
            # 训练分支使用 None。
            None
            # 根据 split 判断。
            if split == "train"
            # 验证/测试使用自定义 collate。
            else isic_eval_collate
        ),
        # worker>0 时跨 epoch 保持子进程，减少重复启动开销。
        persistent_workers=(
            # 0 worker 时必须为 False。
            int(num_workers) > 0
        ),
    )
