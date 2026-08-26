# os 用于遍历原始数据目录、拼接文件路径以及创建输出目录。
import os
# shutil 在当前脚本中未被调用；保留原始导入，不改变脚本依赖关系。
import shutil
# time() 用于统计从开始处理到当前病例所消耗的时间。
from time import time

# NumPy 负责强度截断、归一化、轴变换以及逐切片保存。
import numpy as np
# SimpleITK 当前未被调用；原脚本保留它通常是为了兼容医学影像读取流程。
import SimpleITK as sitk
# nibabel 读取 Synapse 数据集的 NIfTI 图像和标签文件。
import nibabel as nib
# scipy.ndimage 当前未被调用；保留原始导入，不增加任何预处理操作。
import scipy.ndimage as ndimage
# h5py 用于把测试病例的完整三维体数据写成 HDF5 文件。
import h5py

# 依次处理训练集和测试集；训练集输出二维切片，测试集保留完整体数据。
splits = ['train', 'test']
# train = True # Set True to process training set and set False for testset

# 遍历两个数据划分，并为每个划分选择不同的输入、标签和输出路径。
for split in splits:
    # 训练划分使用 Synapse TrainSet，并生成供二维训练加载器读取的 NPZ 文件。
    if (split == 'train'):
        # 原始训练 CT 的 NIfTI 文件目录。
        ct_path = '../data/synapse/Abdomen/RawData/TrainSet/img'  # set your path to your trainset directory
        # 与训练 CT 一一对应的器官分割标签目录。
        seg_path = '../data/synapse/Abdomen/RawData/TrainSet/label'
        # 每个轴向切片保存为一个独立 .npz 文件的目标目录。
        save_path = '../data/synapse/train_npz_new/'
    # 测试划分保留病例的深度维度，以便按体数据统计 Dice、HD95 等指标。
    else:
        # 原始测试 CT 目录；注意这里相对路径与上面的训练路径起点不同，这是原代码约定。
        ct_path = './data/synapse/Abdomen/RawData/TestSet/img'  # set your path to your testset directory
        # 原始测试标签目录。
        seg_path = './data/synapse/Abdomen/RawData/TestSet/label'
        # 每个测试病例保存为一个 .npy.h5 文件的目标目录。
        save_path = './data/synapse/test_vol_h5_new/'

    # 首次处理时创建当前划分的输出目录。
    if os.path.exists(save_path) is False:
        # os.mkdir 只创建最后一级目录，因此其父目录必须已经存在。
        os.mkdir(save_path)

    # CT 窗口上界：高于 275 HU 的值会被截断到 275。
    upper = 275
    # CT 窗口下界：低于 -125 HU 的值会被截断到 -125。
    lower = -125

    # 记录本次划分开始处理的时间，后续打印累计分钟数。
    start_time = time()

    # 逐个读取 CT 文件；标签文件名通过把 img 替换为 label 得到。
    for ct_file in os.listdir(ct_path):

        # nib.load 返回带仿射和头信息的 NIfTI 对象，此处随后只使用其体素数组。
        ct = nib.load(os.path.join(ct_path, ct_file))
        # 按同名规则定位对应分割标注，保证图像和标签来自同一病例。
        seg = nib.load(os.path.join(seg_path, ct_file.replace('img', 'label')))

        # Convert them to numpy format,
        # get_fdata() 将 CT 体数据解码成浮点 NumPy 数组。
        ct_array = ct.get_fdata()
        # 标签也转为数组；后续不会对类别编号做连续值归一化。
        seg_array = seg.get_fdata()

        # 通过固定腹部 CT 窗口抑制范围外极端强度，稳定网络输入分布。
        ct_array = np.clip(ct_array, lower, upper)

        # print([np.min(ct_array), np.max(ct_array)])

        # normalize each 3D image to [0, 1]
        # 线性映射：-125 HU 对应 0，275 HU 对应 1；这不是按病例均值/方差标准化。
        ct_array = (ct_array - lower) / (upper - lower)

        # print([np.min(ct_array), np.max(ct_array)])

        # NIfTI 数组原顺序按该数据约定为 (H, W, D)，改成训练代码使用的 (D, H, W)。
        ct_array = np.transpose(ct_array, (2, 0, 1))
        # 标签执行完全相同的轴置换，以维持每个体素与 CT 的空间对应。
        seg_array = np.transpose(seg_array, (2, 0, 1))

        # 打印当前原始文件名，便于发现处理停在哪个病例。
        print('file name:', ct_file)
        # 打印转换后的 (切片数, 高, 宽)，用于检查数据方向和病例深度。
        print('shape:', ct_array.shape)

        # 去掉第一个点号后的扩展部分，作为输出文件命名的病例主体。
        ct_number = ct_file.split('.')[0]
        # 测试集需要保留完整三维病例，而不是拆成互相独立的切片。
        if (split == 'test'):
            # 将原始 imgXXXX 命名转换为测试加载器列表使用的 caseXXXX.npy.h5。
            new_ct_name = ct_number.replace('img', 'case') + '.npy.h5'
            # 创建一个新的 HDF5 文件；'w' 会写入当前新文件。
            hf = h5py.File(os.path.join(save_path, new_ct_name), 'w')
            # 以键 image 保存归一化后的完整 CT，形状为 (D, H, W)。
            hf.create_dataset('image', data=ct_array)
            # 以键 label 保存相同空间形状的完整标签体。
            hf.create_dataset('label', data=seg_array)
            # 显式关闭文件，确保缓冲数据和 HDF5 元数据写入磁盘。
            hf.close()
            # 测试病例已经整体保存，不再进入下面的逐切片训练数据生成逻辑。
            continue

        # 训练集沿深度轴遍历，把每张轴向切片制作成独立训练样本。
        for s_idx in range(ct_array.shape[0]):
            # 取第 s_idx 张 CT 切片，输出形状为 (H, W)。
            ct_array_s = ct_array[s_idx, :, :]
            # 取同一深度位置的标签切片，保证像素级对齐。
            seg_array_s = seg_array[s_idx, :, :]
            # 将切片序号补齐到三位，例如 7 -> 007，使文件名按字典序排列正确。
            slice_no = "{:03d}".format(s_idx)
            # 构造 caseXXXX_sliceYYY 形式的样本基名，与 Synapse_dataset 的列表项约定一致。
            new_ct_name = ct_number.replace('img', 'case') + '_slice' + slice_no
            # 在同一 NPZ 中分别以 image、label 键保存图像和标签二维数组。
            np.savez(os.path.join(save_path, new_ct_name), image=ct_array_s, label=seg_array_s)

        # 打印当前划分从 start_time 起累计耗时，单位由秒换算为分钟。
        print('already use {:.3f} min'.format((time() - start_time) / 60))
        # 输出分隔线，让连续病例的日志更容易阅读。
        print('-----------')
