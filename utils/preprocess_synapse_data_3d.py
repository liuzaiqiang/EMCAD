# os 用于目录遍历、路径拼接和输出目录创建。
import os
# shutil 在当前脚本中未调用；保留原始导入，不改变任何运行逻辑。
import shutil
# time() 用于记录预处理累计耗时。
from time import time

# NumPy 负责 CT 强度处理、轴变换和多切片样本保存。
import numpy as np
# SimpleITK 当前未调用；原代码保留该医学影像依赖。
import SimpleITK as sitk
# nibabel 负责读取 Synapse 的 NIfTI CT 与标签。
import nibabel as nib
# scipy.ndimage 当前未调用；这里没有额外执行重采样或形态学操作。
import scipy.ndimage as ndimage
# h5py 用于保存测试集完整三维体数据。
import h5py

# 同时生成训练与测试格式；训练样本由相邻三层 CT 组成，测试仍保留整卷。
splits = ['train', 'test']
#train = True # Set True to process training set and set False for testset 

# 为 train、test 两个划分分别执行路径选择和预处理。
for split in splits:
    # 训练集被转换成中心切片监督的三帧输入 NPZ。
    if(split == 'train'):
        # Synapse 原始训练 CT 的 NIfTI 目录。
        ct_path = '../data/synapse/Abdomen/RawData/TrainSet/img' # set your path to your trainset directory
        # 与训练 CT 对齐的分割标签目录。
        seg_path = '../data/synapse/Abdomen/RawData/TrainSet/label' 
        # 三相邻切片训练样本的输出目录；mframes 表示 multiple frames。
        save_path = '../data/synapse/train_npz_mframes/'
    # 测试集不拆片，保存完整体数据供逐切片推理后重组评测。
    else:
        # Synapse 原始测试 CT 目录。
        ct_path = '../data/synapse/Abdomen/RawData/TestSet/img' # set your path to your testset directory
        # Synapse 原始测试标签目录。
        seg_path = '../data/synapse/Abdomen/RawData/TestSet/label'
        # 测试病例 HDF5 文件输出目录。
        save_path = '../data/synapse/test_vol_h5_mframes/'
    
    # 只在输出目录不存在时创建它。
    if os.path.exists(save_path) is False:
        # os.mkdir 要求父目录已经存在；不递归创建上层目录。
        os.mkdir(save_path)

    # 腹部 CT 窗口上界，单位按原始数据的 HU 语义理解。
    upper = 275 
    # 腹部 CT 窗口下界。
    lower = -125

    # 记录当前划分开始处理的时间。
    start_time = time()
    # 用一个足够大的初值追踪所有病例中的最小切片数；变量名沿用原代码。
    min_size= 10000
    # 遍历当前划分中的全部 CT 文件。
    for ct_file in os.listdir(ct_path):

        # 读取 CT NIfTI 对象。
        ct = nib.load(os.path.join(ct_path, ct_file))
        # 通过 img -> label 的命名替换定位对应标签。
        seg = nib.load(os.path.join(seg_path, ct_file.replace('img', 'label')))

        #Convert them to numpy format, 
        # 将 CT 体素解码成浮点数组。
        ct_array = ct.get_fdata()
        # 将离散分割标签解码成数组。
        seg_array = seg.get_fdata()

        # 把窗口范围外的 CT 强度截断到 [-125, 275]。
        ct_array = np.clip(ct_array, lower, upper)
    
        #print([np.min(ct_array), np.max(ct_array)])
    
        #normalize each 3D image to [0, 1] 
        # 采用固定窗口的线性缩放，把截断后的 CT 映射到 [0, 1]。
        ct_array = (ct_array - lower) / (upper - lower)
    
        #print([np.min(ct_array), np.max(ct_array)])
    
        # 将原始 (H, W, D) 轴顺序转换为 (D, H, W)，让第一维可按切片遍历。
        ct_array = np.transpose(ct_array, (2, 0, 1))
        # 标签执行相同轴变换，保持体素位置严格对应。
        seg_array = np.transpose(seg_array, (2, 0, 1))
    
        # 输出当前处理的病例文件名。
        print('file name:', ct_file)
        # 输出轴变换后的体数据形状。
        print('shape:', ct_array.shape)
        
        # 若当前病例深度更小，则更新数据集中已观察到的最小切片数。
        if(ct_array.shape[0] < min_size):
            # 保存新的最小深度，脚本结束时打印用于数据完整性检查。
            min_size = ct_array.shape[0]

        # 取扩展名前的 imgXXXX 作为病例编号主体。
        ct_number = ct_file.split('.')[0]
        # 测试病例直接以完整三维形式写入 HDF5。
        if(split == 'test'):
	    # 将 imgXXXX 改为 caseXXXX，并使用加载器预期的 .npy.h5 后缀。
    	    new_ct_name = ct_number.replace('img', 'case')+'.npy.h5'
	    # 创建目标 HDF5 文件。
    	    hf = h5py.File(os.path.join(save_path, new_ct_name), 'w')
	    # 保存归一化后的 CT，形状为 (D, H, W)。
    	    hf.create_dataset('image', data=ct_array)
	    # 保存对应标签体。
    	    hf.create_dataset('label', data=seg_array)
	    # 关闭文件并刷新写入内容。
    	    hf.close()
	    # 测试数据不进入训练样本的三帧切片生成循环。
    	    continue
    	
        # 三帧窗口需要 s_idx、s_idx+1、s_idx+2，因此循环上界比总深度少 2。
        for s_idx in range(ct_array.shape[0]-2):
            #ct_array_s = np.zeros()
	    # 先转为 (H, W, D)，再截取连续三层，得到网络输入 (H, W, 3)。
    	    ct_array_s = np.transpose(ct_array, (1, 2, 0))[:, :, s_idx:s_idx+3]
	    # 打印每个三帧样本形状，便于确认最后一维确实为 3。
    	    print(ct_array_s.shape)
	    # 监督标签取三层窗口的中间层 s_idx+1，使上下相邻层仅提供空间上下文。
    	    seg_array_s = seg_array[s_idx+1, :, :]
	    # 三位补零的起始层编号用于稳定排序输出文件。
    	    slice_no = "{:03d}".format(s_idx)
	    # 构造 caseXXXX_sliceYYY 样本名。
    	    new_ct_name = ct_number.replace('img', 'case') + '_slice' + slice_no
	    # 保存三通道相邻切片输入和二维中心层标签。
    	    np.savez(os.path.join(save_path, new_ct_name), image=ct_array_s, label=seg_array_s)

        
        # 打印当前划分的累计预处理时间，单位为分钟。
        print('already use {:.3f} min'.format((time() - start_time) / 60))
        # 日志病例分隔线。
        print('-----------')
    # 处理完整个划分后输出观察到的最小病例深度；原日志文字写作 max_size，但变量实际记录最小值。
    print('max_size '+str(min_size))
