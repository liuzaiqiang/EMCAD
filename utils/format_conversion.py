# os 提供目录遍历、路径拼接和目录创建功能。
import os
# shutil 当前未被本文件调用；保留原始导入，仅说明其通常用于文件复制/移动。
import shutil
# libtiff.TIFF 用于读取 TIFF 掩膜；依赖需要通过项目注释中的 pip 命令安装。
from libtiff import TIFF  # pip install libtiff
# scipy.misc 提供旧版 imsave 接口；代码使用它把读取出的数组写成 PNG/JPG。
from scipy import misc
# random 用于无放回地随机抽取数据索引。
import random


# 将单个 tif/tiff 文件读取为数组并保存成目标图片格式；格式由 _dst_path 后缀决定。
def tif2png(_src_path, _dst_path):
    """
    Usage:
        formatting `tif/tiff` files to `jpg/png` files
    :param _src_path:
    :param _dst_path:
    :return:
    """
    # 以只读模式打开源 TIFF，避免对原始标注文件产生任何修改。
    tif = TIFF.open(_src_path, mode='r')
    # 解码当前 TIFF 图像页为 NumPy 风格数组。
    image = tif.read_image()
    # 根据目标路径扩展名编码并保存数组；这里没有重新映射像素标签值。
    misc.imsave(_dst_path, image)


# 从输入列表长度对应的索引区间中随机选择 550 个互不重复的索引。
def data_split(src_list):
    """
    Usage:
        randomly spliting dataset
    :param src_list:
    :return:
    """
    # random.sample 是无放回抽样；若 src_list 少于 550 个元素会直接抛出 ValueError。
    counter_list = random.sample(range(0, len(src_list)), 550)

    # 返回索引而不是样本本身，调用方可用同一索引同步切分图像与掩膜。
    return counter_list


# 只有直接执行该文件时才运行批量格式转换；被其他模块导入时不会触碰磁盘数据。
if __name__ == '__main__':
    # 原始 TIFF 掩膜目录；这是示例相对路径，需要从项目预期工作目录运行。
    src_dir = '../Dataset/train_dataset/CVC-EndoSceneStill/CVC-612/test_split/masks_tif'
    # 转换后 PNG 掩膜的目标目录。
    dst_dir = '../Dataset/train_dataset/CVC-EndoSceneStill/CVC-612/test_split/masks'

    # 若目标目录不存在则创建；exist_ok=True 允许目录已存在。
    os.makedirs(dst_dir, exist_ok=True)
    # 遍历源目录中的每个文件名；当前实现没有额外过滤非 TIFF 文件。
    for img_name in os.listdir(src_dir):
        # 源路径保持原文件名，目标路径只把字符串后缀 .tif 替换为 .png。
        tif2png(os.path.join(src_dir, img_name),
                # 第二个实参是对应的 PNG 输出路径；图像和掩膜内容由 tif2png 原样转换。
                os.path.join(dst_dir, img_name.replace('.tif', '.png')))
