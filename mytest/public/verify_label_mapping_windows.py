import sys
from pathlib import Path

import h5py
import numpy as np


path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(
    "D:/files/Medical_Image_Segmentation_Projects/data/Synapse/case0001.npy.h5"
)
if not path.is_file():
    raise FileNotFoundError(f"找不到 H5 文件: {path}")

with h5py.File(path, "r") as f:
    label = f["label"][:]

values, counts = np.unique(label, return_counts=True)
print("文件:", path)
print("标签值:", values)
print("像素数量:", counts)
