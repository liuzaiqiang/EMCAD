import h5py
import numpy as np
from pathlib import Path


# Use the same test-volume directory as test_synapse.py by default.
path = Path(__file__).resolve().parents[1] / "data" / "synapse" / "test_vol_h5_new" / "case0001.npy.h5"

if not path.is_file():
    raise FileNotFoundError(
        f"找不到 H5 文件: {path}\n"
        "请把 path 改成实际的 case0001.npy.h5 路径。"
    )

with h5py.File(path, "r") as f:
    label = f["label"][:]

values, counts = np.unique(label, return_counts=True)
print("文件:", path)
print("标签值:", values)
print("像素数量:", counts)
