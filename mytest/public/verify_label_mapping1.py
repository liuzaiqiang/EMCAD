import h5py
import numpy as np

path = "D:/files/Medical_Image_Segmentation_Projects/data/Synapse/test_vol_h5/case0001.npy.h5"

with h5py.File(path, "r") as f:
    label = f["label"][:]

print(np.unique(label, return_counts=True))