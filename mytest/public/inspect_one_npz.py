import numpy as np

path = r"D:\files\Medical_Image_Segmentation_Projects\data\Synapse\train_npz\case0005_slice000.npz"

with np.load(path, allow_pickle=False) as data:
    print("文件中的键：", data.files)
    image = data["image"]
    label = data["label"]

    print("image shape:", image.shape)
    print("image dtype:", image.dtype)
    print("label shape:", label.shape)
    print("label dtype:", label.dtype)
    print("label values:", np.unique(label))

    values, counts = np.unique(label, return_counts=True)
    for value, count in zip(values, counts):
        print(f"label={int(value)}: {int(count):,} voxels")
