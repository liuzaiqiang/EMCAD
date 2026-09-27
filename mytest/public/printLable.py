import matplotlib.pyplot as plt
import numpy as np

path = r"D:\files\Medical_Image_Segmentation_Projects\data\Synapse\train_npz\case0005_slice000.npz"
with np.load(path, allow_pickle=False) as data:
    image = data["image"]
    label = data["label"]

plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
plt.imshow(image, cmap="gray")
plt.title("image")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(label, cmap="tab10", vmin=0, vmax=8)
plt.title("label")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(image, cmap="gray")
plt.imshow(label, cmap="tab10", vmin=0, vmax=8, alpha=0.45)
plt.title("overlay")
plt.axis("off")

plt.tight_layout()
plt.show()
