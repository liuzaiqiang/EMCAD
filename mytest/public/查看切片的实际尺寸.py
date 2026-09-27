import numpy as np
data = np.load("D:\\files\\Medical_Image_Segmentation_Projects\\data\\Synapse\\train_npz\\case0005_slice000.npz")
print(data["image"].shape)  # 输出就是这张切片的实际尺寸  (512, 512)    
#本项目,npz切片都是原始切片尺寸