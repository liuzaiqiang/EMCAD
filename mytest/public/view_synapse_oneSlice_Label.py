# ============================================================
# Synapse 数据集 npz 文件读取与标签可视化
# 功能：读取一个 npz 文件，显示原图、标签、叠加效果
# ============================================================

# ---------- 1. 导入需要的库 ----------
import numpy as np       # 用于读取 npz 文件和数组操作
import matplotlib.pyplot as plt  # 用于画图显示

# ---------- 2. 指定要查看的 npz 文件路径 ----------
# r 开头表示原始字符串，避免反斜杠被转义
file_path = r"/root/shared-nvme/lzq_MedicalImageSegmentation/data/Synapse/train_npz/case0040_slice187.npz"
print(file_path)
# ---------- 3. 加载 npz 文件 ----------
# np.load() 会返回一个 NpzFile 对象，类似字典，可以用 key 访问里面的数组
data = np.load(file_path)

# ---------- 4. 查看文件里有哪些内容 ----------
# data.files 会列出所有 key，通常是 ['image', 'label']
print("文件里包含的数组：", data.files)

# ---------- 5. 取出图像和标签数组 ----------
image = data['image']  # 用 key 'image' 取出 CT 图像
label = data['label']  # 用 key 'label' 取出分割标签

# ---------- 6. 打印基本信息（帮助理解数据） ----------
print(f"\n图像形状: {image.shape}")          # 通常是 (512, 512)
print(f"图像像素范围: [{image.min():.2f}, {image.max():.2f}]")
print(f"\n标签形状: {label.shape}")            # 和图像一样大
print(f"标签里的唯一值: {np.unique(label)}")   # 显示这个切片里出现了哪些器官
print(f"标签值含义: 0=背景, 1=脾脏, 2=右肾, 3=左肾, 4=胆囊, 5=食管, 6=肝脏, 7=胃, 8=主动脉")

# ---------- 7. 创建画布，并排显示三张图 ----------
# figsize=(15, 5) 表示画布宽15英寸、高5英寸
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# --- 第1张：原始 CT 图像 ---
# cmap='gray' 表示用灰度色彩映射
axes[0].imshow(image, cmap='gray')
axes[0].set_title('Original CT Image', fontsize=12)  # 标题
axes[0].axis('off')  # 关闭坐标轴

# --- 第2张：分割标签 ---
# cmap='tab10' 是 matplotlib 内置的10种区分色，适合显示分类标签
axes[1].imshow(label, cmap='tab10')
axes[1].set_title('Segmentation Label', fontsize=12)
axes[1].axis('off')

# --- 第3张：标签半透明叠加在原图上 ---
# 先画原图（底层），再画标签（上层）
# alpha=0.5 表示标签层透明度50%，这样能同时看到原图和标签
axes[2].imshow(image, cmap='gray')
axes[2].imshow(label, cmap='tab10', alpha=0.5)
axes[2].set_title('Overlay (Image + Label)', fontsize=12)
axes[2].axis('off')

# ---------- 8. 调整布局并显示 ----------
plt.tight_layout()  # 自动调整子图间距，防止标题重叠
plt.show()           # 弹出窗口显示图片
