# ============================== 初学者阅读总览 ==============================
# 这是较早的 SLDGroup 风格息肉二分类测试脚本，输入 RGB 图像，输出 1 通道前景 logits。
# 主流程：DataLoader -> EMCAD 四级输出 -> 取最后一级 -> 双线性恢复原尺寸 -> sigmoid ->
# 每图 min-max 归一化 -> 0.5 阈值 -> Dice/IoU/敏感度/特异度/精确率/HD95 -> Excel。
# 二分类只有 1 个前景通道：p(息肉)=sigmoid(logit)，背景概率隐含为 1-p，不需要第二通道。
# 论文对应：EMCAD 结构见第 3.2 节与图 2；二分类训练/352x352 设置见第 4.1 节；
# 息肉分割结果见第 4.2.1 节/表 1。逐图归一化、0.2 真值阈值、空掩膜 HD95=100、
# Excel 累积以及 strict=False 加载属于本脚本工程策略，阅读结果时应与论文指标协议区分。
# ========================================================================

# os 负责测试数据、checkpoint、预测图和 Excel 路径。
import os
# time 当前脚本没有实际使用，是保留导入。
import time
# torch 负责张量、GPU 推理与 checkpoint 加载。
import torch
# F 提供预测/标签尺寸恢复所需的 interpolate。
import torch.nn.functional as F
# NumPy 把二值预测转换为 uint8 图像。
import numpy as np
# pandas 构造逐图结果表并读写 Excel。
import pandas as pd
# OpenCV 把预测掩膜保存为 PNG 等图像文件。
import cv2
# argparse 定义测试命令行接口。
import argparse
# tqdm 显示批次推理进度。
from tqdm import tqdm

# Project-specific imports
# EMCADNet 封装编码器、EMCAD 解码器和四个二分类分割头。
from lib.networks import EMCADNet
# get_loader 配对 images/masks，训练尺寸归一化，并在测试时返回原始尺寸与文件名。
from utils.dataloader_polyp import get_loader
# medpy 的 hd95 计算预测边界与真值边界间对称 95% Hausdorff 距离。
from medpy.metric.binary import hd95

# 计算一个预测与标签的软/硬 Dice。输入应同形，通常是二值 [H,W] Tensor；输出标量 Tensor。
def dice_coefficient(predicted, labels):
# 若二者不在同一设备，把标签迁移到预测所在设备。
    if predicted.device != labels.device:
# .to 不改变数值，只改变设备。
        labels = labels.to(predicted.device)
# 平滑项防止预测和标签都空时分母为 0。
    smooth = 1e-6
# contiguous 确保内存连续，view(-1) 把所有像素拉成一维。
    predicted_flat = predicted.contiguous().view(-1)
# 标签同样展平，确保逐像素对应。
    labels_flat = labels.contiguous().view(-1)
# 二值时乘积和等于交集像素数。
    intersection = (predicted_flat * labels_flat).sum()
# Dice 分母为预测前景数 + 真值前景数。
    total = predicted_flat.sum() + labels_flat.sum()
# Dice=(2*交集+平滑)/(两集合大小和+平滑)。
    return (2. * intersection + smooth) / (total + smooth)

# 计算 Jaccard/IoU。输入输出设备与形状约定同上。
def iou(predicted, labels):
# 设备不一致时迁移标签。
    if predicted.device != labels.device:
# 与预测对齐设备。
        labels = labels.to(predicted.device)
# 防止空集分母为 0。
    smooth = 1e-6
# 展平预测。
    predicted_flat = predicted.contiguous().view(-1)
# 展平标签。
    labels_flat = labels.contiguous().view(-1)
# 交集像素数。
    intersection = (predicted_flat * labels_flat).sum()
# 并集=|P|+|G|-|P交G|。
    union = predicted_flat.sum() + labels_flat.sum() - intersection
# IoU=(交集+平滑)/(并集+平滑)。
    return (intersection + smooth) / (union + smooth)

# 计算混淆矩阵派生指标与 HD95。pred/gt 应为同形 0/1 Tensor；返回四个 Python 数值。
def get_binary_metrics(pred, gt):
# 真阳性：预测 1 且真值 1。
    tp = (pred * gt).sum().item()
# 真阴性：预测 0 且真值 0。
    tn = ((1 - pred) * (1 - gt)).sum().item()
# 假阳性：预测 1 但真值 0。
    fp = (pred * (1 - gt)).sum().item()
# 假阴性：预测 0 但真值 1。
    fn = ((1 - pred) * gt).sum().item()
    
# 敏感度/召回率=TP/(TP+FN)，1e-8 防止真值无前景时除零。
    sensitivity = tp / (tp + fn + 1e-8)
# 特异度=TN/(TN+FP)。
    specificity = tn / (tn + fp + 1e-8)
# 精确率=TP/(TP+FP)。
    precision = tp / (tp + fp + 1e-8)
    
# HD95 可能因空掩膜、维度或 medpy 内部问题抛异常，因此此处包裹 try。
    try:
# 只有预测和真值都至少含一个前景像素时才真正计算表面距离。
        if pred.sum() > 0 and gt.sum() > 0:
# 转 CPU NumPy 后计算；未传 voxel spacing，因此单位是像素。
            hd_val = hd95(pred.cpu().numpy(), gt.cpu().numpy())
# 任一为空时用固定惩罚值 100.0，而不是 NaN 或数据集对角线。
        else:
# 该约定会直接影响平均 HD95 的数值解释。
            hd_val = 100.0
# 裸 except 会把所有异常都转成 100，便于批处理继续但会隐藏真实错误类型。
    except:
# 异常兜底值。
        hd_val = 100.0
        
# 返回敏感度、特异度、精确率、HD95 四元组。
    return sensitivity, specificity, precision, hd_val

# 对一个数据集划分执行推理。输入：已加载模型、数据根、划分名、参数、可选保存目录；
# 输出：逐图宏平均 Dice、逐图宏平均 IoU、每张图的详细指标字典列表。
def test(model, path, dataset, opt, save_base=None):
# 组合到具体划分目录，例如 <path>/test。
    data_path = os.path.join(path, dataset)
# 图像目录。
    image_root = f'{data_path}/images/'
# 掩膜目录。
    gt_root = f'{data_path}/masks/'
# 切换评估模式，固定 BatchNorm/Dropout。
    model.eval()
    
# 构建测试 DataLoader。
    test_loader = get_loader(
# 输入图像与真值目录。
        image_root=image_root, gt_root=gt_root, 
# 批大小和网络输入尺寸。
        batchsize=opt.test_batchsize, trainsize=opt.img_size,
# 禁止打乱；split='test' 使 loader 返回原尺寸和文件名。
        shuffle=False, split='test', color_image=opt.color_image
# DataLoader 构造结束。
    )
    
# 累计逐图 Dice、IoU 和图像计数；最终做 per-image macro mean。
    DSC, IOU, total_images = 0.0, 0.0, 0
# 保存每张图的完整指标，供 pandas/Excel 使用。
    detailed_results = []

# 推理不需要梯度图，no_grad 降低显存与计算开销。
    with torch.no_grad():
# 每个 pack 在测试模式含 images、gts、original_shapes、names。
        for pack in tqdm(test_loader, desc=f"Inference on {dataset}"):
# images 通常 [B,3,352,352]；gts 可能保留原尺寸列表/批；original_shapes 记录每图 H/W。
            images, gts, original_shapes, names = pack       
# 图像和标签移到默认 GPU；标签转 float 便于插值和二值运算。
            images, gts = images.cuda(), gts.cuda().float()

# EMCAD 返回四个同输出分辨率的 1 通道 logits 列表。
            ress = model(images)
# 兼容只返回单 Tensor 的模型。
            if not isinstance(ress, list):
# 统一包装为列表。
                ress = [ress]
            # Take the primary output (EMCADNet usually uses the last item for final prediction)
# 取最终/最高分辨率分割头，形状通常 [B,1,352,352]，尚未 sigmoid。
            predictions = ress[-1]
            
# 批内逐图恢复到各自原尺寸并计算指标。
            for i in range(len(images)):
# loader 把原尺寸按两个序列组织；这里解释为 H、W。
                h_orig, w_orig = int(original_shapes[0][i]), int(original_shapes[1][i])
                
# 取第 i 个 [1,H,W] logits，并补 batch 维为 [1,1,H,W]。
                p = predictions[i].unsqueeze(0)
# 双线性缩放到原图尺寸，sigmoid 转前景概率，再 squeeze 为 [H_orig,W_orig]。
                pred_resized = F.interpolate(p, size=(h_orig, w_orig), mode='bilinear', align_corners=False).sigmoid().squeeze()
# 对每张图单独做 min-max 归一化到近似 [0,1]；这会改变 sigmoid=0.5 的绝对概率语义。
                pred_resized = (pred_resized - pred_resized.min()) / (pred_resized.max() - pred_resized.min() + 1e-8)
                
# 取第 i 个真值并补 batch 维。
                g = gts[i].unsqueeze(0)
# 最近邻恢复标签尺寸，避免产生类别插值；squeeze 回二维。
                gt_resized = F.interpolate(g, size=(h_orig, w_orig), mode='nearest').squeeze()

# 归一化预测以 0.5 阈值转硬掩膜。
                input_binary = (pred_resized >= 0.5).float()
# 真值以 0.2 阈值二值化，容忍掩膜读取/缩放后的非严格 0/1 值。
                target_binary = (gt_resized >= 0.2).float()

# 当前图 Dice 转成 Python float。
                d = dice_coefficient(input_binary, target_binary).item()
# 当前图 IoU。
                io = iou(input_binary, target_binary).item()
# 当前图敏感度、特异度、精确率和 HD95。
                sens, spec, prec, hd = get_binary_metrics(input_binary, target_binary)

# 累加当前图 Dice。
                DSC += d
# 累加当前图 IoU。
                IOU += io
# 已处理图像数加 1。
                total_images += 1

# 追加一条可直接转换为 DataFrame 的结果记录。
                detailed_results.append({
# 文件名与两项重叠指标。
                    'Name': names[i], 'Dice': d, 'IoU': io,
# 以下四项先格式化为 4 位小数再转 float，因此详细表精度被截到 4 位。
                    'Sensitivity': float('{:.4f}'.format(sens)),
                    'Specificity': float('{:.4f}'.format(spec)),
                    'Precision': float('{:.4f}'.format(prec)),
# HD95 同样保留 4 位小数。
                    'HD95': float('{:.4f}'.format(hd))
# 当前记录结束。
                })

# 提供保存目录时导出硬二值预测图。
                if save_base:
# 0/1 乘 255 并转 uint8，得到黑白掩膜。
                    pred_img = (input_binary.cpu().numpy() * 255).astype(np.uint8)
# 使用原图文件名写入保存目录。
                    cv2.imwrite(os.path.join(save_base, names[i]), pred_img)

# 返回逐图宏平均 Dice/IoU 与明细；若 total_images=0，此处会除零。
    return DSC / total_images, IOU / total_images, detailed_results

# 仅直接运行脚本时解析参数并执行测试。
if __name__ == '__main__':
# 创建命令行解析器。
    parser = argparse.ArgumentParser()
# 必填实验标识，用于推导模型目录、预测目录和结果文件名。
    parser.add_argument('--run_id', type=str, required=True)
# 编码器结构必须与训练权重一致。
    parser.add_argument('--encoder', type=str, default='pvt_v2_b2')
# MSCB 通道扩张倍数。
    parser.add_argument('--expansion_factor', type=int, default=2)
# MSDC 多尺度卷积核列表。
    parser.add_argument('--kernel_sizes', type=int, nargs='+', default=[1, 3, 5])
# LGAG 卷积核尺寸。
    parser.add_argument('--lgag_ks', type=int, default=3)
# MSCB 激活函数。
    parser.add_argument('--activation_mscb', type=str, default='relu6')
# 传入旗标后使用串行深度卷积；默认并行。
    parser.add_argument('--no_dw_parallel', action='store_true', default=False)
# 传入旗标后多尺度聚合用 concat；默认 add。
    parser.add_argument('--concatenation', action='store_true', default=False)
# 数据集名称，用于拼接 target/<dataset_name>。
    parser.add_argument('--dataset_name', type=str, default='ClinicDB')
# 评估划分名，默认 test；此处未用 choices 限制字符串。
    parser.add_argument('--split', type=str, default='test')
# 模型输入正方形尺寸，论文二分类常用 352。
    parser.add_argument('--img_size', type=int, default=352)
# 推理批大小。
    parser.add_argument('--test_batchsize', type=int, default=1)
# 是否按彩色图读取；该参数未声明 type/action，命令行显式传值会成为字符串。
    parser.add_argument('--color_image', default=True)
# 息肉 prepared data 根目录。
    parser.add_argument('--test_path', type=str, default='../data/polyp/target/')
# 解析命令行。
    opt = parser.parse_args()

    # --- Paths ---
# 预测掩膜保存到 run_id/数据集/划分三级目录。
    save_base = f'./predictions_polyp/{opt.run_id}/{opt.dataset_name}/{opt.split}'
# 递归创建预测目录。
    os.makedirs(save_base, exist_ok=True)
# 创建逐次实验 Excel 目录。
    os.makedirs('results_polyp', exist_ok=True)
# 训练脚本约定最优权重名为 <run_id>-best.pth。
    model_path = os.path.join(f'./model_pth/{opt.run_id}/', f'{opt.run_id}-best.pth')
# 把数据集目录接到 test_path；test() 稍后再追加 split。
    opt.test_path = f'{opt.test_path}/{opt.dataset_name}/'

    # --- Model Loading ---
# 按训练结构重建一通道二分类 EMCAD。
    model = EMCADNet(
# 1 个前景 logit 通道，背景为隐式互补概率。
        num_classes=1, 
# MSDC 多尺度核。
        kernel_sizes=opt.kernel_sizes, 
# MSCB 扩张倍数。
        expansion_factor=opt.expansion_factor, 
# no_dw_parallel 取反得到正向开关。
        dw_parallel=not opt.no_dw_parallel, 
# concatenation 取反得到 add 开关。
        add=not opt.concatenation, 
# LGAG 核尺寸。
        lgag_ks=opt.lgag_ks, 
# 激活类型。
        activation=opt.activation_mscb, 
# 编码器名称。
        encoder=opt.encoder, 
# 推理直接加载完整 checkpoint，不预先加载 ImageNet 权重。
        pretrain=False # Always False for inference
# 构造后立即移到默认 CUDA；没有 CPU 回退。
    ).cuda()
    
# 加载权重：无 map_location；strict=False 会忽略不匹配/缺失键，可能掩盖结构配置错误。
    model.load_state_dict(torch.load(model_path), strict=False)
# 明确切换 eval；test() 内还会再次调用。
    model.eval()

    # --- Run Inference ---
    # Adjust test path to match main script behavior
# 对指定划分执行推理并返回宏平均与逐图记录。
    mean_dice, mean_iou, results = test(model, opt.test_path, opt.split, opt, save_base=save_base)

    # --- Individual Excel ---
# 将逐图字典列表转为表格，每行一张图。
    df = pd.DataFrame(results)
# 对所有数值列求均值，字符串 Name 自动排除。
    mean_row = df.mean(numeric_only=True).to_dict()
# 给均值行补上可识别名称。
    mean_row['Name'] = 'AVERAGE'
# 把均值行追加到表尾并重建连续索引。
    df = pd.concat([df, pd.DataFrame([mean_row])], ignore_index=True)
# 保存本次 run 的独立 Excel。
    df.to_excel(f'results_polyp/Results_{opt.run_id}_{opt.dataset_name}_{opt.split}.xlsx', index=False)

    # --- Persistent Summary ---
# 所有运行共享的汇总工作簿路径。
    summary_file = 'All_Runs_Summary_Polyp.xlsx'
# 构造本次实验的一行摘要。
    avg_data = {
# 实验标识、网络、数据集。
        'run_id': opt.run_id, 'network': 'EMCADNet', 'dataset': opt.dataset_name,
# 划分和由 test() 原始累计得到的 Dice/IoU。
        'split': opt.split, 'dice': mean_dice, 'iou': mean_iou,
# 其余指标取前面 DataFrame 均值行；明细中已截到 4 位小数。
        'sensitivity': mean_row['Sensitivity'], 'specificity': mean_row['Specificity'],
# 精确率和 HD95。
        'precision': mean_row['Precision'], 'HD95': mean_row['HD95']
# 摘要字典结束。
    }
# 单行 DataFrame 便于与已有表拼接。
    df_new = pd.DataFrame([avg_data])

# 汇总文件已存在时先读取旧记录。
    if os.path.exists(summary_file):
# 读取已有工作簿。
        df_existing = pd.read_excel(summary_file)
# 在末尾追加本次记录，不做 run_id 去重。
        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
# 覆盖写回完整汇总表。
        df_combined.to_excel(summary_file, index=False)
# 首次运行时直接创建汇总表。
    else:
# 只写当前一行。
        df_new.to_excel(summary_file, index=False)

# 终端提示评估完成及汇总文件位置。
    print(f"Evaluation complete. Summary appended to {summary_file}")
