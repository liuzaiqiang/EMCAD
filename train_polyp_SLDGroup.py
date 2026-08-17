# ============================== 初学者阅读总览 ==============================
# 这是较早的 SLDGroup 风格息肉训练脚本，一次启动连续做 5 个 run。
# 训练链：RGB 图像/二值 mask -> 0.75/1/1.25 多尺度 -> EMCAD 四个 1 通道 logits ->
# 四个单头 structure loss + 四头 logits 求和的 structure loss -> 反向传播 -> AdamW。
# structure loss=边界加权 BCE + 边界加权 IoU；二分类背景由 1-sigmoid(logit) 隐含表示。
# 每个 epoch 保存 last，随后分别评估 test 和 val；仅依据 val Dice 更新 best，
# 但 test 指标也被每轮查看，严格科研流程中应意识到这会增加对测试集的观察次数。
# 论文对应：EMCAD 结构见第 3.2 节/图2，多阶段损失见第 3.3 节，二分类设置见第 4.1 节。
# 论文一般二分类设置为 352x352、batch 16、AdamW lr/wd=1e-4、200 epoch；
# 本脚本默认 batch 8、lr 5e-4 加余弦调度，属于当前工程配置，不是逐项原样复刻。
# ========================================================================

# os 创建日志/权重目录并拼接路径。
import os
# time 生成 run 时间戳并累计训练耗时。
import time
# logging 把每轮验证/测试结果写入独立 run 日志。
import logging
# argparse 定义网络与训练参数。
import argparse
# datetime 用于打印带日期时间的训练进度。
from datetime import datetime

# NumPy 在本文件当前执行路径未直接使用，是保留导入。
import numpy as np
# torch 提供模型、GPU、优化器和 checkpoint 保存。
import torch
# nn 仅出现在下方被三引号禁用的 DataParallel 示例中。
import torch.nn as nn
# F 提供池化、BCE、插值和 sigmoid。
import torch.nn.functional as F
# Variable 是旧版 PyTorch 包装；现代 Tensor 已支持 autograd，但这里保留现有调用。
from torch.autograd import Variable
# CosineAnnealingLR 在 epoch 维执行余弦学习率衰减。
from torch.optim.lr_scheduler import CosineAnnealingLR

# Project-specific imports
# EMCADNet 封装编码器、解码器与四个二分类输出头。
from lib.networks import EMCADNet
# get_loader 配对图像/掩膜，并按 train/test 模式返回不同批结构。
from utils.dataloader_polyp import get_loader as get_loader
# clip_gradient 做逐参数梯度截断；adjust_lr 是旧阶梯衰减；AvgMeter 记录 loss；cal_params_flops 统计复杂度。
from utils.utils import clip_gradient, adjust_lr, AvgMeter, cal_params_flops


# 边界感知二分类结构损失。输入 pred/mask 通常 [B,1,H,W]；pred 是 logits，mask 是 0/1 float；
# 返回一个标量 Tensor。参数 w 只对最终 wbce+wiou 总和做权重缩放。
def structure_loss(pred, mask, w=1):
# 用 31x31 局部均值与原 mask 的差异构造边界权重；边缘附近权重大于内部平坦区域。
    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
# 对 logits 直接算逐像素 BCE；with_logits 数值上比先 sigmoid 再 BCE 稳定。
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduction='none')
# 按边界权重加权，再对 H/W 求和并除以权重和，得到每个 batch/通道的加权 BCE。
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

# 将 logits 转为前景概率。
    pred = torch.sigmoid(pred)
# 加权交集。
    inter = ((pred * mask) * weit).sum(dim=(2, 3))
# 这里先计算加权概率和+标签和，后面减 inter 得到并集。
    union = ((pred + mask) * weit).sum(dim=(2, 3))
# 加 1 平滑后的加权 IoU loss=1-IoU。
    wiou = 1 - (inter + 1) / (union - inter + 1)

# BCE 与 IoU loss 相加、乘 w，再对 batch/通道取均值。
    return (w * (wbce + wiou)).mean()

# 计算展平后的 Dice；输入通常是阈值后的 0/1 Tensor，输出标量 Tensor。
def dice_coefficient(predicted, labels):
# 对齐设备。
    if predicted.device != labels.device:
# 标签迁移到预测设备。
        labels = labels.to(predicted.device)
# 平滑避免双空掩膜除零。
    smooth = 1e-6
# 展平所有像素。
    predicted_flat = predicted.contiguous().view(-1)
# 展平标签。
    labels_flat = labels.contiguous().view(-1)
# 交集像素数。
    intersection = (predicted_flat * labels_flat).sum()
# 两集合大小之和。
    total = predicted_flat.sum() + labels_flat.sum()
# Dice 公式。
    return (2. * intersection + smooth) / (total + smooth)

# 计算展平后的 IoU/Jaccard；输入输出约定同 Dice。
def iou(predicted, labels):
# 设备对齐。
    if predicted.device != labels.device:
# 标签迁移。
        labels = labels.to(predicted.device)
# 平滑项。
    smooth = 1e-6
# 展平预测。
    predicted_flat = predicted.contiguous().view(-1)
# 展平标签。
    labels_flat = labels.contiguous().view(-1)
# 交集。
    intersection = (predicted_flat * labels_flat).sum()
# 并集。
    union = predicted_flat.sum() + labels_flat.sum() - intersection
# IoU 公式。
    return (intersection + smooth) / (union + smooth)

# 评估一个 val/test 划分。输入模型、数据集根、划分名和 opt；
# 返回逐图宏平均 Dice、逐图宏平均 IoU、图像数。
def test(model, path, dataset, opt):
# 组合 <path>/<dataset>，dataset 在调用处为 'test' 或 'val'。
    data_path = os.path.join(path, dataset)
# 图像目录。
    image_root = f'{data_path}/images/'
# 掩膜目录。
    gt_root = f'{data_path}/masks/'
# 评估模式固定 BatchNorm/Dropout。
    model.eval()
    
# 构建不打乱的评估 DataLoader。
    test_loader = get_loader(
# 图像/标签路径。
        image_root=image_root, gt_root=gt_root, 
# 测试批大小与网络输入尺寸。
        batchsize=opt.test_batchsize, trainsize=opt.img_size,
# split='test' 使 loader 返回原尺寸；color_image 控制 RGB/灰度。
        shuffle=False, split='test', color_image=opt.color_image
# loader 构造结束。
    )
    
# 累计每图 Dice、IoU 与数量。
    DSC, IOU, total_images = 0.0, 0.0, 0
# 关闭梯度记录，降低测试显存。
    with torch.no_grad():
# pack 依次来自 DataLoader。
        for pack in test_loader:
# images 为缩放后的模型输入；gts 为真值；original_shapes 保存每图原 H/W；最后名称在此忽略。
            images, gts, original_shapes, _ = pack       
# 图像移到默认 CUDA；即使后面 device 选择 CPU，这里仍强制要求 CUDA。
            images = images.cuda()
# 标签移到 CUDA 并转 float，供插值和阈值使用。
            gts = gts.cuda().float()

# 前向得到四级 1 通道 logits。
            ress = model(images)
# 兼容单输出模型。
            if not isinstance(ress, list):
# 包装成列表。
                ress = [ress]
            # Take the primary output
# 取最终输出 P[-1] 作为测试预测。
            predictions = ress[-1]
            
# 批内逐图处理不同原尺寸。
            for i in range(len(images)):
                # Note: original_shapes in some loaders is [W, H], in others [H, W]
                # We ensure it matches your specific data loader's return order
# 当前代码把 original_shapes[0/1] 解释为 H/W；必须与 loader 返回约定一致。
                h_orig, w_orig = int(original_shapes[0][i]), int(original_shapes[1][i])
                
                # 1. Prediction Resize (Bilinear for soft maps)
# 取第 i 个 [1,h,w] logits 并补 batch 维。
                p = predictions[i].unsqueeze(0)
# 双线性恢复到原尺寸，适合连续 logits/概率图。
                pred_resized = F.interpolate(p, size=(h_orig, w_orig), mode='bilinear', align_corners=False)
# sigmoid 转前景概率并去除单例维。
                pred_resized = pred_resized.sigmoid().squeeze()
                
                # 2. Local Normalization
# 每图单独 min-max 归一化，之后的 0.5 不再等同原始 sigmoid 概率 0.5。
                pred_resized = (pred_resized - pred_resized.min()) / (pred_resized.max() - pred_resized.min() + 1e-8)
                
                # 3. GT Resize (NEAREST to maintain binary mask integrity)
# 取标签并补 batch 维。
                g = gts[i].unsqueeze(0)
# 最近邻恢复离散掩膜，避免灰度过渡；squeeze 为二维。
                gt_resized = F.interpolate(g, size=(h_orig, w_orig), mode='nearest').squeeze()

# 历史调试打印，已注释，不执行。
                #print(pred_resized.shape, gt_resized.shape, g.shape)

                # 4. Binary Thresholding
# 预测归一化值 >=0.5 判前景。
                input_binary = (pred_resized >= 0.5).float()
# 真值 >=0.2 判前景，兼容非严格 0/1 掩膜。
                target_binary = (gt_resized >= 0.2).float() 

                # Applying original thresholding (0.5 for pred, 0.2 for target) 
# 当前图 Dice 加入总和。
                DSC += dice_coefficient(input_binary, target_binary).item()
# 当前图 IoU 加入总和。
                IOU += iou(input_binary, target_binary).item()
# 图像计数加 1。
                total_images += 1

# 返回 per-image macro mean；若目录为空 total_images=0 会除零。
    return DSC / total_images, IOU / total_images, total_images

# 执行一个 epoch 的训练，并在 epoch 末评估 test/val、保存 last/best。
# 输入 train_loader、model、optimizer、当前 epoch、opt、model_name；无显式返回，
# 通过 global 变量更新 best/test_dice_at_best_val/total_train_time/dict_plot。
def train(train_loader, model, optimizer, epoch, opt, model_name):
# 切回训练模式。
    model.train()
# 声明这些名字来自模块主流程，而不是函数局部变量。
    global best, test_dice_at_best_val, total_train_time, dict_plot
    
# 记录本 epoch 开始时间。
    epoch_start = time.time()
# AvgMeter 保存 loss 序列并显示最近若干项平均。
    loss_record = AvgMeter()
# 论文二分类常见的三尺度训练比例。
    size_rates = [0.75, 1, 1.25] 
# 每 epoch 的 DataLoader 批次数，用于进度打印。
    total_step = len(train_loader)

# 从 1 开始编号训练 batch。
    for i, (images, gts) in enumerate(train_loader, start=1):
# 每个原始 batch 依次做三个尺度，每个尺度都执行一次优化器更新。
        for rate in size_rates:            
# 清除上一次尺度/批次梯度。
            optimizer.zero_grad()
# 用旧式 Variable 包装并强制移到 CUDA；gts 转 float。
# 注意 images/gts 在 rate 循环内被重新赋值：第一个 0.75 尺度后变量已变成缩放后的 GPU Tensor。
            images, gts = Variable(images).cuda(), Variable(gts).float().cuda()
    
# 只有 rate !=1 时执行尺寸变换。
            if rate != 1:
# 目标边长先乘比例，再四舍五入到最接近的 32 倍数，适配多级下采样。
                trainsize = int(round(opt.img_size * rate / 32) * 32)
# 图像用双线性插值；align_corners=True 是此旧脚本的既定设置。
                images = F.interpolate(images, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
# 标签用最近邻，保持二值语义。
                gts = F.interpolate(gts, size=(trainsize, trainsize), mode='nearest')
# 现有重赋值的直接结果：rate=1 时复用上一轮 rate=0.75 已缩放的 images/gts，
# 并不会自动恢复原始 opt.img_size；这里只说明真实执行顺序，不改逻辑。
            
# 前向得到四级 logits 列表。
            P = model(images)
# 兼容单输出模型，但下面固定访问 P[0]..P[3]，因此实际仍要求至少 4 个输出。
            if not isinstance(P, list):
# 单 Tensor 包装成列表。
                P = [P]
# 对最粗一级输出单独计算 structure loss。
            loss_p1 = structure_loss(P[0], gts)
# 第二级输出损失。
            loss_p2 = structure_loss(P[1], gts)
# 第三级输出损失。
            loss_p3 = structure_loss(P[2], gts)
# 最终级输出损失。
            loss_p4 = structure_loss(P[3], gts)
# 四个 logits 逐元素相加后再计算聚合输出损失；相加的是 logits，不是概率。
            loss_p1234 = structure_loss(P[0]+P[1]+P[2]+P[3], gts)

# 五项损失权重当前全部为 1。
            weights = [1, 1, 1, 1, 1]
# 总损失为四个单头损失加一个聚合头损失；没有除以 5。
            loss = weights[0]*loss_p1 + weights[1]*loss_p2 + weights[2]*loss_p3 + weights[3]*loss_p4 + weights[4]*loss_p1234

# 对共享编码器、EMCAD 解码器和四个分割头反向传播。
            loss.backward()
# 将每个参数梯度原地截断到 [-clip,+clip]，抑制梯度爆炸。
            clip_gradient(optimizer, opt.clip)
# AdamW 更新一次；因此每个原始 batch 共更新三次。
            optimizer.step()
            
# 只在 rate==1 分支记录 loss，避免三尺度都计入显示器。
            if rate == 1:
# loss.data 是当前尺度标量 Tensor，n=opt.batchsize 用于 AvgMeter 加权统计。
                loss_record.update(loss.data, opt.batchsize)
                
# 每 100 个 batch 或最后一个 batch 打印训练状态。
        if i % 100 == 0 or i == total_step:
# 显示时间、epoch、step、当前优化器学习率和 AvgMeter 最近 loss。
            print(f'{datetime.now()} Epoch [{epoch:03d}/{opt.epoch:03d}], Step [{i:04d}/{total_step:04d}], '
# 续接同一 f-string 输出。
                  f'LR: {optimizer.param_groups[0]["lr"]:.6f}, Loss: {loss_record.show():.4f}')
        
# 累加该 epoch 训练阶段耗时；后面的验证/测试耗时不计入 total_train_time。
    total_train_time += (time.time() - epoch_start)
    
    # Save Last
# 当前 run 的权重目录。
    save_path = opt.train_save
# 确保目录存在。
    os.makedirs(save_path, exist_ok=True)
# 每 epoch 覆盖 <model_name>-last.pth；只保存模型 state_dict。
    torch.save(model.state_dict(), os.path.join(save_path, f"{model_name}-last.pth"))

    # Validation and Testing
# 保存本 epoch 两个划分的 Dice。
    epoch_results = {}
# 当前顺序先 test 后 val；两者都在每个 epoch 查看。
    for ds in ['test', 'val']:
# 对指定划分计算宏平均 Dice/IoU。
        d_dice, d_iou, _ = test(model, opt.test_path, ds, opt)
# 仅保存 Dice 供后面选择/记录。
        epoch_results[ds] = d_dice
# 写入 run 日志。
        logging.info(f'Epoch: {epoch}, Dataset: {ds}, Dice: {d_dice:.4f}, IoU: {d_iou:.4f}')
# 同步打印。
        print(f'Epoch: {epoch}, Dataset: {ds}, Dice: {d_dice:.4f}, IoU: {d_iou:.4f}')
# 把每轮 Dice 追加到全局绘图字典；本脚本当前未实际绘图。
        dict_plot[ds].append(d_dice)

    # Check if Best Validation Dice
# 只依据 val Dice 严格提升来更新 best，不以 test 选权重。
    if epoch_results['val'] > best:
# 记录提升前后数值。
        logging.info(f"### Best Model Saved (Dice improved from {best:.4f} to {epoch_results['val']:.4f}) ###")
# 终端提示。
        print(f"### Best Model Saved (Dice improved from {best:.4f} to {epoch_results['val']:.4f}) ###")
# 更新全局最好验证 Dice。
        best = epoch_results['val']
# 同时记录该最佳验证时点对应的 test Dice；这不是独立只测一次的最终测试协议。
        test_dice_at_best_val = epoch_results['test'] # Track test dice at peak val
# 保存最优模型到固定文件。
        torch.save(model.state_dict(), os.path.join(save_path, f"{model_name}-best.pth"))
    
# 仅直接运行脚本时启动五次训练。
if __name__ == '__main__':
    # Initial defaults
# 数据集名在解析器创建前硬编码；train_path/test_path 默认值据此立即生成。
    dataset_name = 'ClinicDB' #'CVC-ColonDB' #'Kvasir' #ETIS-LaribPolypDB' #BCAI-IGH
    
# 创建命令行解析器。
    parser = argparse.ArgumentParser()
    # network related parameters
# 编码器名称。
    parser.add_argument('--encoder', type=str,
# 默认 PVTv2-B2；可改 PVTv2/ResNet 已实现变体。
                        default='pvt_v2_b2', help='Name of encoder: pvt_v2_b2, pvt_v2_b0, resnet18, resnet34 ...')
# MSCB 扩张倍数。
    parser.add_argument('--expansion_factor', type=int,
                        default=2, help='expansion factor in MSCB block')
# MSDC 多尺度卷积核列表。
    parser.add_argument('--kernel_sizes', type=int, nargs='+',
                        default=[1, 3, 5], help='multi-scale kernel sizes in MSDC block')
# LGAG 卷积核大小。
    parser.add_argument('--lgag_ks', type=int,
                        default=3, help='Kernel size in LGAG')
# MSCB 激活类型。
    parser.add_argument('--activation_mscb', type=str,
                        default='relu6', help='activation used in MSCB: relu6 or relu')
# 出现后关闭并行深度卷积。
    parser.add_argument('--no_dw_parallel', action='store_true', 
                        default=False, help='use this flag to disable depth-wise parallel convolutions')
# 出现后用 concat 聚合，默认 add。
    parser.add_argument('--concatenation', action='store_true', 
                        default=False, help='use this flag to concatenate feature maps in MSDC block')
# 出现后不加载编码器 ImageNet 预训练权重。
    parser.add_argument('--no_pretrain', action='store_true', 
                        default=False, help='use this flag to turn off loading pretrained enocder weights')
# PVT 预训练文件目录。
    parser.add_argument('--pretrained_dir', type=str,
                        default='./pretrained_pth/pvt/', help='path to pretrained encoder dir')
# 该参数参与 run_id，但 train() 当前固定使用“五路 paper 风格损失”，没有读取 opt.supervision 分支。
    parser.add_argument('--supervision', type=str,
                    default='mutation', help='loss supervision: mutation, deep_supervision or last_layer')    
# 总 epoch 数，默认 200。
    parser.add_argument('--epoch', type=int, default=200)
# AdamW 初始学习率 5e-4；注释说明无 scheduler 时通常用 1e-4。
    parser.add_argument('--lr', type=float, default=0.0005) # base learning rate is 0.0005 for CosineAnnealingLR and 0.0001 for no scheduler
# 训练 batch，当前 8。
    parser.add_argument('--batchsize', type=int, default=8)
# val/test 推理 batch。
    parser.add_argument('--test_batchsize', type=int, default=8)
# 网络输入尺寸 352。
    parser.add_argument('--img_size', type=int, default=352)
# 梯度值截断阈值。
    parser.add_argument('--clip', type=float, default=0.5)
# adjust_lr 阶梯衰减倍率。
    parser.add_argument('--decay_rate', type=float, default=0.1)
# 阶梯衰减周期 300；默认仅训练 200 epoch，因此 epoch//300 始终 0，decay 因子为 1。
    parser.add_argument('--decay_epoch', type=int, default=300)
# 是否读取彩色图；无 type/action，命令行传值可能成为字符串。
    parser.add_argument('--color_image', default=True)
# 是否做 loader 数据增强；同样无 type/action。
    parser.add_argument('--augmentation', default=True)
# 训练划分根目录，默认由上方硬编码 dataset_name 生成。
    parser.add_argument('--train_path', type=str, default=f'../data/polyp/target/{dataset_name}/train/')
# 数据集根目录，test() 再追加 val/test。
    parser.add_argument('--test_path', type=str, default=f'../data/polyp/target/{dataset_name}/')
# 单 run 权重目录会在循环中覆盖此初始空字符串。
    parser.add_argument('--train_save', type=str, default='') 
# 解析全部参数。
    opt = parser.parse_args()

# 固定连续执行 5 个 run，用不同初始化/时间戳形成独立实验。
    for run in [1,2,3,4,5]:
# 保存 val/test 每轮 Dice 序列。
        dict_plot = {'val': [], 'test': []}
# 当前 run 最好验证 Dice。
        best = 0.0
# 最好验证时点对应的测试 Dice。
        test_dice_at_best_val = 0.0
# 仅累计训练阶段秒数。
        total_train_time = 0

# 目录名片段：concat 或 add。
        if opt.concatenation:
# 拼接模式。
            aggregation = 'concat'
# 默认相加模式。
        else: 
# 记录 add。
            aggregation = 'add'
        
# 目录名片段：series 或 parallel。
        if opt.no_dw_parallel:
# 串行深度卷积。
            dw_mode = 'series'
# 默认并行。
        else: 
# 记录 parallel。
            dw_mode = 'parallel'

# 当前时分秒用于降低 5 个 run 的重名概率。
        timestamp = time.strftime('%H%M%S')
# run_id 编入数据集、结构、batch、学习率、epoch、增强、run 序号和时间。
        run_id = (f"{dataset_name}_{opt.encoder}_EMCAD_kernel_sizes_{opt.kernel_sizes}_dw_{dw_mode}_{aggregation}_lgag_ks_{opt.lgag_ks}_ef{opt.expansion_factor}_act_mscb_{opt.activation_mscb}_bs{opt.batchsize}_cas_lr{opt.lr}_"
# 第二段续接 epoch 等信息。
                      f"e{opt.epoch}_aug{opt.augmentation}_run{run}_t{timestamp}")
# 清理列表字符串中的方括号和逗号空格。
        run_id = run_id.replace('[', '').replace(']', '').replace(', ', '_')
# 本 run 权重目录。
        opt.train_save = f'./model_pth/{run_id}/'
        
# 创建日志根目录。
        os.makedirs('logs', exist_ok=True)
# 创建权重目录。
        os.makedirs(opt.train_save, exist_ok=True)
        
# force=True 使五个 run 每次重新配置 logging 到各自文件。
        logging.basicConfig(filename=f'logs/train_log_{run_id}.log', level=logging.INFO, 
# 日志格式带完整时间。
                            format='[%(asctime)s] %(message)s', force=True)


        # Build model
# 历史模型构造示例，已注释，不执行。
        #model = EMCADNet(dw_parallel=dw_parallel, expansion_factor=expansion_factor, add=add, kernel_sizes=kernel_sizes, att_ks=att_ks, activation=activation, encoder=encoder, pretrain=pretrain, head=head, bbox=False, cds=False) # head='SAH'
# 创建 1 通道二分类 EMCAD；四个头都输出前景 logits。
        model = EMCADNet(num_classes=1, kernel_sizes=opt.kernel_sizes, expansion_factor=opt.expansion_factor, dw_parallel=not opt.no_dw_parallel, add=not opt.concatenation, lgag_ks=opt.lgag_ks, activation=opt.activation_mscb, encoder=opt.encoder, pretrain= not opt.no_pretrain, pretrained_dir=opt.pretrained_dir)

# 设备变量表面支持 CPU 回退。
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 三引号包围的是未执行的字符串常量，内部 DataParallel 示例不会运行。
        '''if torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            model = nn.DataParallel(model)'''

# 将模型移到 device；但 train()/test() 内使用 .cuda()，完整脚本实际仍要求 CUDA。
        model.to(device)

# 打印主干与解码器名称。
        print(f"Encoder: {opt.encoder} | Decoder: EMCAD")
# 统计参数量/FLOPs并通过 logging 记录；该辅助函数可能执行一次虚拟前向。
        cal_params_flops(model, opt.img_size, logging)
# 创建 AdamW，位置参数 opt.lr 作为学习率，weight_decay=1e-4。
        optimizer = torch.optim.AdamW(model.parameters(), opt.lr, weight_decay=1e-4)
# 余弦调度在 opt.epoch 轮内从当前学习率逐步降至 1e-6。
        scheduler = CosineAnnealingLR(optimizer, T_max=opt.epoch, eta_min=1e-6)

# 创建训练 DataLoader。
        train_loader = get_loader(
# 训练图像和掩膜目录。
            image_root=f'{opt.train_path}/images/', gt_root=f'{opt.train_path}/masks/',
# batch 大小和网络输入尺寸。
            batchsize=opt.batchsize, trainsize=opt.img_size, 
# 打乱、增强、训练模式和彩色读取选项。
            shuffle=True, augmentation=opt.augmentation, split='train', color_image=opt.color_image
# loader 构造结束。
        )

# epoch 从 1 到 opt.epoch，便于日志直接显示人类轮次。
        for epoch in range(1, opt.epoch + 1):
# 旧阶梯调度先运行；默认 decay_epoch=300、epoch<=200 时因子始终 1，通常不改变学习率。
# 若改小 decay_epoch，它会与下面 CosineAnnealingLR 叠加，学习率不再是单一余弦曲线。
            adjust_lr(optimizer, opt.lr, epoch, opt.decay_rate, opt.decay_epoch)
# 训练一轮、评估 val/test、保存 last/best。
            train(train_loader, model, optimizer, epoch, opt, run_id)
# epoch 末推进余弦调度器。
            scheduler.step()
        # FINAL SUMMARY
        
# 组装本 run 最终摘要文本。
        summary = (f"\n{'='*40}\nFINAL RESULTS: {run_id}\n"
# 最好验证 Dice。
                   f"Best Val Dice: {best:.4f}\n"
# 对应 epoch 的 test Dice。
                   f"Test Dice at Best Val: {test_dice_at_best_val:.4f}\n"
# 只含训练循环、不含验证测试的累计耗时。
                   f"Total Train Time: {total_train_time:.2f}s\n{'='*40}")
# 打印摘要。
        print(summary)
# 写入当前 run 日志。
        logging.info(summary)
        