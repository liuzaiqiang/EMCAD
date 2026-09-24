# ============================== 初学者阅读总览 ==============================
# 本文件实现 Synapse 的训练闭环，是理解“数据如何经过模型并产生梯度”的核心入口。
# 主链路：Synapse_dataset -> DataLoader -> EMCADNet 四个 logits -> 监督组合 ->  0.3*交叉熵+0.7*Dice -> backward -> AdamW.step -> 整体病例验证 -> 保存权重。
# 论文对应：
# 第 3.3 节解释四个分割头、多阶段损失和输出聚合；
# 第 4.1 节给出  Synapse 的 224x224、300 epoch、batch size 6、AdamW、lr/weight_decay=1e-4、CE:Dice=0.3:0.7。
# mutation 对 4 个输出的 15 个非空子集逐一计算损失，是当前仓库代码中的监督实现细节；阅读时要与论文第 3.3 节的公式/描述对照。
# 训练以二维切片为样本：[B,1,H,W]；验证以三维病例为样本：[1,D,H,W]，
# val_single_volume 再逐切片送入网络，并按8个前景器官汇总 Dice。
# ========================================================================

# argparse 当前未被直接使用，属于从通用训练模板保留的工程导入。
import argparse
# logging 同时把训练过程写入 log.txt 并输出到终端。
import logging
# os 用于拼接 checkpoint 与 TensorBoard 日志路径。
import os
# random 用于给每个 DataLoader worker 设置不同但可复现的随机种子。
import random
# sys.stdout 被注册为 logging 的终端输出流。
import sys
# time 当前执行路径未直接使用，属于保留导入。
import time
# NumPy 用于输出索引、验证指标累加和均值计算。
import numpy as np
# tqdm 为 epoch/病例循环提供进度条。
from tqdm import tqdm

# torch 提供张量运算、自动求导、设备迁移和权重保存。
import torch
# nn 主要用于多 GPU时的 nn.DataParallel。
import torch.nn as nn
# optim 提供 AdamW；被注释的 SGD 方案也来自这里。
import torch.optim as optim
# SummaryWriter 将标量写成 TensorBoard 可读取的事件文件。
from tensorboardX import SummaryWriter
# CrossEntropyLoss 对每个像素执行 9 类分类，输入 logits、目标为整数类别编号。
from torch.nn.modules.loss import CrossEntropyLoss
# DataLoader 负责批处理、打乱、并行读取与 pin memory。
from torch.utils.data import DataLoader
# transforms.Compose 串联训练样本变换；当前只有 RandomGenerator。
from torchvision import transforms
# AMP 组件在当前 Synapse 训练函数中没有实际使用，属于保留导入。
from torch.cuda.amp import GradScaler, autocast

# Synapse_dataset 读取二维训练 NPZ/三维 H5；RandomGenerator 做同步增强和缩放。
from utils.dataset_synapse import Synapse_dataset, RandomGenerator
# powerset生成监督组合；DiceLoss 计算多类软Dice；两个volume函数负责整体验证。
from utils.utils import powerset, one_hot_encoder, DiceLoss, val_single_volume


# 训练过程中调用的整病例评估函数；它返回所有病例、所有前景类别的平均 Dice 标量。
# 注意：split 名为 test_vol，是否属于“验证集”取决于你的实际列表划分；若它是官方测试集，每个 epoch 用它挑 best.pth 会造成测试集参与模型选择。这里仅忠实说明现有行为，不改逻辑。
def inference(args, model, best_performance):
    # 用完整体目录和 test_vol.txt 创建病例级数据集；单样本通常 image/label=[D,H,W]。
    db_test = Synapse_dataset(base_dir=args.volume_path, split="test_vol",
                              list_dir=args.list_dir, nclass=args.num_classes)
    # batch_size=1 表示每次评估一个病例；不打乱才能保持列表与日志顺序稳定。
    testloader = DataLoader(db_test, batch_size=1,
                            shuffle=False, num_workers=1)
    # 记录本轮需要处理的病例数，即 DataLoader 的迭代次数。
    logging.info("{} test iterations per epoch".format(len(testloader)))
    # 切换到eval模式，冻结 BatchNorm 运行统计并关闭 Dropout 的随机行为。
    model.eval()
    # 先用标量 0 初始化；第一次相加NumPy数组后会变成形状 [num_classes-1] 的向量。
    metric_list = 0.0
    # enumerate 给出病例序号，tqdm 显示进度；sampled_batch 是 DataLoader 拼成的字典。

    # enumerate(testloader)：testloader 是一个 PyTorch 的数据加载器，每次迭代返回一个批次的数据（通常是一个包含图像、标签和病例名的字典）。
    # enumerate 函数为每次迭代包裹了一个从 0 开始的索引 i_batch，这样在循环体内既能获取当前批次的序号，也能获取对应的数据 sampled_batch。

    # tqdm(...)：tqdm 是一个快速、可扩展的进度条工具。将 enumerate(testloader) 的迭代器传入 tqdm 后，它会在终端实时渲染出一个进度条，自动显示已处理的批次数、总批次数、耗时以及预计剩余时间等信息。
    # 该for循环每执行一次，就处理一个测试病例；循环的总执行次数完全取决于测试集划分（如 test_vol.txt 列表）中包含的病例数量
    for i_batch, sampled_batch in tqdm(enumerate(testloader)):
        # 读取原始病例最后两个空间维度；h、w 在本函数后续没有使用，是保留的调试信息。
        h, w = sampled_batch["image"].size()[2:]
        # image/label 通常为 [1,D,H,W]；case_name 从长度为 1 的字符串列表中取出。
        image, label, case_name = sampled_batch["image"], sampled_batch["label"], sampled_batch['case_name'][0]
        # val_single_volume 会逐切片缩放到 [img_size,img_size]，前向推理、argmax，再还原原尺寸。
        # 返回长度 num_classes-1 的列表；每个元素是对应前景类别在该病例上的 Dice。
        metric_i = val_single_volume(image, label, model, classes=args.num_classes,
                                     patch_size=[args.img_size, args.img_size],
                                     # case 用于日志语义；z_spacing 传入但当前 val_single_volume 的 Dice 路径不使用物理间距。
                                     case=case_name, z_spacing=args.z_spacing)
        # 把当前病例的 8 类 Dice 转为 NumPy 数组并累加到跨病例总和。
        # metric_list在for循环内的含义：在遍历测试集的 for 循环中，metric_list 依次累加每个病例的评估指标。此时它的数值含义是所有已遍历病例的各前景类别 Dice 系数之和。它是一个形状为 [num_classes-1] 的一维 NumPy 数组，其中每个元素对应当前已处理病例在某个前景类别上的 Dice 值总和。
        metric_list += np.array(metric_i)
    # 除以病例数得到每个前景类别的病例平均 Dice，形状仍是 [8]。
    # 在 for 循环结束后，代码执行了 metric_list = metric_list / len(db_test)。此时它的数值含义变为每个前景类别在所有测试病例上的平均 Dice 系数。数组形状仍为 [num_classes-1]，其中第 i 个元素代表第 i+1 个前景类别在整个测试集上的宏平均 Dice。
    # metric_list的值的格式的举例：array([0.82, 0.75, 0.88, 0.79, 0.91, 0.70, 0.85, 0.77])
    metric_list = metric_list / len(db_test)
    # 再沿类别维求均值，得到单个 macro mean Dice，用作 checkpoint 选择指标。
    # 随后，代码通过 np.mean(metric_list, axis=0) 对该数组沿类别维度再次求均值，得到所有前景类别的总体宏平均 Dice，用作模型 checkpoint 选择的依据。
    performance = np.mean(metric_list, axis=0)
    # 提前判断并更新历史最优值，使后续日志语义准确  2026-09-11凌晨添加
    if best_performance <= performance:
        best_performance = performance
    # 同时记录当前性能和进入函数前的历史最好性能，便于观察是否刷新 best.pth。
    logging.info('Testing performance in val model: mean_dice : %f, best_dice : %f' % ( performance, best_performance))
    # 返回 Python/NumPy 标量，供训练主循环比较大小。
    return performance


# Synapse 训练主函数：args 提供配置，model 是已创建的 EMCADNet，snapshot_path 是本次实验目录。
def trainer_synapse(args, model, snapshot_path):
    # 配置文件日志；每条记录带时分秒和毫秒，写入当前实验目录的 log.txt。
    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        # message 是调用 logging.info 传入的正文。
                        format='[%(asctime)s.%(msecs)03d] %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S')
    # 再挂一个 stdout handler，使同一条日志也实时显示在终端。
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    # 把全部命令行配置写入日志，这是复现实验时最基本的配置快照。
    logging.info(str(args))
    # 缓存基础学习率，后面创建优化器并在每次迭代写回参数组。
    base_lr = args.base_lr
    # 缓存类别总数；9 包含背景，将用于构造多类 DiceLoss。
    num_classes = args.num_classes
    # DataLoader 总 batch size 按“单卡 batch size x 期望 GPU 数”计算。
    # 后面若启用 DataParallel，PyTorch 会再把该批次切分到多张卡。
    batch_size = args.batch_size * args.n_gpu
    # 训练集按train.txt逐行读取二维 NPZ 切片。
    db_train = Synapse_dataset(base_dir=args.root_path, list_dir=args.list_dir, split="train", nclass=args.num_classes,
                               # Compose 目前只有一个变换：随机旋转/翻转，并缩放为统一的 224x224。
                               transform=transforms.Compose(
                                   # 变换输出 image=[1,H,W] float32、label=[H,W] int64。
                                   [RandomGenerator(output_size=[args.img_size, args.img_size])]))

    # 打印训练切片总数；这是切片数，不是患者/体数据数。
    print("The length of train set is: {}".format(len(db_train)))


    # DataLoader 为每个 worker 调用此函数，使不同 worker 不共享完全相同的 Python 随机序列。
    # 如果数据增强使用 Python 的 random，就可能造成：
    #       不同样本使用相同的旋转角度；
    #       不同样本进行相同的翻转；
    #        随机裁剪模式高度重复；
    #        多 worker 的随机增强失去独立性。
    # 这会降低数据增强的多样性。
    def worker_init_fn(worker_id):
        # worker0使用seed，worker1使用seed+1，依此类推。
        # 所有 worker 使用同一个实验的随机种子体系，但每个 worker 拥有自己的随机子序列。
        random.seed(args.seed + worker_id)

    # 构造训练批次：shuffle=True 每个epoch重排切片；pin_memory=True可加快CPU->GPU 拷贝。
    trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True,
                             worker_init_fn=worker_init_fn)
    # 下面两行原注释提示 Windows 多进程读取可能需要 num_workers=0；当前真正执行的是上面的 8。
    # windows下 num_workers需要改为0
    # trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True,worker_init_fn=worker_init_fn)
    # 选择设备；虽然这里有 CPU 回退，入口 train_synapse.py 在此之前已经 model.cuda()，所以整体仍要求 CUDA。
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 只有机器可见GPU数大于1 且用户要求 n_gpu>1 时，才包裹数据并行。
    if torch.cuda.device_count() > 1 and args.n_gpu > 1:
        # 打印的是全部可见 GPU 数，不一定等于 args.n_gpu。
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        # DataParallel 默认使用所有可见设备，并在 batch 维切分输入、在主卡汇总四级输出。
        model = nn.DataParallel(model)
    # 确保模型最终位于选择的设备；对 DataParallel 而言主模块位于默认 CUDA 设备。
    model.to(device)

    # 切换训练模式，使 BatchNorm 更新统计量、Dropout（若有）启用随机失活。
    model.train()
    # 交叉熵输入 [B,9,H,W] 原始 logits，目标 [B,H,W] long；内部完成 log-softmax/NLL。
    ce_loss = CrossEntropyLoss()
    # DiceLoss 会 softmax 后把整数标签 one-hot，逐类计算并平均 9 个类别的 Dice loss。
    dice_loss = DiceLoss(num_classes)

    # 这是保留的 SGD 备选方案
    # optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    # 当前实际优化器是 AdamW；weight decay=1e-4 与论文第 4.1 节设置一致。
    optimizer = optim.AdamW(model.parameters(), lr=base_lr, weight_decay=0.0001)
    # TensorBoard 事件写入 snapshot_path/log，可用 total_loss 与 lr 曲线观察训练。
    writer = SummaryWriter(snapshot_path + '/log')
    # 全局迭代计数从 0 开始，每处理一个 batch 后加 1。
    iter_num = 0
    # 外层 epoch 上限直接来自 --max_epochs。
    max_epoch = args.max_epochs
    # 实际最大迭代数由 epoch 数乘每 epoch 批次数决定；不会使用 args.max_iterations。
    max_iterations = args.max_epochs * len(trainloader)
    # 写入每 epoch 批次数和预计总迭代数，便于核对训练是否完整。
    logging.info("{} iterations per epoch. {} max iterations ".format(len(trainloader), max_iterations))
    # 历史最好验证 Dice；初始 0，后续大于等于它时覆盖 best.pth。
    best_performance = 0.0
    # tqdm 包装 epoch 范围，ncols=70 固定终端进度条宽度。创建一个带进度条的 epoch 循环。ncols=70 限制进度条宽度。
    iterator = tqdm(range(max_epoch), ncols=70)

    # 外层循环遍历 epoch_num=0..max_epoch-1。
    for epoch_num in iterator:
        # 每个 epoch 开始前确保模型处于训练模式  2026-08-31 16:45新增
        model.train()
        # 内层循环逐批读取字典；i_batch 是当前 epoch 内的 batch 序号。
        for i_batch, sampled_batch in enumerate(trainloader):
            # DataLoader 堆叠后，image_batch 通常 [B,1,224,224]，label_batch 通常 [B,224,224]。
            image_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            # 将图像和标签移到默认 GPU
            # squeeze(1)只在标签含冗余通道维[B,1,H,W] 时移除它。
            # CrossEntropyLoss 最终要求标签形状 [B,H,W]，每个值是 0.......8的类别索引。
            image_batch, label_batch = image_batch.cuda(), label_batch.squeeze(1).cuda()
            # 训练模式前向：EMCADNet 返回由粗到细的四个全分辨率 logits，均为 [B,9,224,224]。
            # 论文第 3.2/3.3 节把它们记作多阶段分割输出；这里变量名P表示 prediction list。
            # 注意，mode='train' 只是你这个 EMCAD forward() 的参数，不能替代 PyTorch 的 model.train()。真正决定 BatchNorm、Dropout 是否处于训练状态的是模型的 training 属性。
            P = model(image_batch, mode='train')
            # 兼容只返回单张量的其他模型：统一包装为列表，后续监督代码只处理 list。
            # 如果模型只返回一个 Tensor，就把它包装为只有一个元素的列表。这样后面统一使用 P[index]，不用为单输出模型和多输出模型各写一套逻辑。
            if not isinstance(P, list):
                # 单输出模型经包装后 len(P)=1。
                P = [P]
            # 监督组合只需根据输出个数计算一次；ss 之后在后续所有 batch/epoch 中复用。
            # 只在第 0 个 epoch 的第 0 个 batch 执行一次。输出数量和监督模式不会在训练中途改变，因此 ss 只需初始化一次，后面 batch 复用。
            if epoch_num == 0 and i_batch == 0:
                # 对当前 EMCADNet，n_outs=4。
                n_outs = len(P)
                # 生成 [0,1,2,3]，分别索引 p4、p3、p2、p1 四级输出。
                out_idxs = list(np.arange(n_outs))
                # mutation 使用幂集监督：4 个输出共有 2^4=16 个子集，空集稍后跳过，实际 15 组。
                if args.supervision == 'mutation':
                    # powerset 返回各类输出索引组合；每组 logits 在通道与空间位置上逐元素相加。
                     # powerset([0,1,2,3]) 生成所有子集：
                    # [], [0], [1], [2], [3],
                    # [0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3],
                    # [0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3],
                    # [0, 1, 2, 3]
                    ss = [x for x in powerset(out_idxs)]
                # deep_supervision 只对每个输出单独计算，共 4 组，不计算输出和的组合。
                elif args.supervision == 'deep_supervision':
                    # 结果为 [[0],[1],[2],[3]]。 四个输出各自单独计算一次损失，不计算输出之间的组合。
                    ss = [[x] for x in out_idxs]
                # 任何其他字符串（通常 last_layer）只监督列表最后一个输出。
                else:
                    # Python 索引 -1 指向 P[-1]，即最终/最高分辨率分割头 p1。
                    # 这就是只监督最终输出的 last_layer 模式。
                    ss = [[-1]] 
                # 打印监督组合，便于确认本次实验究竟累加了多少项损失。
                print("深监督组合：",ss)
            # 当前 batch 的总损失从 0 开始；首次加 Tensor 后自动变为带梯度的标量 Tensor。
            loss = 0.0
            # 每个监督组内部采用 30% 交叉熵 + 70% Dice，与论文第 4.1 节一致。
            w_ce, w_dice = 0.3, 0.7
            # 遍历监督组合；mutation 为 16 次循环，其中空集不产生损失。s 是当前组合的索引列表。例如 mutation 中可能是 [0]、[1,3] 或 [0,1,2,3]。
            for s in ss:
                # 当前组合的聚合 logits 初始化为 0；加上第一个输出后成为 [B,9,H,W] Tensor。
                iout = 0.0
                # 空集合没有可监督输出，直接进入下一组合。空集没有任何输出可相加。如果不跳过，iout 会保持普通数字 0.0，既没有预测形状，也没有梯度关系，不能送入损失函数。这里的 continue 只跳过当前组合，不会跳过整个 batch。
                if (s == []):
                    # continue 只跳过当前 s，不跳过整个 batch。
                    continue
                # 把该组合中指定的预测头逐元素求和；这里相加的是 logits，不是 softmax 概率。
                for idx in range(len(s)):
                    # s[idx]是P的索引；各输出已上采样到相同 [B,9,H,W]，因此可以直接相加。
                    iout += P[s[idx]] 

                # 多类交叉熵惩罚每个像素的类别预测；label_batch[:] 只是取完整标签，long保证整数类型。CE 在每个像素比较 9 个类别 logits 与真实类别，内部完成 log-softmax 和负对数似然。
                # label_batch[:] 表示取全部标签；关键操作是后面的 .long()，保证 CE 接收整数类别索引。
                loss_ce = ce_loss(iout, label_batch[:].long())
                # DiceLoss 内部 softmax=True，把 logits 转为 9 类概率并与 one-hot 标签计算重叠。训练阶段使用连续概率计算 Dice，而不是 argmax 后的离散图，这样梯度才能回到网络。
                loss_dice = dice_loss(iout, label_batch, softmax=True)
                # 把该组合的加权损失累加到总损失；没有再除以组合数。
                # 因此 mutation(15组)的 loss 数值尺度天然大于 deep_supervision(4组)，两者不可直接横比。
                loss += (w_ce * loss_ce + w_dice * loss_dice)

            # 候选融合模式的显式训练项；fusion_loss_weight=0 时 baseline 路径完全不变。
            fusion_model = model.module if hasattr(model, 'module') else model
            if getattr(args, 'fusion_loss_weight', 0.0) > 0 and getattr(fusion_model, 'fusion_mode', 'p1') != 'p1':
                loss = loss + float(args.fusion_loss_weight) * fusion_model.fusion_auxiliary_loss(
                    P, label_batch, ce_loss, dice_loss,
                    reliability_loss_weight=getattr(args, 'reliability_loss_weight', 1.0))
            """
            用两个输出的极小例子模拟
            若只有 P=[P0,P1]，使用 mutation：
            powerset([0,1]) = [[], [0], [1], [0,1]]
            执行顺序：
                []：跳过；
                [0]：用 P0 算一次 CE+Dice；
                [1]：用 P1 算一次 CE+Dice；
                [0,1]：用 P0+P1 再算一次；
                三次结果相加，得到 batch loss。
            当前 EMCAD 有 4 个输出，所以是 15 次非空组合，而不是 3 次。
            """
            # 清除上一个 batch 残留梯度；默认把梯度张量写成 0。
            optimizer.zero_grad()
            # 从总损失沿 15/4/1 条监督路径反向传播，梯度最终汇合到共享编码器和解码器参数。同一个输出若出现在多个组合中，会收到多个路径的梯度贡献；
            loss.backward()
            # AdamW 根据当前梯度、学习率和权重衰减更新一次全部可训练参数。
            optimizer.step()
            # 因此 mutation 不是“多打印几个损失”，而是确实改变了梯度。四个输出头以及它们之前共享的 decoder、encoder 都会受到这些监督路径影响。

            """
            P 是概率吗？不是。它是 logits。概率只在 CE 内部或 DiceLoss 的 softmax=True 分支中产生。
            四个输出代表四个类别吗？不是。它们代表四个解码阶段；每个输出内部都有 9 个类别通道。
            为什么 loss 可能很大？因为一个 batch 中可能累加 15 组损失，尤其 mutation 的数值尺度天然较大。
            为什么 Dice 要 one-hot？CE 使用类别索引图；Dice 要逐类别比较预测概率图和目标掩膜，所以需要把标签变成 [B,9,H,W]。
            核心思想是：模型产生多个尺度的分割 logits，训练器按监督策略选择单个输出或输出组合，用 CE 和 Dice 共同约束，再把所有监督路径的损失交给反向传播。
            """
            # 原作者保留的多项式学习率衰减公式
            # lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9 # we did not use this

            # 当前学习率始终等于 base_lr，即 constant learning rate。
            lr_ = base_lr
            # 遍历优化器参数组；当前通常只有一个组。
            for param_group in optimizer.param_groups:
                # 把组学习率写回常数；因创建 AdamW 时已是 base_lr，这里主要用于明确策略。
                param_group['lr'] = lr_

            # 一个 batch 更新完成后，全局 step 加 1。
            iter_num = iter_num + 1
            # 将本step学习率写入 TensorBoard 的 info/lr 曲线。
            writer.add_scalar('info/lr', lr_, iter_num)
            # 将包含所有监督组合之和的总损失写入 info/total_loss。
            writer.add_scalar('info/total_loss', loss, iter_num)
            # 每 50个全局 step 打印一次细粒度训练日志。
            if iter_num % 50 == 0:
                # loss.item() 把单元素 GPU Tensor 同步取回为 Python 数值用于格式化。
                logging.info('iteration %d, epoch %d : loss : %f, lr: %f' % (
                    iter_num, epoch_num, loss.item(), lr_))

        # 一个 epoch 结束后记录最后一个 batch 的 loss；它不是该 epoch 所有 batch 的平均值。
        # 它在一个 epoch 的所有训练批次都完成后才执行。
        # 记录对象：代码直接使用了 loss.item() 和 lr_，而没有对整个 epoch 的损失进行累加或求平均。因此，它记录的是当前 epoch 中最后一个批次的损失值和学习率，而非该 epoch 的平均损失。
        # 这段代码则确保无论 epoch 内包含多少个批次，每个 epoch 结束时都会有一行固定的记录，方便按 epoch 维度快速追踪训练状态。

        logging.info('iteration %d, epoch %d : loss : %f, lr: %f' %
                     (iter_num, epoch_num, loss.item(), lr_))

        # 固定使用last.pth 文件名，所以每个 epoch 都覆盖为最新模型状态。注意，这一步只是拼接路径
        save_mode_path = os.path.join(snapshot_path, 'last.pth')
        # 这里只保存 state_dict，不含优化器状态、epoch 和随机数状态，不能完整无缝恢复训练。
        # 多 GPU 时 model.state_dict() 的键通常带 module.前缀；单GPU测试加载时需注意格式一致性。
        # 写入模型参数 将模型当前的 state_dict（即所有可学习参数的字典）序列化并写入上述路径。
        # 覆盖更新：由于文件名硬编码为 last.pth，每个 epoch 结束后都会覆盖该文件。因此，它始终保存的是训练过程中最后一个 epoch 的模型状态，而非历史最优状态（最优状态由 best.pth 保存）。
        # 非完整快照：这里仅保存了模型参数，未包含优化器状态、epoch 编号或随机数状态，因此无法用于无缝恢复中断的训练流程。
        torch.save(model.state_dict(), save_mode_path)

        # 在完整体数据上计算宏平均 Dice；函数内部会执行 model.eval()。
        performance = inference(args, model, best_performance)
        # 重要的现有代码行为：本函数只在进入 epoch 循环前调用过一次 model.train()，
        # inference 后没有在下一 epoch 开头再次切回 train；因此第 2 个 epoch 起模型保持 eval 模式。
        # 这会影响 BatchNorm/Dropout，但依照用户约束这里只解释，不修改该逻辑。——已经修改

        # 每 50 个 epoch 另存一个阶段性 checkpoint。
        save_interval = 50

        # 当前指标大于或等于历史最好值时更新；相等也会覆盖已有 best.pth。
        if (best_performance <= performance):
            # 先更新内存中的最好 Dice，供下一次 inference 日志与比较使用。
            best_performance = performance
            # 最优权重总是写到固定文件 best.pth。
            save_mode_path = os.path.join(snapshot_path, 'best.pth')
            # 保存触发该最好指标时的全部模型参数。
            torch.save(model.state_dict(), save_mode_path)
            # 在文本日志中记录实际保存路径。
            logging.info("save model to {}".format(save_mode_path))

        # epoch_num 从 0 开始，但判断用 epoch_num+1，所以第 50、100、150...轮触发。
        if (epoch_num + 1) % save_interval == 0:
            # 文件名使用零基 epoch_num，因此第 50 轮会保存为 epoch_49.pth。
            save_mode_path = os.path.join(
                snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
            # 写入阶段性模型权重。
            torch.save(model.state_dict(), save_mode_path)
            # 记录阶段性文件位置。
            logging.info("save model to {}".format(save_mode_path))

        # 最后一个 epoch 时再次确保保存最终权重；默认 300 轮对应 epoch_num=299。
        if epoch_num >= max_epoch - 1:
            # 最终文件名同样采用零基编号，例如 epoch_299.pth。
            save_mode_path = os.path.join(
                snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
            # 保存最终状态；若刚好也命中 save_interval，可能对同一文件写两次。
            torch.save(model.state_dict(), save_mode_path)
            # 写入最终保存日志。
            logging.info("save model to {}".format(save_mode_path))
            # 主动关闭 tqdm 进度条，避免终端残留未结束状态。
            iterator.close()
            # 跳出 epoch 循环；在最后一轮此操作与自然结束等价。
            break

    # 刷新并关闭 TensorBoard writer，确保事件文件完整落盘。
    writer.close()
    # 返回人类可读状态字符串；train_synapse.py 当前没有接收或打印这个返回值。
    return "Training Finished!"
