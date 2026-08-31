# ============================== 初学者阅读总览 ==============================
# 本文件是 Synapse 多器官分割的“启动入口”，自身不实现网络层，也不执行具体的反向传播。
# 执行链为：解析命令行 -> 固定随机性 -> 整理数据集配置 -> 生成实验目录名 -> 创建 EMCADNet -> 把模型放到 GPU -> 调用 trainer.py 中的 trainer_synapse。
# 论文对应关系：网络整体见仓库论文第 3.2 节及图 2；多阶段输出监督见第 3.3 节；
# Synapse 的 224x224、300 epoch、batch size 6、AdamW、1e-4 学习率/权重衰减、 0.3*CrossEntropy+0.7*Dice 等设置见论文第 4.1 节 Implementation details。
# 这里的 num_classes=9 表示“背景 1 类 + 8 个前景器官”，不是 9 个前景器官。
# 典型训练张量：输入 image=[B,1,224,224]，标签 label=[B,224,224]； EMCADNet 内部先把单通道 CT 映射为 3 通道，再输出 4 个 [B,9,224,224] logits。
# ========================================================================

# argparse 负责把命令行参数（例如 --batch_size 6）转换为 args 对象。
import argparse
# logging 在本入口中虽被导入，但实际日志配置位于 trainer.py；这是保留的工程导入。
import logging
# os 用于检查并创建保存权重和日志的实验目录。
import os
# random 控制 Python 标准库层面的随机数据增强选择。
import random
# NumPy 用于设置其独立随机数生成器的种子。
import numpy as np
# torch 提供张量、随机种子、CUDA 与模型运行能力。
import torch
# nn 在本文件当前执行路径中未直接使用；模型类内部使用 torch.nn。
import torch.nn as nn
# cudnn 开关控制卷积算法选择方式，影响速度与可复现性。
import torch.backends.cudnn as cudnn

# EMCADNet 封装“编码器 + EMCAD 解码器 + 四个分割头”；结构细节在 lib/networks.py。
from lib.networks import EMCADNet
# trainer_synapse 承担 DataLoader、损失、反向传播、验证和 checkpoint 保存。
from trainer import trainer_synapse

# 创建参数解析器；下方所有 add_argument 都把一个可配置项注册到该解析器。
parser = argparse.ArgumentParser()

# 训练数据根目录：每个 Synapse 二维切片通常对应一个 .npz 文件。
parser.add_argument('--root_path', type=str, default='../data/synapse/train_npz', help='root dir for data')
# 完整体数据目录：验证/测试按病例读取 .npy.h5，   而不是逐切片 .npz。
parser.add_argument('--volume_path', type=str, default='../data/synapse/test_vol_h5',
                    help='root dir for validation volume data')
# 数据集键名稍后用于查询 dataset_config 和 trainer 映射表；当前只注册 Synapse。
parser.add_argument('--dataset', type=str, default='Synapse', help='experiment_name')
# 划分列表目录应包含 train.txt、test_vol.txt 等文本清单。
parser.add_argument('--list_dir', type=str, default='./lists/lists_Synapse', help='list dir')
# 多分类输出通道数；9=背景(0)+8个器官(1..8)，须与标签编号一致。
parser.add_argument('--num_classes', type=int, default=9, help='output channel of network')
# network related parameters
# 编码器名称决定主干结构和四级特征通道数；论文主结果主要使用 PVTv2-B0/B2。
parser.add_argument('--encoder', type=str, default='pvt_v2_b2',
                    help='Name of encoder: pvt_v2_b2, pvt_v2_b0, resnet18, resnet34 ...')
# MSCB 通道扩张倍数，控制中间特征宽度，也影响参数量与计算量。
parser.add_argument('--expansion_factor', type=int, default=2, help='expansion factor in MSCB block')
# MSDC 的多尺度深度卷积核；论文第 5.2 节消融后采用 [1,3,5]。
parser.add_argument('--kernel_sizes', type=int, nargs='+',
                    # nargs='+' 表示命令行可传入一个或多个整数，例如 --kernel_sizes 1 3 5。
                    default=[1, 3, 5], help='multi-scale kernel sizes in MSDC block')
# LGAG 的空间卷积核大小；它控制跳跃连接门控时观察的局部邻域。
parser.add_argument('--lgag_ks', type=int, default=3, help='Kernel size in LGAG')
# MSCB 激活函数选择；ReLU6 会把正激活截断到 6，常用于轻量网络。
parser.add_argument('--activation_mscb', type=str, default='relu6', help='activation used in MSCB: relu6 or relu')
# store_true 参数默认 False；命令行出现该旗标后变 True，从而关闭并行深度卷积。
parser.add_argument('--no_dw_parallel', action='store_true', default=False,
                    help='use this flag to disable depth-wise parallel convolutions')
# 出现该旗标后，MSDC 多尺度分支用通道拼接；默认则逐元素相加。
parser.add_argument('--concatenation', action='store_true', default=False,
                    help='use this flag to concatenate feature maps in MSDC block')
# 出现该旗标后不加载编码器预训练权重；注意“编码器架构”和“是否预训练”是两个独立选择。
parser.add_argument('--no_pretrain', action='store_true', default=False,
                    help='use this flag to turn off loading pretrained enocder weights')
# PVT 预训练权重目录；EMCADNet 会按 encoder 名称拼接具体 .pth 文件名。
parser.add_argument('--pretrained_dir', type=str, default='./pretrained_pth/pvt/',
                    help='path to pretrained encoder dir')
# 四输出监督策略：mutation=非空输出组合；deep_supervision=各输出单独；其余走最终输出。
parser.add_argument('--supervision', type=str, default='mutation',
                    help='loss supervision: mutation, deep_supervision or last_layer')
# 此参数在当前 trainer.py 中不控制循环终止，只参与实验目录命名；实际迭代数由 epoch 数决定。
parser.add_argument('--max_iterations', type=int, default=50000, help='maximum epoch number to train')
# 实际外层训练轮数；论文 Synapse 设置为 300 epoch。
parser.add_argument('--max_epochs', type=int, default=300, help='maximum epoch number to train')
# 单 GPU 每批切片数；trainer 中还会乘 args.n_gpu 得到 DataLoader batch_size。
parser.add_argument('--batch_size', type=int, default=6, help='batch_size per gpu')
# AdamW 的基础学习率；当前 trainer 保持常数，不使用被注释掉的多项式衰减。
parser.add_argument('--base_lr', type=float, default=0.0001, help='segmentation network learning rate')
# 输入二维切片的目标高宽；这里一个整数同时用于 H 和 W。
parser.add_argument('--img_size', type=int, default=224, help='input patch size of network input')
# 期望使用的 GPU 数；大于 1 时 trainer.py 才尝试 nn.DataParallel。
parser.add_argument('--n_gpu', type=int, default=1, help='total gpu')
# 1 表示确定性模式，0 表示允许 cuDNN benchmark 选择更快但可能不完全可复现的算法。
parser.add_argument('--deterministic', type=int, default=1, help='whether use deterministic training')
# 同一份数据、代码和环境下，固定种子用于尽量复现实验随机序列。
parser.add_argument('--seed', type=int, default=2222, help='random seed')
# 真正解析当前进程的命令行；未显式传入的选项采用上方 default。
args = parser.parse_args()

# 只有直接运行 python train_synapse.py 时才进入；被其他模块 import 时不会自动训练。
if __name__ == "__main__":
    # 非确定性分支优先运行速度：benchmark 会为当前输入形状搜索较快卷积算法。
    if not args.deterministic:
        # 开启自动算法基准测试；输入尺寸固定时通常能加速。
        cudnn.benchmark = True
        # 允许使用非确定性 CUDA 算法，因此重复运行可能产生细微差异。
        cudnn.deterministic = False
    # 确定性分支优先复现性，代价可能是速度下降。
    else:
        # 禁止基准搜索，避免算法选择随运行环境波动。
        cudnn.benchmark = False
        # 要求 cuDNN 尽量采用确定性实现；仍不代表跨硬件/版本绝对逐位一致。
        cudnn.deterministic = True

    # 固定 Python random；RandomGenerator 中的旋转/翻转分支依赖它。
    random.seed(args.seed)
    # 固定 NumPy 随机数；数据增强和 worker 内的 NumPy 操作可能依赖它。
    np.random.seed(args.seed)
    # 固定当前进程的 PyTorch CPU 随机数生成器。
    torch.manual_seed(args.seed)
    # 固定当前 CUDA 设备的随机数生成器；多 GPU 更完整的写法通常是 manual_seed_all。
    torch.cuda.manual_seed(args.seed)

    # 保存用户选择的数据集键；下面所有配置通过这个键取值。
    dataset_name = args.dataset
    # 数据集配置表把“通用训练代码”与“特定数据路径/类别数”连接起来。
    dataset_config = {
        # 当前入口仅实现 Synapse；传入其他名称会在索引时触发 KeyError。
        'Synapse': {
            # 二维训练切片目录。
            'root_path': args.root_path,
            # 完整 3D 病例目录，当前 trainer 把它用于每个 epoch 后的评估。
            'volume_path': args.volume_path,
            # 数据划分清单目录。
            'list_dir': args.list_dir,
            # 背景加 8 个器官的总类别数。
            'num_classes': args.num_classes,
            # 体数据 z 方向间距；这里固定 1，主要影响带物理距离的指标/保存元数据。
            'z_spacing': 1,
            # 结束 Synapse 配置字典。
        },
        # 结束整个数据集配置映射。
    }
    # 用配置表回填 args，确保下游 trainer 只读取统一的 args 接口。
    args.num_classes = dataset_config[dataset_name]['num_classes']
    # 回填训练切片根目录。
    args.root_path = dataset_config[dataset_name]['root_path']
    # 回填完整体数据根目录。
    args.volume_path = dataset_config[dataset_name]['volume_path']
    # 动态增加 z_spacing 属性，供验证函数使用。
    args.z_spacing = dataset_config[dataset_name]['z_spacing']
    # 回填划分列表目录。
    args.list_dir = dataset_config[dataset_name]['list_dir']

    # 把布尔开关转换为便于阅读的实验名片段；该字符串不改变模型计算。
    if args.concatenation:
        # concatenate 模式在目录名中记为 concat。
        aggregation = 'concat'
    # 默认逐元素相加模式。
    else:
        # add 与传给 EMCADNet 的 add=True 对应。
        aggregation = 'add'

    # 同样把深度卷积分支执行方式编码进实验目录名。
    if args.no_dw_parallel:
        # 关闭 parallel 后记作 series。
        dw_mode = 'series'
    # 默认使用论文消融支持的并行多尺度深度卷积。
    else:
        # parallel 表示多个尺度分支并行提取上下文。
        dw_mode = 'parallel'

    # 手工运行编号；若要真正区分多次运行，需要修改它或通过外部目录管理。
    run = 1
    # 拼出包含主干、卷积核、聚合方式、门控核、扩张倍数、激活和监督策略的实验标识。
    args.exp = (args.encoder + '_EMCAD_kernel_sizes_' + str(
        args.kernel_sizes) + '_dw_' + dw_mode + '_' + aggregation + '_lgag_ks_' + str(args.lgag_ks) + '_ef'
                + str(
                args.expansion_factor) + '_act_mscb_' + args.activation_mscb + '_loss_' + args.supervision + '_output_final_layer_Run' + str(
                run) + '_' + dataset_name + str(args.img_size))
    # 构造权重保存目录；外层目录含 args.exp，内层再次记录主要结构超参数。
    snapshot_path = "model_pth/{}/{}".format(args.exp, args.encoder + '_EMCAD_kernel_sizes_' + str(
        args.kernel_sizes) + '_dw_' + dw_mode + '_' + aggregation + '_lgag_ks_' + str(args.lgag_ks) + '_ef'
                                             + str(
        args.expansion_factor) + '_act_mscb_' + args.activation_mscb + '_loss_' + args.supervision + '_output_final_layer_Run' + str(
        run))
    # 把 Python 列表字符串中的括号和逗号空格清理掉，避免目录名出现 [1, 3, 5]。
    snapshot_path = snapshot_path.replace('[', '').replace(']', '').replace(', ', '_')

    # 预训练开启时追加 _pretrain；关闭时保持原目录名不变。
    snapshot_path = snapshot_path + '_pretrain' if not args.no_pretrain else snapshot_path
    # 非默认 max_iterations 只改变目录后缀；取字符串前两位形成类似 30k 的标签。
    snapshot_path = snapshot_path + '_' + str(args.max_iterations)[
        0:2] + 'k' if args.max_iterations != 50000 else snapshot_path
    # 非默认 epoch 数追加到目录名，便于区分实验。
    snapshot_path = snapshot_path + '_epo' + str(args.max_epochs) if args.max_epochs != 300 else snapshot_path
    # batch size 总是写入目录名。
    snapshot_path = snapshot_path + '_bs' + str(args.batch_size)
    # 非默认学习率才追加，默认 1e-4 不重复写。
    snapshot_path = snapshot_path + '_lr' + str(args.base_lr) if args.base_lr != 0.0001 else snapshot_path
    # 输入尺寸总是写入目录名。
    snapshot_path = snapshot_path + '_' + str(args.img_size)
    # 只有种子不等于 1234 时才追加；当前默认 2222 因而会出现 _s2222。
    snapshot_path = snapshot_path + '_s' + str(args.seed) if args.seed != 1234 else snapshot_path

    # 首次运行该配置时创建目录；已存在时复用，其中旧文件可能被本训练流程覆盖。
    # if not os.path.exists(snapshot_path):
        # 递归建立外层和内层目录。
    # os.makedirs(snapshot_path)

    # === 简化后的 snapshot_path（Windows / Linux 通用）===
    exp_name = f"run_seed{args.seed}"
    snapshot_path = os.path.join("model_pth", exp_name)

    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)

    # 创建完整分割网络：这些参数会继续传入 EMCAD 解码器，决定真正的模型结构。
    # 对 Synapse，num_classes=9，所以四个预测头各输出 9 通道原始 logits；这里不做 softmax。
    model = EMCADNet(num_classes=args.num_classes, kernel_sizes=args.kernel_sizes,
                     expansion_factor=args.expansion_factor, dw_parallel=not args.no_dw_parallel,
                     add=not args.concatenation, lgag_ks=args.lgag_ks, activation=args.activation_mscb,
                     encoder=args.encoder, pretrain=not args.no_pretrain, pretrained_dir=args.pretrained_dir)

    # 把模型参数移动到默认 CUDA 设备；本入口没有 CPU 回退，因此无 CUDA 时会直接报错。
    model.cuda()

    # 仅打印构建成功提示；实际参数量还会由 EMCADNet 构造函数打印。
    print('Model successfully created.')

    # 训练器映射允许按数据集名选择函数；当前仍然只支持 Synapse。
    trainer = {'Synapse': trainer_synapse, }
    # 调用真正训练函数，并把配置、已上 GPU 的模型、实验输出目录传入。
    trainer[dataset_name](args, model, snapshot_path)
