# ============================== 初学者阅读总览 ==============================
# 本文件加载训练得到的 Synapse checkpoint，对完整 3D CT 病例执行逐切片推理。
# 数据流：病例 [1,D,H,W] -> 每张切片缩放到 224x224 -> EMCAD 四级 logits ->
# 取最后一级 logits -> softmax + argmax -> 恢复原尺寸 -> 拼回 [D,H,W] 预测体 ->
# 对 8 个前景器官分别计算 Dice、HD95、Jaccard、ASD，并可保存 NIfTI/叠加图。
# 论文对应：模型结构见第 3.2 节与图 2；Synapse 结果见第 4.2.2 节相关表格；
# 数据集与指标定义见补充材料第 7.1、7.2 节。本文件中的目录命名、PNG/NIfTI 输出、
# best.pth 不存在时回退到最后 epoch 等，属于仓库工程实现，不是模型结构本身。
# ========================================================================

# argparse 解析测试路径、网络结构和保存选项。
import argparse
# logging 将逐病例与逐类别指标写入文件并同步输出到终端。
import logging
# os 负责 checkpoint、日志和预测目录的拼接/创建。
import os
# random 用于固定 Python 随机数状态；纯推理通常不应含随机增强。
import random
# sys.stdout 被添加为 logging 的流处理器。
import sys
# NumPy 用于把病例指标列表转数组并做宏平均。
import numpy as np
# torch 用于构建模型、加载 state_dict 和执行 GPU 推理。
import torch
# cuDNN 开关与训练入口一致，用于速度/确定性的取舍。
import torch.backends.cudnn as cudnn
# nn 在当前文件执行路径中未直接使用，是保留导入。
import torch.nn as nn
# DataLoader 每次提供一个完整病例。
from torch.utils.data import DataLoader
# tqdm 展示病例级推理进度。
from tqdm import tqdm

# Synapse_dataset 在 test_vol 分支读取完整 .npy.h5 病例。
from utils.dataset_synapse import Synapse_dataset
# test_single_volume 负责逐切片前向、恢复尺寸、计算 4 项指标并按需保存结果。
from utils.utils import test_single_volume

# EMCADNet 必须用与训练 checkpoint 一致的结构参数重新实例化。
from lib.networks import EMCADNet

# 创建命令行解析器。
parser = argparse.ArgumentParser()

# 完整体测试目录；默认名带 _new，与训练入口默认 volume_path 不同，需人工确认划分一致。
parser.add_argument('--volume_path', type=str,
                    default='../data/synapse/test_vol_h5_new', help='root dir for validation volume data')
# 数据集键用于查询后面的 dataset_config；当前只支持 Synapse。
parser.add_argument('--dataset', type=str,
                    default='Synapse', help='experiment_name')
# 9 个网络输出通道=背景+8个器官；测试指标循环会跳过背景类别 0。
parser.add_argument('--num_classes', type=int,
                    default=9, help='output channel of network')
# 列表目录应包含 test_vol.txt，每行对应一个 H5 病例名。
parser.add_argument('--list_dir', type=str,
                    default='./lists/lists_Synapse', help='list dir')

# network related parameters
# 编码器必须与训练时一致，否则 state_dict 的参数名/形状通常无法严格加载。
parser.add_argument('--encoder', type=str,
                    default='pvt_v2_b2', help='Name of encoder: pvt_v2_b2, pvt_v2_b0, resnet18, resnet34 ...')
# MSCB 扩张倍数属于网络结构，测试时必须复用训练配置。
parser.add_argument('--expansion_factor', type=int,
                    default=2, help='expansion factor in MSCB block')
# MSDC 多尺度核列表属于网络结构；论文默认 [1,3,5]。
parser.add_argument('--kernel_sizes', type=int, nargs='+',
                    default=[1, 3, 5], help='multi-scale kernel sizes in MSDC block')
# LGAG 卷积核大小属于结构参数。
parser.add_argument('--lgag_ks', type=int,
                    default=3, help='Kernel size in LGAG')
# MSCB 激活类型必须与训练网络一致。
parser.add_argument('--activation_mscb', type=str,
                    default='relu6', help='activation used in MSCB: relu6 or relu')
# 出现该旗标表示深度卷积分支采用串行而非默认并行。
parser.add_argument('--no_dw_parallel', action='store_true',
                    default=False, help='use this flag to disable depth-wise parallel convolutions')
# 出现该旗标表示多尺度分支采用 concat；默认 add。
parser.add_argument('--concatenation', action='store_true',
                    default=False, help='use this flag to concatenate feature maps in MSDC block')
# 测试时是否加载编码器预训练权重本身不应影响 checkpoint 覆盖后的数值，
# 但该开关也参与本脚本的目录名重建，所以必须与训练命名保持一致。
parser.add_argument('--no_pretrain', action='store_true',
                    default=False, help='use this flag to turn off loading pretrained enocder weights')
# 构造模型时 PVT 预训练文件目录；随后完整 checkpoint 会覆盖模型参数。
parser.add_argument('--pretrained_dir', type=str, default='./pretrained_pth/pvt/',
                    help='path to pretrained encoder dir')
# 监督策略不参与测试前向，却参与 checkpoint 目录名，因此仍需匹配训练命令。
parser.add_argument('--supervision', type=str,
                    default='mutation', help='loss supervision: mutation, deep_supervision or last_layer')

# max_iterations 在这里不控制任何循环，只参与复刻训练目录名。
parser.add_argument('--max_iterations', type=int, default=30000, help='maximum epoch number to train')
# max_epochs 用于目录名，以及 best.pth 缺失时推导 epoch_(max_epochs-1).pth。
parser.add_argument('--max_epochs', type=int, default=300, help='maximum epoch number to train')
# batch_size 不用于病例 DataLoader（后者固定 1），只参与 checkpoint 路径命名。
parser.add_argument('--batch_size', type=int, default=6,
                    help='batch_size per gpu')
# base_lr 同样只用于重建非默认学习率目录后缀。
parser.add_argument('--base_lr', type=float, default=0.0001, help='segmentation network learning rate')
# 每张切片送入网络前缩放到的正方形尺寸；应与训练输入尺寸一致。
parser.add_argument('--img_size', type=int, default=224, help='input patch size of network input')
# store_true 却设置 default=True，导致此变量始终为 True，命令行没有对应的关闭旗标。
# 因而当前代码总会创建预测目录并保存结果；这是现有工程行为，不是论文要求。
parser.add_argument('--is_savenii', action="store_true", default=True, help='whether to save results during inference')

# 预测保存根目录的初始值；主函数后面会在 is_savenii 分支中重新赋值。
parser.add_argument('--test_save_dir', type=str, default='predictions', help='saving prediction as nii!')
# 1 选择确定性 cuDNN 设置，0 优先 benchmark 性能。
parser.add_argument('--deterministic', type=int, default=1, help='whether use deterministic training')
# 固定 Python、NumPy、PyTorch 和 CUDA 随机状态。
parser.add_argument('--seed', type=int, default=2222, help='random seed')
# 解析命令行并生成全局 args。
args = parser.parse_args()

# 若总类别数为 14，则准备 13 个前景类别名称；背景仍隐含为索引 0。
if (args.num_classes == 14):
    # 名称顺序必须与标签整数编号 1..13 一一对应，日志和可视化才不会错位。
    classes = ['spleen', 'right kidney', 'left kidney', 'gallbladder', 'esophagus', 'liver', 'stomach', 'aorta',
               'inferior vena cava', 'portal vein and splenic vein', 'pancreas', 'right adrenal gland',
               'left adrenal gland']
# 默认 Synapse 配置是 9 类，因此进入此分支。
else:
    # 8 个名称依次对应标签 1..8；标签 0 背景不单独报告。
    classes = ['spleen', 'right kidney', 'left kidney', 'gallbladder', 'pancreas', 'liver', 'stomach', 'aorta']


# 整个测试集推理函数；test_save_path 控制 NIfTI/PNG 输出位置。
def inference(args, model, test_save_path=None):
    # args.Dataset 在主函数中被绑定为 Synapse_dataset；test_vol 分支读取完整 3D 病例。
    db_test = args.Dataset(base_dir=args.volume_path, split="test_vol", list_dir=args.list_dir, nclass=args.num_classes)
    # 每批一个病例且不打乱；num_workers=1 用一个子进程读取 H5。
    testloader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=1)
    # 记录病例级迭代总数。
    logging.info("{} test iterations per epoch".format(len(testloader)))
    # eval 模式固定 BatchNorm 统计并关闭 Dropout 随机性。
    model.eval()
    # 第一次加 NumPy 数组后，metric_list 会成为 [8,4]：8 类 x 4 指标。
    metric_list = 0.0
    # 逐病例迭代；tqdm 包装 enumerate 显示测试进度。
    for i_batch, sampled_batch in tqdm(enumerate(testloader)):
        # 取得病例张量最后两个空间维；h、w 后续未使用，属于保留调试变量。
        h, w = sampled_batch["image"].size()[2:]
        # image/label 通常 [1,D,H,W]；case_name 从 batch 的单元素字符串列表取出。
        image, label, case_name = sampled_batch["image"], sampled_batch["label"], sampled_batch['case_name'][0]
        # test_single_volume 内部逐切片推理，返回 8 行，每行是 (Dice,HD95,Jaccard,ASD)。
        # 网络四个输出中只取 P[-1]；softmax 将 9 通道 logits 变概率，argmax 选像素类别。
        metric_i = test_single_volume(image, label, model, classes=args.num_classes,
                                      patch_size=[args.img_size, args.img_size],
                                      # z_spacing=1 用于保存体数据间距；class_names 用于叠加图图例和日志语义。
                                      test_save_path=test_save_path, case=case_name, z_spacing=1, class_names=classes)
        # 把当前 [8,4] 指标矩阵加到跨病例累计值。
        metric_list += np.array(metric_i)
        # 先沿类别维求当前病例的 4 项宏平均并写日志。
        logging.info('idx %d case %s mean_dice %f mean_hd95 %f, mean_jacard %f mean_asd %f' % (i_batch, case_name,
                                                                                               np.mean(metric_i,
                                                                                                       axis=0)[0],
                                                                                               np.mean(metric_i,
                                                                                                       axis=0)[1],
                                                                                               np.mean(metric_i,
                                                                                                       axis=0)[2],
                                                                                               np.mean(metric_i,
                                                                                                       axis=0)[3]))
    # 除以病例数，得到每个器官跨病例平均指标，形状仍为 [8,4]。
    metric_list = metric_list / len(db_test)
    # 遍历前景类别编号 1..num_classes-1；不计算背景类别 0。
    for i in range(1, args.num_classes):
        # metric_list 的第 i-1 行对应标签 i，classes 同样用 i-1 取器官名。
        logging.info('Mean class (%d) %s mean_dice %f mean_hd95 %f, mean_jacard %f mean_asd %f' % (i, classes[i - 1],
                                                                                                   metric_list[i - 1][
                                                                                                       0],
                                                                                                   metric_list[i - 1][
                                                                                                       1],
                                                                                                   metric_list[i - 1][
                                                                                                       2],
                                                                                                   metric_list[i - 1][
                                                                                                       3]))
    # 再对 8 个器官求平均，取第 0 列得到总体 mean Dice。
    performance = np.mean(metric_list, axis=0)[0]
    # 第 1 列是总体 mean HD95，单位取决于 utils 中度量和体素间距设置。
    mean_hd95 = np.mean(metric_list, axis=0)[1]
    # 第 2 列是总体 mean Jaccard/IoU。
    mean_jacard = np.mean(metric_list, axis=0)[2]
    # 第 3 列是总体 mean ASD（平均表面距离）。
    mean_asd = np.mean(metric_list, axis=0)[3]
    # 记录最终四项宏平均结果。
    logging.info(
        'Testing performance in best val model: mean_dice : %f mean_hd95 : %f, mean_jacard : %f mean_asd : %f' % (
            performance, mean_hd95, mean_jacard, mean_asd))
    # 函数只返回状态字符串，真正数值保存在日志中。
    return "Testing Finished!"


# 仅直接运行本文件时执行测试；import 时只定义参数、类别表和函数。
if __name__ == "__main__":

    # 非确定性模式允许 cuDNN 搜索较快算法。
    if not args.deterministic:
        # benchmark=True 对固定输入尺寸可能提速。
        cudnn.benchmark = True
        # 允许非确定性实现。
        cudnn.deterministic = False
    # 默认 deterministic=1，进入可复现性优先分支。
    else:
        # 关闭运行时算法基准搜索。
        cudnn.benchmark = False
        # 要求使用确定性 cuDNN 算法；跨版本/硬件仍未必逐位一致。
        cudnn.deterministic = True
    # 固定 Python 随机数。
    random.seed(args.seed)
    # 固定 NumPy 随机数。
    np.random.seed(args.seed)
    # 固定 PyTorch CPU 随机数。
    torch.manual_seed(args.seed)
    # 固定当前 CUDA 设备随机数。
    torch.cuda.manual_seed(args.seed)

    # 数据集配置表把名称映射到类、路径、类别数和体素间距。
    dataset_config = {
        # 当前只注册 Synapse。
        'Synapse': {
            # 保存 Dataset 类本身，稍后赋给 args.Dataset 再实例化。
            'Dataset': Synapse_dataset,
            # 完整体数据根目录。
            'volume_path': args.volume_path,
            # test_vol.txt 所在目录。
            'list_dir': args.list_dir,
            # 总类别数，含背景。
            'num_classes': args.num_classes,
            # z 方向间距配置；当前 inference 调用仍直接传 z_spacing=1。
            'z_spacing': 1,
            # 结束 Synapse 子配置。
        },
        # 结束配置表。
    }
    # 读取命令行选择的数据集键。
    dataset_name = args.dataset
    # 把配置表中的类别数回填给下游函数。
    args.num_classes = dataset_config[dataset_name]['num_classes']
    # 回填完整体路径。
    args.volume_path = dataset_config[dataset_name]['volume_path']
    # 动态挂载 Dataset 类；inference 中通过 args.Dataset(...) 构造实例。
    args.Dataset = dataset_config[dataset_name]['Dataset']
    # 回填列表目录。
    args.list_dir = dataset_config[dataset_name]['list_dir']
    # 动态增加 z_spacing 属性。
    args.z_spacing = dataset_config[dataset_name]['z_spacing']
    # 输出 no_pretrain 布尔值，帮助核对 checkpoint 路径是否应带 _pretrain。
    print(args.no_pretrain)

    # 将聚合方式转换成训练目录名中使用的文本片段。
    if args.concatenation:
        # concat 对应 EMCADNet(add=False)。
        aggregation = 'concat'
    # 默认 add 对应 EMCADNet(add=True)。
    else:
        # 保存字符串只为重建路径。
        aggregation = 'add'

    # 将深度卷积执行模式转换成目录名片段。
    if args.no_dw_parallel:
        # series 对应 EMCADNet(dw_parallel=False)。
        dw_mode = 'series'
    # 默认 parallel 对应论文采用的并行多尺度分支。
    else:
        # 保存目录名片段。
        dw_mode = 'parallel'

    # 固定运行编号必须与 train_synapse.py 中 run=1 一致。
    run = 1

    # 用与训练入口相同的规则重建实验标识；任一结构参数不一致都会指向错误目录。
    args.exp = args.encoder + '_EMCAD_kernel_sizes_' + str(
        args.kernel_sizes) + '_dw_' + dw_mode + '_' + aggregation + '_lgag_ks_' + str(args.lgag_ks) + '_ef' + str(
        args.expansion_factor) + '_act_mscb_' + args.activation_mscb + '_loss_' + args.supervision + '_output_final_layer_Run' + str(
        run) + '_' + dataset_name + str(args.img_size)
    # 重建内层 checkpoint 目录。
    snapshot_path = "model_pth/{}/{}".format(args.exp, args.encoder + '_EMCAD_kernel_sizes_' + str(
        args.kernel_sizes) + '_dw_' + dw_mode + '_' + aggregation + '_lgag_ks_' + str(args.lgag_ks) + '_ef' + str(
        args.expansion_factor) + '_act_mscb_' + args.activation_mscb + '_loss_' + args.supervision + '_output_final_layer_Run' + str(
        run))
    # 清理 kernel_sizes 列表字符串，规则必须与训练入口一致。
    snapshot_path = snapshot_path.replace('[', '').replace(']', '').replace(', ', '_')

    # 训练时使用预训练编码器的目录带 _pretrain。
    snapshot_path = snapshot_path + '_pretrain' if not args.no_pretrain else snapshot_path
    # 非默认 max_iterations 追加类似 30k 的字符串；此值不表示本脚本实际执行迭代。
    snapshot_path = snapshot_path + '_' + str(args.max_iterations)[
        0:2] + 'k' if args.max_iterations != 50000 else snapshot_path
    # 非默认 epoch 数追加目录后缀。
    snapshot_path = snapshot_path + '_epo' + str(args.max_epochs) if args.max_epochs != 300 else snapshot_path
    # batch size 只为匹配训练目录名。
    snapshot_path = snapshot_path + '_bs' + str(args.batch_size)
    # 非默认学习率只为匹配训练目录名。
    snapshot_path = snapshot_path + '_lr' + str(args.base_lr) if args.base_lr != 0.0001 else snapshot_path
    # 追加输入尺寸。
    snapshot_path = snapshot_path + '_' + str(args.img_size)
    # 当前默认 seed=2222，因此追加 _s2222。
    snapshot_path = snapshot_path + '_s' + str(args.seed) if args.seed != 1234 else snapshot_path

    # 按与 checkpoint 一致的结构构造空模型；num_classes=9 决定四个分割头通道数。
    model = EMCADNet(num_classes=args.num_classes, kernel_sizes=args.kernel_sizes,
                     expansion_factor=args.expansion_factor, dw_parallel=not args.no_dw_parallel,
                     add=not args.concatenation, lgag_ks=args.lgag_ks, activation=args.activation_mscb,
                     encoder=args.encoder, pretrain=not args.no_pretrain, pretrained_dir=args.pretrained_dir)
    # 把模型移到默认 GPU；本测试入口没有 CPU 回退。
    model.cuda()

    # 下方是历史 checkpoint 路径示例，整行已注释，不参与运行。
    # snapshot_path = 'model_pth/'+args.encoder+'_EMCAD_wi_normal_dw_parallel_add_Conv2D_cec_cdc1x1_dwc_cs_ef2_k_sizes_1_3_5_ag3g_relu6_up3_relu_to1_3ch_relu_loss2p4_w1_out1_nlrd_mutation_True_cds_False_cds_decoder_FalseRun'+str(run)+'_Synapse224/'+args.encoder+'_EMCAD_wi_normal_dw_parallel_add_Conv2D_cec_cdc1x1_dwc_cs_ef2_k_sizes_1_3_5_ag3g_relu6_up3_relu_to1_3ch_relu_loss2p4_w1_out1_nlrd_mutation_True_cds_False_cds_decoder_FalseRun'+str(run)+'_50k_epo300_bs6_lr0.0001_224_s2222'
    # 首选加载训练过程中按验证 Dice 选择的 best.pth。
    snapshot = os.path.join(snapshot_path, 'best.pth')
    # 打印解析出的 checkpoint 路径，便于发现参数命名不匹配。
    print(">>>>>>snapshot值(也是best.pth要放的位置)：", snapshot)
    # 若 best.pth 不存在，则回退到零基编号的最后 epoch 文件，例如 epoch_299.pth。
    if not os.path.exists(snapshot): snapshot = snapshot.replace('best', 'epoch_' + str(args.max_epochs - 1))
    # torch.load 读取 state_dict，load_state_dict 默认 strict=True；没有 map_location，要求当前 CUDA 环境兼容。
    model.load_state_dict(torch.load(snapshot))
    # 以正斜杠切分路径得到内层目录名；在纯 Windows 反斜杠路径上此写法需要留意。
    snapshot_name = snapshot_path.split('/')[-1]

    # 测试日志按实验标识放入 test_log/test_log_<exp>。
    log_folder = 'test_log/test_log_' + args.exp
    # 递归创建日志目录，存在时不报错。
    os.makedirs(log_folder, exist_ok=True)
    # 文件名取 snapshot_name；配置时间格式与训练日志一致。
    logging.basicConfig(filename=log_folder + '/' + snapshot_name + ".txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    # 同时把日志输出到控制台。
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    # 记录完整测试参数。
    logging.info(str(args))
    # 记录 checkpoint 目录短名。
    logging.info(snapshot_name)

    # 当前 is_savenii 因 default=True 实际总会进入该分支。
    if args.is_savenii:
        # 把保存根目录改为 checkpoint 目录下的 predictions。
        args.test_save_dir = os.path.join(snapshot_path, "predictions")
        # 再按实验标识和 snapshot_name+'2' 建立更深的输出目录；末尾 2 是现有命名约定。
        test_save_path = os.path.join(args.test_save_dir, args.exp, snapshot_name + '2')
        # 创建输出目录；test_single_volume 会在其中写 PNG 和 NIfTI 文件。
        os.makedirs(test_save_path, exist_ok=True)
    # 理论上的不保存分支；以当前参数定义无法通过命令行触发。
    else:
        # None 表示不提供保存路径，但 utils 中的逐切片叠加图代码也依赖路径，需注意当前实现耦合。
        test_save_path = None
    # 启动整测试集推理；返回的状态字符串未被接收，最终结果查看日志与输出文件。
    inference(args, model, test_save_path)
