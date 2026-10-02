"""在不改动训练逻辑的前提下，记录模型参数量和计算量。"""

import logging

import torch


def log_model_complexity(model, input_shape, prefix="model", output_path=None):
    """对已经构建完成的模型做一次性复杂度探测，并写入当前训练日志。

    这里的探测只用于实验记录，不参与反向传播，也不修改模型结构、权重或
    优化器。THOP 原生返回 MACs（乘加次数），论文中常见的 FLOPs 口径将一次
    乘法和一次加法分别计数，因此这里按 ``FLOPs = 2 * MACs`` 写入日志。
    """
    # DataParallel 只是训练时的外层包装。复杂度工具应当分析真正的网络模块，
    # 否则可能把并行封装当成网络本身，或者在不同 GPU 数量下得到不一致结果。
    target = model.module if isinstance(model, torch.nn.DataParallel) else model

    # 参数量与输入尺寸即使没有安装 thop 也可以记录；参数量统计包含可训练和
    # 不可训练参数，和论文中通常报告的 total parameters 口径一致。
    parameter_count = sum(parameter.numel() for parameter in target.parameters())
    records = [
        "{}_parameters={}".format(prefix, parameter_count),
        "{}_complexity_input_shape={}".format(prefix, tuple(input_shape)),
    ]
    logging.info("%s_parameters=%d", prefix, parameter_count)
    logging.info("%s_complexity_input_shape=%s", prefix, tuple(input_shape))

    # 训练入口在这里传入的是已经迁移到 CPU/GPU 的模型。保存原始 training 状态，
    # 使本次探测结束后恢复到调用前状态，避免 BatchNorm/Dropout 状态影响后续训练。
    was_training = target.training
    try:
        # thop 是 requirements.txt 中声明的可选统计依赖。导入放在函数内部，
        # 这样即使用户暂时没有安装它，训练仍可启动并至少记录参数量。
        from thop import profile

        target.eval()
        # dummy input 必须和模型参数位于同一设备，否则 GPU 训练时会设备不匹配。
        device = next(target.parameters()).device
        dummy_input = torch.zeros(tuple(input_shape), device=device)
        # 关闭梯度，避免复杂度探测额外分配反向图显存；verbose=False 防止 THOP
        # 把逐层统计刷屏，只把汇总结果写入训练日志。
        with torch.no_grad():
            macs, _ = profile(target, inputs=(dummy_input,), verbose=False)
        logging.info("%s_macs=%.0f", prefix, macs)
        logging.info("%s_flops=%.0f (2 * MACs)", prefix, 2.0 * macs)
        records.extend([
            "{}_macs={:.0f}".format(prefix, macs),
            "{}_flops={:.0f} (2 * MACs)".format(prefix, 2.0 * macs),
        ])
    except Exception as exc:
        # 统计工具对自定义算子或某些模型分支可能不兼容。复杂度记录失败不应
        # 让已经可以正常训练的实验失败，因此把具体原因持久化后继续训练。
        logging.warning("%s_flops_unavailable=%s", prefix, exc)
        records.append("{}_flops_unavailable={}".format(prefix, exc))
    finally:
        # 只有调用前处于 train 模式时才切回 train；调用前本来是 eval 时保持原状。
        if was_training:
            target.train()
        if output_path is not None:
            with open(output_path, "w", encoding="utf-8") as stream:
                stream.write("\n".join(records) + "\n")
