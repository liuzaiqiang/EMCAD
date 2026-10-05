"""
实验环境与运行开销测量工具。

本模块只负责观测，不参与 EMCADNet 的构造、前向计算逻辑、损失函数或
checkpoint 选择。所有计时都在调用点显式包围现有代码，因此不会改变模型结构。
GPU 计时前后的 synchronize 是必要的：CUDA kernel 默认异步提交，如果省略同步，
CPU 的 wall-clock 计时会在 GPU 真正完成前返回，得到明显偏小的时间。
"""

import json
import os
import platform
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch


def environment_summary(device: torch.device) -> Dict[str, Any]:
    """返回论文复现实验需要的硬件、软件、输入设备摘要。"""
    result = {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "hostname": platform.node(),
        "torch_version": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "torch_cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "device": str(device),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
    }
    if device.type == "cuda" and torch.cuda.is_available():
        index = device.index if device.index is not None else torch.cuda.current_device()
        result.update({
            "gpu_index": int(index),
            "gpu_name": torch.cuda.get_device_name(index),
            "gpu_capability": "%d.%d" % torch.cuda.get_device_capability(index),
            "gpu_count_visible_to_torch": int(torch.cuda.device_count()),
        })
    else:
        result.update({"gpu_index": None, "gpu_name": None, "gpu_capability": None,
                       "gpu_count_visible_to_torch": 0})
    return result


def reset_peak_memory(device: torch.device) -> None:
    """清零本次测量的 CUDA 峰值计数器；CPU 运行时安全地跳过。"""
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)


def synchronize(device: torch.device) -> None:
    """让异步 CUDA kernel 完成，保证时间和显存读数已经稳定。"""
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(device)


def peak_memory_mb(device: torch.device) -> Dict[str, Optional[float]]:
    """读取 allocated/reserved 两种峰值，单位 MB，便于和 nvidia-smi 区分。"""
    if device.type != "cuda" or not torch.cuda.is_available():
        return {"peak_allocated_mb": None, "peak_reserved_mb": None}
    return {
        "peak_allocated_mb": round(torch.cuda.max_memory_allocated(device) / 1024**2, 3),
        "peak_reserved_mb": round(torch.cuda.max_memory_reserved(device) / 1024**2, 3),
    }


@dataclass
class InferenceBenchmark:
    """累积一次测试运行的前向时间、端到端时间和样本数。"""
    forward_seconds: float = 0.0
    end_to_end_seconds: float = 0.0
    forward_units: int = 0
    samples: int = 0

    def result(self, device: torch.device, warmup: int, repetitions: int,
               batch_size: int, input_size: Any, precision: str) -> Dict[str, Any]:
        """生成带有测量口径说明的结构化结果。"""
        units = max(self.forward_units, 1)
        samples = max(self.samples, 1)
        return {
            "measurement_policy": {
                "forward": "model forward only; CUDA synchronized before and after",
                "end_to_end": "existing evaluation path, including preprocessing, forward, postprocessing and requested output saving",
                "memory": "peak torch.cuda allocated/reserved after reset",
                "warmup_iterations": warmup,
                "timed_repetitions": repetitions,
            },
            "forward_time_ms_per_unit": round(self.forward_seconds * 1000.0 / units, 4),
            "end_to_end_time_ms_per_sample": round(self.end_to_end_seconds * 1000.0 / samples, 4),
            "forward_units": self.forward_units,
            "evaluated_samples": self.samples,
            "inference_batch_size": batch_size,
            "input_size": input_size,
            "precision": precision,
            **peak_memory_mb(device),
        }


def timed_call(fn, device: torch.device):
    """执行一个已有 callable 并返回 (result, elapsed_seconds)，不改变 callable。"""
    synchronize(device)
    start = time.perf_counter()
    value = fn()
    synchronize(device)
    return value, time.perf_counter() - start


def log_environment(logger, device: torch.device, prefix: str = "BENCHMARK_ENV") -> Dict[str, Any]:
    """把环境摘要逐项写入现有日志，并返回字典供 JSON 保存。"""
    summary = environment_summary(device)
    for key, value in summary.items():
        logger.info("%s_%s=%s", prefix, key.upper(), value)
    return summary


def log_json(logger, prefix: str, payload: Dict[str, Any]) -> None:
    """用一行 JSON 写入日志，既适合人读，也方便后处理脚本解析。"""
    logger.info("%s=%s", prefix, json.dumps(payload, ensure_ascii=False, sort_keys=True))
