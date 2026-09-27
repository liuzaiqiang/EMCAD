#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
用法:
    python3 gpu_stats.py [日志路径] [小时数]
示例:
    python3 gpu_stats.py ~/gpu_logs/gpu_monitor.log 12
    python3 gpu_stats.py                          # 默认日志路径 + 12 小时
"""
import sys
from datetime import datetime, timedelta
from collections import defaultdict

LOG = sys.argv[1] if len(sys.argv) > 1 else "/root/gpu_logs/gpu_monitor.log"
HOURS = float(sys.argv[2]) if len(sys.argv) > 2 else 12.0


def to_float(s):
    """安全转数字，遇到 [N/A] 或空值返回 None，不让脚本崩"""
    try:
        return float(s.strip())
    except (ValueError, AttributeError):
        return None


# 按 GPU 编号分组，每张卡存一个列表
rows = defaultdict(list)
cutoff = datetime.now() - timedelta(hours=HOURS)
skipped = 0

with open(LOG, "r", errors="ignore") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        parts = line.split(",")
        # 期望格式: 时间,index,mem_used,mem_total,util,temp,power
        if len(parts) < 7:
            skipped += 1
            continue
        try:
            ts = datetime.strptime(parts[0].strip(), "%Y-%m-%d_%H:%M:%S")
        except ValueError:
            skipped += 1
            continue
        if ts < cutoff:          # 只保留最近 N 小时
            continue
        idx = parts[1].strip()
        rows[idx].append({
            "ts":    ts,
            "mem":   to_float(parts[2]),
            "total": to_float(parts[3]),
            "util":  to_float(parts[4]),
            "temp":  to_float(parts[5]),
            "power": to_float(parts[6]),
        })

if not rows:
    print(f"[警告] 日志 {LOG} 中没有最近 {HOURS} 小时的有效数据。")
    print(f"       可能是监控刚启动、日志路径不对，或时间格式不匹配。跳过行数: {skipped}")
    sys.exit(0)

print(f"统计窗口: 最近 {HOURS} 小时（{cutoff.strftime('%Y-%m-%d %H:%M')} 至今）")
print(f"日志文件: {LOG}")
print("-" * 72)

for idx in sorted(rows, key=lambda x: int(x) if x.isdigit() else 999):
    data = rows[idx]
    mems = [d["mem"] for d in data if d["mem"] is not None]
    utils = [d["util"] for d in data if d["util"] is not None]
    temps = [d["temp"] for d in data if d["temp"] is not None]
    powers = [d["power"] for d in data if d["power"] is not None]

    # 找显存峰值对应的那一条记录（用于输出峰值出现时刻）
    peak = max(data, key=lambda d: d["mem"] if d["mem"] is not None else -1)

    print(f"GPU {idx}  |  有效记录 {len(data)} 条  |  "
          f"{data[0]['ts'].strftime('%H:%M:%S')} ~ {data[-1]['ts'].strftime('%H:%M:%S')}")
    if mems:
        print(f"  显存(MiB): 峰值 {max(mems):.0f}  最低 {min(mems):.0f}  "
              f"平均 {sum(mems)/len(mems):.0f}  当前 {mems[-1]:.0f}"
              f" / {peak['total']:.0f}（峰值出现在 {peak['ts'].strftime('%m-%d %H:%M:%S')}）")
    if utils:
        print(
            f"  利用率(%):  峰值 {max(utils):.0f}  平均 {sum(utils)/len(utils):.0f}")
    if temps:
        print(
            f"  温度(℃):    峰值 {max(temps):.0f}  平均 {sum(temps)/len(temps):.0f}")
    if powers:
        print(
            f"  功耗(W):    峰值 {max(powers):.1f}  平均 {sum(powers)/len(powers):.1f}")
    print("-" * 72)
