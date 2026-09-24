# ICASSP 2027 医学图像分割投稿冲刺清单

核对时间：2026-08-05，中国时间。

官方信息要点：

- ICASSP 2027 会议时间：2027-05-16 至 2027-05-21，Toronto, Canada。
- Full Paper Submission Deadline：2026-09-16。
- 医学图像方向可考虑的范围：Biomedical Imaging & Signal Processing，也可视论文侧重点考虑 Image, Video & Multidimensional Signal Processing / Machine Learning for Signal Processing。
- 常规会议论文：最多 4 页技术内容，可选第 5 页仅放参考文献。
- 官方页强调初筛会检查篇幅、结构、研究主题、语言、作者信息、作者顺序，并进行 plagiarism/self-plagiarism 检查。

官方链接：

- Call for Papers: https://2027.ieeeicassp.org/call-for-papers/
- Important Dates: https://2027.ieeeicassp.org/important-dates/
- Publishing & Paper Presentation Options: https://2027.ieeeicassp.org/publishing-and-paper-presentation-options/
- Editorial Policies: https://2027.ieeeicassp.org/about/editorial-policies/

## 一句话目标

不要在 40 多天里追求“做一个大而全的新系统”。

目标应是：一个清楚问题 + 一个可信改动 + 至少一个标准公开数据集 + 固定验证流程 + 有消融、有可视化、有失败记录、有英文 4 页论文。

## 总体节奏

### 8 月 5 日至 8 月 9 日：冻结选题和实验协议

必须产出：

- 论文题目草案 1 句。
- 核心假设 1 句：我的方法为什么可能比 baseline 好。
- Baseline 代码能完整 train/eval 一遍。
- 数据集、划分、指标、训练 epoch、输入尺寸、batch size、GPU、随机种子写死。
- 建立实验目录模板。

不要做：

- 不要同时换 backbone、loss、augmentation、decoder、dataset。
- 不要一边看 test set 一边选 checkpoint。
- 不要跑了实验但没有命令、配置、日志和代码版本。

### 8 月 10 日至 8 月 18 日：跑 baseline 与复现实验

必须产出：

- 原论文或公开 baseline 的可复现结果。
- 至少 1 个主数据集上的 mean Dice / HD95 或任务通用指标。
- 每个 case 的结果 CSV，而不是只有一个平均数。
- 训练曲线图：loss、validation Dice、validation HD95。
- baseline 的失败样例 6 到 12 张。

判断标准：

- 如果 baseline 都跑不稳，不要急着加创新点。
- 如果复现结果和论文差距很大，先查数据预处理、输入尺寸、split、评价脚本和 checkpoint 选择。

### 8 月 19 日至 8 月 27 日：只加一个主要创新点

必须产出：

- Method-A：baseline + 你的主要模块或策略。
- 与 baseline 完全相同的数据划分、训练轮数、优化器、学习率计划、输入尺寸和评价脚本。
- 主表初版：Baseline vs Method-A。
- 至少 1 个消融：去掉关键模块或替换成简单版本。

优先选择的创新类型：

- 小模块：替换 decoder 中一个注意力/融合/多尺度块。
- 小损失：针对边界、小目标或类别不平衡的一项约束。
- 小策略：更稳健的提示、伪标签筛选、跨尺度一致性或轻量后处理。

不建议现在做：

- 大模型全面重训。
- 三个以上模块堆叠。
- 同时换多个公开数据集和多个任务。
- 需要复杂人工标注或临床协作的新数据。

### 8 月 28 日至 9 月 3 日：补消融、泛化和可靠性

至少完成：

- 主数据集完整对比。
- 1 到 2 个关键消融。
- 复杂度指标：Params、FLOPs 或 inference time 至少选一个。
- 可视化：成功样例、失败样例、边界/小目标样例。

如果时间允许：

- 第二公开数据集外部验证。
- 3 个随机种子的均值和标准差。
- 类别级结果表。
- 统计检验或 bootstrap 置信区间。

### 9 月 4 日至 9 月 8 日：论文初稿必须成形

4 页论文建议分配：

- Abstract：120 到 160 词。
- Introduction：约 0.6 页，问题、痛点、贡献 3 点。
- Related Work：约 0.4 页，重点证明你知道相关工作。
- Method：约 1 页，图 + 公式 + 模块解释。
- Experiments：约 1.4 页，数据、设置、主表、消融、可视化。
- Conclusion：约 0.15 页。
- 第 5 页：只放 references。

每天都要同步写作，不要等实验全部结束再写。

### 9 月 9 日至 9 月 12 日：内部审稿和修补

检查问题：

- Reviewer 能否在 30 秒内看懂贡献是什么？
- 实验是否支持结论，还是结论说得太满？
- 是否有强 baseline？
- 是否有公平比较？
- 是否报告了训练设置和评价协议？
- 是否引用了最近 2 年同方向关键论文？
- 图表是否能独立读懂？
- 是否有拼写、语法、格式、页数问题？

此阶段只修最影响可信度的问题，不再做大改。

### 9 月 13 日至 9 月 15 日：投稿前冻结

必须完成：

- PDF 编译无错误、无字体问题、无超页。
- OpenReview/投稿系统账号、作者信息、单位、邮箱、COI 检查完成。
- 标题、摘要、作者顺序、track/EDICS 和 PDF 内容一致。
- 所有作者确认最终版本。
- 备份源码、PDF、BibTeX、图表原始文件、实验结果。

建议把个人内部 deadline 设为 2026-09-15 晚上，不要赌 2026-09-16 当天网络和格式检查。

### 9 月 16 日：只做提交和应急

当天不要再重跑核心实验。只允许：

- 修 typo。
- 修 PDF 格式。
- 修引用和作者信息。
- 上传并确认系统显示 submitted。
- 保存投稿确认邮件或截图。

## 每个实验必须保存什么

每次训练都建一个独立实验目录：

```text
experiments/
  EXP_YYYYMMDD_HHMM_shortname/
    config.json
    command.txt
    git_status.txt
    git_diff.patch
    environment.txt
    train.log
    metrics_per_epoch.csv
    metrics_per_case.csv
    best_checkpoint.txt
    notes.md
    predictions/
    figures/
```

`notes.md` 最少写：

```text
# EXP_YYYYMMDD_HHMM_shortname

Hypothesis:
What changed:
Dataset and split:
Seed:
Expected result:

Actual result:
Best validation epoch:
Final test result:
Anomalies:
Failure cases:
Next decision:
```

## 科研细节红线

- 数据泄漏：同一个 patient/case 不能同时出现在 train/val/test。
- 选模型：只能用 validation 选 best checkpoint，final test 只用于最终报告。
- 公平对比：baseline 和你的方法必须使用同一 split、同一输入尺寸、同一训练预算、同一评价脚本。
- 单变量原则：一次实验只改变一个核心因素，否则无法解释提升来自哪里。
- 负结果要留档：失败实验会帮你写 limitation，也能避免重复浪费 GPU。
- 表格不要只报平均 Dice：尽量补 HD95、类别级指标、per-case 结果和方差。
- 不要过度调 test set：反复看 test 后再改方法，本质上会污染最终测试。
- 不要选择性报告：如果某数据集下降，要么解释，要么降低结论范围。
- 医学图像要报告预处理：spacing、resize/crop、normalization、windowing、slice/volume 评价方式。
- 伦理和数据许可要清楚：公开数据集写引用、license、是否需要 IRB；私有数据必须有合规来源。

## 第一篇 ICASSP 医学分割论文的最低可投稿形态

最低形态：

- 1 个清楚医学分割问题。
- 1 个公开数据集，最好加 1 个外部验证数据集。
- 1 个强 baseline。
- 1 个主要改动。
- 2 个消融。
- 1 张方法图。
- 1 张主结果表。
- 1 张可视化图。
- 1 个失败分析段落。

更稳形态：

- 主数据集 + 外部数据集。
- baseline + 2 到 4 个 SOTA 方法。
- 关键模块消融、复杂度分析、随机种子方差。
- 代码和实验记录可追溯。

## 每天晚上 15 分钟复盘

每天记录 5 行：

```text
Today:
Experiment IDs:
Best result:
Problem found:
Tomorrow first action:
```

如果连续两天没有可解释进展，立刻缩小题目，不要硬堆复杂度。
