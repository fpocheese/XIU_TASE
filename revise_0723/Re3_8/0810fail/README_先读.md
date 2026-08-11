# 原始二维 Monte Carlo 失败与延迟协同分析

本目录针对审稿意见

> Notably, one of the recommendations is to include failure-case analysis for unsuccessful interception and delayed cooperative engagement scenarios because understanding framework limitations remains essential for practical battlefield deployment reliability.

对原始二维 ART-MAPPO Monte Carlo 文件进行了可追溯复核。分析没有修改源数据，也没有补造未保存的失败轨迹。

## 最重要的结论

1. 论文报告的 1000 次试验中，Case 1 有 987 次严格成功、13 次未成功；Case 2 有 971 次严格成功、29 次未成功。按整数计数，ISR 分别为 **98.70%** 和 **97.10%**。
2. 原评估程序只在 `break_flag=True` 时写入五项终端指标，因此现有 `*_eval.txt` 均为成功回合，无法从中恢复 13/29 个未成功回合的飞行轨迹、失效拦截器或直接原因。
3. 可以严格分析的是“成功但协同延迟较大”的尾部样本。Case 1 的 $E_{co\text{-}time}$ 中位数、95% 分位数和最大值分别为 0.0492、0.0708 和 0.0900 s；Case 2 分别为 0.0767、0.1521 和 0.3167 s。
4. Case 2 最慢 5% 成功回合的平均交战时间为 36.655 s，其余成功回合为 33.244 s，增加 3.411 s（10.26%）；$E_{co\text{-}time}$ 与 $E_t$ 的 Spearman 相关系数为 0.426。证据表明连续机动下的主要可靠性边界表现为交战时间延长并伴随协同误差尾部扩展，而不是大范围末端脱靶。
5. **0.10 s 仅作为辅助诊断线，不是原代码的成功/失败阈值。** 主分析使用各工况成功样本中 $E_{co\text{-}time}$ 的上 5% 尾部来定义 delayed cooperative engagement。

## 文件说明

- `原始二维MC_失败与延迟协同分析.md`：中文完整数据审计、定量分析、证据边界和论文写作建议。
- `reviewer_response_failure_analysis_EN.md`：英文审稿回复建议稿。
- `reviewer_response_failure_analysis_ZH.md`：英文回复的中文对应说明。
- `manuscript_text_failure_analysis_EN.tex`：可选的英文正文补充段落和表格 LaTeX；未自动写入论文。
- `failure_delay_table.tex`：单独的定量汇总表。
- `failure_delay_analysis.pdf/.svg/.png`：延迟协同尾部与交战时间关系图，PNG 为 600 dpi。
- `source_data_audit.csv`：源文件、哈希、行数、NaN/Inf 与样本选择审计。
- `interception_failure_inventory.csv`：严格成功/未成功总数及失败轨迹是否可识别。
- `paper_matched_success_metrics.csv`：与论文成功数相匹配的成功样本及尾部标记。
- `delay_summary.csv`：$E_{co\text{-}time}$ 的均值、分位数、极值和阈值计数。
- `delayed_tail_comparison.csv`：上 5% 延迟尾部与其余成功样本的五项指标对比。
- `delay_metric_correlations.csv`：$E_{co\text{-}time}$ 与其他指标的 Pearson/Spearman 相关性。
- `top10_delayed_success_cases.csv`：每个工况延迟最大的 10 个严格成功样本。
- `metric_distribution_summary.csv`：论文匹配队列和全部归档行的分布审计。
- `analysis_summary.json`：机器可读的核心结果。
- `analyze_original_2d_mc_failures.py`：完整可复现分析脚本（上一级目录）。

## 使用前必须确认

两个源文件分别包含 1961 和 1920 行成功记录，多于论文单批 1000 次试验对应的 987 和 971 条成功记录，而且没有 episode ID 或批次分隔符。本分析依据文件追加顺序，取前 987/971 行构造“论文匹配队列”；全部归档行的统计也保存在 `metric_distribution_summary.csv` 中，且总体结论一致。正式投稿前，作者应确认前 N 行确实对应论文所报告的那一批试验。

另外，正文“971 successes”与“97.14%”存在算术不一致：$971/1000=97.10\%$。二者必须在最终稿中统一。
