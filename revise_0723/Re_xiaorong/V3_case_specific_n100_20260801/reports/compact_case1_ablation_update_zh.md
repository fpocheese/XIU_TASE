# Case 1 紧凑消融章节更新说明

## 最终稿结构

高亮稿在两个仿真工况介绍结束后新增 `Component Ablation Study`。本节只含：

1. 一段三个对照组的定义；
2. 一段基于真实训练日志和蒙特卡洛数据的解释；
3. 一张四算法总表。

未加入消融箱式图，也未按 Case 1/Case 2 拆分表格。本次按最新要求仅报告 Case 1。

## 1000 次测试数据闭环

每个算法使用三个互不重叠且四算法严格配对的随机种子块：

- 原正式测试：98401--98500，共 100 次；
- 新增测试：100001--100500，共 500 次；
- 补充测试：102001--102400，共 400 次。

因此每个算法恰好 1000 次，四算法共 4000 条 episode。所有网络在测试前冻结，
测试过程没有反向传播、优化器更新或再训练。检查结果为：4000 条数据全部属于
Case 1，关键指标无 NaN/Inf，四算法均为 1000/1000 完整拦截。

## 论文表格数据

| Variant | Training AUC ($10^3$) | Std. final return ($10^3$) | ISR (%) | $E_n$ (g) | $E_{miss}$ (m) | $E_{co\text{-}time}$ (s) | $E_t$ (s) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Full ART-MAPPO | 535.07 | 108.91 | 100.0 | 0.0315 | 1.431 | 0.0768 | 26.830 |
| w/o trust-aware mechanism | 535.67 | 107.54 | 100.0 | 0.0307 | 1.414 | 0.0667 | 27.248 |
| w/o GRU temporal encoder | 535.04 | 106.25 | 100.0 | 0.0341 | 1.431 | 0.0656 | 26.938 |
| w/o attention-residual backbone | 535.04 | 106.01 | 100.0 | 0.0338 | 1.426 | 0.0699 | 26.863 |

四组 AUC 与末段回报标准差接近，说明名义训练回报出现平台，不能据此强行构造普遍
排序。机制日志和连续测试指标给出更具体的作用边界：去掉 trust 后策略熵下降幅度
由 52.4% 扩大到 78.5%，且平均交战时间增加 0.417 s；去掉 GRU 后最终 Critic loss
为完整模型的 5.31 倍，同时 $E_n$ 增加 0.00262 g、$E_t$ 增加 0.107 s；去掉容量
匹配的 attention-residual 后，$E_n$ 增加 0.00227 g、$E_t$ 增加 0.032 s。完整模型
的 $E_{co\text{-}time}=0.0768$ s，仍明显低于 0.5 s 协同阈值。结果支持三个模块的
互补贡献，同时不声称完整模型必须在每一个单项指标上最小。

## 权威文件

- 高亮稿 PDF：`manuscript/main_highlight_compiled_v3.pdf`
- 高亮稿源文件：`manuscript/main_highlight_v3.tex`
- 新增章节：`manuscript/ablation_section_insert.tex`
- 论文表格：`tables/paper_compact_ablation_n1000.tex` 和 `.csv`
- 4000 条合并 episode：`analysis/compact_case1_n1000/case1_ablation_episodes_n1000.csv`
- 1000 对配对效应：`analysis/compact_case1_n1000/case1_paired_terminal_effects_n1000.csv`
- 数据审计：`analysis/compact_case1_n1000/case1_ablation_n1000_audit.csv`
- 原始新增 500 次：`formal_evaluation_case1_n500/`
- 原始补充 400 次：`formal_evaluation_case1_n400/`
