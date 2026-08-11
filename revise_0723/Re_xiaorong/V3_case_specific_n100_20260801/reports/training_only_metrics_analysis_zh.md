# 三项新增训练指标的定义、结果与证据边界

本文档只分析训练过程，不使用任何拦截成功率或冻结策略测试结果。

## 1. 指标核查结果

| 用户提出的指标 | 原分析中是否已有 | 本次处理 |
|---|---|---|
| Training AUC ($\uparrow$) | 已有，原字段为 `return_auc_time_normalized` | 保留并在论文训练表中直接展示绝对值；同时保留相对Full归一化值 |
| Final Average Episodic Return ($\uparrow$) | 已有，原字段为 `final_window_return` | 在训练表中新增；明确使用最后58次更新 |
| Std. of Final Return ($\downarrow$) | 原表没有 | 新增原始末窗口标准差、20更新平滑后的末窗口标准差和变异系数 |

三个指标均由八个原始 `training_metrics.csv` 自动计算，没有手工填表。

## 2. 计算定义

### Training AUC

采用训练步数归一化的回报曲线面积：

\[
\mathrm{AUC}_{R}=\frac{1}{x_T-x_1}\int_{x_1}^{x_T}R(x)\,\mathrm{d}x.
\]

这里 $R(x)$ 是原始 `mean_episode_return`，$x$ 是环境步数。时间归一化后，AUC仍具有回报单位，可在相同Case、相同训练预算的四个变体之间比较。

### Final Average Episodic Return

八条正式训练轨迹均有585次更新。最终窗口按现有分析约定取最后10%，即最后58次更新：

\[
\bar R_{\mathrm{final}}=\frac{1}{58}\sum_{u=528}^{585}R_u.
\]

### Std. of Final Return

原始单次训练内部的末窗口样本标准差定义为：

\[
s_{R,\mathrm{final}}=
\sqrt{\frac{1}{57}\sum_{u=528}^{585}
(R_u-\bar R_{\mathrm{final}})^2}.
\]

同时输出：

- `final_window_smoothed_return_std`：先用现有20更新因果移动平均处理，再计算最后58点的标准差；
- `final_window_return_cv_percent`：原始标准差除以末窗口均值的百分比。

必须注意：上述标准差衡量的是一个训练运行内部的更新间波动，不是跨训练种子的最终回报标准差。当前每个单元只有一个训练种子，因此严格的跨种子 $\operatorname{Std}(\bar R_{\mathrm{final}})$ 无法计算。

## 3. 实际结果

表中AUC、最终回报和标准差均以 $10^3$ 回报单位表示。

| Case | Variant | Training AUC $\uparrow$ | Final avg. return $\uparrow$ | Raw final std. $\downarrow$ | Smoothed final std. $\downarrow$ | CV (%) $\downarrow$ |
|---|---|---:|---:|---:|---:|---:|
| Case 1 | Full | 535.07 | 534.16 | 108.91 | 5.876 | 20.39 |
| Case 1 | No trust | 535.67 | 535.66 | 107.54 | 5.997 | 20.08 |
| Case 1 | No GRU | 535.04 | 533.40 | 106.25 | 5.899 | 19.92 |
| Case 1 | No attention--residual | 535.04 | 533.41 | 106.01 | 5.755 | 19.87 |
| Case 2 | Full | 606.96 | 608.14 | 151.91 | 7.873 | 24.98 |
| Case 2 | No trust | 606.95 | 608.13 | 146.56 | 7.982 | 24.10 |
| Case 2 | No GRU | 606.96 | 608.14 | 154.14 | 8.320 | 25.35 |
| Case 2 | No attention--residual | 606.96 | 608.14 | 163.48 | 8.506 | 26.88 |

## 4. 三项指标能否分别证明三个模块的作用

### 4.1 Training AUC不能单独证明trust提高整体学习效率

Case 1中No trust的AUC为535.67，高于Full的535.07约0.111%；Case 2中Full与No trust只相差约0.0009%。所以当前AUC结果表明的是：trust机制没有损害总体学习回报，但不能证明它提高了名义训练的整体学习效率。

Trust最直接的训练证据仍是策略熵：去除trust后，熵降幅由Case 1的52.4%扩大到78.5%，由Case 2的46.8%扩大到69.7%。因此严谨表述应为：trust抑制探索过早塌缩，同时保持与Full近似的AUC；不能写成“trust显著提高AUC”。

若要把Training AUC作为trust的主要指标，需要增加多个训练种子，并在更难或非平稳训练分布下观察Full的AUC是否稳定高于No trust。

### 4.2 Final Average Episodic Return只在Case 1提供很小的GRU/attention--residual支持

Case 1中Full最终回报为534.16，No GRU/No attention--residual分别为533.40/533.41，Full高约0.14%。Case 2四组最终回报差异低于0.001%，且No attention--residual数值上略高于Full。因此该指标不能证明GRU或attention--residual在两个Case中都提高最终策略质量。

GRU更强的训练证据依旧是Critic loss：去除GRU后，Case 1和Case 2末窗口Critic loss分别增加到Full的5.31倍和6.69倍。最终回报相同而Critic误差显著增大，可解释为底层解析制导和任务结构维持了回报天花板，但GRU明显提高了价值学习稳定性。

Attention--residual在当前最终回报中没有一致优势，不能据此宣称提高最终策略质量。

### 4.3 Std. of Final Return只在Case 2支持attention--residual和GRU的稳定性

Case 2中，No GRU的原始末窗口标准差比Full高1.47%，No attention--residual高7.62%；采用20更新平滑后，两者分别比Full高5.68%和8.05%。该方向与“GRU和attention--residual改善训练稳定性”的解释一致，且attention--residual差异更大。

但Case 1中Full的原始标准差高于三个消融，平滑标准差也没有形成Full一致最优排序。因此当前数据只能支持“Case 2中的稳定性改善”，不能推广为两个Case的普遍结论。

原始回报标准差还受到向量环境episode完成批次交替影响。它不是纯粹的参数更新噪声。论文若使用该指标，建议优先报告平滑后标准差，并明确它是单运行内部的描述性指标。

## 5. 当前训练数据能够支持的模块结论

| 模块 | 当前最强训练指标 | 新增指标的辅助作用 | 严谨结论 |
|---|---|---|---|
| Trust | Entropy decline、final entropy、entropy retention | AUC显示trust未牺牲整体回报 | Trust减缓探索塌缩，但当前AUC不证明其提高学习效率 |
| GRU | Critic loss mean/std | Case 1最终回报小幅提高；Case 2末窗口波动降低 | GRU显著稳定价值学习；最终回报优势不一致 |
| Attention--residual | 容量匹配；Case 2 entropy与末窗口波动 | Case 2平滑末窗口标准差降低8.05% | 在Case 2提供训练稳定裕度；Case 1和最终回报尚无一致优势 |

## 6. 推荐论文用法

TABLE VII可以加入三项指标，但建议将其称为“training-dynamics descriptors”，不要把三列机械地一一指定为三个模块的证明。

推荐表述：

> The training AUCs and final returns remain nearly unchanged across the four variants, indicating a nominal return ceiling. The trust-aware mechanism is instead distinguished by substantially slower entropy collapse at comparable AUC. Removing the GRU increases the terminal Critic loss by 5.31 and 6.69 times in Cases 1 and 2, respectively. In Case 2, removing the GRU and attention-residual backbone also increases the smoothed final-window return fluctuation by 5.68% and 8.05%, whereas this ordering is not reproduced in Case 1. We therefore interpret the stability evidence as case dependent and refrain from claiming a universal final-return advantage.

真正要对三个新指标进行显著性检验，需要每个“模块×Case”至少5个独立训练种子，然后以训练种子为统计样本报告mean、standard deviation、95% CI和效应量。当前58个末窗口更新不能替代独立训练种子。
