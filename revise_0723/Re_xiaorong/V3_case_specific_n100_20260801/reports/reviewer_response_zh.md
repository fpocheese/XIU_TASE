# 审稿意见

> Then, it would be helpful to add an ablation investigation demonstrating the independent contribution of the trust-aware mechanism, GRU temporal encoder, and attention-residual backbone within ART-MAPPO architecture.

# 回复

感谢审稿专家的建设性建议。我们在论文的两个工况中补充了逐组件消融实验：完整 ART-MAPPO、关闭 trust-aware、关闭 GRU、关闭 attention-residual。四种结构在两个工况中分别训练，因此共得到八个独立模型。所有训练采用同一训练种子、优化器、奖励函数、动力学和599,040环境步预算。No attention-residual 使用参数量匹配的替代骨干（actor 参数732.8k对733.2k），排除了容量差异解释。检查点仅由与正式测试不重合的20次验证决定；冻结模型随后在每个“结构×工况”上进行100次配对 Monte Carlo 测试，总计800次，测试期间无反向传播和优化器更新。

新增训练曲线揭示了三个组件的互补作用。关闭 trust 后策略熵明显更快塌缩：Case 1 末期熵由1.159降到0.517，Case 2由1.301降到0.756，而完整模型保留了0.535/0.523的 guided-action fraction。这与原文对 trust-aware 的训练侧定位一致，即在解析制导先验和学习探索之间进行受控调制。关闭 GRU 后 Case 1 的末窗口 Critic loss 从0.0583升到0.3095（5.31倍），Case 2从0.0348升到0.2330（6.69倍），波动也明显增大，表明 GRU 的核心独立贡献是稳定时序价值估计。参数量匹配的 No attention-residual 在名义工况下与完整模型的回报 AUC 和 Critic loss接近，但 Case 2 最终策略熵更低（1.192对1.301），且 Case 1 终端法向过载略高（0.0340 g对0.0316 g）。因此，修订稿将该骨干的贡献客观描述为表征与安全裕度，而不宣称数据不支持的普遍优势。

八个冻结策略单元的目标覆盖、协同成功和严格任务成功率均为100/100，Wilson 95%区间为96.3%--100%，配对 McNemar 检验为 $p=1$。这一成功率天花板说明解析制导层和任务冗余使四种结构都能完成名义任务，二元成功率本身无法区分组件。为此我们进一步报告仿真章节定义的四个连续指标。在 Case 1 中，完整模型相对 No trust、No GRU、No attention-residual 分别缩短交战时间0.419 s、0.108 s和0.032 s，并相对后两者降低约0.00238 g的终端法向过载；与此同时其组内时间误差大0.0067--0.0111 s，但仍远小于0.5 s协同阈值。Case 2 四组分布高度重叠，主要可辨识的配对效应是完整模型比 No GRU 缩短0.0278 s，其余差异很小或置信区间包含0。

因此，修订稿没有强行要求完整模型在每个名义指标上都排名第一，而是用训练动力学和配对连续指标回答组件的独立贡献：trust-aware 保持受控探索，GRU 稳定时序 Critic 学习，attention-residual 提供容量受控的表征/安全裕度；共同的100%名义成功率则表明系统存在有益冗余。我们已在高亮版中加入两工况的 reward、Critic loss 和 Policy entropy 曲线、100次冻结策略成功率、四项终端指标箱线图、训练效应表，以及成功率天花板和单训练种子限制的明确讨论。
