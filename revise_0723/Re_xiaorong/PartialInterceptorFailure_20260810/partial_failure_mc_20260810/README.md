# 两架拦截器失效蒙特卡洛实验（2026-08-10）

## 1. 结论先行

本次实验采用远端稳定版 xiu_onpolicy_3d_fix/stable_V2 的既有 Case 1/Case 2
策略，在不训练、不反向传播、不更新权重的条件下，对每个工况独立执行 100 次
蒙特卡洛推理。每次试验在固定目标分配完成后、首个制导指令生成前，从 20 架
拦截器中均匀无放回地随机选择两架，使其在整个交战过程中不可用。

这一严格协议下，Case 1 仍取得 92.0% ISR；Case 2 仅取得 44.0% ISR。结果表明，
框架在 Case 1 中具有较强的任务冗余，但 Case 2 的正弦机动威胁对固定分配下的
部分失效更敏感。该结果适合用于“失效边界与局限性”分析，不能表述为两个工况
均对两机失效高度鲁棒。

## 2. 与审稿意见的对应关系

若延迟、噪声、丢包和三维动力学确实在生成论文全部结果的同一环境中启用，并且
论文明确报告其注入位置、参数和统计协议，则现有名义工况结果可以回应
“real-world applicability”中关于通信/感知非理想性和三维动力学的部分要求。
本次两机失效实验直接补充了“partial interceptor failures”的验证。

但对远端稳定分支的代码审计发现：

- 已实现目标感知状态延迟、三轴独立高斯位置/速度噪声；
- 已实现拦截器指令的一阶执行机构滞后；
- 未找到独立的友机通信延迟实现；
- 未找到 1% 通信数据包丢失及“保持上一有效友机消息”的实现。

而且本次稳定评测实际采用 50 ms 目标感知延迟、位置噪声 3 m、速度噪声
0.3 m/s，以及 Case 1/Case 2 分别 250/400 ms 的一阶指令滞后。因此，该稳定分支
与当前论文文字中的 100 ms 感知延迟、50 ms 指令响应和 1% 丢包并不完全一致。
在这些实现和数值闭环前，不应声称本次数据已经验证了独立友机通信延迟或 1%
数据包丢失。

## 3. 实验协议

- 远端稳定源码：/home/a2rl/xiu_onpolicy_3d_fix/stable_V2
- 独立实验副本：/home/a2rl/xiu_onpolicy_3d_fix/partial_failure_mc_20260810
- 策略：Case 1/Case 2 已训练 actor，确定性推理
- 训练操作：无
- 回合数：每个工况 100
- 随机种子：Case 1 为 89001--89100；Case 2 为 90001--90100
- 失效数量：每回合恰好两架，均匀无放回抽样
- 失效时刻：目标分配后、首次制导前
- 失效状态：不运动、三轴指令为零、不能触发命中
- 在线任务重分配：无
- 固定分组规模：[3, 3, 3, 3, 2, 2, 2, 2]
- 命中半径：3 m
- 协同容差：0.5 s
- 仿真步长：0.05 s

ISR 定义为一个回合内 8 个进攻飞行器均至少被一架现役拦截器命中。四项连续
指标只在 ISR 成功回合中统计，并且只使用现役拦截器的真实命中事件：

- \(E_{co\text{-}time}\)：同目标现役命中拦截器到达时间相对组均值的平均绝对偏差；
- \(E_n\)：命中时刻终端横向加速度 \(|n_y|\) 的均值；
- \(E_{miss}\)：真实三维命中距离的均值；
- \(E_t\)：8 个目标全部完成拦截的任务完成时刻。

## 4. 五项主要结果

| Case | ISR (95% Wilson CI) | \(E_{co\text{-}time}\) (s) | \(E_n\) (g) | \(E_{miss}\) (m) | \(E_t\) (s) |
|---|---:|---:|---:|---:|---:|
| Case 1 | 92.0% (85.0%--95.9%) | 0.0113 ± 0.0027 | 0.2616 ± 0.0472 | 1.6582 ± 0.1341 | 33.7859 ± 0.9481 |
| Case 2 | 44.0% (34.7%--53.8%) | 0.8520 ± 1.0111 | 0.2628 ± 0.0459 | 2.0530 ± 0.1583 | 61.6489 ± 11.6553 |

连续指标为成功回合的均值 ± 样本标准差，样本数分别为 92 和 44。Case 2 的
\(E_{co\text{-}time}\) 中位数为 0.1224 s、四分位区间为
0.0239--1.7963 s；均值明显高于中位数，说明部分成功回合仍存在严重延迟协同，
分布不宜只用均值描述。

附加严格诊断（不属于用户要求的五个主要指标）：

| Case | 全部现役拦截器命中率 | 严格协同成功率 |
|---|---:|---:|
| Case 1 | 82.0% | 82.0% |
| Case 2 | 15.0% | 6.0% |

“严格协同成功”要求每个目标组的全部现役拦截器均命中，且组内到达时间极差不
超过 0.5 s。

## 5. 失败原因闭环

| Case | ISR 失败回合 | 目标组被两次失效完全清空 | 仍有现役拦截器但目标未覆盖 |
|---|---:|---:|---:|
| Case 1 | 8 | 5 | 3 |
| Case 2 | 56 | 5 | 51 |

Case 1 的失败主要由两次随机失效恰好清空两机目标组造成；其余目标组的剩余
制导能力基本保持。Case 2 中相同结构性原因仍只有 5 次，另外 51 次至少存在一个
仍有现役拦截器的目标未被覆盖。其中目标 7 和目标 4 分别在 24 和 17 个回合中
未覆盖，二者均为两机组；目标 6 也有 10 次未覆盖。说明 Case 2 的主要限制是：
固定分配在失效后降低了机动目标对应组的有效冗余，而现有策略没有失效感知的
在线重分配/重规划机制。\(E_t\) 的显著增长和
\(E_{co\text{-}time}\) 的长尾共同反映了延迟协同交战，而 \(E_n\) 仍与 Case 1
接近，仅说明控制幅值保持有界，不能抵消任务完成率和协同性的下降。

## 6. 审稿回复建议（英文）

> Thank you for this important suggestion. We added a partial-interceptor-failure
> study using the frozen policies and the same three-dimensional evaluation
> environment. In each of 100 independent trials per case, two of the 20
> interceptors were sampled uniformly without replacement and made unavailable
> after target assignment but before the first guidance command; no retraining,
> online reassignment, or post-selection of trials was performed. Mission
> success required all eight attackers to be intercepted by the remaining
> operational vehicles. The resulting interception success rates were 92.0%
> in Case 1 and 44.0% in Case 2. For successful trials, the mean
> \(\{E_{co\text{-}time},E_n,E_{miss},E_t\}\) values were
> \(\{0.0113~\mathrm{s},0.2616~g,1.6582~\mathrm{m},33.7859~\mathrm{s}\}\)
> and
> \(\{0.8520~\mathrm{s},0.2628~g,2.0530~\mathrm{m},61.6489~\mathrm{s}\}\),
> respectively. Failure decomposition shows that only five failed trials in
> each case resulted from both failed interceptors removing all vehicles
> assigned to one target; 51 of the 56 Case-2 failures occurred despite at
> least one operational interceptor remaining for the uncovered target.
> Accordingly, the added experiment identifies a clear practical limitation:
> the fixed assignment provides useful redundancy in Case 1, whereas the more
> demanding maneuver pattern in Case 2 requires failure-aware online
> reassignment or replanning. We have revised the discussion to state this
> limitation explicitly rather than claiming unconditional fault tolerance.

关于 packet loss 的回复必须在真实加入并运行丢包模型后另行补充；不要把上述
失效结果误写成对 1% 通信丢包的验证。

## 7. 文件说明

- formal_case1_n100/、formal_case2_n100/
  - episodes.csv：逐回合失效编号、ISR、附加成功率及四项指标；
  - targets.csv：逐回合逐目标的分组规模、现役数、命中数和指标；
  - hit_events.csv：逐命中事件的时刻、终端 \(|n_y|\) 和三维距离；
  - successful_metrics.csv：仅 ISR 成功回合；
  - summary.json/csv：远端脚本直接生成的统计量；
  - manifest.json：运行参数与“不训练”声明；
  - *_representative_success.npz：首个成功回合的真实轨迹、动作和事件。
- analysis/combined_summary.csv：五项主要结果及附加诊断；
- analysis/metric_descriptive_statistics.csv：均值、标准差、中位数、四分位数和范围；
- analysis/failure_cause_summary.csv：失败机理分类；
- analysis/uncovered_target_summary.csv：逐目标脆弱性；
- analysis/validation_report.json：样本数、种子及 NaN/Inf 校验；
- figures/two_defender_failure_mc_boxplots.{pdf,svg,png}：V10 风格 1×4 箱线图；
- run_two_defender_failure_mc.py：远端固定策略评测脚本；
- summarize_partial_failure_results.py：校验和统计脚本；
- plot_partial_failure_boxplots.py：可复现绘图脚本；
- formal_case1_n100.log、formal_case2_n100.log：完整运行日志。

## 8. Case 2 严格同步成功子集重绘

根据后续要求，另行生成了一个不覆盖原图的条件子集版本。图内仍只显示
“Case 1”和“Case 2”；筛选规则仅记录在代码和 manifest 中：

- Case 1：92 个 ISR 成功回合；
- Case 2：6 个同时满足 ISR 成功和严格协同成功的回合；
- 严格协同成功要求全部现役拦截器命中，且所有目标组到达时间极差不超过 0.5 s。

筛选后 Case 2 的 \(E_{co\text{-}time}\) 为
\(0.0271\pm0.0071\) s，范围为 0.0198--0.0365 s。该图用于观察严格同步成功
样本的条件分布，不代表 Case 2 全部 100 次试验或全部 44 个 ISR 成功回合的
无条件统计分布。

相关文件位于 figures/case2_sync_subset/；筛选及绘图代码为
plot_case2_synchronized_subset_boxplots.py。
