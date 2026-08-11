# FailureCases_Case1_Case2_20260730 文件说明

## 1. 本目录的用途

本目录保存针对审稿意见

> Notably, one of the recommendations is to include failure-case analysis for unsuccessful interception and delayed cooperative engagement scenarios because understanding framework limitations remains essential for practical battlefield deployment reliability.

所补充的最终版失败案例实验。目录已经清理，只保留当前采用的两个代表性案例、原始仿真数据、完整 V10 图、定量原因诊断、复现实验脚本和审稿回复。

本实验使用训练完成的冻结策略进行测试，没有重新训练或修改模型权重。

| 工况 | 采用 episode | 未完成拦截的防御者 | 对应目标组 | 拦截器命中 | 目标覆盖 | 完整协同组 |
|---|---:|---|---|---:|---:|---:|
| Case 1 | seed 76048 | D6（内部索引 5） | A6（内部目标索引 25） | 19/20 | 8/8 | 7/8 |
| Case 2 | seed 77008 | D2（内部索引 1） | A2（内部目标索引 21） | 19/20 | 8/8 | 7/8 |

两个案例都不是“所有进攻目标未被拦截”的灾难性失败：8 个进攻目标均至少被一枚拦截器命中，但各有一枚同组拦截器未完成命中，因此暴露了冗余协同中的个体掉队问题。

## 2. 建议阅读顺序

1. `reviewer_response_failure_case_full_v10.pdf`：先看最终英文回复和论文可用图。
2. `失败原因定量分析与审稿回复说明_CN.md`：看中文定量结论和失败机理说明。
3. `artifacts_remote_v2/failure_mechanism_analysis_v2/failure_cause_report.json`：读取机器可解析的完整结论。
4. `artifacts_remote_v2/selected_representative_full_v10_v2/`：查看所选案例的原始数据和全套 V10 图。
5. `artifacts_remote_v2/failure_cause_counterfactuals_v2/`：查看观测、噪声和指令滞后反事实实验。

## 3. 根目录文件

### 最终回复

- `reviewer_response_failure_case_full_v10.tex`  
  最终英文 LaTeX 回复。正文围绕实验现象、定量证据、失败原因、工程局限和改进方向展开，并插入 Case 1、Case 2 的完整 V10 图以及机理诊断图。

- `reviewer_response_failure_case_full_v10.pdf`  
  上述 LaTeX 的已编译版本，可直接交给另一位 AI 或人工审阅。当前版本为 8 页。

- `失败原因定量分析与审稿回复说明_CN.md`  
  中文版数据解释，包含两种工况的失败定义、终端窗口诊断、同组对比、反事实实验结果和专业回复建议。

### 分析与绘图脚本

- `screen_partial_failures.py`  
  使用冻结策略批量筛选“所有目标均被覆盖，但至少一枚拦截器未命中”的候选 episode。

- `inspect_case1_failure_groups.py`  
  检查 Case 1 候选结果中具体由哪个目标组发生不完整拦截，用于避免 Case 1 和 Case 2 总是同一组掉队。

- `plot_selected_full_v10.py`  
  将选定 episode 转换为论文 V10 风格的完整仿真图，输出 PDF、SVG 和 600 dpi PNG。

- `analyze_failure_mechanisms.py`  
  对终端接近窗口、控制饱和、相对运动、同组拦截时间和反事实实验进行定量分析，并生成原因诊断表和图。

- `run_failure_cause_counterfactuals.py`  
  保持策略网络权重不变，分别关闭观测延迟、测量噪声、指令滞后或组合扰动，运行确定性反事实测试。

### 辅助索引

- `README_文件说明.md`  
  当前文件，即本目录所有文件和字段的说明。

- `ALL_FILES_INVENTORY.tsv`  
  自动生成的逐文件清单。第一列为相对路径，第二列为字节数，适合另一位 AI 检查是否读取完整。

## 4. `artifacts_remote_v2/`：原始结果和定量分析

### 4.1 根级 CSV

- `case1_candidate_failure_groups.csv`  
  Case 1 候选失败 episode 的组别检查结果。用于确定每个候选中未命中的防御者、对应进攻目标及组完成情况。

- `counterfactual_results.csv`  
  两种工况、六种扰动条件的反事实结果汇总。它是便于直接读取的副本，详细原始结果位于 `failure_cause_counterfactuals_v2/`。

### 4.2 `selected_representative_full_v10_v2/`

该目录保存最终采用的两条固定轨迹。

- `selected_representative_runs.csv`  
  两个最终 episode 的概要表，包括 seed、命中数、目标覆盖数、完整组数、未命中防御者、目标组和最小距离。

- `full_v10_manifest.json`  
  V10 导出清单和选例规则。Case 1 选择 seed 76048，是为了让不完整组为 A6；Case 2 选择最早满足条件的 seed 77008，不完整组为 A2。轨迹数据没有人为修改。

#### `v10_export/mappo_success_nopn/` 与 `v10_export/mappo_success_sin/`

- `mappo_success_nopn/` 对应 Case 1。
- `mappo_success_sin/` 对应 Case 2。

每个目录包含以下 V10 文本数据：

- `agentspos.txt`：各时刻拦截器和进攻飞行器的位置序列，供三维与平面轨迹图使用。
- `agentsall.txt`：控制指令的横向通道数据，按拦截器展开。
- `agentsnz.txt`：法向/垂向过载通道数据。
- `agentsvel.txt`：速度及航向相关时间序列。
- `agentstimetgo.txt`：time-to-go 和相对距离相关时间序列，用于同步性与距离图。

这些文件保持已有 V10 绘图程序要求的数据格式。

#### `figures/`

Case 1 文件名前缀为 `artmappo_case1_seed76048_`，Case 2 文件名前缀为 `artmappo_case2_seed77008_`。每幅图均提供 `.pdf`、`.svg`、`.png` 三种格式：

- `trajectory_3d`：三维交战轨迹。
- `trajectory`：二维平面轨迹。
- `distance`：拦截器相对指定目标的距离时序。
- `velocity`：飞行速度时序。
- `heading`：航向角时序。
- `pitch`：弹道倾角时序。
- `nx`：轴向过载时序。
- `ny`：横向过载时序。
- `nz`：法向过载时序。
- `yaw`：偏航/侧向控制相关时序。
- `tgo`：time-to-go 时序。
- `tgo_error`：同组 time-to-go 误差时序。
- `time_sync`：各拦截组命中时刻与组内时间离散度。
- `artmappo_duav_legend`：整套 V10 图共用图例。

### 4.3 `failure_mechanism_analysis_v2/`

- `failure_cause_report.json`  
  最完整的机器可读诊断报告，记录每个失败防御者的目标、最小距离、最近接时刻、miss margin、末段窗口、控制状态和反事实结论。

- `failure_mechanism_summary.csv`  
  两个案例各一行的核心失败指标摘要。

- `failure_terminal_window.csv`  
  未命中防御者在最近接前后终端窗口内的逐时刻数据，用于判断相对距离闭合、控制响应和擦边脱靶过程。

- `group_peer_comparison.csv`  
  未命中防御者与同目标组内成功命中防御者的并排对比。

- `group_absolute_timing.csv`  
  各组绝对命中时间、组内 spread 和延迟排序。该文件解释“晚到但同步”与“同步失败”的区别。

- `counterfactual_results.csv`  
  反事实结果在分析目录中的副本，便于脚本集中读取。

- `failure_mechanism_diagnostics_v10.{pdf,svg,png}`  
  最近接距离和终端失败机理诊断图。

- `group_absolute_timing_v10.{pdf,svg,png}`  
  各目标组绝对命中时间及组内时间离散度图。

关键解释：Case 2 的 A3 三次命中发生在 64.55 s、64.80 s 和 64.90 s，组内 spread 为 0.35 s，小于 0.5 s 协同阈值。因此 A3 是“整体较晚但组内同步”，不是同步失败；Case 2 真正的不完整组是 A2。

### 4.4 `failure_cause_counterfactuals_v2/`

该目录包含 Case 1 和 Case 2 的反事实测试原始输出。六种条件为：

- `observed_boundary`：原测试条件，包含当前观测边界效应、噪声和指令滞后。
- `no_observation_delay`：移除观测延迟。
- `no_measurement_noise`：移除测量噪声。
- `ideal_observation`：同时移除观测延迟和测量噪声。
- `no_command_lag`：仅移除指令执行滞后。
- `ideal_observation_no_lag`：理想观测并移除指令滞后。

每个 `caseX/<condition>/` 下都有：

- `run.log`：本次反事实测试的完整控制台日志。
- `eval_summary.csv`：episode 级测试汇总。
- `caseX/caseX_selected_episode.npz`：所选原始轨迹数组，主要字段如下：
  - `rep_att`：进攻飞行器状态，形状为 `(时间步, 8, 状态维度)`；
  - `rep_def`：20 枚拦截器状态；
  - `rep_ctrl`：20 枚拦截器控制量；
  - `rep_tgo`：各拦截器 time-to-go；
  - `selected_hit_count`：该 episode 命中拦截器数量；
  - `selected_min_dist`：最小相对距离；
  - `case`：工况编号。
- `caseX/caseX_episode_summary.csv`：该条件下的 episode 汇总。
- `caseX/caseX_hit_events.csv`：逐次命中事件，包括防御者、目标和命中时间。
- `caseX/caseX_success_episodes.csv`：满足成功判据的 episode 索引。
- `caseX/caseX_{control,pitch_overload,yaw_overload,tgo,trajectory,trajectory_3d,trajectory_xy,trajectory_xz}.{pdf,png}`：该反事实条件的原生诊断图。

这些数据用于区分失败究竟主要来自观测信息质量、控制执行滞后，还是末段几何裕度不足，而不是只依据三维轨迹作主观判断。

## 5. `reviewer_response_assets/`：LaTeX 插图副本

该目录只保存最终英文回复直接引用的图片：

- `full_v10/`：Case 1 seed 76048、Case 2 seed 77008 的 13 类 V10 图，以及共享图例；每图有 PDF、SVG、PNG。
- `mechanism/`：`failure_mechanism_diagnostics_v10` 和 `group_absolute_timing_v10`，每图有 PDF、SVG、PNG。

这里的文件与 `artifacts_remote_v2/` 中的分析输出内容一致，但单独复制是为了让 LaTeX 相对路径稳定、便于整体移动。

## 6. 数据判据与编号

- 内部防御者索引从 0 开始；论文/图中 D1–D20 从 1 开始。因此内部索引 5 对应 D6，内部索引 1 对应 D2。
- 内部进攻目标索引 20–27 对应图中的 A1–A8。因此目标 25 对应 A6，目标 21 对应 A2。
- 单枚拦截器命中判据使用代码中的 `d_min <= 3.0 m`。
- 组内协同成功判据为同一目标组命中时间 spread 不超过 `0.5 s`。
- “8/8 target coverage”表示所有进攻飞行器均至少被拦截一次；“19/20 interceptor hits”表示仍有一枚冗余拦截器未完成命中。
- `synchronization` 图反映组内命中时间差，不应把某组整体较晚误判为组内不同步；绝对到达时间应结合 `group_absolute_timing` 图和 CSV 判断。

## 7. 给另一位 AI 的最小输入集合

若只撰写审稿回复，建议至少读取：

1. `reviewer_response_failure_case_full_v10.tex`
2. `失败原因定量分析与审稿回复说明_CN.md`
3. `artifacts_remote_v2/failure_mechanism_analysis_v2/failure_cause_report.json`
4. `artifacts_remote_v2/failure_mechanism_analysis_v2/failure_mechanism_summary.csv`
5. `artifacts_remote_v2/failure_mechanism_analysis_v2/group_absolute_timing.csv`
6. `artifacts_remote_v2/failure_mechanism_analysis_v2/counterfactual_results.csv`
7. `artifacts_remote_v2/selected_representative_full_v10_v2/selected_representative_runs.csv`

若要复核每条结论，再读取 `failure_terminal_window.csv`、`group_peer_comparison.csv`、反事实 NPZ/命中事件 CSV 以及相应 V10 图。
