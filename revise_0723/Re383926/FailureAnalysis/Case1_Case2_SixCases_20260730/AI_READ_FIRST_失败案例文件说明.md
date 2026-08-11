# AI READ FIRST：失败案例实验文件说明

## 0. 最重要的版本说明

本目录同时保留了早期分析和最终 V2 分析。撰写审稿回复时，应以以下 V2 结果为准：

| 工况 | 最终代表 seed | 漏失拦截器 | 不完整目标组 | 个体命中 | 目标覆盖 | 完整组 |
|---|---:|---|---|---:|---:|---:|
| Case 1 | 76048 | \(D_6\) | \(A_6\) | 19/20 | 8/8 | 7/8 |
| Case 2 | 77008 | \(D_2\) | \(A_2\) | 19/20 | 8/8 | 7/8 |

早期 Case 1 使用 seed 76014，其不完整组也是 \(A_2\)。该版本仅为历史留档，不应再用于最终回复。凡路径中含有以下内容，均视为旧版：

- `artifacts_remote/selected_representative_full_v10/`
- `artifacts_remote/failure_mechanism_analysis/`
- 文件名中的 `case1_seed76014`
- 根目录中的早期短版回复文件

最终 V2 的权威数据位于：

- `artifacts_remote_v2/failure_mechanism_analysis_v2/`
- `artifacts_remote_v2/selected_representative_full_v10_v2/`
- `artifacts_remote_v2/failure_cause_counterfactuals_v2/`
- 根目录的 `reviewer_response_failure_case_full_v10.tex/.pdf`

## 1. 建议另一个 AI 的阅读顺序

1. `reviewer_response_failure_case_full_v10.tex`  
   最新、完整、专业英文审稿回复。包含最终数值、反事实结果表、全部图的插入方式和工程局限讨论。
2. `失败原因定量分析与审稿回复说明_CN.md`  
   最新中文机理解释，适合快速理解结论。
3. `artifacts_remote_v2/failure_mechanism_analysis_v2/failure_mechanism_summary.csv`  
   两个最终代表案例的核心定量数据。
4. `artifacts_remote_v2/failure_mechanism_analysis_v2/group_absolute_timing.csv`  
   每个目标组的绝对命中时刻、组内时差和不完整组的右删失延迟。
5. `artifacts_remote_v2/counterfactual_results.csv`  
   观测延迟、测量噪声和指令滞后的冻结策略反事实结果。
6. `artifacts_remote_v2/failure_mechanism_analysis_v2/failure_mechanism_diagnostics_v10.pdf`  
   终端距离、距离变化率、LOS 角速率和控制指令的直接机理图。
7. `artifacts_remote_v2/failure_mechanism_analysis_v2/group_absolute_timing_v10.pdf`  
   区分“绝对拦截较晚”和“组内不同步”的关键图。
8. `artifacts_remote_v2/selected_representative_full_v10_v2/figures/`  
   两个代表案例的全部 V10 仿真图。

## 2. 必须保持一致的事实

### 2.1 Case 1

- seed：76048；
- 漏失内部 ID：5，对应论文编号 \(D_6\)；
- 内部目标 ID：25，对应 \(A_6\)；
- 最近点：33.10 s；
- 最近距离：3.243 m；
- 相对 3 m 命中半径的脱靶量：0.243 m；
- 最近点后一个积分步长内距离变化率由负转正；
- 严格组完成延迟下界：\(75-33.10=41.90\) s；
- \(D_{14}\) 在 33.10 s 命中同一目标 \(A_6\)，因此目标仍被消灭。

### 2.2 Case 2

- seed：77008；
- 漏失内部 ID：1，对应论文编号 \(D_2\)；
- 内部目标 ID：21，对应 \(A_2\)；
- 最近点：26.15 s；
- 最近距离：3.574 m；
- 相对命中半径的脱靶量：0.574 m；
- \(D_{18}\) 和 \(D_{10}\) 分别在 26.10 s 和 26.15 s 命中 \(A_2\)；
- 严格组完成延迟下界：\(75-26.10=48.90\) s。

### 2.3 Case 2 的 \(A_3\)

\(A_3\) 的三次命中时刻为 64.55、64.80 和 64.90 s：

\[
\Delta t_{A_3}=64.90-64.55=0.35~\mathrm{s}<0.5~\mathrm{s}.
\]

所以 \(A_3\) 是 **delayed but synchronized engagement**：

- 从绝对时间看，它比其他组晚很多；
- 从组内协同误差看，它满足 0.5 s 阈值；
- 不能把 \(A_3\) 写成 synchronization failure。

## 3. 根目录文件逐项说明

| 文件 | 含义 | 是否用于最终回复 |
|---|---|---|
| `reviewer_response_failure_case_full_v10.tex` | 最新完整英文审稿回复，含图、表、定量原因和改进建议 | **是，首选** |
| `reviewer_response_failure_case_full_v10.pdf` | 上述 LaTeX 的已编译 8 页版本 | **是，首选** |
| `失败原因定量分析与审稿回复说明_CN.md` | 最新 V2 中文分析 | **是** |
| `AI_READ_FIRST_失败案例文件说明.md` | 本文件，提供版本和数据导航 | **是** |
| `ALL_FILES_INVENTORY.tsv` | 目录内 1278 个文件的完整相对路径和字节数清单 | 检索用 |
| `reviewer_response_failure_case.tex/.pdf` | 早期短版回复，只包含较早的失败案例表述 | 否，历史留档 |
| `REVIEWER_RESPONSE_EN.md` | 最早的英文 Markdown 回复草稿 | 否 |
| `PAPER_INSERTION_EN.tex` | 较早的论文正文插入段落草稿 | 否，数值可能不是 V2 |
| `实验说明与结果分析_CN.md` | 六个候选失败样本的早期中文说明 | 可作背景，不作为最终两个代表案例结论 |
| `README.md` | 早期实验包总览，主要描述 `artifacts_remote/` | 仅作目录背景 |
| `QA_VALIDATION.json` | 六案例早期数据 QA 结果 | 旧版 QA |
| `FULL_V10_QA.json` | 早期完整 V10 输出的数量和有限值检查 | 旧版 QA |
| `SOURCE_INTEGRITY_AUDIT.md` | 原始成功工程、实验副本、模型和源码哈希审计 | 可用于说明未改训练权重 |
| `missfont.log` | LaTeX/字体工具生成的无关日志 | 不使用 |

### 根目录脚本

| 脚本 | 功能 |
|---|---|
| `screen_partial_failures.py` | 在远端对两个工况各筛选 100 个冻结策略 evaluation episodes，查找边界失败 |
| `replay_selected_failures.py` | 复现早期选择的 6 个真实失败样本并保存原生轨迹 |
| `postprocess_and_plot_v10.py` | 将六案例轨迹整理为早期 V10 诊断图和长表 CSV |
| `plot_selected_full_v10.py` | **最终 V2 绘图脚本**；Case 1 seed 76048、Case 2 seed 77008，各生成 13 类 V10 图 |
| `inspect_case1_failure_groups.py` | 从 100 次筛选命中事件中识别 Case 1 各候选样本的不完整目标组 |
| `run_failure_cause_counterfactuals.py` | **最终 V2 反事实脚本**；冻结策略下分别去除延迟、噪声和指令滞后 |
| `analyze_failure_mechanisms.py` | **最终 V2 机理分析脚本**；计算最近点、距离变化率、LOS 角速率、控制指令、同组时刻并绘图 |

## 4. `reviewer_response_assets/`：LaTeX 直接引用的图

### 4.1 `reviewer_response_assets/mechanism/`

每张图均提供 PDF、SVG 和 PNG：

- `failure_mechanism_diagnostics_v10.*`  
  4×2 终端机理图。两列对应 Case 1/Case 2，四行依次是：
  1. 相对距离与 3 m 命中阈值；
  2. 距离变化率 \(\dot d\)；
  3. LOS 角速率；
  4. \(|n_y|\)、\(|n_z|\) 与分量限制。
- `group_absolute_timing_v10.*`  
  两个工况各目标组的绝对命中时刻。橙色虚线表示到 75 s 仍不完整；Case 2 的 \(A_3\) 位于约 64.9 s，说明其绝对延迟但组内同步。

### 4.2 `reviewer_response_assets/full_v10/`

最终有效文件前缀：

- `artmappo_case1_seed76048_*`
- `artmappo_case2_seed77008_*`

旧的 `artmappo_case1_seed76014_*` 仅为历史留档，不应在新回复中引用。

每个有效 seed 有以下 13 个图 stem，每个 stem 有 PDF/SVG/PNG：

| stem | 图的内容 |
|---|---|
| `trajectory` | \(x-y\) 平面轨迹、命中点和漏失最近点 |
| `trajectory_3d` | 三维轨迹 |
| `nx` | 轴向过载/控制指令 \(n_x\) |
| `ny` | 偏航平面过载 \(n_y\) |
| `nz` | 俯仰平面过载 \(n_z\) |
| `velocity` | 20 架拦截器速度历史 |
| `heading` | 旧版 V10 航向角表示 |
| `yaw` | 显式偏航角 \(\psi_D\) |
| `pitch` | 显式俯仰角 \(\theta_D\) |
| `tgo` | 各拦截器的 time-to-go |
| `tgo_error` | 同目标组内 time-to-go mismatch |
| `distance` | 拦截器到分配目标的距离 |
| `time_sync` | 每组命中时刻差；不完整组用到 75 s 的右删失下界表示 |

`artmappo_duav_legend.*` 是上述多拦截器时序图共用的 D-UAV 图例。

`failure_cases_case1_case2_v10.pdf` 是早期六案例中的组合图，不属于最终 V2 两代表案例图。

格式说明：

- PDF：论文优先使用的矢量图；
- SVG：可编辑矢量备份；
- PNG：600 dpi 预览。

## 5. `artifacts_remote_v2/`：最终数据和最终图

### 5.1 `case1_candidate_failure_groups.csv`

记录 Case 1 在 100 次筛选中出现的候选失败组。主要用途：

- 证明不完整组并非固定为某一个目标；
- 说明 seed 76048 对应 \(D_6/A_6\)；
- 还能看到 A1、A2、A5、A7 等其他真实候选。

字段：

- `eval_seed`：可复现种子；
- `hit_count`：20 个拦截器中的命中数；
- `complete_groups`：8 个目标组中的完整组数；
- `missed_defender_ids`：零基内部拦截器 ID；
- `missed_paper_labels`：论文编号；
- `incomplete_internal_target_ids`：内部目标 ID；
- `incomplete_group_labels`：论文目标组编号；
- `source_events/source_episode`：该结果在 100 次筛选原始数据中的来源。

### 5.2 `counterfactual_results.csv`

最终反事实汇总，共 2 个工况 × 6 个条件：

- `observed_boundary`：1 步延迟、3 m 位置噪声、0.3 m/s 速度噪声和 nominal command lag；
- `no_observation_delay`：只去掉观测延迟；
- `no_measurement_noise`：只去掉位置/速度测量噪声；
- `ideal_observation`：无延迟且无噪声；
- `no_command_lag`：只去掉指令滞后；
- `ideal_observation_no_lag`：理想观测且无指令滞后。

关键字段：

- `defender_hit_count`：个体命中数；
- `target_coverage_count`：被拦截的进攻飞行器数量；
- `complete_coordinated_group_count`：完整协同目标组数量；
- `max_observed_hit_spread_s`：完整组中的最大命中时差；
- `originally_missed_defender_hit`：原漏失成员在该反事实条件下是否恢复命中；
- `originally_missed_defender_min_distance_m`：原漏失成员的最近距离；
- `native_npz`：远端生成位置。相同原始文件已回传至本地 `failure_cause_counterfactuals_v2/`。

### 5.3 `failure_cause_counterfactuals_v2/`

这是 12 组反事实 evaluation 的完整原始结果。目录形式：

`case{1|2}/{condition}/`

每个条件包含：

- `run.log`：实际评估命令输出；
- `eval_summary.csv`：该次评估摘要；
- `case1/` 或 `case2/` 原生数据子目录。

原生数据子目录中：

- `case*_selected_episode.npz`：完整固定策略轨迹；
- `case*_episode_summary.csv`：episode 级命中、覆盖和协同统计；
- `case*_hit_events.csv`：每个命中事件的位置、速度、过载、距离、\(t_{go}\) 和时刻；
- `case*_success_episodes.csv`：成功条件记录；
- `case*_control.*`：原生控制曲线；
- `case*_pitch_overload.*`：俯仰过载；
- `case*_yaw_overload.*`：偏航过载；
- `case*_tgo.*`：原生 \(t_{go}\) 图；
- `case*_trajectory.*`：原生轨迹图；
- `case*_trajectory_3d.*`：原生三维轨迹；
- `case*_trajectory_xy.*`、`case*_trajectory_xz.*`：平面投影。

NPZ 主要数组：

- `rep_att`: `(1501, 8, 3)`，8 个进攻飞行器三维位置；
- `rep_def`: `(1501, 20, 3)`，20 个拦截器三维位置；
- `rep_ctrl`: `(1501, 20, 3)`，三轴控制/过载指令；
- `rep_tgo`: `(1501, 20)`，time-to-go；
- `selected_hit_count`、`selected_min_dist`、`case`：episode 元数据。

所有反事实都是 inference/evaluation；没有训练、反向传播或 optimizer step。

### 5.4 `failure_mechanism_analysis_v2/`

| 文件 | 含义 |
|---|---|
| `failure_mechanism_summary.csv` | 两个最终案例的核心最近点与终端机理统计 |
| `failure_terminal_window.csv` | 最近点前后 \(-2\) 至 \(+2\) s 的距离、闭合速度、\(\dot d\)、LOS 角速率、姿态误差、\(t_{go}\) 和控制指令采样 |
| `group_peer_comparison.csv` | 不完整组内部每名拦截器是否命中、命中时刻、事件距离 |
| `group_absolute_timing.csv` | 16 个“工况×目标组”的首个/最后命中时刻、组内时差、完整性和延迟下界 |
| `counterfactual_results.csv` | 反事实汇总的副本，便于本目录自包含 |
| `failure_cause_report.json` | 将机理统计和反事实结果合并的机器可读报告 |
| `failure_mechanism_diagnostics_v10.*` | 终端机理诊断图 |
| `group_absolute_timing_v10.*` | 绝对组命中时刻图 |

### 5.5 `selected_representative_full_v10_v2/`

- `figures/`：最终两案例的 27 个图 stem（2×13 加共享图例），每个提供 PDF/SVG/PNG；
- `selected_representative_runs.csv`：最终两个 seed、漏失成员、目标、最近距离和选例依据；
- `full_v10_manifest.json`：图类型、选例规则、未训练声明和数值摘要；
- `v10_export/`：V10 绘图兼容文本数组。

`v10_export/` 中：

- `mappo_success_nopn/`：Case 1；
- `mappo_success_sin/`：Case 2；
- `agentspos.txt`：D-UAV 和 A-UAV 的平面位置列；
- `agentsall.txt`：每个 D-UAV 的 \(n_x,n_y\) 交错列；
- `agentsnz.txt`：每个 D-UAV 的 \(n_z\)；
- `agentsvel.txt`：水平速度和偏航角交错列；
- `agentstimetgo.txt`：\(t_{go}\) 和到分配目标距离交错列。

## 6. `artifacts_remote/`：旧版、筛选证据和源码留档

该目录不是全部都应舍弃；其中部分是 V2 的原始筛选来源，但“最终代表结果”已经由 `artifacts_remote_v2/` 取代。

### 6.1 仍可作为证据使用

- `partial_failure_screen_nominal_n100/`  
  两个工况各 100 次冻结策略筛选。每个工况有 `chunk_00` 至 `chunk_04`，每块 20 episodes。
  - `screen_all_episodes.csv`：200 episodes 总表；
  - `partial_failure_candidates.csv`：所有满足部分失败条件的候选；
  - `candidate_counts.json`：候选数量；
  - `screen_manifest.json`：实际参数和命令；
  - 每个 chunk 的 `case*_episode_summary.csv` 和 `case*_hit_events.csv`：20 episodes 的真实统计和事件；
  - `run.log`：执行日志。
- `selected_six_failure_cases/`  
  早期复现的 6 个真实失败案例：Case 1 seeds 76014/76048/76052，Case 2 seeds 77008/77020/77023。  
  最终 V2 仍从这里读取 Case 1 seed 76048 和 Case 2 seed 77008 的 observed-boundary 原生轨迹。
- `failure_case_analysis_v10/`  
  六个样本的长表和早期诊断图：
  - `failure_case_metrics.csv`：案例级指标；
  - `defender_metrics.csv`：拦截器级指标；
  - `target_group_metrics.csv`：目标组级指标；
  - `defender_trajectories_long.csv`、`attacker_trajectories_long.csv`：长格式轨迹；
  - `case*_seed_*_diagnostic_v10.*`：单案例诊断图；
  - `case*_three_failure_cases_v10.*`：每工况三案例组合；
  - `failure_cases_case1_case2_v10.*`：两工况组合。
- `models/`  
  Case 1/2 的 `actor.pt` 和 `critic.pt`，证明 evaluation 使用的冻结模型。
- `presets/paper_case_presets_original_assignment_verified.npz`  
  固定论文工况初始状态和目标分配。
- `state/source_hashes_before.txt`  
  原成功工程在实验前的源码哈希。
- `code/`  
  远端独立实验副本源码、环境定义和绘图脚本的完整留档。

### 6.2 不用于最终结论的旧代表结果

- `selected_representative_full_v10/`：旧 Case 1 seed 76014 + Case 2 seed 77008；
- `failure_mechanism_analysis/`：旧 Case 1 seed 76014 的机理分析；
- `reviewer_response_assets/full_v10/artmappo_case1_seed76014_*`：旧 Case 1 图。

## 7. 关键定义，避免写错

### 7.1 内部 ID 与论文编号

- 拦截器内部 ID 从 0 开始，论文编号从 1 开始：
  - internal 5 = \(D_6\)；
  - internal 1 = \(D_2\)。
- 目标内部 ID 为 20–27：
  - internal 20 = \(A_1\)；
  - internal 21 = \(A_2\)；
  - internal 25 = \(A_6\)。

### 7.2 两种时间指标

1. **组内 synchronization spread**
   \[
   \Delta t_j=\max_{i\in\mathcal G_j} t_i^{\rm hit}
             -\min_{i\in\mathcal G_j} t_i^{\rm hit}.
   \]
   完整组只要 \(\Delta t_j\le0.5\) s 就视为同步。
2. **绝对拦截/组完成时刻**  
   表示该组在整场交战中何时完成。它可能很晚，但组内仍同步。

不完整组没有有限的全员完成时刻，因此用右删失下界：

\[
\Delta t_j^{\rm lower}=75~\mathrm{s}-t_{j,\rm first\,peer\,hit}.
\]

### 7.3 失败层级

- 个体层：一个已分配拦截器未命中；
- 目标层：因同组冗余成员命中，8/8 进攻飞行器仍全部被拦截；
- 严格协同组层：一个成员漏失使该目标组不完整。

不要写成“整个任务失败”或“有进攻飞行器突防”。

## 8. 数据支持的原因结论

- 直接几何机理：最近距离仍大于 3 m，随后 \(\dot d\) 反号，由接近转为分离；
- 两个案例的控制分量未达到饱和，不能写成控制能力不足；
- Case 1：测量噪声主导的窄裕度失败。去掉噪声可恢复 20/20 和 8/8；单独去掉延迟或指令滞后不能恢复；
- Case 2：延迟、噪声、指令动态、高闭合速度和持续目标机动的非线性耦合。不能将 50 ms 延迟写成唯一原因；
- Case 2 的 \(A_3\)：绝对时刻延迟，但组内同步；
- 当前框架局限：最近点掠过后没有显式在线 miss detection、恢复模式或事件触发重分配；
- 合理改进：预测最近距离+\(\dot d\ge0\) 触发、目标重分配、备用拦截器、不确定性感知估计和鲁棒训练。

## 9. 最适合审稿回复正文的图

若篇幅有限，优先顺序为：

1. `failure_mechanism_diagnostics_v10.pdf`：证明为什么漏失；
2. `group_absolute_timing_v10.pdf`：证明 A3 是 delayed but synchronized，并展示两个不完整组；
3. 两个 `trajectory` 或 `trajectory_3d`：展示总体交战；
4. 两个 `time_sync`：展示 A6/A2 右删失与其余组同步；
5. 其余 \(n_x,n_y,n_z,t_{go},t_{go}\) error、distance、attitude 图可放补充材料。

## 10. 审稿意见原文

> Notably, one of the recommendations is to include failure-case analysis for unsuccessful interception and delayed cooperative engagement scenarios because understanding framework limitations remains essential for practical battlefield deployment reliability.

回复应围绕“实验现象—定量原因—失败层级—可靠性含义—改进措施”组织，避免描述内部工作过程或用户曾提出的操作要求。
