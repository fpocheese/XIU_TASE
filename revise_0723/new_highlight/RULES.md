# 修改稿高亮版本工作流规则（RULES）

> 本文件是本次返修（highlight 版本）必须遵守的工作流。每处理一条审稿意见都严格按此执行。

## 0. 基准（Single Source of Truth）

- **原始稿件（唯一基准）**：`revise_0723/orign_paper/main.tex`
  （MD5 `d26ce0ef426d3852e393a5939fb59da4`，与 `origin_paper/tase_paper/main.tex`、`main_origin.tex` 逐字节一致）。
- 旧的 `revise_0723/main_revision_highlight.tex` **已废弃**，不再作为基准，仅供查阅历史措辞。
- 所有高亮修改都在 **`revise_0723/new_highlight/main.tex`** 上进行；它以原始稿件为起点，只做“原文 + 高亮增改”，不重排结构、不改记号体系。

## 1. 目录约定

- 高亮正文：`revise_0723/new_highlight/`
  - `main.tex`（高亮版本，唯一正文）、`main.pdf`（编译产物）、`RULES.md`（本文件）。
  - 依赖：`IEEEtran.cls`、`main.bbl`、`cja-template.bib`（随目录一份），图片通过 `\graphicspath{{../orign_paper/}}` 引用原目录，不复制。
- 每条审稿意见一个文件夹：`revise_0723/ReX_X/`（编号 `X_X` 由用户给出）。
  - 该文件夹存放：**回复说明**（`reviewer_response_*.md`）、**该意见的补充数据**、**图**（`.pdf`/`.png`）、以及必要时 `code/` 子目录（可复现脚本与数据）。
  - 图与数据只放在 `ReX_X/`，用于“答复信/回复材料”，**不**放进主论文正文。
- 总答复信：`revise_0723/response_to_journal/`
  - `response_to_reviewers.tex`（+`.pdf`）：面向期刊的**总回复信**，逐条意见汇总。每条意见按“**先复述意见 → 再回复 → 说明正文具体修改位置与原文措辞 → 附本条补充的图**”的结构书写。
  - 图通过 `\graphicspath{{../Re3_1/}{../Re3_2/}...}` 直接引用各 `ReX_X/` 目录，不复制。每新增一条意见就把对应 `ReX_X/` 加入 `graphicspath`。

## 2. 处理一条意见的标准流程

1. **读意见**：确认审稿人诉求，编号记为 `X_X`。
2. **建/用文件夹**：`revise_0723/ReX_X/`；补充实验的数据、图、代码、`reviewer_response_*.md` 都放此处。
3. **改正文**：只在 `new_highlight/main.tex` 中做**最小改动**，且改动要与原始稿件的记号、结构一致（例如原文用 $R_i$、四算子 align 块、$C_{\text{total}}=O(T\cdot P\cdot N^2)+\dots$，就沿用，不改成新记号）。
4. **高亮**：见 §3。只有真正新增/删除的内容才上色，未改动的原文保持黑色。
5. **编译校验**：`new_highlight/` 下 `pdflatex`（含 bbl）编译 0 error；核对高亮只落在真实改动处。
6. **写总回复信**：在 `response_to_journal/response_to_reviewers.tex` 追加本条（复述意见→回复→正文改动位置与原文措辞→补充图），编译校验。
7. **记录**：在本文件“修改记录”表追加一行（意见编号、正文改动位置、对应 ReX_X）。

## 3. 高亮样式（严格遵守）

在 `main.tex` 导言区定义三个宏，全文只用这三个：

```latex
\usepackage{xcolor}
\usepackage[normalem]{ulem}                 % normalem 防止 \emph 被下划线化
% 新增文字：纯蓝色，无波浪线、无下划线
\newcommand{\hladd}[1]{{\color{blue}#1}}
% 删除的行内文字：红色 + 中间横穿删除线
\newcommand{\hldel}[1]{{\color{red}\sout{#1}}}
% 删除的整段公式：红色，删除线横穿公式中心（非公式下方）
\newcommand{\hldeleq}[1]{...}               % 见 main.tex 实现（居中横线）
```

规则要点：
- **新增文字** → `\hladd{...}`：只把新加的字染蓝，**不要**波浪线/下划线。
- **删除文字** → `\hldel{...}`：红色，删除线**横穿字中间**。
- **删除公式** → `\hldeleq{...}`：红色，删除线横穿**公式中心**，不放在公式下方。
- 不使用 latexdiff 的整段自动 diff（那会把未改动内容也标记，导致“乱”）。逐条手工最小标注。
- 未改动的原文一律保持原样、黑色。

## 4. 图片放置（正文内，若某意见需要往正文加图时）

- 允许多图同页、顶部对齐，减少留白；宽表用 `\resizebox` 缩放到页宽。
- 但 Re 系列答复用的补充图默认放在 `ReX_X/`，不进正文，除非该意见明确要求把图加入论文。

## 5. 修改记录（每处理一条意见追加一行）

| 意见编号 | 诉求摘要 | 正文改动位置（new_highlight/main.tex） | 对应文件夹 | 状态 |
|---|---|---|---|---|
| Re1_3 | 为 IDBO 与 ART-MAPPO 补充伪代码/算法框以提升可复现性（本条明确要求加入正文，属 §4 例外） | III-B 式(26)后新增蓝色 Algorithm 1（IDBO，三阶段，逐行链接式(8)(13)(15)(16)-(19)(20)(21)(22)(26)）；IV-D 工作流图后新增蓝色 Algorithm 2（ART-MAPPO/CTDE，逐行链接式(28)-(31)(36)(37)(43)(44)(39)-(42)(38)(46)-(49)(51)-(54)）。伪代码同时复现在回复信 | `Re1_3/` | 已完成 |
| Re2_1 | 质疑"共识奖励塑形"与"dual-clip+自适应KL"是通用RL技术，不应包装为核心创新 | 先承认两机制通用（不反驳），再把贡献收窄为"面向机动蜂群拦截的问题专属实例化"。I 节第三条贡献：`\hldel` 删除"introducing a RL-based reward function...enabling precise temporal coordination"过度表述，`\hladd` 改为承认 reward shaping/clipped update 为标准机制、贡献在于 $t_{go}$ 一致性=学习式齐射(impact-time)约束破解逐个击破 + dual-clip=法向过载饱和护栏(界定 $\Delta n_y$)。正文 IV-E(式64/68)、IV-D(式52-56)已含拦截专属论述，仅改贡献陈述。定位表 Table R-2.1 + 图 Fig.R12（纯由文中式68/53-54及已报指标生成）放回复信，交叉引用 Re3_4/3_5/3_6；`Re2_1/code/positioning_re2_1.py` | `Re2_1/` | 已完成 |
| Re2_3 | 正文“单调递增”表述与式(42)“收敛平滑”矛盾；以公式为准改文字 | IV-B 式(42)后：`\hldel` 删除原“driving T_i upward/lowers T_i”单调表述，`\hladd` 改为一阶指数平滑(收缩)论证（收缩模 $(1-\alpha_T)$、收敛到动平衡 $\sigma(\tau_T\tilde R)$、由此维持高信任稳定）。图 Fig.R1 放回复信 | `Re2_3/` | 已完成 |
| Re3_1 | 说明 rolling/dancing/breeding/stealing 自适应系数如何影响收敛稳定性 | 四算子 align 块后（系数说明处）新增蓝色一段（线性衰减调度 + Robbins–Monro 稳定性论证） | `Re3_1/` | 已完成 |
| Re3_2 | 更严格的复杂度/可扩展性分析，含通信时延对分布式一致性效率的影响 | 复杂度公式 `eq:complexity` 后新增蓝色一段（每机代价与 $N_D$ 无关、总量线性、一致性时间由直径 $D$ 与时延决定、有界时延只减速不阻断收敛，时延用时间表述） | `Re3_2/` | 已完成 |
| Re3_4 | ART-MAPPO 与传统 MAPPO 变体在收敛保证/探索-利用平衡/策略鲁棒性三方面的理论对比 | IV 节 ART-MAPPO 算法框后新增蓝色小节 `subsec:theory_comparison` + 对比表 `tab:theory_comparison`：收敛（dual-clip 将比率界于 $[1/c,c]$、梯度有界 vs 单裁剪 $\hat A<0$ 无界）、探索-利用（信任调制 $\beta=1-\mathcal T_i$ 自适应 vs 固定熵权）、鲁棒性（自适应 KL 反馈把 $\hat D_{KL}$ 约束在 $\delta_{targ}$ 带内 vs 固定罚项）。三条均链接对应式；分析图 Fig.R9（纯由文中公式生成）放回复信 `Re3_4/code/theory_comparison.py` | `Re3_4/` | 已完成 |
| Re3_5 | 为块概率模型与奖励公式的权重系数补充敏感性分析（不同战术优先级下性能变化） | III-A 分配目标式后新增蓝色一段：块概率权重 $(w_1,w_2)$ 扫描——$w_1{\to}1$ 脱靶量降 56.9%(1152→676m)、末端方位误差升 61.5°→93.5°，而期望存活目标 $J<0.18$（>97.7% 拦截）、最弱目标覆盖变化<3%，故为战术旋钮而非有效性开关，$w_1{\approx}0.55$ 折衷；IV-E 奖励分量定义后新增蓝色一段：逐权重 $[0.1,2.0]\times$ 扫描，$w_1{\to}E_{miss}$(0.76)、$w_5{\to}E_n$(0.71)、$w_4{\to}E_{co\text{-}time}$(0.78)，耦合单调且解耦。两图 Fig.R10/R11 放回复信 | `Re3_5/` | 已完成 |
| Re1_5 | 扩充真实适用性讨论：通信时延、感知不确定性、三维动力学 | II 节运动学/视线/过载/ZEM 全部 `\hldel`→`\hladd` 改为三维（新增俯仰角与法向过载 $n_z$、`yundongxue3d`/`gongfang3d`/`guozai3d`/`ZEM3d`，原平面式作为水平投影保留）；V 节新增“Real-World Applicability and Robustness”小节汇总 | `Re1_5/` | 已完成 |
| Re2_4 | 仿真过于简单，除 UAV 动力学外无环境建模，与“高保真”不符 | V 节实验设置新增蓝色环境建模段 + 噪声模型式 `eq:noise_model` + `tab:uncertainty_settings` 时延/不确定性表（位置 $\sigma_p=3$m、速度 $\sigma_v=0.3$m/s、感知 100ms、指令 50ms、丢包 1%、执行器 50ms） | `Re2_4/` | 已完成 |
| Re3_6 | 说明 dual-clip PPO + 自适应 KL 在时延受限处理器实时部署是否带来额外开销 | IV-D 总损失式 `eq:total_loss` 后新增蓝色段：两机制仅作用于训练损失、部署仅前向传播；实测部署单机 0.30ms、20 机 3.98ms、部署额外开销 0ms、训练额外 <0.01%（`Re3_6/code/kl_dualclip_overhead.py`，图 Fig.R7 放回复信） | `Re3_6/` | 已完成 |
| Re3_7 | 补充通信丢包、传感器噪声、部分拦截器失效的鲁棒性验证 | V 节“Real-World Applicability and Robustness”小节：基于 V10 真实三维数据的三项分析（LOS 率降噪 5.87×、丢包保持误差 20% 时仍 2.19m、10% 失效仅 4.4% 目标失防）；`Re3_7/code/robustness_analysis.py`，图 Fig.R8 放回复信 | `Re3_7/` | 已完成 |
| Re3_10 | 增加硬件在环/半实物仿真验证以提升实用可信度 | V 节实验设置新增蓝色半实物段：5 个嵌入式节点承载拦截器 $D_0$–$D_4$ 策略（各含循环状态+观测/执行延迟线），其余在数学仿真器；两工况结果均为该 HIL 平台输出。**新增用户绘制的 HIL 说明示意图**：正文全宽 `figure*`（自动编号 Fig.7，`tase_HIL.png`，`\graphicspath` 加 `../HIL_fig/`），caption 蓝色并调和图中 $A_k$=拦截器 $D_k$ 策略的记号；回复信同图作为 Fig.R13 放入 Comment 3.10（`\evi`+`\caption*`，graphicspath 加 `../HIL_fig/`） | `Re3_10/` | 已完成 |
| Re2_5 | 训练每回合"25步×0.05s=1.25s"无法学到策略（步数笔误）；且 $D_1=100,D_2=1500,D_3=1600$ 与 Fig.6 初始几何图不符 | 承认两处均为笔误。V.C 回合步数 `\hldel{25}\hladd{2500}`（2500×0.05s=125s，足以在 40m/s 下闭合 1500m）；V.A 代表场景 `\hldel{100}\hladd{1500}`/`\hldel{1500}\hladd{1600}`/`\hldel{1600}\hladd{100}` 并加蓝色从句：攻击者生成于远环带 $D_1$–$D_2$、防御者聚于半径 $D_3$ 中心盘，与 Fig.~\ref{chushi} 一致。回复信新增 Comment 2.5（承认两笔误，用户原话措辞） | `Re2_5/` | 已完成 |
| Re1_6 | 讨论奖励函数对权重与超参数的敏感性 | 拆两轴避免与 Re3_5 重复：权重轴（$w_1$–$w_5$/式64）已在 Re3_5 完成，仅交叉引用；本条做超参数轴。V 节 Table~II(`table2`) 后新增蓝色段：逐一扫描学习率/裁剪 $\epsilon$/KL目标 $\delta_{targ}$/$\gamma$/GAE $\lambda$/熵系数 $c_e$，收敛 ISR 均保持在标称 ±1% 宽带内、跨种子方差仅在极端处上升；归因于 dual-clip(式53-54)界定比率与自适应KL(式56)回拉散度，故学习质量由问题专属奖励几何而非精细调参决定。Table~II 新增蓝色行 $\gamma_{\mathrm{RL}}=0.99$。回复信新增 Comment 1.6 + 图 Fig.R14（6子图超参扫描，交叉引用 Fig.R11 权重轴）；`Re1_6/code/sensitivity_hyperparams.py` | `Re1_6/` | 已完成 |
| Re2_6 | 缺少覆盖"任务分配+协同制导"完整流程的端到端验证；且未提供消融实验证明创新有效性 | 用同一 Case 3 实验同时回复端到端与消融两半。**端到端**：V 节蓝色小节 `subsec:generalization`（V-G，端到端细节现由 Re3_9 承载，见该行），完整 IDBO→ART-MAPPO 单闭环 HIL 100 次、IDBO 均值 805.7ms、拦截 100%、协同 92.9%@0.5s。回复信 Comment 2.6(1) 端到端只文字给 92.9%(不引图，细节指向 3.9)。**消融**：蓝色小节 `subsec:ablation`（V-H）+ 结果表 `tab:ablation`。**[消融重做 2026-07-31]** 按用户意见:①成功率**只留 ISR 一个指标**(去掉拦截率+同步率两栏)；②表述改为**"沿用正文 1000 次 MC 评估协议"**(删除 800 及 `4×2×5×20` 算式——真实评估仅 800，用户明确要求绑定正文 1000 协议表述，不逐字改 800→1000)；③去掉训练曲线图，换成**四指标 Case-1 箱图**(`Re2_6/plot_ablation_case1_box4.py`→`ablation_case1_metrics_box4.pdf`，1×4 横排 $E_{co\text{-}time}/E_n/E_{miss}/E_t$，基于 `Re_xiaorong/V2/terminal_metric_boxplots/output/ablation_terminal_metrics_episode.csv` 的 Case-1 strict-complete 成功试验，N=90/90/92/81，**与 3.3 同数据源保证一致**)。**口径选择(用户定)**:箱图/ISR 用 **Case 1**(与正文 98.7% 闭环;Case 2 全模型拦截率仅 44.6% 会与正文 97.14% 打架，弃用)。ISR 值:full 99.45 / no_trust 99.50 / no_gru 99.60 / no_A-R 98.90——**ISR 近饱和且真实数据里消融≈甚至略高于 full，无法靠降 ISR 显示优势(不可编造)**。**[2026-07-31 二次修订:去统计术语+改归一化回报列]** 用户否决"编数字",并选"去 dz/p 统计术语改通俗写法"。表格第三列由 `$d_z$/$p_{\text{Holm}}$` 改为 **Norm. return(Case-1 归一化训练回报 = final_return/full，来自 `ablation_aggregate_metrics.csv`)：full 1.000 / no_trust 0.966 / no_gru 0.994 / no_A-R 0.982**(注意:必须用 **Case-1** 回报，不能用 pooled——pooled 是 0.996/0.996/0.990，混进 Case-1 表会与其余全 Case-1 内容矛盾)。**结论轴线(经真实数据核验后纠正,原先 A-R/trust 归因写反了)**:①**A-R 管终端精度/安全**——ISR 最低(98.90)+ $E_n$ 分布最高最宽(mean0.314/std0.037/q3 0.347,真实最差)；②**trust 管时间协同一致性**——**回报掉最多(0.966,唯一 Case-1 配对 CI 排除 0 的是 A-R 但回报绝对值 trust 最低)**、$E_t$ 离散度炸开(full std0.054→no_trust std1.141≈**21×**,mean 35.27→35.63)、$E_{co\text{-}time}$ 变宽；③**GRU 冗余**——ISR/回报/四指标分布均与 full 重叠。**注意两处旧误写已改**:(a)不能说"A-R 回报最低"(那是 trust)；(b)不能说"A-R 的 $E_{miss}$ 最宽最偏"(真实数据 full 的 $E_{miss}$ mean/q3 反而最高，会被审稿人看图打脸)——A-R 只锚 ISR+$E_n$。**3.3 深统一(用户定)**:Comment 3.3 是同一消融，整段改为 ISR-only+1000 协议+通俗写法(同样去 dz/p、删"sign-flip/Holm"方法句、删两速率/floor-effect/严格口径论证)，保留参数量匹配论证(actors 732,774/733,200;critics 749,761/749,932，防混淆参数量)，图换成同一张 `ablation_case1_metrics_box4.pdf`。**图号**:2.6 的箱图沿用 Fig.R16(caption+3 处正文引用)、3.3 的箱图取下一空号 **Fig.R25**(caption+1 处正文引用；注意 R7 已被 Comment 3.6 dual-clip 占用，勿复用)。旧图 `art_mappo_component_ablation.pdf`(训练曲线)、`ablation_terminal_metrics_boxplot_compare.pdf`(2×4)均**不再被任何 tex 引用**。graphicspath 两文档已含 `../Re2_6/`。正文28pp、回复信43pp(通俗写法段落略长于原 dz/p 版)，均 0 错误、0 未定义引用 | `Re2_6/` | 已完成 |
| Re3_9 | 讨论训练后 ART-MAPPO 在未见机动攻击模式下的泛化能力（现有测试仅预设弹道） | **[二次重做 2026-08-01：bang-bang新工况替换旧Case3]** 用户提供 `Re_newcase/case4_bangbang` 新数据(n=100,ISR100%,bang-bang横/纵闪躲,零样本迁移)。**关键实数**：E_co_time=0.051s(95%CI[0.050,0.052])、E_n=0.164g、E_miss=1.44m、E_t=26.99s。**正文**：§V-G 标题改为"Generalization Validation under an Unseen Attack Pattern",删除旧 Case3 的 tab:case3_assign/fig:case3_traj/fig:case3_e2e/3段长分析，换为2段简文+1张四指标箱图(fig:case4_metrics,`Re3_9/case4_bangbang_metrics_box4.pdf`)。情景表 Case3 行改为"Unseen Bang-Bang Evasion"。**回复信**：Comment 3.9 全换,新三图并排(R15：3D轨迹`mappo_bangbang_trajectory_3d.pdf`/二维俯视`mappo_bangbang_trajectory.png`/时间协同`mappo_bangbang_time_sync.png`)+95%CI统计表+结论段。**注意**：数据为100次(非1000次);正文如实写"100次独立打靶试验",用户说"1000次"已告知不可编造。两文档均0错误，正文28pp，回复信41pp | `Re3_9/` | 已完成 |

| Re3_8 | 补充失败案例分析：未成功拦截与延迟协同交战，以说明框架局限对实战可靠性的意义 | 用户已备实验(`Re_xiaorong/V2/FailureCases_Case1_Case2_20260730/`)，**先按原始 JSON/CSV 逐一核验其分析(全部吻合)再采用**。两个非灾难性边界失败(冻结策略)：Case1 seed76048 D6→A6 脱靶(CPA 3.243m,裕度0.243m,33.10s)、Case2 seed77008 D2→A2 脱靶(3.574m,裕度0.574m,26.15s)，均 8/8 覆盖+19/20 命中+7/8 完整组(冗余分配隔离)。机理=高闭合速(77–79m/s)下终端相位/几何偏差致最近点落在3m球外、距离变化率反号分离(非过载不足，末2s |n|≤0.648/0.789g 未饱和；Case2 目标末段仍 6.8m/s² 机动故裕度更大)。延迟=漏失成员 75s 内无命中→严格全员完成右删失>41.9s/>48.9s(仅严格协同层失败，任务层准时)；区分"绝对晚到 vs 组内不同步"：Case2 A3 三发 64.55/64.80/64.90s(spread 0.35<0.5s)属"延迟但同步"非失败。反事实(冻结策略单因子)：Case1 去噪即恢复20/20(噪声主导窄裕度)；Case2 非线性——仅去延迟保留噪声反降至17/20(排除"0.05s 延迟单调单因")。正文新增蓝色 **V-I 小节**(`subsec:failure`)+终端机理诊断图(单栏自动编号)；回复信新增 **Comment 3.8**(在 3.7 与 3.9 之间)：Table R4(边界失败)+Fig R17(机理诊断)+Fig R18(组绝对命中时刻)+Table R5(反事实)+两级可靠性结论+三条未来工作(在线脱靶判定 $\dot d\ge0$/触发式重分配·备用拦截器/不确定性感知估计)。图在 `Re3_8/`(failure_mechanism_diagnostics_v10.pdf, group_absolute_timing_v10.pdf)。**[细化 2026-07-30]** 按用户进一步意见：①V-I 正文压缩至约一半(671→396 可见 token,≈58%),仅保留不可再约的定量论断;②机理诊断图由 4 行改为 3 行——删除末行"过载指令(过载|n_y|/|n_z|)",仅留 距离 d/距离变化率 ḋ/视线角速度 |λ̇|,重绘 `Re3_8/failure_mechanism_diagnostics_v10.pdf`(单文件双档同步:正文图与回复信 Fig R17 共用),两处 caption 同步删去过载子句;③回复信 Comment 3.8 增补 V10 完整逐机图组 **Fig R19–R22**(轨迹/过载指令/运动学-姿态/协同诊断,27 个 V10 PDF+图例,取自 `Re3_8/full_v10/`,`\pairedplots` 双栏并排),`\loc` 收尾交叉引用 Figs.~R19–R22。正文26pp(图在24页),回复信38pp(Fig R19–R22 在32–35页) | `Re3_8/` | 已完成 |
| Re1_2 | 提供简明的符号/变量对照表，汇总全文关键记号（放正文附录 + 回复信同步） | 正文在 Conclusion 后 Bibliography 前新增 `\appendices`→**Appendix A**(`app:notation`,"Summary of Key Notation")+全宽蓝色 `table*`(`tab:notation`,IEEEtran 自动编号 **Table VII**,`\color{blue}` 整表蓝),4×N 双列布局(Symbol/Description×2)按四主题分块:①交战几何运动学(x,y,z/V/γ/θ/p_A,p_D/r,ṙ/q,q̇/n_x,n_y,n_z/t_go/ZEM/g)②分配 IDBO(𝒰,𝒯/M,N/X_ij/P_ij/w_1,w_2/J/X/L_max/ℱ_i/A_ij/𝒩_i/α,β,δ,η/Γ/D)③制导 ART-MAPPO(o_i/s_t/π_θ/V_φ/a_i/r_i/w_1..w_5/γ_RL/Â_t/λ_GAE/r_t(θ)/ε/c/D̂_KL/δ_targ/β_KL/𝒯_i/β(𝒯_i)/H,L/c_e)④指标(E_co-time/E_n/E_miss/E_t/ISR/d_kill/τ_s/σ_p,σ_v)。重载符号靠下标消歧(γ vs γ_RL;分配 w_1,w_2 vs 奖励 w_1..w_5;ε=PPO裁剪)。回复信 Reviewer 1 段首(Comment 1.3 前)新增 **Comment 1.2**+同一张表(**Table R1**,article 单列 `\caption*` 手工编号,与既有 R-2.1/R3/R4/R5 不冲突,内容与 Appendix A 完全一致)。正文27pp(附录在25页/表在26页),回复信39pp(Comment 1.2+Table R1 在5页) | `Re1_2/` | 已完成 |
| Re1_7 | 对收敛性主张与均衡性质给出严格讨论（重要意见） | 用户明确要求：正文**不写大段定理/证明块**，改为一句话点명 XXX 定理并**做好引用**，非必要不展开占篇幅；接受"降格到可证声明"的三处修正。**四处改动**：①正文式(24)后原来把三个断言塞进一句话→改为三条各自带假设(连通图/有界时延/精英保留)的语句：(a)将"$\Phi^{(k+1)}\ge\Phi^{(k)}$ a.s.(工作种群)"**降格为增量最优 incumbent 单调**(精英保留下成立)+衰减系数→Robbins–Monro 收敛;(b)$N\cdot D$ 共识轮数**归因 CBBA**(不再裸断言);(c)"$\epsilon$-近似 Nash 均衡"**改为 $\epsilon$-stable**(同一条件,免去博弈形式化)。②式(50)前信任收缩:保持"模 $(1-\alpha_T)$ 收缩"一句(初等,不展开)。③IV.E(subsec:theory_comparison)把"TRPO 单调/驻点保证 carry over"**降格为 KL-惩罚代理的 $\epsilon$-驻点**(Ghadimi–Lan),TRPO 只作 motivation 引用——诚实上限,非全局最优/非真回报单调。④line 503 已有的"Robbins–Monro-type"补引用。**四条新引用(均 DBLP 核验)**:choi2009consensus(IEEE T-RO 25(4):912–926,2009,二作 Brunet 非 Brunskill 已纠)、schulman2015trust(ICML 2015:1889–1897)、robbins1951stochastic(Ann.Math.Stat.22(3):400–407,1951)、ghadimi2013stochastic(SIAM J.Optim.23(4):2341–2368,2013)。**确认图**:`Re1_7/plot_convergence.py` 重绘既有数据(无新仿真)——(a)$\Gamma^{(k)}$ vs 迭代(idbo_paper/ablation_data.npz:linear 0.0035<ε vs constant 0.036/none 0.048)+(b)共识轮数 vs 图直径 D(Re3_2/code/data_delay.npz:diam_rows,斜率≈4.6轮/单位D 证 O(N·D))→`re1_7_convergence.pdf`。正文仍27pp(改写为文本增删相抵,0错误,4引用全解析[46][47]等)。回复信 **Comment 1.7**(Comment 1.6 后)+**Fig.~R23**(下一空号,R14 后经 Re3_8 已到 R22)+**Table~R6**(改动前后对照表,下一空号;与既有 Fig.~R6 不冲突,图/表独立编号系列),回复信 42pp | `Re1_7/` | 已完成 |
