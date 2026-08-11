# 第一条审稿意见：公式、符号与图像可读性审阅方案

审阅对象：`origin_paper/XIU_tase_paper/main.tex`。本文件只提出方案，不修改论文正文。

## 一、总判断与闭环重组方案

审稿人的判断成立。第三章有 22 个编号公式，第四章有 41 个编号公式，共 63 个；真正造成阅读负担的不是公式绝对数量，而是以下四类内容连续处于同一叙事层级：核心贡献、标准教材公式、实现细节、流程图/算法框的重复表达。

建议把正文公式压缩到约 20--24 个：第三章保留约 8--9 个，第四章保留约 12--15 个。其余内容并非简单删除，而是采用三种去向：标准公式改为引用或行内定义；实现细节移至附录/算法框；同一模块的连续公式合并为一个编号公式或一张“量—定义—物理含义”表。

建议第三章按如下因果链重写：

`相对几何量 -> 配对拦截评分 -> 全局分配目标与约束 -> 连续候选搜索 -> 离散竞价 -> 邻域共识 -> 一致分配与停止条件`。

建议第四章按如下因果链重写：

`物理观测 -> 特征交互/残差编码 -> GRU 时序状态 -> actor/critic -> trust 调节探索 -> rollout 奖励 -> GAE -> dual-clip + KL 更新 -> 新策略`。

每个核心公式后只回答三件事即可形成物理闭环：输入是什么、该映射改变了什么、输出送到哪里。例如：ZEM 越小使配对评分越接近 1；组内 `t_go` 偏差越大使协调奖励越负；trust 越高使启发式动作采样概率越低；KL 超阈值只限制下一轮策略更新幅度，并不直接保证过载约束。

## 二、第三章逐式审阅（22 个编号公式）

| 公式 | 含义 | 建议 | 必须补充或校正的逻辑 |
|---|---|---|---|
| `eq:interception_probability` | 用 ZEM 可行性和航向几何构造配对拦截评分。 | 保留，与后两式合并为一个编号的 aligned/cases 公式。 | `w_1,w_2` 与奖励权重重复；改为 `\omega_Z,\omega_q`。若未做概率标定，应称 score 而非 probability。 |
| `eq:zem_prob` | ZEM 越小，配对得分越接近 1。 | 合并到上式。 | 目前 ZEM 有符号，而 `\overline{ZEM}` 可能因正负抵消接近零；建议用 RMS/平均绝对值并加 `\varepsilon_{num}`。说明量纲归一化。 |
| `eq:angular_prob` | 航向与 LOS 偏差越小，几何得分越高。 | 合并到上式。 | `P_\sigma,S_\sigma` 容易与标准差/策略方差混淆；改为 `p_{ij}^{ang},e_{ij}^{ang}`。 |
| `eq:optimization_formulation` | 最小化所有目标的剩余生存概率，同时满足一机一目标、每目标至少一机和容量上限。 | 核心公式，保留。 | 乘积解释依赖“各拦截事件条件独立”及 `P_ij` 为校准概率；若只是评分，改成加权成本目标。显式给出可行条件 `N_D >= N_A`、`N_D <= L_max N_A`。 |
| `eq:hierarchical` | 抽象表示“局部优化后再共识”。 | 删除。 | 与三阶段公式、流程图、Algorithm 1 完全重复，且输出 `\mathcal A_IDBO` 未定义；输入用二值 `X`，后文优化的却是连续偏好。 |
| `eq:idbo_fitness` | 本地效用由拦截评分、对抗优势和超容量惩罚组成。 | 核心公式，保留；与惩罚式合并。 | `x_ij\in[0,1]` 后再 sigmoid 只映射到 `[0.5,0.731]`；要么令潜变量 `\xi_ij\in\mathbb R` 再 sigmoid，要么删 sigmoid。 |
| `eq:saturation_penalty` | 邻域估计人数超过 `L_max` 时处罚该候选。 | 合并进 fitness，或定义一个简短的 `\hat n_ij` 后行内给出 hinge penalty。 | 说明邻域估计如何覆盖非邻居；否则局部“未超载”不等于全局可行。 |
| `eq:adversarial_advantage` | 以预计拦截时间、相对速度、相遇角和竞争程度描述战术优势。 | 核心贡献可保留，但建议拆成“3 个归一化正项 + 竞争项”的可解释形式。 | 速度项可能为负，应用 clipping；`p_i` 在式中未使用；`\Delta t_ij,\theta_ij,T_max,v_max` 需明确定义；“energetic”应改为 velocity compatibility。 |
| `eq:rolling_update` | 局部梯度与衰减随机项驱动 exploitation/exploration。 | 四个 DBO 更新式整体移至附录，正文只留一个统一算子和四行语义表。 | 必须与实际代码一致；当前可用 IDBO 代码并没有该梯度更新。 |
| `eq:dancing_update` | 以优势调制乘性扰动并叠加高斯噪声。 | 同上。 | `\circ` 和 `\odot` 未统一；更新后如何投影回可行域未写。 |
| `eq:breeding_update` | 向最优与次优候选靠拢。 | 同上。 | 说明 `best/second` 是本机种群还是全网候选，以及优势向量为负时的行为。 |
| `eq:stealing_update` | 从竞争 UAV 借用偏好。 | 当前形式不能直接保留，先改正再移附录。 | 向量方程中 `j` 是自由下标；分子 `A_ij` 与求和 UAV `k` 不匹配。必须写成逐分量式或明确的逐元素向量权重。 |
| `eq:thresholding` | 将连续偏好二值化。 | 建议删除并改为 argmax/auction winner indicator。 | 当前“优势越高阈值越高”反而更难选中；`max_k A_ik` 可能非正；阈值还可能产生零个或多个目标，违反每机恰好一个目标的约束。 |
| `eq:bid_formulation` | 用本地质量、偏好和相对优势生成竞价。 | 保留但改为目标特定效用 `u_ij`。 | 当前 `F_i(x_i)` 对所有目标是同一标量，目标区分主要又回到偏好/优势；用 `u_ij` 更直观。 |
| `eq:cbba_update` | 合并邻居 winner lists 并保留每目标最高的 `L_max` 个竞价。 | 核心共识式，保留。 | 分布式节点应维护本地副本 `Z_{i,j}`，而不是看似全局的 `Z_j`；与前文 winner vector `z_i` 统一。 |
| `eq:phase1` | 概括本地连续优化。 | 删除。 | 已由公式、流程图和算法框表达。 |
| `eq:phase2` | 概括竞价共识。 | 删除。 | 同上；还能减少一批只出现一次的函数符号。 |
| `eq:phase3` | 概括优势更新。 | 删除。 | `H` 没有具体定义，且与 attention heads 的 `H` 冲突。 |
| `eq:convergence` | 邻域 winner lists 的不一致度低于阈值时停止。 | 保留，但改成标准成对 disagreement。 | 当前“向量减邻域平均再取 `l_0`”对包含 UAV ID 的 winner list 不自然；建议用邻接边上 `1[Z_i\ne Z_l]` 的平均。 |
| `eq:fitness_monotonic` | elitist acceptance 使接受的效用不下降。 | 若能补足条件则保留为 Proposition；否则改成谨慎的文字说明。 | 邻居估计变化会改变 `F_i`，所以各步 aggregate fitness 未必单调；plateau 也可能循环。需固定快照、确定性 tie-breaking 与有限状态。 |
| `eq:epsilon_nash` | 声称最终分配是 epsilon-Nash。 | 暂不建议保留。 | 文中没有从共识误差到单边效用增益的误差界；同一个 `epsilon` 同时承担停止阈值和博弈误差，没有推导。最好弱化为“locally stable fixed point”。 |
| `eq:complexity` | 给出优化、通信、优势更新的计算量。 | 改为一行复杂度表或正文行内表达。 | 文中称“per-iteration”却含总迭代数 `T`；`P=O(N)` 也未说明依据。应分别给 per-iteration 和 total-run，并使用 `N_D,N_A,N_pop`。 |

第三章建议最终保留：综合配对评分、分配模型、局部效用/容量惩罚、对抗优势、统一 IDBO 更新算子、目标竞价、top-`L_max` 共识、共识误差；收敛命题是否保留取决于能否补足证明。

## 三、第四章逐式审阅（41 个编号公式）

| 公式 | 含义 | 建议 | 必须补充或校正的逻辑 |
|---|---|---|---|
| `eq:encoder` | 两层 MLP 将四维物理观测映射到潜变量。 | 压成 `h_i^0=f_enc(o_i)` 一行，层宽移到参数表。 | 若后续 attention token 代表各物理通道，应先分别嵌入每个 `o_i^k`，不能先全连接混合后再 reshape 并声称 token 对应通道。 |
| `eq:qkv` | 标准 attention 的 Q/K/V 投影。 | 与 attention 和输出式合并为一个 MHA 公式，或引用 Transformer 后只写差异项。 | 使用 head 索引 `h`，避免 `j` 与目标索引冲突。 |
| `eq:attention` | 计算带可观测性 mask 的注意力权重。 | 合并后保留，因为 mask 是声称的任务差异。 | 如果实验和代码没有 sensor blanking/mask，应删 mask 及相关贡献陈述。 |
| `eq:mha_out` | 多头输出拼接、投影并与输入残差相加。 | 与上两式合并。 | 当前维度不闭合：每个 token 拼接后为 `d`，再 flatten 为 `n_o d`，但 `W^O` 写成 `d x d`。需重设 `d_h=d/(n_oH)` 或不 flatten。 |
| `eq:res_block_inner1` | 残差块第一层。 | 三式合成 `h^{l+1}=ReLU(h^l+F_l(h^l))`。 | 标准层内细节可放图注/附录。 |
| `eq:res_block_inner2` | 残差块第二层。 | 合并。 | 同上。 |
| `eq:res_block_out` | identity shortcut 输出。 | 合并并保留核心残差关系。 | 物理直觉应写成“保留低层运动学信息并提供直接梯度通道”。 |
| `eq:jacobian` | 用 Jacobian 说明梯度通道。 | 删除。 | 后面的范数下界一般不成立，`I + dz/dh` 可能相消；ReLU inactive 时也可能为零，不能据此证明不消失。 |
| `eq:gru` | GRU 汇总历史交战信息。 | 保留一行或与 actor 合并。 | 建议将 hidden state 改为 `h_{i,t}^{GRU}`；当前 `c_t` 与 bid vector `c_i`、dual-clip `c` 冲突，且缺 agent 下标。 |
| `eq:actor_dist` | actor 输出连续过载指令的高斯分布。 | 核心公式，保留。 | 若采用参数共享 MAPPO，写 `pi_theta` 而非每机 `pi_{theta_i}`；若修订稿使用 3D v10 图，则 action 必须加入 `n_z`。 |
| `eq:critic_input` | critic 拼接全体局部观测形成 centralized state。 | 可与 actor/critic 的 CTDE 关系合并。 | 此处 `N` 被当作 interceptor 数，但第三章 `N` 是 target 数；必须改为 `N_D`。还需说明是否有额外全局信息。 |
| `eq:running_stats_mu` | return 的指数滑动均值。 | 与方差、标准化和 trust 合并为一个编号递推块，或放 Algorithm 2。 | `N` 改为 `N_D`；说明只在 training episode 间更新。 |
| `eq:running_stats_sigma` | return 的指数滑动方差。 | 合并。 | 方差最好用更新前均值或当前 batch variance，避免递推含义不清。 |
| `eq:trust_normalized` | 将单机 return 转成相对 swarm 的 z-score。 | 与 trust 式合并。 | `epsilon_0` 改为 `epsilon_num`，与 PPO clipping 区分。 |
| `eq:trust_evolution` | 对相对表现做 sigmoid 和时间平滑得到 trust。 | trust 模块核心式，保留。 | 明确 trust 是训练探索调节量，不是部署阶段的安全/可靠性认证。 |
| `eq:exploration_policy` | 在 learned policy 与 guided policy 间按 trust 混合采样。 | 核心式，保留；与 guided mixture 合并成一个块。 | 写清只用于训练；部署时是否令 `beta=0`。 |
| `eq:guided` | guided policy 在 PN/探测/均匀探索之间混合。 | 合并，但先重定义 guided actions。 | `argmax Q_phi(o,a)` 与前文只有 state-value critic `V_phi(s)` 矛盾，也不能等同 PN；`argmin Q` 主动选最差动作与“安全探索”矛盾。应直接定义 clipped PN command 和安全边界 probe。 |
| `eq:expected_action` | 给出混合分布的期望动作。 | 删除。 | 是前两式的直接推论，训练实际是 sample 而不是执行期望值，不增加算法信息。 |
| `eq:gae_A` | GAE 的折扣 TD 残差累加。 | 与 TD residual 合成一个编号，或引用 GAE 并移附录。 | 这是标准公式而非贡献。 |
| `eq:gae_delta` | 一步 TD residual。 | 合并。 | `r_i^t` 与 PPO ratio `r_t(theta)` 冲突；ratio 改为 `rho_t(theta)`。 |
| `eq:standard_clip` | 标准 PPO clipped surrogate。 | 与 dual-clip 合并，或引用 PPO 后行内定义 `L_clip`。 | 标准知识无需独占一个展示公式。 |
| `eq:dual_clip_final` | 负优势样本增加 objective floor，降低极端重要性比造成的破坏性更新。 | 核心优化式，保留。 | 文字应描述 surrogate 被截断，而不是概率比被硬约束。 |
| `eq:rt_bound` | 声称 dual-clip 对 ratio 给出区间约束。 | 删除。 | 该结论不成立：dual-clip 不强制 `r_t` 落入 `[1/c,c]`，更没有 `1/c` 下界；正优势时 ratio 也可越界，只是 surrogate 饱和。 |
| `eq:kl_approx` | 旧策略采样下 KL 的样本估计。 | 改为行内定义或与自适应系数合并。 | 称 empirical estimate 更准确。 |
| `eq:kl_adapt` | KL 过大时提高惩罚，过小时降低惩罚。 | 核心式，保留。 | 不要把 KL 阈值直接解释为结构过载界；它只控制 policy drift，需通过实验说明与控制平滑性的关联。 |
| `eq:value_loss` | critic 的均方误差。 | 行内定义或移附录。 | 标准公式，无需独立编号。 |
| `eq:total_loss` | 汇总 dual-clip、KL、value 与 entropy。 | 核心式，保留。 | 明确 actor/critic 分别优化哪些项，避免读者误解共同参数。cosine schedule 移表格。 |
| `eq:observation_space` | 定义 4 维局部观测。 | 保留一行，并用四行表代替随后五个编号公式。 | 应说明各量归一化/裁剪；当前各分量单位和尺度完全不同。 |
| `eq:o1` | LOS-rate 推导的法向制导需求，并做跳变拒绝。 | 主公式表中保留简化值，piecewise 滤波规则移实现细节。 | 说明阈值是传感器预处理而非策略结构。 |
| `eq:o2` | defender heading 与 LOS 的 lead-angle error。 | 放观测表。 | 角度差应用 wrap 到 `[-pi,pi]`。 |
| `eq:o3` | 单步控制 effort。 | 放观测表并改成上一时刻指令。 | 决策前 observation 不能使用尚未生成的当前动作；应为 `n_{x,i}^{t-1},n_{y,i}^{t-1}`，且它是 effort proxy 不是严格能量。 |
| `eq:o4` | 为同步最早到达者构造速度误差。 | 放观测表。 | `min t_go -> 0` 会导致发散；需 denominator floor、归一化和无有效组员时的定义。 |
| `eq:reward_function` | 五个任务目标的加权和。 | 核心式，保留。 | assignment 的 `w_1,w_2` 与这里重复；改为语义化权重 `lambda_dist...`。 |
| `eq:r_dist` | 接近目标的有界 dense reward。 | 五个分量改成一张表，不再分别编号。 | 距离应先按场景尺度归一化，否则 `alpha_d` 有单位且跨场景不可比。 |
| `eq:r_angle` | 惩罚 heading/LOS error。 | 放 reward 表。 | `w_2 alpha_a` 是重复可辨识权重；合并成一个系数。 |
| `eq:r_hit` | 进入 lethal radius 的稀疏成功奖励。 | 放 reward 表。 | 明确只触发一次，避免在 kill radius 内每步重复累加。 |
| `eq:r_coord` | 惩罚本机 `t_go` 偏离组均值。 | 放 reward 表。 | reward 是 environment 计算，不是 centralized critic 计算；执行阶段若不在线学习，不需要“critic 计算 reward”。 |
| `eq:r_energy` | 惩罚大过载。 | 放 reward 表。 | 若 3D 模型使用 `n_z`，这里也必须加入；可考虑惩罚 command rate 以支持“平滑”论述。 |
| `E1` | 组内 time-to-go 的平均绝对偏差。 | 移到仿真章，与四个评价量合为一个 aligned 公式或表。 | 必须给出评价时刻；若在各自终端时刻计算，`t_go` 都趋零，不能代表到达时间分散。建议直接用 arrival times。 |
| `E2` | 平均 terminal normal overload。 | 与评价量合并。 | 公式缺时间下标/terminal window；定义为 `n_y(t_f)` 或最后 `Delta T` 的均值。3D 时需说明是 resultant lateral load 还是分轴。 |
| `E3` | 平均 terminal miss distance。 | 与评价量合并。 | `d_{tf_i}` 改成更清楚的 `d_i(t_{f,i})`。 |

第四章建议最终保留：观测向量、任务化 attention、残差块、GRU/actor/critic、trust 更新、混合探索、GAE（可选）、dual-clip、KL 自适应、总损失、总奖励，以及移到仿真章后的统一评价式。

## 四、需要优先修正的数学/实现一致性风险

这些问题比“删几个公式”更重要，否则压缩后逻辑仍不闭环：

1. attention 的 token 构造和矩阵维度目前不闭合；mask 的物理通道解释也与“先 MLP 混合再 reshape”冲突。
2. residual Jacobian 的非消失梯度下界不成立，应删除证明性表述。
3. guided action 使用未定义的 `Q_phi`，而文章定义的是 `V_phi(s)`；`argmin Q` 也不等于安全 probe。
4. dual-clip 的 `eq:rt_bound` 不是由目标函数推出的硬约束，应删除。
5. observation 的 `o3` 有时间因果问题，`o4` 有除零/放大问题。
6. 第三章连续偏好与二值 assignment 都使用 `x_ij`，并且 threshold 不保证 one-hot feasibility。
7. 可用的 `dbo_code/python_project/idbo.py` 实现的是离散 repair、spiral、Levy flight、opposition learning 和 adaptive random neighborhood search，不是论文四个“优势驱动”更新式；`scenario.py` 的编码也是“每个目标选择一个 UAV”，与文中的“每个 defender 选择一个 target、many-to-one”相反。如果这些文件确实生成了论文实验，必须先对齐模型、公式、算法框和代码；如果不是，应明确标记为非论文实现，避免返修材料混用。
8. v10 已绘制 `n_z`、yaw、pitch 和 3D 轨迹，而原文 Section II/IV 仍是二维模型及 `[n_x,n_y]` action。若返修稿采用 v10/v11 的 3D 结果，动力学、action、observation、reward 和指标必须同步扩为 3D。

## 五、符号体系审计与统一方案

### 最严重的复用/冲突

| 当前符号 | 冲突 | 建议 |
|---|---|---|
| `M,N,m` | 第三章 `M=interceptors,N=targets`，第四章 `N=interceptors`，指标又用 `m`。 | 全文固定 `N_D,N_A`，组规模用 `|G_j|`。 |
| `x_i,x_ij,X_ij,x,y` | 位置、连续偏好、二值 assignment 混用。 | 位置 `p_i`；连续偏好 `xi_ij`；二值分配只用 `X_ij`。 |
| `A` | attacker、action space、对抗优势、attention matrix、GAE advantage。 | attacker 用 target set；action space `mathscr A`；对抗优势 `chi_ij`；attention `Alpha_h`；GAE 保留 `Ahat_t`。 |
| `P` | MDP transition、拦截概率、概率矩阵、population size。 | 配对量 `p_ij^int`；population `N_pop`；MDP transition 只在预备知识出现一次。 |
| `r` | 相对距离、reward、随机向量、PPO ratio。 | 距离 `R_ij` 或 `d_ij`；reward `r_i^t`；随机量 `xi`；PPO ratio `rho_t(theta)`。 |
| `sigma` | sigmoid、标准差、角度评分、policy std、噪声 std。 | sigmoid 保留 `sigma(.)`；角度量改 `e^ang`；所有标准差加清晰下标。 |
| `gamma` | heading 与 RL discount。 | 2D heading 用 `psi`（也与 v10 一致）；RL discount 保留 `gamma_RL`。 |
| `theta` | actor 参数、aspect angle。 | actor 保留 `theta`；aspect angle改 `vartheta_ij`。 |
| `epsilon` | consensus、Nash、PPO clip、numerical stabilizer。 | `epsilon_con,epsilon_NE,epsilon_PPO,epsilon_num`。 |
| `c` | bid vector、GRU state、dual-clip constant、loss weights。 | bid `b_ij`；GRU state `h_i,t^GRU`；dual clip `c_DC`；损失系数保留下标。 |
| `t,T` | target symbol、连续时间/步、IDBO iteration、episode horizon、decay constant。 | target 用 `a_j` 或 `T_j`；time step `tau`；assignment iteration `k`；horizon `H_ep`；time constant `T_d`。 |
| `w_1,w_2` | assignment weights 与 reward weights。 | assignment `omega_Z,omega_ang`；reward `lambda_dist,...`。 |
| `z` | winner list 与 residual hidden `z_1,z_2`。 | winner list `W_i,j`；network intermediate `u_l`。 |
| `V` | speed、critic value、attention value、velocity matrix。 | speed `V_i` 保留；attention value写 `Q_h,K_h,U_h`；全局 velocity matrix避免再写粗体 `V`。 |

### 定义缺失或定义位置不佳

- sigmoid 在第三章首次使用时没有定义，到第四章 trust 才定义；应在首次出现处定义一次。
- Hadamard product 同时用 `circ`、`odot`，应统一为 `odot`。
- `Delta t_ij,T_max,v_max,vartheta_ij,sigma_A,tau` 等未进入 notation table；反而部分只用一次的中间量占用正文。
- `p_i` 在第三章公式后定义但未用于该式，直到 reward 才真正使用；应在动力学建模处统一定义 position/velocity vector。
- `E_t` 被称为第四个指标但没有公式；notation table 中有它，正文只用文字说明。
- `n_x,n_y` 实际是乘以 `g` 的无量纲 load-factor commands，不应称“acceleration”而应称“load factor/overload command”。

### 建议的 notation table 结构

不要做一个按出现顺序堆叠的长表，分为四组最清楚：

1. **Sets and indices**：`N_D,N_A,i,j,k,tau,G_j,N_i`。
2. **Physical quantities**：`p,V,psi,q,d,t_go,n_x,n_y(,n_z)`，每项带单位。
3. **Assignment quantities**：`X_ij,xi_ij,p_ij^int,chi_ij,b_ij,W_i,j,L_max`。
4. **Learning quantities**：`o_i^tau,a_i,pi_theta,V_phi,h_i,tau^GRU,T_i,r_i^tau,G_i^k,rho_tau,Ahat_tau,beta_KL`。

## 六、图像降拥挤建议与 v11 选择原则

v10 的核心问题是：20 条曲线即使使用 20 种颜色，单栏图在打印、缩放和灰度阅读时仍无法可靠追踪；独立的 20 项图例还要求读者不断在图例和曲线间跳视。

v11 生成三种候选：

- **A：target-group small multiples**。每个目标组只有 2--3 架 defender，使用线型区分成员。保留全部个体信息，最适合 `n_y/n_z`、speed、yaw/pitch 和 `t_go`，也是当前最推荐的正文方案。代价是需要双栏宽度。
- **B：group mean + min--max band**。把 20 条线降为 8 条 group mean，并用浅色带保留组内离散范围。最适合证明“组内同步”和整体控制趋势，可放单栏；不适合声称每一架 UAV 都严格满足界限，除非同时给出 peak/max 统计。
- **C：heatmap**。每行一架 defender，按 assigned target 排序，用颜色表达时序值。最适合展示全体是否出现饱和区、异常个体或同步结构；精确读数弱，适合作为补充材料或概览。

正文建议：对过载用 A（能看到每机是否越界），对 `t_go` 用 B（叙事重点是组内收敛），完整 20 机 heatmap 放补充材料。速度/heading 若只用于说明基本运动趋势，可用 B，或者只保留 A 中最有代表性的工况，把另一工况移补充材料。

v11 不改变任何 v10 文件，读取相同的 `agents*.txt` 导出，并同时输出 PNG/PDF。
