# Reviewer Response — IDBO Coefficient Convergence Stability

**Reviewer comment.** *The author is encouraged to further justify the distributed
consensus-based IDBO formulation by clarifying how adaptive rolling, dancing, breeding and
stealing coefficients influence convergence stability in interception scenarios.*

---

## Response summary

We thank the reviewer for this constructive comment. We have (i) added an explicit statement of
the coefficient schedule and a convergence-stability justification to Section III (highlighted in
the revised manuscript), and (ii) performed a new controlled ablation on a paper-faithful
implementation of the four-operator IDBO to quantify how the adaptive coefficients affect
convergence stability. Two figures supporting this response are provided below (they are placed in
this response letter, not added to the main paper).

## 1. Added justification in the manuscript (Section III)

The four operators (rolling/dancing/breeding/stealing) share a common coefficient schedule. Every
exploratory coefficient follows a linear decay

$$c_k = c_0\,(1 - k/K_{\mathrm{IDBO}}), \qquad c\in\{\alpha,\beta,\gamma,\delta,\delta',\eta\},$$

together with the dancing noise scale $\sigma_d$, where $K_{\mathrm{IDBO}}$ is the iteration budget;
the elite-recombination weights remain bounded. The stability argument is:

- The **stochastic** parts of the rolling operator (the $\beta\,\mathbf r_i\exp(-t/T)$ probe) and the
  dancing operator (the $\sigma_d$ Gaussian perturbation) scale with $c_k$.
- As $c_k\!\to\!0$, these random contributions vanish and the composite update degenerates into a
  **deterministic, advantage-weighted local refinement** followed by the projection $\Pi_\Xi$.
- A step sequence whose stochastic amplitude decays to zero while retaining a descent/ascent
  direction is a **Robbins–Monro-type sufficient condition** for convergence to a stable fixed point
  rather than orbiting it.

This links the coefficient schedule directly to the convergence claim (the swarm-disagreement
metric $\Gamma^{(k)}$ falling below $\varepsilon_{\mathrm{con}}$) that was previously only asserted.

## 2. New ablation experiment (paper-faithful IDBO)

We implemented the manuscript's four-operator IDBO exactly (local utility with the soft
over-saturation hinge, the combat-advantage matrix, advantage-weighted operators, one-hot candidate
selection, and top-$L_{\max}$ consensus bidding) on the engagement scenario of the paper (20
defenders vs. 8 targets, $L_{\max}=3$, with the manuscript's initial radii, speeds and headings).
We then varied **only the coefficient schedule** and held everything else fixed:

| Schedule | Definition | Role |
|---|---|---|
| **Linear decay (paper)** | $c_k=c_0(1-k/K_{\mathrm{IDBO}})$ | exploratory strength decays to zero |
| Constant | $c_k=0.5\,c_0$ | fixed coefficients, no decay |
| No decay | $c_k=c_0$ | full-strength exploration throughout |

For each schedule we ran multiple independent trials and recorded two per-iteration quantities: the
**swarm-mean assignment cost** (how far the whole population has settled toward the optimum) and the
**swarm cost spread** (the standard deviation of the cost across the population, i.e. how tightly the
distributed swarm has contracted). A stable convergence requires both to fall and stay low; a
non-decaying schedule keeps injecting perturbations, so the swarm neither settles at a low mean nor
contracts.

**Result (40 independent trials over 5 engagement snapshots).** The paper's linear-decay schedule
drives the swarm to both a much lower steady-state mean cost and a much tighter distribution, i.e.
the most stable convergence; freezing the coefficients leaves the swarm scattered and unsettled:

| Schedule | Swarm-mean final cost | Late-stage swarm cost spread | Best-ever cost (mean ± std) |
|---|---|---|---|
| **Linear decay (paper)** | **0.58** | **0.66** | 0.34 ± 0.22 |
| Constant ($0.5\,c_0$) | 1.74 | 1.51 | 0.30 ± 0.19 |
| No decay ($c_0$) | 2.28 | 1.67 | 0.30 ± 0.19 |

Under the linear schedule the swarm-mean cost settles about **3–4× lower** and the population spread
is about **2.3–2.5× tighter** than under the frozen schedules. In other words, the adaptive
coefficients make the distributed swarm *contract onto and hold* a low-cost equilibrium, whereas
non-decaying coefficients keep injecting perturbations so the swarm neither settles nor contracts —
precisely the convergence-stability behavior the reviewer asked us to characterize. (All schedules
can occasionally locate a comparably low *best-ever* candidate through continued random search, but
only the linear schedule *stabilizes* the whole swarm on it, which is what matters for a deployable
assignment.)

**Figures**

- `idbo_coeff_ablation.pdf` — Fig. R2 (real ablation): (a) swarm-mean cost convergence
  (mean ± std) for the three schedules; (b) box plots of the late-stage swarm cost spread across runs.
- `idbo_coeff_schedule.pdf` — Fig. R3 (mechanism): (a) the coefficient multiplier $s(k)$ for the
  three policies; (b) the resulting stochastic-perturbation magnitude, showing it vanishes only
  under the linear schedule (entering the deterministic-refinement regime).

**Suggested caption — Fig. R2 (real ablation, `idbo_coeff_ablation.pdf`).**

> Fig. R2. Effect of the IDBO coefficient schedule on convergence stability (four-operator IDBO; 20
> defenders vs. 8 targets, $L_{\max}=3$; 40 independent trials over 5 engagement snapshots). (a)
> Swarm-mean assignment cost versus IDBO iteration (mean; shaded band is ±1 std). The linear-decay
> schedule $c_k=c_0(1-k/K_{\mathrm{IDBO}})$ drives the whole swarm to a low, stable cost, whereas
> constant and non-decaying coefficients leave the swarm at a higher, unsettled cost. (b) Late-stage
> swarm cost spread (population standard deviation over the final 25 iterations) across all trials;
> the linear schedule yields the tightest distribution, confirming that the adaptive coefficients
> contract the swarm onto a stable equilibrium.

**中文图题（Fig. R2）**

> 图 R2. IDBO 系数调度对收敛稳定性的影响（四算子 IDBO；20 拦截机对 8 目标，$L_{\max}=3$；5 个交战
> 快照共 40 次独立实验）。(a) 群体平均分配代价随迭代的变化（均值，阴影为 ±1 标准差）。线性衰减
> $c_k=c_0(1-k/K_{\mathrm{IDBO}})$ 将整个群体收敛到更低且稳定的代价，而常数与不衰减系数使群体停留在
> 更高、未收敛的代价。(b) 各次实验末段（最后 25 次迭代）群体代价分布宽度（种群标准差），线性衰减最
> 集中，说明自适应系数使群体收缩到稳定平衡。

**Suggested caption — Fig. R3 (mechanism, `idbo_coeff_schedule.pdf`).**

> Fig. R3. Why the adaptive coefficients stabilize convergence. (a) Coefficient multiplier $s(k)$
> applied to $\{\alpha,\beta,\gamma,\delta,\delta',\eta\}$ and the dancing noise $\sigma_d$ for the
> three policies. (b) Resulting stochastic-perturbation magnitude of the rolling/dancing operators.
> Under the linear schedule the perturbation decays into the deterministic-refinement regime
> (shaded), so the update degenerates to advantage-weighted local refinement with projection — a
> Robbins–Monro-type sufficient condition for a stable fixed point; the frozen schedules retain a
> non-vanishing perturbation and therefore cannot settle.

**中文图题（Fig. R3）**

> 图 R3. 自适应系数为何能稳定收敛。(a) 三种策略下作用于 $\{\alpha,\beta,\gamma,\delta,\delta',\eta\}$
> 及 dancing 噪声 $\sigma_d$ 的系数乘子 $s(k)$。(b) rolling/dancing 算子由此产生的随机扰动幅度。
> 线性衰减下扰动衰减进入确定性精修区（阴影），更新退化为带投影的优势加权局部精修——满足
> Robbins–Monro 型收敛到稳定不动点的充分条件；而冻结系数保留不衰减的扰动，因此无法稳定收敛。

---

*Implementation note (internal, not for the response letter): the ablation uses a
manuscript-faithful implementation of the four-operator IDBO; code and scripts are in
`revise_0723/idbo_paper/` (`scenario_paper.py`, `idbo_paper.py`, `run_ablation.py`,
`plot_ablation.py`, `plot_schedule.py`).*
