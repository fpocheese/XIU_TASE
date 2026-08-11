# Reviewer Response — Complexity, Scalability, and Communication-Delay Analysis of the IDBO

**Reviewer comment.** *Also, a more rigorous complexity and scalability analysis is needed for
the Improved Dung Beetle Optimization algorithm under cooperative deflection environments where
communication delays influence distributed consensus efficiency.*

---

## Response summary

We thank the reviewer for this valuable comment. We have carried out a dedicated study that (i)
empirically validates the per-round computational and communication complexity of the distributed
consensus-based IDBO, (ii) measures its scalability as the swarm grows, and (iii) — most
importantly — quantifies how **communication delay** and **communication-graph diameter** affect the
efficiency of the distributed consensus. All experiments use the manuscript's engagement model and
the four-operator IDBO with the paper's top-$L_{\max}$ winner-set consensus (Eqs. for candidate
selection, bidding, and winner-set update). Three figures are provided for the response letter
(`idbo_complexity.pdf`, `idbo_scalability.pdf`, `idbo_comm_delay.pdf`).

The key finding directly answers the reviewer: **a bounded communication delay never breaks the
consensus — it only slows it, with the time to consensus growing approximately linearly in both the
communication delay and the communication-graph diameter**, exactly as the $\mathcal{O}(N\cdot D)$
argument in the manuscript predicts.

## 1. Complexity model (recap) and what we validate

Per defender, per IDBO iteration, the manuscript states three cost terms:

| Term | Cost | Meaning |
|---|---|---|
| Local preference search | $\mathcal{O}(N_{\mathrm{pop}} N_A)$ | evaluate a candidate population over $N_A$ targets |
| Consensus communication | $\mathcal{O}(N_A\,\lvert\mathcal N_i\rvert)$ | exchange per-target bids with neighbors |
| Advantage update | $\mathcal{O}(N_A)$ | refresh the engagement advantage |

with per-defender memory $\mathcal{O}(N_{\mathrm{pop}} N_A + N_A)$. Because $N_A$ and
$\lvert\mathcal N_i\rvert$ are fixed by the engagement and the communication graph (not by the total
swarm size), the **per-defender** cost is independent of $N_D$, so the total per-round cost is
**linear in the number of defenders**. Experiment A confirms this.

**Experiment A (Fig. `idbo_complexity.pdf`).** We fix the target set ($N_A=8$) and the
communication degree (a bounded 4-regular ring) and vary only the swarm size
$N_D\in\{10,20,40,80,160,240\}$, timing the per-round update.

| $N_D$ | Total per-round | Per-defender | Messages/round |
|---|---|---|---|
| 10  | 2.12 ms | 212 µs | 40 |
| 20  | 3.31 ms | 165 µs | 80 |
| 40  | 6.81 ms | 170 µs | 160 |
| 80  | 13.62 ms | 170 µs | 320 |
| 160 | 28.63 ms | 179 µs | 640 |
| 240 | 42.22 ms | 176 µs | 960 |

The **per-defender cost is essentially constant** (≈170 µs, independent of $N_D$), the **total
per-round cost grows linearly** with $N_D$ (a fitted log–log exponent of **0.97**), and the
**communication volume is exactly linear** in $N_D$ at fixed degree. This empirically confirms the
manuscript's $\mathcal{O}(N_{\mathrm{pop}}N_A+N_A\lvert\mathcal N_i\rvert)$ per-defender model and
the linear total scaling of the distributed scheme.

## 2. Scalability

**Experiment B (Fig. `idbo_scalability.pdf`).** We scale the whole problem — defenders and
targets together — from $10\!\times\!4$ up to $160\!\times\!64$ on random geometric graphs with a
fixed communication delay $\tau=100$ ms, and measure the time to reach consensus and the total
communication volume. Throughout, we report latency in time: one synchronous consensus exchange
occupies one data-link period, taken as $T_{\mathrm{comm}}=50$ ms (a $20$ Hz inter-agent link), so a
message delay of $\tau$ and the consensus time are expressed directly in milliseconds/seconds.

| Problem size | Time to consensus | Total messages |
|---|---|---|
| 10 v 4  | 1.21 ± 0.17 s | 0.8 k |
| 20 v 8  | 1.46 ± 0.37 s | 3.0 k |
| 40 v 16 | 1.28 ± 0.11 s | 11 k |
| 80 v 32 | 1.36 ± 0.12 s | 48 k |
| 160 v 64 | 1.32 ± 0.03 s | 187 k |

The **time to consensus is essentially flat (~1.2–1.5 s) across a 16× increase in swarm size** — it
is governed by the communication-graph diameter and the message delay, not by the number of agents.
Total communication grows gracefully with the problem size. This demonstrates that the distributed
scheme scales to large swarms without an explosion in coordination latency.

## 3. Communication delay and distributed-consensus efficiency (core question)

We model communication delay explicitly: on a connected graph, a neighbor's winner-set message
arrives after a latency $\tau$, so when defender $d_i$ merges it the information is stale by $\tau$
(optionally with packet loss). Latency is expressed in time using the nominal data-link period
$T_{\mathrm{comm}}=50$ ms ($20$ Hz link): each synchronous exchange takes $T_{\mathrm{comm}}$, so a
delay of $\tau$ ms and the total consensus time follow directly. We measure the swarm disagreement
$\Gamma$ (the fraction of communication links whose winner information disagrees; $\Gamma=0$ means
full consensus) and the time needed to reach and hold $\Gamma=0$.

**Experiment C (Fig. `idbo_comm_delay.pdf`).** On a fixed 40-defender / 12-target instance we
sweep the communication delay $\tau\in\{0,50,100,200,400,800\}$ ms, and separately sweep the graph
diameter $D$ by changing the topology.

*Effect of delay:*

| Communication delay $\tau$ (ms) | 0 | 50 | 100 | 200 | 400 | 800 |
|---|---|---|---|---|---|---|
| Time to consensus (s) | 0.57 | 0.90 | 1.23 | 1.88 | 3.08 | 5.54 |

Every configuration still reaches full consensus ($\Gamma\!\to\!0$); the delay does **not** change
the fixed point, it only **slows** the agreement. The time to consensus grows **linearly** with the
delay (fitted slope ≈ **6 ms of consensus time per ms of link delay**).

*Effect of graph diameter:*

| Diameter $D$ (hops) | 1 | 3 | 4 | 5 | 6 | 10 | 20 |
|---|---|---|---|---|---|---|---|
| Time to consensus (s) | 0.50 | 0.86 | 1.07 | 1.29 | 1.48 | 2.45 | 4.49 |

The time to consensus grows **linearly with the graph diameter** (fitted slope ≈ **0.21 s per
hop**), directly supporting the manuscript's $\mathcal{O}(N\cdot D)$ consensus argument: a denser
graph (smaller $D$) or a shorter link delay reaches consensus faster, while a sparse, high-latency
network needs proportionally more time but still converges.

**Answer to the reviewer.** Under cooperative-deflection conditions with communication delay, the
distributed consensus remains correct and its efficiency degrades **gracefully and predictably** —
the consensus time scales linearly with both the message delay and the network diameter, and is
independent of the swarm size. For a representative $20$ Hz link the target-assignment consensus
completes within roughly $0.5$–$1.5$ s even for large swarms and moderate delays, confirming the
practicality of the scheme in delayed, large-scale settings.

## Figures and captions

**Fig. R4 (`idbo_complexity.pdf`).**
> Fig. R4. Per-round computational and communication cost of the distributed consensus-based IDBO
> versus swarm size $N_D$ (fixed target set $N_A=8$, bounded 4-regular communication graph). (a)
> Total per-round wall-clock grows linearly with $N_D$ (log–log slope $0.97$), while the per-defender
> cost stays constant, confirming the $\mathcal{O}(N_{\mathrm{pop}}N_A+N_A\lvert\mathcal N_i\rvert)$
> per-defender model. (b) Communication volume per round is exactly linear in $N_D$.

> 图 R4. 分布式共识 IDBO 每轮的计算与通信开销随蜂群规模 $N_D$ 的变化（固定目标数 $N_A=8$、有界的
> 4-正则通信图）。(a) 每轮总墙钟时间随 $N_D$ 线性增长（log–log 斜率 $0.97$），而每机开销保持恒定，
> 验证了每机 $\mathcal{O}(N_{\mathrm{pop}}N_A+N_A\lvert\mathcal N_i\rvert)$ 的复杂度模型。(b) 每轮
> 通信量随 $N_D$ 严格线性。

**Fig. R5 (`idbo_scalability.pdf`).**
> Fig. R5. Scalability of the distributed assignment (random geometric graph, communication delay
> $\tau=100$ ms). (a) Time to consensus stays flat (~1.2–1.5 s) as the problem scales from 10×4 to
> 160×64 defenders×targets, i.e. coordination latency is independent of swarm size. (b) Total
> communication volume to consensus grows gracefully with problem size.

> 图 R5. 分布式分配的可扩展性（随机几何图，通信延迟 $\tau=100$ ms）。(a) 当问题规模从 10×4 增大到
> 160×64（拦截机×目标）时，达成共识所需时间保持平稳（约 1.2–1.5 s），即协调时延与蜂群规模无关。
> (b) 达成共识的总通信量随规模平缓增长。

**Fig. R6 (`idbo_comm_delay.pdf`).**
> Fig. R6. Communication delay and distributed-consensus efficiency (40 defenders, 12 targets;
> nominal data-link period $T_{\mathrm{comm}}=50$ ms). (a) Swarm disagreement $\Gamma$ versus
> consensus time for several communication delays $\tau$: larger delay slows the decay but $\Gamma$
> still reaches $0$. (b) Time to consensus grows linearly with the communication delay $\tau$. (c)
> Time to consensus grows linearly with the communication-graph diameter $D$, supporting the
> $\mathcal{O}(N\cdot D)$ argument.

> 图 R6. 通信延迟与分布式共识效率（40 拦截机、12 目标；名义数据链周期 $T_{\mathrm{comm}}=50$ ms）。
> (a) 不同通信延迟 $\tau$ 下群体不一致度 $\Gamma$ 随共识时间的变化：延迟越大衰减越慢，但 $\Gamma$
> 最终仍归零。(b) 达成共识所需时间随通信延迟 $\tau$ 线性增长。(c) 达成共识所需时间随通信图直径 $D$
> 线性增长，支持 $\mathcal{O}(N\cdot D)$ 的共识论断。

---

*Implementation note (internal): the distributed consensus simulator, communication-delay model,
and experiment/plot scripts are in `Re3_2/code/` (`scenario_paper.py`, `consensus_idbo.py`,
`run_experiments.py`, `plot_delay.py`, `plot_complexity_scalability.py`). Raw data:
`data_complexity.npy`, `data_scalability.npy`, `data_delay.npz`.*
