# Manuscript audit: reviewer coverage and evidence integrity

## 1. Audit scope and evidence scale

- Snapshot: 2026-07-18; manuscript under review is `/home/uav/00gao_xueshu/DT_PAPER/XIU_tase_paper_V1/main.tex` (1384 lines).
- Baselines: repository `HEAD:main.tex` (original submission) and `/home/uav/00gao_xueshu/DT_PAPER/XIU_code/main.tex`.
- Review source: `Revision.md`. It contains Reviewer 1--3 only; the AE summary is absent. The AE text used below is independently present at `yijian.md:3-5` and in the review DOCX.
- `A`: direct source/diff/raw-data evidence; `B`: traceable code or artifact with limited validation; `C`: qualitative/selected-run evidence only; `0`: absent or contradicted.
- Status terms: `covered` means the requested change is present and defensible; `partial` means text exists but evidence/logic is insufficient; `open` means not answered.

## 2. Three-version comparison

- Paper working tree vs `HEAD`: `main.tex` has `+253/-24`; only `main.tex` and compiled `main.pdf` are tracked modifications. No revised figure asset is tracked. `[A]`
- Defensible additions relative to `HEAD`: de-emphasized contribution claim (`main.tex:212-223`); intuitive transition prose (`349`, `369`, `386`, `394`, `414`, `425`, `447`, `604`, `675`, `680`, `692`, `727`, `751`, `836`); IDBO pseudocode (`alg:idbo`, `545-581`); ART-MAPPO pseudocode (`alg:artmappo`, `913-952`); empirical-convergence caveat (`1097-1107`); notation appendix (`app:notation`/`tab:notation`, `1340-1382`). `[A]`
- High-risk additions relative to `HEAD`: purported convergence/equilibrium argument (`subsec:convergence_equilibrium`, `485-532`) and unsupported qualitative sensitivity claims (`subsec:reward_sensitivity`, `854-892`). `[A]`
- `/XIU_code/main.tex` differs from the paper working tree only by 25 lines at workspace lines `894-918`: a one-seed “reward sensitivity preview,” `fig:reward_sensitivity_preview`. `[A]`
- The simulation setup, 25-step statement, 2-D cases, Fig. 5 asset, Fig. 6 asset, Monte Carlo section, and all old case figures are unchanged from `HEAD`; therefore the main empirical reviewer concerns remain unresolved. `[A]`

## 3. Blocking findings: delete or rebuild before submission

### C1. Training curves do not come from the claimed interception environment

- Paper claim: 20-vs-8 ART-MAPPO interception training, 10,000 episodes, 25 steps (`subsec:training`, `1089-1117`).
- Figure hashes for `fig_a_reward.png`, `fig_b_critic_loss.png`, and `fig_c_entropy.png` exactly match `0620septimedone/.../results/simple_converge_v7/`. `[A]`
- Their generator uses `SimpleConvergeEnv(num_agents=4,num_landmarks=4,episode_length=25)` (`train_simple_converge_v7.py:97-122,132`), not the 20-vs-8 UAV environment.
- That environment is a normalized 2-D box with `dt=0.1`, `max_speed=1`, four landmarks, and a synthetic dense coverage reward (`simple_converge.py:13-47,141-203`). `[A]`
- The final plotting script explicitly blends ART-MAPPO data into a prescribed ceiling `MAX_REWARD=9.0`, adds Gaussian noise, and anneals the uncertainty band (`plot_results_ieee_v3.py:6-13,33-43,128-206`). Critic-loss and entropy curves are also reconstructed (`export_plot_data.py:98-183`). `[A]`
- Raw arrays do contain five seeds and yield terminal mean `8.9712` and seed SD `0.6363`, but the displayed curve and convergence diagnostics are not raw statistics. The stated “improvements” use `(ours-baseline)/ours`; conventional relative improvements are 8.92% vs IQL, 13.98% vs MAPPO, 16.81% vs IPPO, and 42.53% vs IA2C, not `8.2/12.3/14.4/29.8%`. `[A]`
- Required action: rebuild Fig. `fig:training_convergence` from unmodified logs produced by the actual interception environment and recompute all statistics. Until then delete `1091-1117`, abstract superiority claims `75-77`, and corresponding conclusion claims `1325-1329`.

### C2. Target-assignment figures do not implement the manuscript formulation

- `convergence.png` and `violin.png` exactly match `dbo_code/python_project/results/`. `[A]`
- The code solves an 8-dimensional “each target chooses one UAV” problem and minimizes distance/heading/load/threat cost (`scenario.py:1-12,61-66,127-146`).
- Manuscript Eq. `eq:optimization_formulation` (`372-380`) instead assigns every one of 20 interceptors to a target and minimizes target survival probability under per-target lower/upper capacities.
- The available IDBO code uses Bernoulli initialization, spiral search, Lévy flight, opposition learning, and neighborhood search (`idbo.py:433-567`); it contains no distributed consensus, bids, winner lists, communication delays, or manuscript adversarial-advantage operators. `[A]`
- Required action: either implement and rerun the exact manuscript algorithm/objective, or rewrite Section III and its claims to match the actual centralized code. Do not use the present figures to support `eq:hierarchical`--`eq:convergence`, distributed scalability, or equilibrium.

### C3. The added convergence/equilibrium “proof” is invalid

- `485-496` assumes elitist acceptance and feasibility projection, but neither Eqs. `eq:rolling_update`--`eq:stealing_update` nor `alg:idbo` specifies them.
- `498-507` invokes finiteness of binary assignments to prove convergence of `Phi`, although `Phi` is defined on continuous, neighbor-coupled preferences; local acceptance does not imply aggregate monotonicity after other agents and `A_ij` change. A finite plateau can also cycle.
- `509-513` claims `O(ND)` consensus despite changing bids, delays, and no synchronous/reliable-message model.
- `515-525` asserts an epsilon-Nash property without defining a game/potential or relating assignment-disagreement epsilon in `eq:convergence` to utility-regret epsilon in `eq:epsilon_nash`.
- `534-540` calls `O(TPN^2)` “per-iteration” although it contains total iteration count `T`; no derivation or scale experiment supports it.
- Required action: delete `eq:fitness_monotonic`, `eq:epsilon_nash`, and all guarantee language. Retain a rigorous limitation statement: fixed-snapshot finite heuristic search has no proven global/Nash guarantee; report empirical convergence, runtime, scale, topology, and delay tests instead.

### C4. Manuscript architecture and available implementation disagree

- Manuscript attention dimensions are inconsistent: `H^(0)` is `n_o x (d/n_o)`, each of `H` heads outputs `n_o x d/H`, so concatenation is `n_o x d`; flattening is length `n_o d`, incompatible with `W^O in R^(d x d)` and residual addition to `h^(0) in R^d` (`614-632`).
- Available actor code applies attention to `actor_features.unsqueeze(1)`, hence sequence length is one (`r_actor_critic_advanced.py:192-205`); this cannot realize the claimed cross-channel attention.
- Repository search finds no implementation of trust update, trust-blended action sampling, `a_opt`, or `a_probe`; yet these are central in `eq:trust_evolution`--`eq:expected_action`.
- Adaptive KL defaults to `kl_coef=0`; multiplicative updates keep it at zero (`r_mappo_advanced.py:47-49,157-164`; `config.py:341-345`). `[A]`
- Required action: reconcile equations with the code actually used for each reported checkpoint, then perform the requested module ablations. Otherwise remove module-level performance attribution.

### C5. Additional mathematical/logical errors

- Trust: from `eq:trust_evolution`, `T_(k+1)-T_k=alpha_T[sigma(tau Rtilde)-T_k]`; above-average reward does not necessarily increase trust as claimed at `690`. It moves trust toward the current sigmoid target. Replace the monotonic language and state the fixed-input contraction factor `1-alpha_T`.
- Guided exploration: `a_opt/a_probe` use `Q_phi(o,a)` (`705`), but the model defines only `V_phi(s)` (`665-668`). Mixed heuristic behavior is also absent from the PPO likelihood ratio, creating an unaddressed off-policy mismatch.
- Dual clip: `eq:dual_clip_final` lower-bounds the surrogate for negative advantages; it does not constrain the actual ratio to `[1/c,c]`. Delete false Eq. `eq:rt_bound` and claims at `745-749`.
- Residual gradient: the norm lower bound at `648` is false because `I+J` can cancel and the ReLU gate can be zero. Replace “preventing” with “can improve gradient flow.”
- IDBO stealing: `eq:stealing_update` (`434-435`) updates a vector but contains free index `j`; denominator can be zero/negative. Write a component-wise equation with normalized nonnegative weights and epsilon denominator.
- Thresholding: `eq:thresholding` (`440-446`) raises the selection threshold for larger advantage, the opposite of the prose; it can divide by a nonpositive maximum. Reverse/normalize the effect and define edge cases.
- 3-D inconsistency: model/action/reward remain 2-D (`255-273`, `423`, `657-661`, `798`, `821-825`, `846-852`); inserting 3-D figures alone would contradict the method.

### C6. The workspace sensitivity preview is not submission evidence

- Workspace-only `fig:reward_sensitivity_preview` uses one seed and an incomplete manifest (`cases.csv` contains an unfinished entry). `[A]`
- Each completed preview configuration contains only one logged episode; `actor_grad_norm=0`, `is_warmup=1`, and `kl_coef=0`, so the actor was not updated. `[A]`
- Required action: remove workspace lines `894-918`. Replace only after all one-factor cases and nominal settings are run in the true interception environment over multiple seeds, with ISR, miss distance, synchronization, and overload metrics plus uncertainty.

## 4. Reviewer-comment mapping

### Associate Editor summary (`yijian.md:5`; absent from `Revision.md`)

- Dense math/notation: `partial`, `[A]`; intuitive prose and `tab:notation` added, but Fig. overload is unchanged and notation still omits/collides on reward/assignment `w_1,w_2`.
- Logical inconsistency: `open`, `[A]`; trust wording changed to smoothing but line `690` remains mathematically wrong.
- Simplistic environment/parameter mismatch: `open`, `[A]`; `255-273`, `970-1044`, and `1089` are unchanged.
- Ablation/convergence/realistic robustness: `open`, `[A]`; no ablation, invalid guarantee argument, and no manuscript robustness experiment.

### Reviewer 1

- R1.1 notation table (`Revision.md:10`): `partial`, `[A]`; `app:notation`/`tab:notation`, `1340-1382`. Add all weights, trust/PPO constants, 3-D states/actions, units; disambiguate duplicate `w_1,w_2`.
- R1.2 pseudocode (`:11`): `covered structurally`, `[A]`; `alg:idbo` and `alg:artmappo`. Correct them after implementation reconciliation; IDBO pseudocode currently describes unavailable code.
- R1.3 intuition (`:12`): `covered`, `[A]`; transition prose listed in Section 2. Remove causal overclaims.
- R1.4 delays/uncertainty/3-D (`:13`): `open`, `[A]`; only future-work sentence `1331-1334`; manuscript is explicitly 2-D. New 3-D figures exist but are not cited.
- R1.5 reward/hyperparameter sensitivity (`:14`): `partial`, `[C]`; qualitative `854-892`; workspace preview invalid (C6).
- R1.6 convergence/equilibrium (`:15`): `partial but unsafe`, `[0]`; `485-540` attempts coverage but is not a valid proof.

### Reviewer 2

- R2.1 novelty overstatement (`Revision.md:23`): `mostly covered`, `[A]`; revised contribution `212-223`. Still soften “three synergistic modules” (`599`) and overload causality (`718-719`, `769`).
- R2.2 unclear Fig. 5 (`:24`): `open`, `[A]`; `fig:algorithm_flow` still uses unchanged `rlsuanfa.pdf` (`783-790`).
- R2.3 trust paradox (`:25`): `partial`, `[A]`; `675-690` now calls it smoothing, but the direction claim is still wrong; use the exact increment equation in C5.
- R2.4 simplistic/high-fidelity environment (`:26`): `open`, `[A]`; 2-D model/setup unchanged. “High-fidelity features” at `673` should be “learned features.”
- R2.5 25 steps and D1/D2/D3/Fig. 6 (`:27`): `open/contradicted`, `[A]`; `dt=0.05` at `972`, radii at `1000-1014`, 25 steps at `1089`, unchanged `chushi01.pdf`. Training evidence is actually the toy environment in C1.
- R2.6 end-to-end plus ablation (`:28`): `open`, `[A]`; `1070` then `1126-1127` is narrative sequencing, not a data-passing experiment. New 3-D preset hard-codes assignment (`new_sim_fig/.../initial_state_audit.txt:10-13`); no ablation exists.

### Reviewer 3 (compound comment at `Revision.md:36`)

- R3.1 adaptive DBO coefficients/stability: `open`, `[0]`; `425-437` merely says all coefficients decay linearly, while only one explicit exponential factor appears and available code uses different operators.
- R3.2 complexity/scalability under delay: `open`, `[0]`; `eq:complexity` and `488-540` lack correct derivation, scale sweep, message model, and delay experiment.
- R3.3 trust/GRU/attention-residual ablation: `open`, `[0]`; no ablation figure/table; implementation mismatch C4 prevents attribution.
- R3.4 ART-MAPPO vs MAPPO theory: `partial`, `[C]`; caveat `1097-1107` correctly denies global optimality but gives no comparative guarantee, exploration, or robustness analysis.
- R3.5 block-probability and reward sensitivity: `open`, `[0/C]`; no sweep of assignment `w_1,w_2`; reward discussion is qualitative and preview invalid.
- R3.6 dual-clip/KL deployment overhead: `open`, `[0]`; no timing/FLOPs/parameter table. These are training-time losses, so state that actor inference overhead is unchanged only if verified from the deployed graph.
- R3.7 packet loss/noise/interceptor failure: `partial outside manuscript`, `[B]`; local package specifies 100-ms sensing delay, 50-ms actuation delay, noise `0.003/0.02`, dropout `0.01` (`hil_channel_config.json:2-10`) and has 10-run summaries, but no partial-interceptor-failure experiment and no manuscript inclusion.
- R3.8 failure/delayed-engagement cases: `open`, `[A]`; Monte Carlo explicitly reports only successful trials (`1294`), excluding the cases requested.
- R3.9 unseen-maneuver generalization: `open`, `[A]`; only two predefined cases (`1030-1043`, `1135-1283`), no held-out maneuver family.
- R3.10 HIL/semi-physical validation: `open`, `[B]`; README says nodes are “intended for NX deployment” (`hil_deployment_package/README.md:8-13`), not evidence of execution on hardware; no processor timing or hardware log exists.

## 5. New simulation package: what it can and cannot support

- `new_sim_fig` provides 3-D MAPPO and PN trajectories for both cases and exact selected-run CSV/NPZ provenance. It can support representative 3-D case illustrations. `[A]`
- Its `eval_summary.csv` has `episodes=1` for each method/case. MAPPO and PN both have all-hit/all-sync rates 1.0; MAPPO spreads are 0.25/0.30 s and PN 0.40/0.35 s. It cannot support success-rate superiority, robustness, or significance. `[A]`
- The available 10-run delay/noise/dropout experiment (`hil_v9_results/eval_summary.csv:2-5`) is preliminary: all-hit is 1.0, but Case 1 HIL all-sync is 0.9 with one 43.65-s outlier. Report exact `n=10` and the failure, not “strict robustness.” `[A]`
- New results conflict with old prose that PN “completely fails,” catastrophically saturates, or has 17.5% ISR (`1165`, `1243`, `1294-1310`). Rebuild the comparison text around the new evidence.

## 6. Minimum in-scope revision sequence

1. Remove the workspace sensitivity preview and all unsupported proof/guarantee language.
2. Resolve the algorithm-code mismatches in C2/C4 before writing any response claiming reproducibility.
3. Replace Fig. `fig:training_convergence` with raw true-environment, multi-seed curves; recompute all percentages and avoid causal attribution without ablation.
4. Correct trust, dual-clip, attention dimensions, residual-gradient wording, IDBO threshold/stealing equations, and the 3-D state/action model.
5. Replace unchanged Fig. `fig:algorithm_flow`; show one unambiguous `h^(0) -> h^(l) -> h^(l+1)` path.
6. Correct the setup: actual episode horizon, `dt`, physical duration, D1/D2/D3 geometry, 3-D dynamics, sensor/actuator delay, noise, dropout, and randomization ranges.
7. Insert the 3-D representative figures only with selected-run wording; do not infer statistics from `n=1`.
8. Run minimal missing experiments: component ablation; reward and assignment-weight sensitivity; assignment scale/delay sweep; packet/noise/failure robustness; failure-case analysis; held-out maneuvers; measured inference latency.
9. Demonstrate end-to-end data flow by exporting the IDBO assignment and consuming that exact file in guidance; include reassignment after an interceptor failure if claimed.
10. Treat the present software endpoint package as processor-in-the-loop emulation unless actual NX execution logs and timing are produced; state HIL/semi-physical validation as a limitation otherwise.

## 7. Safe material to retain

- Retain the narrowed contribution wording at `212-223`.
- Retain concise equation-intuition transitions after removing “guarantees,” “perfectly,” “proves,” “catastrophic,” and unsupported direct-causation language.
- Retain both algorithm boxes as structural aids only after aligning every step with executable code.
- Retain the notation appendix after completing and disambiguating it.
- Retain the empirical non-global-optimality caveat at `1097-1107`, but attach it to rebuilt raw training evidence.
- Retain 3-D selected trajectories and exact channel settings as transparent representative/configuration evidence; keep statistical claims separate.
