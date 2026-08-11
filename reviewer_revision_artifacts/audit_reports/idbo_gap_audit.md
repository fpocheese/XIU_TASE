# IDBO / Sensitivity / Convergence Evidence-Gap Audit

Date: 2026-07-18  
Scope: Reviewer 1 comments 5--6, Reviewer 2 comment 6, and the IDBO coefficient,
complexity, delayed-consensus, weighting-sensitivity, convergence, and equilibrium
parts of Reviewer 3. No manuscript file was edited by this audit.

## Evidence scale

| Grade | Meaning |
|---|---|
| A | Raw data, generating code, fixed parameters, and an independent rerun agree. |
| B | Direct code/data evidence supports only a bounded claim; no full requested validation. |
| C | Indirect, preview-only, incompletely traceable, or methodologically weak evidence. |
| D | Missing, contradicted by the implementation/data, or addressing a different problem. |

## Executive verdict

1. **The current `dbo_code` experiment is not an implementation of the paper's
   distributed many-to-one IDBO.** It solves a centralized 8-dimensional problem in
   which each target chooses one of 20 UAVs
   (`idbo_code/python_project/scenario.py:1-12,63-66,127-146`). The manuscript uses a
   \(20\times8\) binary matrix in which every defender chooses one target and each
   target receives 1--\(L_{\max}\) defenders
   (`XIU_tase_paper_V1/main.tex:349-380`). This is a direction, dimension, objective,
   feasibility, and architecture mismatch. **Grade D for manuscript-level validation.**
2. The existing 20-run convergence arrays are exactly reproducible for that centralized
   surrogate. They support only: (a) monotonic best-so-far cost, and (b) zero exhaustive
   single-coordinate improvement gap at termination. **Grade A for this bounded claim.**
3. They do **not** support distributed consensus: final population disagreement is
   \(0.338625\pm0.011578\), nonzero in 20/20 runs
   (`idbo_convergence_equilibrium_summary.csv:7-8`). **Grade D for consensus/delay claims.**
4. `reward_sensitivity_preview.pdf` has no raw table or generator in the worktree, is
   one-seed/reduced-budget by its own caption, compares returns under different reward
   definitions, and omits energy, synchronization, PPO/KL, and block-probability weights.
   It must not be used as final evidence. **Grade D for a final sensitivity result.**
5. Reviewer 2 comment 6 is not answered by the three response documents. The 3D
   guidance environment uses either a fixed mapping or Hungarian assignment plus local
   swaps, not IDBO (`simple_world_comm_3d.py:186-192,208-285`). There is no IDBO-to-
   ART-MAPPO end-to-end run and no ART-MAPPO component ablation. **Grade D.**

## Audited artifacts

| Artifact | Finding |
|---|---|
| Paper `dbo_code/python_project` | Same base optimizer/scenario/results as workspace; it lacks only the added history diagnostics. |
| Workspace `idbo_code/python_project/idbo.py` | Adds disagreement, switch-rate, and coordinate-gap logging at `433-455,458,483-491,599-609`; it does not add communication or consensus. |
| `results/results.pkl` | SHA-256 identical in paper/workspace: `462456...a3a8`; 30 runs x 300 iterations for six centralized optimizers. |
| `response_convergence_equilibrium.tex/pdf` | Compiles to 4 pages, but labels evidence as preview and promises later experiments (`tex:50-65,76-89`). |
| `response_reviewer_comments_5_6.tex/pdf` | Compiles to 7 pages; these are Reviewer 1 comments 5--6, not Reviewer 2 comment 6 (`tex:14-19,107-110`). |
| `response_reward_sensitivity.tex/pdf` | Compiles to 3 pages; explicitly one seed/reduced budget and future multi-seed work (`tex:43-49,64-70`). |
| `reward_sensitivity_preview.pdf` | Matplotlib 3.7.1, created 2026-06-17; no generator or raw result file found. |
| `rl_convergence_preview.pdf` | No generating script/raw episode table found; ten episodes cannot establish convergence. |
| Response build logs | No LaTeX errors, undefined references, overfull/underfull warnings, or package warnings were found. This validates typesetting only. |

## Paper-code mismatch in detail

### Problem and objective

- Paper: probability-based many-to-one assignment with binary \(X_{ij}\), all 20
  defenders assigned, every attacker covered, and a per-target capacity
  (`XIU_tase_paper_V1/main.tex:349-380`).
- Code: an 8-entry vector, one UAV index per target; unused UAVs are allowed
  (`scenario.py:4-10,63-66`). Its objective is distance + heading + workload balance
  + threat exposure with weights 0.35/0.20/0.25/0.20
  (`scenario.py:54-58,72-135`), not the paper's survival-probability objective.
- The paper claims a topology with 3 defenders for attackers 1--4 and 2 defenders for
  attackers 5--8 (`XIU_tase_paper_V1/main.tex:1070-1076`). The audited code cannot
  produce this topology because each of its eight targets receives exactly one UAV.

### Operators and coefficients

- Paper operators use \(\alpha,\beta,\gamma,\delta,\delta',\eta,\sigma_d\), described
  only as linearly decaying adaptive coefficients
  (`XIU_tase_paper_V1/main.tex:425-437`). No initial/final values or schedule equations
  are supplied.
- Active code instead uses standard DBO rolling/dancing logic with random `b_val` and
  a fixed 0.1 term (`idbo.py:499-512`), a variable spiral with `k_spiral=5`
  (`idbo.py:523-546`), Lévy \(\beta=1.5\), scale 0.01 and nonlinear weight
  (`idbo.py:548-566`), probabilistic opposition learning (`idbo.py:568-579`), and
  adaptive random-neighborhood sampling 6--15 (`idbo.py:397-430`).
- No code implements adversarial advantage \(A_{ij}\), local neighbor estimates,
  bids/winner lists, top-\(L_{\max}\), message delays, or the manuscript equations
  at `main.tex:394-483`. Thus coefficient or delay experiments cannot be obtained by
  merely rerunning the present script.

### End-to-end coupling

- Guidance presets contain a fixed 20-entry assignment
  (`paper_case_presets.json:4-24,209-229`).
- The only dynamic guidance assignment uses SciPy `linear_sum_assignment`, then
  swap refinement (`simple_world_comm_3d.py:208-285`).
- Repository search found no import/call of IDBO in the ART-MAPPO environment.
  Therefore separate assignment plots plus guidance plots are not an end-to-end test.

## Reviewer-by-reviewer sufficiency

| Comment | Existing support | Grade | Decision |
|---|---|---:|---|
| R1-5 reward/hyperparameter sensitivity (`Revision.md:14`) | Qualitative discussion plus untraceable one-seed reward plot; no energy/PPO/KL sweep. | D | Do not submit the preview as quantitative evidence. |
| R1-6 convergence/equilibrium (`Revision.md:15`) | Centralized surrogate best-cost and coordinate-gap diagnostics reproduce exactly; ART preview is ten episodes/one seed. | B/D | Keep only a narrowly worded best-so-far lemma; do not claim distributed/Nash or MARL equilibrium. |
| R2-6 complete workflow and ablation (`Revision.md:28`) | No IDBO-to-guidance pipeline; no ART module ablation; not addressed in response files. | D | Requires new end-to-end and ablation evidence. |
| R3 IDBO operator coefficients (`Revision.md:36`) | Paper coefficients absent from code; no ablation/sensitivity. | D | Reconcile implementation first, then ablate. |
| R3 complexity/scalability (`Revision.md:36`) | One fixed-size wall-clock comparison only; hardware unspecified. | C | Cannot support asymptotic or onboard real-time claims. |
| R3 delayed consensus (`Revision.md:36`) | No communication graph, messages, or delay in IDBO code; measured disagreement stays nonzero. | D | Requires an actual distributed simulator. |
| R3 block-probability weights (`Revision.md:36`) | No sweep of \(w_1,w_2\) in \(P_{ij}\); code uses a different objective. | D | Requires paper-objective experiment. |
| R3 reward weights (`Revision.md:36`) | Preview varies only angle, distance, hit and one coordination setting. | D | Retrain multi-seed and report invariant mission metrics. |

## Reproducible numerical evidence

### Existing 30-run centralized benchmark

Source: `idbo_code/python_project/results/results.pkl`; protocol is documented at
`README.md:23-28` and defaults at `main.py:395-420`.

| Algorithm | final cost mean ± std | mean runtime, s |
|---|---:|---:|
| DBO | 0.192291 ± 0.002579 | 1.2293 |
| IDBO | 0.188586 ± 0.000850 | 1.5447 |

For this surrogate, IDBO lowers mean cost by 1.927% relative to DBO but takes 25.652%
more wall time. The complete pickle also contains 30x300 curves and times for PSO,
GWO, DBO, SSA, BOA, and IDBO. This does not establish scaling or distributed runtime.

Documented regeneration command (not rerun in this audit because it is the old problem):

```bash
cd /home/uav/00gao_xueshu/DT_PAPER/XIU_code/idbo_code/python_project
python3 main.py --N 50 --iter 300 --runs 30 --outdir /tmp/idbo_baseline_repro
```

`main.py` has no explicit seed argument; add per-run seed logging before treating a
fresh full rerun as archival evidence.

### Independently rerun convergence diagnostic

Command actually run:

```bash
cd /home/uav/00gao_xueshu/DT_PAPER/XIU_code/idbo_code/python_project
python3 convergence_equilibrium_experiment.py \
  --N 50 --iter 150 --runs 20 --seed 2026 \
  --outdir /tmp/idbo_audit_repro
```

All four regenerated `.npy` files were byte-identical to the archived arrays. Only
elapsed time changed (49.03 s archived; 61.72 s audit). Verified results:

- final cost \(0.188666\pm0.000711\);
- final incumbent switch rate 0 in 20/20 runs;
- final coordinate local-improvement gap 0 in 20/20 runs;
- all 20 best-cost curves are monotone non-increasing;
- median last stabilization iteration 52.5;
- final population disagreement range 0.3125--0.3550, nonzero in 20/20 runs.

The generator and definitions are at
`convergence_equilibrium_experiment.py:22-49,103-150` and
`idbo.py:433-455,599-609`.

## Figures that are currently unusable for final claims

1. **`idbo_convergence_equilibrium_preview.pdf`: unusable as distributed-consensus
   evidence.** It omits the archived disagreement series, whose final mean is 0.338625,
   and evaluates the inverse centralized assignment. It may be retained only if relabeled
   as a centralized surrogate optimizer diagnostic.
2. **`reward_sensitivity_preview.pdf`: unusable.** No raw data/generator; one seed and
   reduced budget (`response_reward_sensitivity.tex:43-49`); no energy/sync or PPO/KL
   sweeps. More importantly, final training returns under different reward weights are
   not directly comparable because the measured objective itself changes. Use invariant
   mission metrics instead.
3. **`rl_convergence_preview.pdf`: unusable for convergence.** Ten episodes, one seed,
   15,000 steps, and no traceable raw data (`response_convergence_equilibrium.tex:76-89`).
4. **Paper `convergence.png` / `violin.png`: usable only for the old centralized
   20-UAV-to-8-target surrogate.** They cannot validate equations `main.tex:349-540`
   or the topology asserted at `main.tex:1070`.
5. **`response_reviewer_comments_5_6.pdf`: not a response to Reviewer 2 comment 6.**
   Its headings/quoted comments are Reviewer 1 comments 5--6 (`tex:14-19,107-110`).

## Theory overclaim audit

1. `main.tex:488-513` does not prove distributed convergence. Local utilities depend
   on stale neighbor estimates; simultaneous accepted local moves need not make the
   aggregate \(\Phi\) monotone. Non-strict acceptance permits plateau cycling. Finite-time
   winner propagation additionally needs fixed unique bids, deterministic tie-breaking,
   reliable delivery, phase separation, and an explicit delay bound.
2. The \(O(ND)\) round claim at `main.tex:509-513` omits the delay bound and message
   model. With delay bound \(B\), even immutable information propagation depends on
   \(D\) and \(B\); whether an additional \(N\) factor appears depends on parallel versus
   serialized target messages.
3. Top-\(L_{\max}\) lists do not alone guarantee the row constraint “each UAV chooses
   exactly one target” or the lower bound “each target gets at least one UAV”
   (`main.tex:375-377,453-459`). Hence feasibility is not established.
4. The \(\epsilon\)-Nash assertion at `main.tex:515-528` lacks a potential-game proof
   linking local utilities/bids to global \(J\), and lacks exhaustive feasible unilateral
   deviation measurements. The current gap instead treats each *target* as a coordinate
   choosing one UAV (`idbo.py:443-455`), the inverse of the paper's player/action model.
5. `main.tex:534-540` calls the expression “per-iteration” but places \(T\) only on the
   optimization term. It also assumes \(P=O(N)\) without a scaling rule. A defensible
   form is \(O(T[P\,C_F(N,d_i)+N d_i+N])\), with \(C_F\), message payload, graph degree,
   and parallelization explicitly defined.
6. PPO clipping and a soft KL penalty regularize updates; they are not by themselves a
   hard bound on parameter or policy change. Replace “bound”/“guarantee” language in
   `response_convergence_equilibrium.tex:67-74` with “mitigate” unless a hard gradient,
   KL projection, or early-stop condition is demonstrated. Do not call empirical reward
   stabilization a local policy equilibrium (`tex:91-97`).
7. “\(w_4\) is the most influential” at `main.tex:870-873` is unsupported by the preview,
   which contains only one reduced \(w_{\rm coord}\) setting. Use “controls” or provide
   multi-seed, metric-based evidence.

## Minimum experiments still required

Implementation reconciliation is gate 0. Choose one truthful route:

- **Route A (recommended):** revise the IDBO formulation to match the available
  Bernoulli/spiral/Lévy/nonlinear-weight/ARNS centralized code and explicitly drop
  distributed-consensus claims; or
- **Route B:** implement the paper's \(20\times8\) objective, constraints, adaptive
  operators, top-\(L_{\max}\) neighbor consensus, delay queues, tie-breaking, and logs.
  Only Route B can answer the current distributed-IDBO wording.

After Route B, the minimum credible run set is:

1. **Coefficient ablation:** nominal, each operator disabled, and 0.5x/1x/1.5x schedule
   for rolling/dancing/breeding/stealing; 30 paired seeds, 20v8, identical evaluation
   budget. Report cost, feasible rate, convergence iterations, unilateral gap, and runtime.
2. **Delay/scaling:** \(M/N=\{20/8,40/16,80/32\}\), delay
   \(B=\{0,1,2,5,10\}\) rounds, at least ring and connected sparse graph, 30 seeds.
   Report consensus success, rounds, disagreement, messages/bytes, cost, and runtime.
3. **Block-probability weights:** \(w_{\rm ZEM}\in\{0.2,0.35,0.5,0.65,0.8\}\),
   \(w_\sigma=1-w_{\rm ZEM}\), 30 fixed snapshots/seeds. Report assignment changes,
   objective, feasibility, and downstream interception metrics.
4. **Reward sensitivity:** retrain at 0.5x/1x/2x for distance, angle, hit, coordination,
   energy, and the implemented sync term; at least five full-budget seeds per setting.
   Evaluate all policies with success rate, miss distance, arrival spread, peak load,
   and control energy, not their differently scaled training returns.
5. **End-to-end R2-6:** pass the actual IDBO assignment into the 3D environment, then
   run both paper cases against fixed/Hungarian/DBO assignments, 30 evaluation seeds.
   Archive the assignment matrix used by every episode. Separately run trust, GRU, and
   attention-residual ablations; the current response set has neither experiment.

No runner for items 1--3 exists in the audited tree. The implementation should expose one
CLI, e.g. `experiments/idbo_revision.py --mode coefficients|delay_scale|probability`,
write JSON/CSV plus seed/config hashes, and generate figures only from those tables.

## Evidence-safe minimum manuscript scope now

Until the missing experiments are run, the current files safely support only this narrow
statement: **on the centralized 20-UAV/8-target surrogate implemented in `scenario.py`,
the elitist IDBO best-so-far cost is monotone and the 20 seeded runs terminate at solutions
with zero single-coordinate improvement gap.** It does not imply distributed consensus,
delay robustness, the manuscript many-to-one assignment, global optimality, or Nash
equilibrium of the UAV assignment game.

For a submission retaining the present distributed formulation, omit all three preview
figures and do not use future tense such as “the full revision experiment will...”
(`response_reward_sensitivity.tex:46-49,68-70`;
`response_convergence_equilibrium.tex:53-65,79-89`). Complete Route B experiments first.
