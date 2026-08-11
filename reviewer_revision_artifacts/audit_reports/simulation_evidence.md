# Simulation and Data Evidence Audit

Date: 2026-07-18  
Scope: simulation/data only; no manuscript files were modified.

## Evidence Scale

- **A — direct and cross-checked:** numeric artifact exists and agrees with an
  independent calculation or regeneration.
- **B — direct but provenance-limited:** output/config exists, but the exact run
  command, complete generator, or statistical replication is missing.
- **C — code capability only:** implementation exists, but no matching result
  artifact proves that it was exercised.
- **D — absent/contradicted:** requested evidence is missing or the available
  artifact does not support the proposed claim.

## Executive Finding

1. **[A]** `new_sim_fig` contains internally consistent single-episode 3D
   MAPPO and PN results for two paper cases. All 20 defenders hit their assigned
   targets in each saved episode, and all eight target groups satisfy the
   0.5-s synchronization threshold.
2. **[A]** The packaged paper figures are byte-identical to their source figure
   folders. The exported text agrees with the saved NPZ positions, controls, and
   TGO to rounding precision.
3. **[D]** The selected paper figure package itself contains **no executed
   delay, sensor-noise, actuator-lag, or packet-loss configuration**. Such
   effects must not be attributed to the figures in `new_sim_fig`.
4. **[B]** A separate `smooth_robust_V1_final_lag030` package contains one
   successful episode per case with 50-ms sensing delay, 3-m position noise,
   0.3-m/s velocity noise, and 0.30-s command lag. It contains no packet loss.
5. **[B/D]** A separate folder labelled HIL evaluates 1% command dropout and
   normalized observation/action noise over ten episodes per case, but all
   policy endpoints execute in the same Python process. This is a
   software-in-the-loop channel-impairment test, not verified hardware-in-loop.
6. **[D]** No executed partial-interceptor-failure experiment or unseen-maneuver
   generalization experiment was found. No physical/semi-physical timing trace
   or hardware execution log was found.

## Artifact Inventory and Lineage

- `new_sim_fig` has 157 files: 100 PNG, 16 PDF, 12 CSV, 5 NPZ, and 24 TXT.
- MAPPO source is declared at `new_sim_fig/summary.txt:6-11`; PN source and
  method are declared at `new_sim_fig/summary.txt:13-25`.
- `diff -qr` confirms that both full folders in `new_sim_fig` are byte-identical
  to their declared sources under `XIU_code/hil_v9_results/`.
- Every file in `new_sim_fig/figures/mappo` is SHA-256 identical to the
  corresponding file in
  `new_sim_fig/mappo/final_case_specific_match_v1_t0/figures_v9_latest`.
  The same holds for PN and its `figures_v9` folder.
- The reset audit states the fixed assignment and zero-difference initial-state
  check at `initial_state_audit.txt:7-33`. It also warns that exported velocity
  is reconstructed by finite differences at `initial_state_audit.txt:35`.

## Executed 3D Case Structure

**[A] Directly present in the selected NPZ/CSV data**

- Time step: 0.05 s (`paper_case_presets_original_assignment_verified.npz`,
  field `dt`; also `paper_case_presets.json:2-3`).
- Agents: 20 defenders, 8 attackers; fixed assignment
  `[20,21,22,23,24,25,26,27,20,21,22,23,24,25,26,27,20,21,22,23]`.
- All saved trajectories have three position coordinates. All defenders start
  at `z=0 m`; all attackers start at `z=120 m`; observed altitude range is
  `0-120 m` in all four saved episodes.
- MAPPO case 1 has 608 samples (`0-30.35 s`); MAPPO case 2 has 746
  (`0-37.25 s`). PN case 1 has 607 (`0-30.30 s`); PN case 2 has 739
  (`0-36.90 s`).
- The 3-D hit radius is 3 m (`new_sim_fig/summary.txt:20-21`; evaluator default
  at `eval_3d_guidance.py:580-583`). Every recorded impact distance is below
  3 m.

**[C/B] Dynamics implementation and provenance**

- The local 3D core stores flight state as speed, pitch, and yaw, and integrates
  `x/y/z` using those angles (`remote_xiu_onpolicy_3d_fix_clean/.../core.py:
  282-302,448-453`).
- It applies axial, yaw-normal, and pitch-normal load commands and clips them
  before integration (`core.py:239-280`).
- The paper-restoration tool assigns attacker altitude and zero initial vertical
  velocity (`tools/remote_apply_paper_restore.py:157-163`) and computes vertical
  tracking load (`:222-225`).
- The actual remote source tree and exact command that produced
  `final_case_specific_match_v1_t0` are not archived with the result. The local
  clean scenario lacks the paper-preset and robust-channel additions; therefore
  exact end-to-end rerun provenance is **B**, despite direct output consistency.

## Safe Quantitative Results for the Manuscript

These values are from one saved deterministic episode per method/case. Report
them as **representative-run results**, not Monte Carlo success probabilities.

| Method | Case | Hits | Target groups hit/synchronized | Mean first/group time* (s) | Mean group spread (s) | Max group spread (s) | Max hit distance (m) |
|---|---:|---:|---:|---:|---:|---:|---:|
| MAPPO | 1 (evasive) | 20/20 | 8/8, 8/8 | 27.2375 | 0.1250 | 0.2500 | 2.4310 |
| MAPPO | 2 (sinusoidal) | 20/20 | 8/8, 8/8 | 28.0250 | 0.16875 | 0.3000 | 2.9057 |
| PN, `N=4` | 1 (evasive) | 20/20 | 8/8, 8/8 | 27.19479 | 0.15625 | 0.4000 | 2.8761 |
| PN, `N=4` | 2 (sinusoidal) | 20/20 | 8/8, 8/8 | 27.76771 | 0.14375 | 0.3500 | 2.3306 |

Sources: MAPPO `eval_summary.csv:1-3` and both `case*_hit_events.csv`; PN
`eval_summary.csv:1-3` and both `case*_hit_events.csv`.

`*` **Do not directly compare the time column across methods.** MAPPO's
evaluator defines `mean_target_time` as the mean of the first hit time in each
target group (`eval_3d_guidance.py:279-311`). PN's stored value equals the
unweighted mean of the eight within-group mean hit times. The shared CSV label
therefore hides different semantics.

Additional direct checks:

- MAPPO per-target spreads, case 1:
  `[0.25,0.15,0.15,0.20,0.10,0.00,0.15,0.00] s`.
- MAPPO per-target spreads, case 2:
  `[0.30,0.15,0.30,0.25,0.10,0.10,0.15,0.00] s`.
- PN per-target spreads, case 1:
  `[0.40,0.15,0.15,0.20,0.10,0.05,0.20,0.00] s`.
- PN per-target spreads, case 2:
  `[0.35,0.15,0.20,0.20,0.00,0.05,0.15,0.05] s`.
- Maximum saved command magnitude per MAPPO channel:
  case 1 `[0.9882,0.9975,0.6061]`; case 2 `[1.0000,1.0000,0.7749]`.
- PN summary states `N=4`, longitudinal TGO averaging, axial limits
  `[-0.1,1]g`, normal limits `[-1,1]g` at `new_sim_fig/summary.txt:16-21`.
  However, the PN generator source and exact run command are absent locally
  (**B**, not fully reproducible).

## Figure-to-Data Consistency

Recommended paper assets:

- MAPPO case 1 3D trajectory:
  `new_sim_fig/figures/mappo/mappo_nopn_trajectory_3d.png`
- MAPPO case 2 3D trajectory:
  `new_sim_fig/figures/mappo/mappo_sin_trajectory_3d.png`
- MAPPO case 1/2 control and timing:
  `mappo_{nopn,sin}_{nx,ny,tgo,tgo_error,time_sync,distance}.png`
- PN corresponding assets:
  `new_sim_fig/figures/pn/pn_{nopn,sin}_*.png`

Checks:

- NPZ-to-export maximum errors are only text-rounding scale:
  positions `<=1.10e-4 m`, controls `<=3.03e-8`, TGO `<=3.82e-6 s`.
- `plot_3d_eval_v9.py:50-105` exports the NPZ arrays; `:108-165` plots the
  actual 3D coordinates.
- Independent MAPPO regeneration produced all 21 expected figures. Seventeen
  are byte-identical. Four (`heading`/`velocity` for both cases) differ slightly
  because archived `agentsvel.txt` differs from a fresh finite-difference export
  at only 0.0013-0.0016% of values (maximum 0.0546); this matches the explicit
  velocity-reconstruction warning in `initial_state_audit.txt:35`.
- The PN figures and data are mutually copied/hashed consistently, but no PN
  generator script was found. Their pixel-level regeneration is therefore **B**.

## Delay, Noise, Lag, and Packet Loss

### Separate robust mathematical-simulation package

**[B]** `hil_v9_results/smooth_robust_V1_final_lag030/README.txt:4-9`
records:

- one sensor-delay step = `0.05 s`, with compensation enabled;
- defender position noise standard deviation = `3.0 m`;
- defender velocity noise standard deviation = `0.3 m/s`;
- command lag time constant = `0.30 s`;
- yaw/pitch limits `[-1,1]g`, axial limits `[-0.1,1]g`;
- 3-m hit radius.

One saved episode per case reports:

| Case | Hits | Synchronized groups | Mean first-hit time (s) | Mean spread (s) | Max spread (s) |
|---|---:|---:|---:|---:|---:|
| 1 | 20/20 | 8/8 | 27.19375 | 0.12500 | 0.25 |
| 2 | 20/20 | 8/8 | 27.73750 | 0.11250 | 0.25 |

Source: `.../results/final_eval_success_lag030/eval_summary.csv:1-3`.
Limitations: one episode only; no packet loss; implementation/command log is
not archived locally. Use as an illustrative robustness run, not a robust
success-rate estimate.

### Channel-impairment package labelled HIL

Stored settings (`hil_channel_config.json:2-10`, implemented at
`run_hil_v9_experiment.py:17-78`):

- five of 20 defenders (`D0-D4`);
- observation delay `2` steps = `0.10 s`;
- action delay `1` step = `0.05 s`;
- normalized observation noise std `0.003`;
- normalized action noise std `0.02`;
- command dropout probability `0.01`.

Ten-episode direct results:

| Mode | Case | All-hit episodes | All-sync episodes | Mean target time (s) | Worst spread (s) |
|---|---:|---:|---:|---:|---:|
| baseline | 1 | 10/10 | 10/10 | 48.7613 | 0.6000 |
| impaired | 1 | 10/10 | 9/10 | 48.7600 | 43.6500 |
| baseline | 2 | 10/10 | 10/10 | 34.9688 | 0.2000 |
| impaired | 2 | 10/10 | 10/10 | 34.9662 | 0.2000 |

Critical limitations:

- The runner instantiates `InterceptorPolicyNode` objects locally
  (`run_hil_v9_experiment.py:30-78,172-197`); no network transport or hardware
  execution is shown. Call this **software emulation**, not HIL.
- Its defaults use a 12-m hit radius, 3-s sync tolerance, dynamic assignment,
  and random initial geometry (`run_hil_v9_experiment.py:407-442`;
  scenario `simple_world_comm_3d.py:161-194`). These criteria differ materially
  from the final 3-m/0.5-s paper cases, so the rates are not directly comparable.
- The case-1 impaired run contains a real delayed-cooperation failure:
  episode 6 hit all 20 defenders but synchronized only 7/8 groups, with a
  43.65-s worst spread (`hil/case1/case1_episode_summary.csv:7`).

## `test/` Audit

- `test/` contains 121 summary CSVs, 58 NPZs, and 48 logs, mostly preliminary
  case-2 tuning/evaluation runs rather than the final paper package.
- `test/eval_summary.csv:2` reports 24 case-2 episodes with zero all-hit and
  zero all-sync episodes; it must not be substituted for the final evaluation.
- `test/case2_eval_summary_20260711_224623.csv:2-19` and
  `test/case2_autotune_until_hit_20260711_230621_eval_summary.csv:2-25`
  document repeated unsuccessful preliminary searches.
- The only local two-case auto-optimization round trained for five episodes per
  case and then scored 0/6 in both cases
  (`test/auto_opt_sync_session3/.../eval_summary.csv:1-3`).
- `test/v9_export/*/target_sync_stats_3d.csv` uses approximately 12-m hit
  distances and has assignment/hit-set mismatches. It belongs to the looser HIL
  workflow and is not evidence for the final 3-m paper cases.

## Completed, Code-Only, and Missing Reviewer Evidence

### Completed enough for bounded manuscript claims

- **[A]** 3D representative trajectories for both cases and both methods.
- **[A]** Per-run 20/20 hits, eight-target coverage, and synchronization spread.
- **[B]** One-run delay/position-noise/velocity-noise/command-lag demonstration.
- **[B]** Ten-run software channel-impairment experiment including 1% dropout,
  but under different geometry and relaxed hit/sync criteria.
- **[A]** A useful failure-case datum: delayed case-1 synchronization failure in
  HIL-emulation episode 6.

### Code support without matching final-case evidence

- **[C]** Delay-line, observation/action noise, and packet-drop mechanisms in
  `run_hil_v9_experiment.py`.
- **[C]** 3D kinematic integration and load clipping in `core.py`.
- **[C]** Paper-state restoration and parametric attacker tools under `tools/`.

### Missing

- **[D]** Partial interceptor failures (agent removal, stuck actuator, lost
  vehicle, degraded subset) under the final 3-m cases.
- **[D]** Generalization to unseen maneuver families/parameters with a held-out
  protocol and repeated seeds.
- **[D]** Genuine HIL or semi-physical validation: no device identifier,
  NX execution log, timestamped transport trace, measured latency/jitter, or
  simulator-to-hardware I/O record.
- **[D]** Statistical final-case evaluation: current final MAPPO/PN and robust
  summaries each have only one episode.

## Reproducibility and Claim Restrictions

Do not claim:

- that `new_sim_fig` figures include delay, noise, or packet loss;
- Monte Carlo robustness or a statistically estimated 100% success rate from
  the one-episode final summaries;
- MAPPO superiority in interception time (the method definitions differ);
- genuine HIL/hardware validation from `hil_v9_results/hil`;
- partial-failure robustness or unseen-maneuver generalization;
- full reproducibility of PN or robust experiments without their generator
  source and exact command line.

Safe wording:

> In the representative 3-D runs, all 20 defenders intercepted their assigned
> attackers within the 3-m criterion in both scenarios. The maximum within-group
> arrival-time spreads were 0.25 s and 0.30 s for ART-MAPPO, compared with
> 0.40 s and 0.35 s for the `N=4` PN reference. A separate illustrative
> perturbation run with 50-ms sensing delay, 3-m position noise, 0.3-m/s
> velocity noise, and a 0.30-s command-lag constant also completed all
> interceptions, but broader statistical robustness remains to be established.

Remote-source audit note: attempts to reach the previously used local server
failed with `No route to host`; cloud hosts were unavailable in the sandbox,
and a broader SSH probe was not authorized. Conclusions above rely on the
authoritative local artifacts and explicitly identify missing remote provenance.
