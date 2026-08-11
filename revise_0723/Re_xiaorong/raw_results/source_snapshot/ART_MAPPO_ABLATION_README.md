# ART-MAPPO component ablation (reviewer experiment)

This directory is an isolated experimental copy of the canonical local
training source at:

`/home/uav/00gao_xueshu/togsy_2025/0620septimedone/on-policy-main`

No source file or checkpoint in that canonical directory is modified.

## Audit corrections required before the ablation

The historical `r_actor_critic_advanced.py` passes a tensor with sequence
length one to multi-head attention, so that attention cannot model physical
channel interactions. It also contains no per-defender return-based trust
memory. The active remote `run_sync_tune_v63.sh` selected `--algo MAPPO`, and
the active 3-D scenario was an older 14-observation/2-action copy.

For this experiment only, the paper-aligned 17-observation/3-action scenario
and uncertainty/command-lag support were recovered from the remote
`stable_V2` archive. The ART implementation is placed in new files:

- `r_actor_critic_art.py`: physical-channel token attention, residual blocks,
  and optional GRU;
- `rMAPPOPolicy_art.py` and `r_mappo_art.py`: policy and corrected
  dual-clip/adaptive-KL optimization;
- `art_ablation_runner.py`: return-based trust memory and the
  PN/probe/uniform training mixture.

Following Algorithm 1 and the PPO ratio in the manuscript, the action actually
executed from the trust-dependent mixture is stored, and its likelihood under
the rollout policy is used as the old-policy likelihood. Thus every stored
transition participates in the update exactly as specified by the paper. At
evaluation, all guided exploration is disabled.

## Controlled variants

| Variant | Trust | GRU | Attention-residual |
|---|---:|---:|---:|
| `full` | on | on | on |
| `no_trust` | off | on | on |
| `no_gru` | on | off | on |
| `no_attention_residual` | on | on | off |

The plain-MLP replacement in `no_attention_residual` is dynamically sized to
match the removed backbone capacity. At hidden size 256, the full and
backbone-ablated actor counts are 732,774 and 733,200 parameters; critic
counts are 749,761 and 749,932.

All other environment, reward, optimizer, rollout-budget, and seed settings
are held fixed. Case 1 and Case 2 are trained and evaluated separately using
paired seeds. Formal evaluation uses the same held-out episode seeds for all
four variants and disables trust-guided exploration.

The case-specific guidance profile is taken from the audited paper trajectory
collection, rather than the obsolete `v63` launcher: Case 1 uses base gain
2.0, guidance time constant 0.55, lead 1.45, and command-lag time constant
0.55 s; Case 2 uses 2.4, 0.40, 1.70, and 0.40 s, respectively. The learned
residual scale is 0.05 in both cases. These values are written to every run
manifest.

## Reproduction entry points

Train one run:

```bash
python onpolicy/scripts/train_art_mappo_ablation_3d.py \
  --variant full --case_3d case1 --seed 701 \
  --save_dir /path/to/output --compare_steps 300000
```

Run a restartable suite:

```bash
python onpolicy/scripts/run_art_mappo_ablation_suite.py \
  --output_root /path/to/training --steps 300000 \
  --seeds 701 702 703 704 705 --cases case1 case2
```

Evaluate and analyze:

```bash
python onpolicy/scripts/run_art_mappo_ablation_eval_suite.py \
  --training_root /path/to/training \
  --output_root /path/to/evaluation \
  --seeds 701 702 703 704 705 --episodes_per_seed_case 20

python onpolicy/scripts/analyze_art_mappo_ablation.py \
  --training_root /path/to/training \
  --evaluation_root /path/to/evaluation \
  --output_dir /path/to/analysis
```

The suite writes atomic status files, per-run manifests, raw CSV curves,
periodic complete checkpoints (including optimizers, schedulers, value
normalization, trust, and random states), and separate logs. The analysis
uses paired seed-and-case effects, exact sign-flip tests, Holm correction,
and 95% Student-t intervals over independent seed-level estimates.

The 2-by-2 paper figure contains Case 1 and Case 2 learning curves plus
per-target interception and synchronization rates.  These target-level rates
remain informative when the stricter all-eight-target trial indicator is
zero.  The strict `all_target_interception` and `all_target_sync` metrics are
still exported at episode, seed, aggregate, and paired-statistics levels; they
are never replaced or omitted from the quantitative report.
