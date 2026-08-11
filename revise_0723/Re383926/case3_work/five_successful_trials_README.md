# Five reproducible successful Case-3 trials

These are the first five strict mission successes, ordered by seed, in the
unchanged 100-episode formal Case-3 evaluation:

`74001, 74002, 74003, 74005, 74009`.

Each trial was deterministically replayed on the remote server with the same:

- frozen `stable_V2` Case-2 actor;
- paper-faithful IDBO assignment;
- Case-3 seeded geometry and hybrid maneuver;
- nonzero learned residual scale (0.20);
- 3 m interception radius; and
- 0.5 s within-target-group coordination threshold.

No training, back-propagation, model update or trajectory editing was
performed. Replay metrics match the corresponding formal CSV rows within
`1e-6`.

## Objective ranking

The ranking rule was fixed before viewing the figures:

1. smaller maximum within-group arrival spread;
2. smaller \(E_{\mathrm{co-time}}\);
3. smaller \(E_{\mathrm{miss}}\);
4. smaller \(E_n\).

Numerically equal time spreads are treated as ties to \(10^{-10}\), avoiding
binary floating-point artifacts.

| Rank | Recommended | Seed | Max spread (s) | \(E_{\mathrm{co-time}}\) (s) | \(E_{\mathrm{miss}}\) (m) | \(E_n\) (g) | \(E_t\) (s) |
|---:|:---:|---:|---:|---:|---:|---:|---:|
| 1 | Yes | 74001 | 0.050 | 0.01424 | 1.34488 | 0.66600 | 36.00625 |
| 2 | No | 74002 | 0.100 | 0.01701 | 1.21324 | 0.86483 | 36.16250 |
| 3 | No | 74005 | 0.100 | 0.01875 | 1.64728 | 0.71069 | 35.85000 |
| 4 | No | 74009 | 0.100 | 0.02222 | 1.31029 | 0.80916 | 35.76250 |
| 5 | No | 74003 | 0.300 | 0.05035 | 1.24649 | 0.86329 | 35.94375 |

Seed 74001 is therefore the objectively recommended representative trial. The
other four trials and all their figures remain available; the ranking does not
delete or hide any trial.

## Per-trial contents

Each `seed_<seed>` directory contains:

- compressed raw trajectory NPZ;
- long-format trajectory CSV;
- assignment and arrival CSV;
- target-group and episode metric CSV;
- standalone 3-D trajectory figure;
- standalone assignment/interception-timing figure; and
- two-panel combined figure.

Every figure is supplied as vector PDF, vector SVG, and 600-dpi PNG.

