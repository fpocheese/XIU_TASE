# Source and model integrity audit

The successful source of record is
`/home/a2rl/xiu_onpolicy_3d_fix/stable_V2`.  The reviewer experiment ran from
the independent copy `/home/a2rl/reviewer_failure_cases_20260730`.

The following SHA-256 values were checked again after all simulations and
plotting:

| Artifact | Original stable source | Experiment copy | Match |
|---|---|---|---|
| Case 1 `actor.pt` | `884802798c6c4b27d046bae274a3d354702d6e8646a73875de940202713393a3` | `884802798c6c4b27d046bae274a3d354702d6e8646a73875de940202713393a3` | yes |
| Case 2 `actor.pt` | `8a644cfe9fea7f70fe0165c25d77193887e9c2b9e391ac83ca5efc85714fd0f2` | `8a644cfe9fea7f70fe0165c25d77193887e9c2b9e391ac83ca5efc85714fd0f2` | yes |
| `eval_3d_guidance.py` | `dae54d847679f080363343baa0c17859ca6c5999afb2de08438ed2776f8b96f3` | `dae54d847679f080363343baa0c17859ca6c5999afb2de08438ed2776f8b96f3` | yes |

No training was run.  The replay manifest records zero optimizer steps.  New
screening, replay, postprocessing, and plotting scripts exist only in the
independent experiment tree.

