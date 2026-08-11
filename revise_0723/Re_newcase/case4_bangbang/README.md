# Case 4 Supplementary Experiment Package

This folder collects the Case 4 bang-bang maneuver supplementary experiment selected for the revised manuscript response.

## Source

- Local source root: `/home/uav/00gao_xueshu/DT_PAPER/XIU_tase_paper/new_sim_fig/case4_bangbang`
- Primary formal result: `final_case4_case2geom_fixed_bangbang_n100_clean`
- Primary plotting source: `final_case4_plot_source_clean`

## Experiment Setting

- Case: `case4`
- Attack pattern: `bangbang`
- Assignment mode: fixed
- Source policy case: `case2`
- Geometry: paper Case 2 geometry with bang-bang attacker maneuver
- Attacker maneuver: finite-dwell bounded bang-bang lateral/vertical evasion
- Evaluation device: CPU
- Episodes: 100

## Key Statistics

| Metric | Value |
| --- | ---: |
| Mission success rate | 100/100 = 1.000 |
| Target coverage success rate | 100/100 = 1.000 |
| All-defenders-hit rate | 100/100 = 1.000 |
| Cooperative success rate | 100/100 = 1.000 |
| Mean cooperative time error, `E_co_time` | 0.050927 s |
| Mean overload metric, `E_n` | 0.163672 g |
| Mean miss distance, `E_miss` | 1.443553 m |
| Mean interception time, `E_t` | 26.993625 s |
| Mean closest approach | 0.826585 m |
| Mean worst closest approach | 2.622197 m |

## Directory Layout

- `data_n100_clean/`: formal 100-episode clean Monte Carlo result, including `summary.json`, `summary.csv`, `episodes.csv`, `targets.csv`, `assignments.csv`, `validation.json`, and chunk-level raw outputs.
- `plot_source_clean/`: selected successful trajectory, hit events, v9 export text files, and plotting source files.
- `figures/figures_v9_3d/`: directly usable Case 4 figures in PNG/PDF form.
- `statistics/`: compact copy of the main statistics and validation files.
- `code_patches/`: code snapshot used for Case 4 evaluation and plotting.

## Notes

The clean Case 4 result is the one intended for the reviewer-response supplementary experiment. A separate noisy/default robustness run exists in the older working results, but it is not the primary selected result here.
