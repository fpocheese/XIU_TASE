# Case 2：$E_t\leq45$ s 成功回合筛选箱图

## 筛选口径

- Case 1：保留全部资产安全的拦截成功回合，共 92 条。
- Case 2：首先要求 8 架攻击机均在进入 3 m 保护区前被至少一架有效拦截器命中，再要求记录的 $E_t\leq45$ s，共 7 条。
- 未修改任何回合的指标数值，只筛选已有真实回合。
- 图中不显示筛选条件，条件保存在脚本、CSV 和 manifest 中。

Case 2 入选回合的 seed 为：90036、90037、90045、90060、90062、90064
和 90077。其中只有 seed 90037 同时满足原评估程序的严格
`cooperative_success` 判据；其余回合满足全部攻击机被拦截和资产安全，且
$E_{co\text{-}time}$ 均很小，但仍有部分有效拦截器未完成命中。因此，本图
是“45 s 内资产安全拦截成功”的条件子集，不应表述为完整 100 次蒙特卡洛
分布或严格协同成功分布。

## 统计结果

| Case | 样本数 | $E_{co\text{-}time}$ (s) | $E_n$ (g) | $E_{miss}$ (m) | $E_t$ (s) |
|---|---:|---:|---:|---:|---:|
| Case 1 | 92 | 0.01128 ± 0.00268 | 0.26160 ± 0.04720 | 1.65825 ± 0.13413 | 33.78587 ± 0.94808 |
| Case 2 | 7 | 0.02173 ± 0.00568 | 0.25828 ± 0.06251 | 1.99600 ± 0.20841 | 37.95000 ± 1.85293 |

## 文件

- `two_defender_failure_mc_boxplots_case2_et_le_45s.png/.pdf/.svg`：最终箱图。
- `case2_asset_safe_success_et_le_45s_subset.csv`：Case 2 的 7 条绘图数据。
- `case1_asset_safe_success_subset.csv`：Case 1 的 92 条对照数据。
- `case2_et45_subset_statistics.csv`：描述统计。
- `case2_et45_subset_manifest.json`：完整筛选规则和样本数。
- `../../plot_case2_et45_success_boxplots.py`：可复现脚本。
