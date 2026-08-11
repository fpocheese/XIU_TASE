#!/usr/bin/env python3
"""Generate compact CSV and LaTeX tables from audited ablation summaries."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd


ORDER = ("full", "no_trust", "no_gru", "no_attention_residual")
LABEL = {"full": "Full ART-MAPPO", "no_trust": "No trust",
         "no_gru": "No GRU", "no_attention_residual": "No attention--residual"}
CASE = {"case1": "Case 1", "case2": "Case 2"}


def tex_escape(text):
    return str(text).replace("_", r"\_").replace("%", r"\%")


def finite(value, fmt=".3f", missing="--"):
    return format(float(value), fmt) if np.isfinite(float(value)) else missing


def rate_cell(row, metric):
    return f"{int(row[f'{metric}_count'])}/100 ({100*row[f'{metric}_rate']:.1f}\\%)"


def terminal_cell(row, metric):
    n = int(row[f"{metric}_eligible_n"])
    if n == 0:
        return "-- ($n$=0)"
    median = row[f"{metric}_median"]
    q25, q75 = row[f"{metric}_q25"], row[f"{metric}_q75"]
    return f"{median:.3f} [{q25:.3f}, {q75:.3f}] ($n$={n})"


def write_tex_table(path, column_spec, header, rows, caption, label, table_star=False, footnote=None):
    env = "table*" if table_star else "table"
    lines = [f"\\begin{{{env}}}[!t]", "\\centering", "\\small",
             f"\\caption{{{caption}}}", f"\\label{{{label}}}",
             "\\renewcommand{\\arraystretch}{1.12}",
             f"\\begin{{tabular}}{{{column_spec}}}", "\\toprule",
             header + " \\\\", "\\midrule"]
    lines.extend(row + " \\\\" for row in rows)
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    if footnote:
        lines.append(f"\\vspace{{1mm}}\\parbox{{0.98\\linewidth}}{{\\footnotesize {footnote}}}")
    lines.append(f"\\end{{{env}}}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def success_tables(testing, outdir):
    records, tex_rows = [], []
    for case in ("case1", "case2"):
        for variant in ORDER:
            row = testing[(testing.case == case) & (testing.variant == variant)].iloc[0]
            record = {"case": case, "variant": variant}
            for metric in ("target_coverage_success", "all_defenders_hit", "cooperative_success", "mission_success"):
                record[f"{metric}_count"] = int(row[f"{metric}_count"])
                record[f"{metric}_rate"] = row[f"{metric}_rate"]
                record[f"{metric}_ci95_low"] = row[f"{metric}_ci95_low"]
                record[f"{metric}_ci95_high"] = row[f"{metric}_ci95_high"]
            records.append(record)
            tex_rows.append(" & ".join((CASE[case], LABEL[variant], rate_cell(row,"target_coverage_success"),
                                        rate_cell(row,"all_defenders_hit"), rate_cell(row,"cooperative_success"),
                                        rate_cell(row,"mission_success"))))
    pd.DataFrame(records).to_csv(outdir/"paper_success_rates_n100.csv",index=False)
    write_tex_table(outdir/"paper_success_rates_n100.tex", "llcccc",
                    "Case & Variant & Target coverage & All defenders & Cooperative & Strict mission",
                    tex_rows, "Frozen-policy results over 100 paired Monte Carlo trials per variant and case.",
                    "tab:ablation_success_n100", table_star=True,
                    footnote="Entries are count/100 (percentage). Wilson 95\\% confidence intervals are provided in the accompanying CSV.")


def training_table(training, outdir):
    records, tex_rows = [], []
    for case in ("case1","case2"):
        for variant in ORDER:
            row=training[(training.case==case)&(training.variant==variant)].iloc[0]
            records.append(row.to_dict())
            conv = finite(row.convergence_environment_steps/1000.0, ".1f") if np.isfinite(row.convergence_environment_steps) else "--"
            tex_rows.append(" & ".join((CASE[case],LABEL[variant],f"{int(row.actor_parameter_count)/1000:.1f}",conv,
                                        finite(row.return_auc_time_normalized/1000.0,".2f"),
                                        finite(row.final_window_return/1000.0,".2f"),
                                        finite(row.final_window_return_std/1000.0,".2f"),
                                        finite(row.final_window_critic_loss,".4f"),
                                        finite(row.policy_entropy_change_percent,".1f"))))
    pd.DataFrame(records).to_csv(outdir/"paper_training_effects.csv",index=False)
    core_columns = [
        "case", "variant", "training_seed", "updates",
        "final_environment_steps", "final_window_updates",
        "return_auc_time_normalized", "return_auc_relative_to_full",
        "final_window_return", "final_return_relative_to_full",
        "final_window_return_std", "final_return_std_relative_to_full",
        "final_window_smoothed_return_std",
        "smoothed_final_return_std_relative_to_full",
        "final_window_return_cv_percent",
    ]
    pd.DataFrame(records)[core_columns].to_csv(
        outdir / "paper_training_core_metrics.csv", index=False
    )
    write_tex_table(outdir/"paper_training_effects.tex", "llrrrrrrr",
                    "Case & Variant & Actor (k) & Conv. (k steps) & Training AUC $\\uparrow$ & Final return $\\uparrow$ & Std. final return $\\downarrow$ & Critic loss $\\downarrow$ & $\\Delta H$ (\\%)",
                    tex_rows,"Effect of each module on training dynamics.","tab:ablation_training_effects",table_star=True,
                    footnote="Training AUC is the horizon-normalized area under the raw episodic-return curve. AUC, final return, and its within-run standard deviation are reported in $10^3$ return units. Final statistics use the last 10\\% of updates (58 updates here). The standard deviation measures update-to-update fluctuation within one run; it is not a cross-seed uncertainty estimate. Convergence is the first sustained attainment of 90\\% of each run's own asymptotic return improvement. No visual smoothing enters any table value.")


def terminal_tables(testing,outdir):
    metrics=("E_co_time_s","E_n_g","E_miss_m","E_t_s")
    headers=(r"$E_{co\text{-}time}$ (s)",r"$E_n$ (g)",r"$E_{miss}$ (m)",r"$E_t$ (s)")
    for case in ("case1","case2"):
        records=[]; rows=[]
        for variant in ORDER:
            row=testing[(testing.case==case)&(testing.variant==variant)].iloc[0]
            record={"case":case,"variant":variant}
            cells=[]
            for metric in metrics:
                for suffix in ("eligible_n","median","q25","q75","mean","std"):
                    record[f"{metric}_{suffix}"]=row[f"{metric}_{suffix}"]
                cells.append(terminal_cell(row,metric))
            records.append(record); rows.append(" & ".join([LABEL[variant]]+cells))
        pd.DataFrame(records).to_csv(outdir/f"paper_terminal_metrics_{case}.csv",index=False)
        write_tex_table(outdir/f"paper_terminal_metrics_{case}.tex","lcccc",
                        "Variant & "+" & ".join(headers),rows,
                        f"Terminal metrics for successful finite trials in {CASE[case]}.",
                        f"tab:ablation_terminal_{case}",table_star=True,
                        footnote="Entries are median [Q1, Q3] with the eligible sample count. No failed or missing terminal event is imputed.")


def paired_table(paired,outdir):
    binary=paired[paired.metric_type.eq("binary") & paired.metric.isin(["target_coverage_success","cooperative_success"])].copy()
    binary.to_csv(outdir/"paper_paired_binary_effects.csv",index=False)
    rows=[]
    for _,row in binary.iterrows():
        metric="Coverage" if row.metric=="target_coverage_success" else "Cooperative"
        rows.append(" & ".join((CASE[row.case],LABEL[row.ablation],metric,
                               f"{100*row.full_minus_ablation:.1f}",
                               f"[{100*row.difference_ci95_low:.1f}, {100*row.difference_ci95_high:.1f}]",
                               finite(row.mcnemar_exact_p,".3g"))))
    write_tex_table(outdir/"paper_paired_binary_effects.tex","lllrrr",
                    "Case & Ablation & Metric & Full--ablation (pp) & Paired 95\\% CI (pp) & McNemar $p$",
                    rows,"Paired episode-level effect of removing each component.",
                    "tab:ablation_paired_effects",table_star=True,
                    footnote="Positive differences favor full ART-MAPPO. The confidence interval is a 20,000-resample paired bootstrap; $p$ is the two-sided exact McNemar value.")


def main():
    p=argparse.ArgumentParser(); p.add_argument("--analysis_dir",type=Path,required=True); p.add_argument("--outdir",type=Path,required=True)
    a=p.parse_args(); a.outdir.mkdir(parents=True,exist_ok=True)
    training=pd.read_csv(a.analysis_dir/"training_effect_summary.csv")
    testing=pd.read_csv(a.analysis_dir/"monte_carlo_n100_summary.csv")
    paired=pd.read_csv(a.analysis_dir/"paired_test_comparisons.csv")
    if len(training)!=8 or len(testing)!=8 or len(paired)!=48: raise RuntimeError("unexpected audited input dimensions")
    success_tables(testing,a.outdir); training_table(training,a.outdir); terminal_tables(testing,a.outdir); paired_table(paired,a.outdir)
    print("generated",len(list(a.outdir.glob("paper_*"))),"paper table files")


if __name__=="__main__": main()
