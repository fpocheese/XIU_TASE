#!/usr/bin/env python3
"""Aggregate reviewer experiments and generate evidence-bound response drafts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
try:
    from scipy.stats import binomtest, wilcoxon
except ImportError:  # tables remain reproducible without optional tests
    binomtest = None
    wilcoxon = None


VARIANT_LABELS = {
    "full": "Full ART-MAPPO",
    "no_trust": "No trust-aware exploration",
    "no_gru": "No GRU",
    "no_attention_residual": "No attention-residual backbone",
}


def write_csv(frame: pd.DataFrame, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def load_csvs(root: Path, filename: str):
    frames = []
    for path in sorted(root.glob(f"**/{filename}")):
        if "_chunks" in path.parts or "checkpoints" in path.parts:
            continue
        try:
            frame = pd.read_csv(path)
        except pd.errors.EmptyDataError:
            continue
        if len(frame):
            frame["source_file"] = str(path)
            frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def load_training(root: Path):
    frames = []
    for path in sorted(root.glob("**/training_metrics.csv")):
        parts = path.parts
        variant = next((v for v in VARIANT_LABELS if v in parts), None)
        case = next((c for c in ("case1", "case2") if c in parts), None)
        seed_part = next((x for x in parts if x.startswith("seed")), None)
        if variant is None or case is None or seed_part is None:
            continue
        frame = pd.read_csv(path)
        frame["variant"] = variant
        frame["case"] = case
        frame["seed"] = int(seed_part.replace("seed", ""))
        frame["source_file"] = str(path)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def training_summary(data: pd.DataFrame):
    rows = []
    if data.empty:
        return pd.DataFrame()
    for (case, variant, seed), group in data.groupby(
        ["case", "variant", "seed"], sort=True
    ):
        group = group.sort_values("environment_steps")
        tail_n = max(1, int(np.ceil(0.2 * len(group))))
        tail = group.iloc[-tail_n:]
        x = group["environment_steps"].to_numpy(dtype=float)
        y = group["mean_episode_return"].to_numpy(dtype=float)
        auc = float(np.trapz(y, x) / max(x[-1] - x[0], 1.0))
        tail_return = float(tail["mean_episode_return"].mean())
        target = 0.9 * tail_return
        if tail_return >= y[0]:
            indices = np.where(y >= target)[0]
        else:
            indices = np.where(y <= target)[0]
        steps_to_90 = float(x[indices[0]]) if indices.size else float("nan")
        rows.append(
            {
                "case": case,
                "variant": variant,
                "variant_label": VARIANT_LABELS[variant],
                "seed": seed,
                "updates": len(group),
                "final_environment_steps": float(x[-1]),
                "return_auc": auc,
                "tail_return_mean": tail_return,
                "tail_return_std": float(
                    tail["mean_episode_return"].std(ddof=1)
                ),
                "steps_to_90pct_tail_return": steps_to_90,
                "tail_critic_loss": float(tail["value_loss"].mean()),
                "tail_policy_entropy": float(tail["entropy"].mean()),
                "tail_target_coverage": float(
                    tail["target_coverage_rate"].mean()
                    if "target_coverage_rate" in tail
                    else np.nan
                ),
                "tail_cooperative_group_rate": float(
                    tail["coordinated_group_rate"].mean()
                    if "coordinated_group_rate" in tail
                    else np.nan
                ),
            }
        )
    return pd.DataFrame(rows)


def aggregate_training(seed_level: pd.DataFrame):
    if seed_level.empty:
        return pd.DataFrame()
    metrics = [
        "return_auc",
        "tail_return_mean",
        "steps_to_90pct_tail_return",
        "tail_critic_loss",
        "tail_policy_entropy",
        "tail_target_coverage",
        "tail_cooperative_group_rate",
    ]
    rows = []
    for (case, variant), group in seed_level.groupby(
        ["case", "variant"], sort=True
    ):
        row = {
            "case": case,
            "variant": variant,
            "variant_label": VARIANT_LABELS[variant],
            "seeds": int(group["seed"].nunique()),
        }
        for metric in metrics:
            values = group[metric].to_numpy(dtype=float)
            values = values[np.isfinite(values)]
            row[f"{metric}_mean"] = (
                float(np.mean(values)) if values.size else np.nan
            )
            row[f"{metric}_std"] = (
                float(np.std(values, ddof=1)) if values.size > 1 else np.nan
            )
        rows.append(row)
    return pd.DataFrame(rows)


def mc_summary(episodes: pd.DataFrame, group_cols):
    if episodes.empty:
        return pd.DataFrame()
    rows = []
    for keys, group in episodes.groupby(group_cols, sort=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_cols, keys))
        row["episodes"] = len(group)
        for metric in [
            "target_coverage_success",
            "all_defenders_hit",
            "cooperative_success",
            "mission_success",
        ]:
            if metric in group:
                count = int(group[metric].sum())
                rate = float(group[metric].mean())
                z = 1.959963984540054
                n = len(group)
                denominator = 1.0 + z * z / n
                center = (rate + z * z / (2.0 * n)) / denominator
                half = (
                    z
                    * np.sqrt(
                        rate * (1.0 - rate) / n
                        + z * z / (4.0 * n * n)
                    )
                    / denominator
                )
                row[f"{metric}_rate"] = rate
                row[f"{metric}_count"] = count
                row[f"{metric}_ci95_low"] = max(0.0, center - half)
                row[f"{metric}_ci95_high"] = min(1.0, center + half)
        for metric in [
            "E_co_time_s",
            "E_n_g",
            "E_miss_m",
            "E_t_s",
            "mean_closest_approach_m",
            "worst_closest_approach_m",
            "mean_agent_return",
            "idbo_runtime_ms",
            "idbo_repaired_cost",
        ]:
            if metric not in group:
                continue
            values = group[metric].to_numpy(dtype=float)
            values = values[np.isfinite(values)]
            row[f"{metric}_mean"] = (
                float(np.mean(values)) if values.size else np.nan
            )
            row[f"{metric}_std"] = (
                float(np.std(values, ddof=1)) if values.size > 1 else np.nan
            )
            row[f"{metric}_n"] = int(values.size)
            if values.size > 1:
                half = 1.959963984540054 * float(
                    np.std(values, ddof=1) / np.sqrt(values.size)
                )
                row[f"{metric}_ci95_low"] = float(np.mean(values) - half)
                row[f"{metric}_ci95_high"] = float(np.mean(values) + half)
            else:
                row[f"{metric}_ci95_low"] = np.nan
                row[f"{metric}_ci95_high"] = np.nan
        if "failure_class" in group:
            counts = group["failure_class"].value_counts()
            for name, count in counts.items():
                row[f"failure_{name}_count"] = int(count)
        rows.append(row)
    return pd.DataFrame(rows)


def _bootstrap_ci(values, rng, draws=10000):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.nan, np.nan
    if values.size == 1:
        return float(values[0]), float(values[0])
    indices = rng.integers(0, values.size, size=(draws, values.size))
    means = values[indices].mean(axis=1)
    low, high = np.percentile(means, [2.5, 97.5])
    return float(low), float(high)


def paired_ablation_statistics(episodes: pd.DataFrame):
    """Full-versus-removal paired effects on identical MC seeds."""
    if episodes.empty or "variant" not in episodes:
        return pd.DataFrame()
    key_cols = [
        column
        for column in ("case", "training_seed", "seed")
        if column in episodes.columns
    ]
    if "case" not in key_cols or "seed" not in key_cols:
        return pd.DataFrame()
    binary_metrics = (
        "target_coverage_success",
        "all_defenders_hit",
        "cooperative_success",
        "mission_success",
    )
    lower_better = (
        "E_co_time_s",
        "E_n_g",
        "E_miss_m",
        "E_t_s",
    )
    rows = []
    rng = np.random.default_rng(383926)
    for case in sorted(episodes["case"].unique()):
        full = episodes[
            (episodes["case"] == case) & (episodes["variant"] == "full")
        ]
        for variant in VARIANT_LABELS:
            if variant == "full":
                continue
            removed = episodes[
                (episodes["case"] == case)
                & (episodes["variant"] == variant)
            ]
            paired = full.merge(
                removed,
                on=key_cols,
                suffixes=("_full", "_ablated"),
                validate="one_to_one",
            )
            for metric in binary_metrics:
                if (
                    f"{metric}_full" not in paired
                    or f"{metric}_ablated" not in paired
                ):
                    continue
                full_values = paired[f"{metric}_full"].to_numpy(dtype=int)
                ablated_values = paired[
                    f"{metric}_ablated"
                ].to_numpy(dtype=int)
                difference = full_values - ablated_values
                low, high = _bootstrap_ci(difference, rng)
                full_only = int(
                    np.sum((full_values == 1) & (ablated_values == 0))
                )
                ablated_only = int(
                    np.sum((full_values == 0) & (ablated_values == 1))
                )
                discordant = full_only + ablated_only
                p_value = (
                    float(
                        binomtest(
                            min(full_only, ablated_only),
                            discordant,
                            0.5,
                            alternative="two-sided",
                        ).pvalue
                    )
                    if binomtest is not None and discordant
                    else np.nan
                )
                rows.append(
                    {
                        "case": case,
                        "comparison": f"full_vs_{variant}",
                        "metric": metric,
                        "effect_definition": "full_minus_ablated",
                        "paired_n": len(difference),
                        "effect_mean": float(np.mean(difference)),
                        "effect_ci95_low": low,
                        "effect_ci95_high": high,
                        "full_only_successes": full_only,
                        "ablated_only_successes": ablated_only,
                        "test": "exact_McNemar",
                        "p_value": p_value,
                    }
                )
            for metric in lower_better:
                full_column = f"{metric}_full"
                ablated_column = f"{metric}_ablated"
                if full_column not in paired or ablated_column not in paired:
                    continue
                valid = np.isfinite(
                    paired[full_column].to_numpy(dtype=float)
                ) & np.isfinite(
                    paired[ablated_column].to_numpy(dtype=float)
                )
                # Positive values mean the Full model has the lower metric.
                difference = (
                    paired.loc[valid, ablated_column].to_numpy(dtype=float)
                    - paired.loc[valid, full_column].to_numpy(dtype=float)
                )
                low, high = _bootstrap_ci(difference, rng)
                p_value = np.nan
                if (
                    wilcoxon is not None
                    and difference.size > 0
                    and np.any(np.abs(difference) > 0.0)
                ):
                    p_value = float(
                        wilcoxon(
                            difference,
                            alternative="two-sided",
                            zero_method="wilcox",
                        ).pvalue
                    )
                rows.append(
                    {
                        "case": case,
                        "comparison": f"full_vs_{variant}",
                        "metric": metric,
                        "effect_definition": "ablated_minus_full",
                        "paired_n": len(difference),
                        "effect_mean": (
                            float(np.mean(difference))
                            if difference.size
                            else np.nan
                        ),
                        "effect_ci95_low": low,
                        "effect_ci95_high": high,
                        "full_only_successes": np.nan,
                        "ablated_only_successes": np.nan,
                        "test": "paired_Wilcoxon",
                        "p_value": p_value,
                    }
                )
    return pd.DataFrame(rows)


def _fmt(value, digits=3):
    if value is None or not np.isfinite(float(value)):
        return "N/A"
    return f"{float(value):.{digits}f}"


def make_response_markdown(
    ablation: pd.DataFrame,
    failure: pd.DataFrame,
    generalization: pd.DataFrame,
    end_to_end: pd.DataFrame,
):
    lines = [
        "# Reviewer-response draft (evidence-bound)",
        "",
        "All numbers below are generated from the archived CSV files. No result is inserted manually.",
        "",
        "## Ablation: independent architectural contributions",
        "",
        "> Comment: Add an ablation investigation demonstrating the independent contribution of the trust-aware mechanism, GRU temporal encoder, and attention-residual backbone.",
        "",
        "Response: We added a controlled component ablation. The four variants use identical environments, paper reward, seeds, training budget, optimizer, held-out checkpoint-selection rule, and 100-episode evaluator; only the named component is removed. Trust is interpreted primarily through training dynamics, while GRU and attention-residual effects are evaluated through both training and frozen-policy mission metrics. Checkpoint selection uses validation seeds that are disjoint from the final Monte Carlo seeds.",
        "",
    ]
    if not ablation.empty:
        for _, row in ablation.sort_values(["case", "variant"]).iterrows():
            lines.append(
                f"- {row['case']}, {VARIANT_LABELS.get(row['variant'], row['variant'])}: "
                f"target coverage={_fmt(row.get('target_coverage_success_rate'))}, "
                f"cooperative success={_fmt(row.get('cooperative_success_rate'))}, "
                f"mission success={_fmt(row.get('mission_success_rate'))}, "
                f"$E_{{co-time}}$={_fmt(row.get('E_co_time_s_mean'))} s."
            )
    lines += [
        "",
        "## 3.8 Failure-case analysis",
        "",
        "> Comment: Include failure-case analysis for unsuccessful interception and delayed cooperative engagement.",
        "",
        "Response: We added mutually exclusive outcome classes—uncovered attacker, incomplete cooperative group, delayed cooperative engagement, and mission success—and retained closest-approach and terminal metrics for every condition. Failed trials remain in the reliability denominator.",
        "",
    ]
    if not failure.empty:
        for _, row in failure.sort_values(["case", "condition"]).iterrows():
            lines.append(
                f"- {row['condition']} ({row['case']}): "
                f"coverage={_fmt(row.get('target_coverage_success_rate'))}, "
                f"mission success={_fmt(row.get('mission_success_rate'))}, "
                f"worst closest approach={_fmt(row.get('worst_closest_approach_m_mean'))} m."
            )
    lines += [
        "",
        "## 3.9 Generalization to unseen maneuvers",
        "",
        "> Comment: Discuss generalization under unseen maneuvering attack patterns.",
        "",
        "Response: The trained policy is frozen and tested without any optimizer or gradient step on chirp, multi-sine, and smoothed jink waveforms that are absent from training. The nominal waveform uses the identical Monte Carlo seed set as the paired reference.",
        "",
    ]
    if not generalization.empty:
        for _, row in generalization.sort_values(
            ["case", "attack_pattern"]
        ).iterrows():
            lines.append(
                f"- {row['attack_pattern']} ({row['case']}): "
                f"coverage={_fmt(row.get('target_coverage_success_rate'))}, "
                f"mission success={_fmt(row.get('mission_success_rate'))}, "
                f"$E_{{miss}}$={_fmt(row.get('E_miss_m_mean'))} m "
                f"(conditional n={int(row.get('E_miss_m_n', 0))})."
            )
    lines += [
        "",
        "## 1.6 End-to-end allocation and guidance",
        "",
        "> Comment: End-to-end verification of task allocation and cooperative guidance is lacking.",
        "",
        "Response: Each episode now executes the complete chain: engagement snapshot, paper-faithful IDBO, deterministic capacity/coverage repair, assignment transfer to shared decentralized actors, and cooperative interception. IDBO runtime, cost, disagreement, repair moves, and guidance outcomes are archived per episode.",
        "",
    ]
    if not end_to_end.empty:
        for _, row in end_to_end.sort_values(
            ["case", "assignment_mode"]
        ).iterrows():
            lines.append(
                f"- {row['case']}, {row['assignment_mode']}: "
                f"mission success={_fmt(row.get('mission_success_rate'))}, "
                f"IDBO runtime={_fmt(row.get('idbo_runtime_ms_mean'))} ms, "
                f"$E_{{co-time}}$={_fmt(row.get('E_co_time_s_mean'))} s."
            )
    lines += [
        "",
        "The discussion must retain any observed degradation or mixed ranking; the data do not justify a universal claim that the full model is best on every secondary metric.",
    ]
    return "\n".join(lines) + "\n"


def make_response_markdown_zh(
    ablation: pd.DataFrame,
    failure: pd.DataFrame,
    generalization: pd.DataFrame,
    end_to_end: pd.DataFrame,
):
    lines = [
        "# 审稿意见回复草稿（数据约束版）",
        "",
        "下列数字均由归档 CSV 自动生成，未人工填数或改数。",
        "",
        "## 架构消融：三个模块的独立贡献",
        "",
        "回复：我们新增了严格控制变量的组件消融。完整模型、去除 trust-aware、去除 GRU、去除 attention-residual 四种变体采用相同工况、论文奖励、随机种子、训练预算、优化器、隔离验证集检查点选择规则和 100 次 Monte Carlo 测试；每次仅移除被考察模块。根据原文机理，trust-aware 的主要证据来自训练效率、稳定性和熵演化，GRU 与 attention-residual 同时从训练指标和冻结策略任务指标评价。验证种子与最终测试种子完全隔离。",
        "",
    ]
    if not ablation.empty:
        for _, row in ablation.sort_values(["case", "variant"]).iterrows():
            lines.append(
                f"- {row['case']}，{VARIANT_LABELS.get(row['variant'], row['variant'])}："
                f"目标全覆盖率={_fmt(row.get('target_coverage_success_rate'))}，"
                f"协同成功率={_fmt(row.get('cooperative_success_rate'))}，"
                f"任务成功率={_fmt(row.get('mission_success_rate'))}，"
                f"$E_{{co-time}}$={_fmt(row.get('E_co_time_s_mean'))} s。"
            )
    lines += [
        "",
        "## 3.8 失败案例分析",
        "",
        "回复：我们将结果划分为互斥的四类：进攻目标未被覆盖、协同组不完整、协同接战延迟以及任务成功，并为每种压力条件保留最近距离和终端指标。失败样本始终保留在可靠性分母中，避免只报告成功轨迹造成选择偏差。",
        "",
    ]
    if not failure.empty:
        for _, row in failure.sort_values(["case", "condition"]).iterrows():
            lines.append(
                f"- {row['condition']}（{row['case']}）："
                f"目标全覆盖率={_fmt(row.get('target_coverage_success_rate'))}，"
                f"任务成功率={_fmt(row.get('mission_success_rate'))}，"
                f"最差最近距离均值={_fmt(row.get('worst_closest_approach_m_mean'))} m。"
            )
    lines += [
        "",
        "## 3.9 未见机动模式泛化",
        "",
        "回复：训练完成后冻结完整策略，在不执行梯度计算或优化器更新的条件下，分别测试训练中未出现的 chirp、multi-sine 和平滑 jink 机动；nominal 条件使用完全相同的 Monte Carlo 种子作为配对参照。",
        "",
    ]
    if not generalization.empty:
        for _, row in generalization.sort_values(
            ["case", "attack_pattern"]
        ).iterrows():
            lines.append(
                f"- {row['attack_pattern']}（{row['case']}）："
                f"目标全覆盖率={_fmt(row.get('target_coverage_success_rate'))}，"
                f"任务成功率={_fmt(row.get('mission_success_rate'))}，"
                f"$E_{{miss}}$={_fmt(row.get('E_miss_m_mean'))} m"
                f"（成功条件样本数 n={int(row.get('E_miss_m_n', 0))}）。"
            )
    lines += [
        "",
        "## 1.6 任务分配—协同制导端到端验证",
        "",
        "回复：每个 episode 均执行完整闭环：由实际交战快照运行论文 IDBO，实施确定性的容量/覆盖可行性修复，将目标编号传入冻结的共享去中心化 actor，再完成协同拦截。程序逐 episode 保存 IDBO 耗时、代价、分歧度、修复次数以及全部制导结果。",
        "",
    ]
    if not end_to_end.empty:
        for _, row in end_to_end.sort_values(
            ["case", "assignment_mode"]
        ).iterrows():
            lines.append(
                f"- {row['case']}，{row['assignment_mode']}："
                f"任务成功率={_fmt(row.get('mission_success_rate'))}，"
                f"IDBO 平均耗时={_fmt(row.get('idbo_runtime_ms_mean'))} ms，"
                f"$E_{{co-time}}$={_fmt(row.get('E_co_time_s_mean'))} s。"
            )
    lines += [
        "",
        "最终文字将如实保留退化、混合排序或统计不显著的结果；数据不支持“完整模型在每个次要指标上均最优”的泛化表述。",
    ]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--training_root", type=Path, required=True)
    parser.add_argument("--evaluation_root", type=Path, required=True)
    parser.add_argument("--supplementary_root", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    training = load_training(args.training_root)
    train_seed = training_summary(training)
    train_agg = aggregate_training(train_seed)
    write_csv(train_seed, args.outdir / "training_seed_summary.csv")
    write_csv(train_agg, args.outdir / "training_aggregate_summary.csv")

    eval_episodes = load_csvs(args.evaluation_root, "episodes.csv")
    ablation_cols = [
        c
        for c in ["case", "variant"]
        if c in eval_episodes.columns
    ]
    ablation = (
        mc_summary(eval_episodes, ablation_cols)
        if ablation_cols
        else pd.DataFrame()
    )
    write_csv(ablation, args.outdir / "ablation_mc_summary.csv")
    paired_ablation = paired_ablation_statistics(eval_episodes)
    write_csv(
        paired_ablation,
        args.outdir / "ablation_paired_statistics.csv",
    )

    supplementary = load_csvs(args.supplementary_root, "episodes.csv")
    failure_data = supplementary[
        supplementary.get("condition", pd.Series(dtype=str))
        .astype(str)
        .str.startswith("failure_")
    ]
    generalization_data = supplementary[
        supplementary.get("condition", pd.Series(dtype=str))
        .astype(str)
        .str.startswith("generalization_")
    ]
    end_to_end_data = supplementary[
        supplementary.get("condition", pd.Series(dtype=str))
        .astype(str)
        .str.startswith("end_to_end_")
    ]
    failure = (
        mc_summary(failure_data, ["case", "condition"])
        if len(failure_data)
        else pd.DataFrame()
    )
    generalization = (
        mc_summary(
            generalization_data,
            ["case", "attack_pattern", "condition"],
        )
        if len(generalization_data)
        else pd.DataFrame()
    )
    end_to_end = (
        mc_summary(
            end_to_end_data,
            ["case", "assignment_mode", "condition"],
        )
        if len(end_to_end_data)
        else pd.DataFrame()
    )
    write_csv(failure, args.outdir / "failure_case_summary.csv")
    write_csv(
        generalization, args.outdir / "unseen_maneuver_summary.csv"
    )
    write_csv(end_to_end, args.outdir / "end_to_end_summary.csv")

    response = make_response_markdown(
        ablation, failure, generalization, end_to_end
    )
    (args.outdir / "reviewer_response_en.md").write_text(
        response, encoding="utf-8"
    )
    (args.outdir / "reviewer_response_zh.md").write_text(
        make_response_markdown_zh(
            ablation, failure, generalization, end_to_end
        ),
        encoding="utf-8",
    )
    manifest = {
        "training_files": int(
            training["source_file"].nunique() if len(training) else 0
        ),
        "ablation_evaluation_episodes": int(len(eval_episodes)),
        "supplementary_episodes": int(len(supplementary)),
        "manual_data_edits": False,
    }
    (args.outdir / "analysis_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
