#!/usr/bin/env python3
import argparse
from pathlib import Path
from types import SimpleNamespace

from onpolicy.scripts import eval_3d_guidance as ev


def raw_args(seed, model, params):
    base = dict(
        seed=seed,
        eval_episodes=1,
        max_steps=2400,
        hit_radius_3d=12.0,
        sync_tol=3.0,
        sync_min_hits=0,
        defender_guidance_base_gain=1.0,
        defender_guidance_tau=0.55,
        defender_guidance_lead=1.2,
        defender_residual_scale=0.05,
        defender_load_limit=1.6,
        defender_axial_min=-0.8,
        defender_axial_max=1.0,
        defender_sync_speed_gain=0.14,
        defender_sync_tgo_ref="mean",
        target_assignment_mode="fixed",
        target_assignment_spread_weight=6.0,
        attack_maneuver_gain=1.20,
        attack_maneuver_offset_gain=1.25,
        attack_maneuver_freq=0.17,
        attack_maneuver_fade_range=450.0,
        case1_lateral_base=0.95,
        case1_lateral_tail=0.40,
        case1_vertical_amp=0.35,
        case2_lateral_amp=1.00,
        case2_vertical_amp=0.25,
        case2_vertical_freq_scale=0.50,
        stochastic_eval=False,
        eval_different_seed=False,
        require_success_plot=False,
        require_all_hit=True,
        model_dir=model,
        model_dir_case1=None,
        model_dir_case2=model,
        outdir="/tmp/unused",
        case1=False,
        case2=True,
    )
    base.update(params)
    return SimpleNamespace(**base)


def run_one(seed, model, params):
    cfg = ev.build_args("case2", raw_args(seed, model, params))
    env, policy, device, _ = ev.collect_model(cfg, Path(model))
    result = ev.evaluate_case(cfg, env, policy, device)
    s = result["case_summary"]
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--seeds", nargs="+", type=int, required=True)
    args = ap.parse_args()

    param_sets = [
        ("base", {}),
        ("freq015_amp080", dict(attack_maneuver_freq=0.15, case2_lateral_amp=0.80, case2_vertical_amp=0.22)),
        ("freq016_amp080", dict(attack_maneuver_freq=0.16, case2_lateral_amp=0.80, case2_vertical_amp=0.22)),
        ("freq017_amp080", dict(attack_maneuver_freq=0.17, case2_lateral_amp=0.80, case2_vertical_amp=0.22)),
        ("dyn_base", dict(target_assignment_mode="dynamic", target_assignment_spread_weight=0.0)),
        ("dyn_freq015_amp080", dict(target_assignment_mode="dynamic", target_assignment_spread_weight=0.0, attack_maneuver_freq=0.15, case2_lateral_amp=0.80, case2_vertical_amp=0.22)),
    ]
    for seed in args.seeds:
        for name, params in param_sets:
            try:
                s = run_one(seed, args.model, params)
                print(
                    f"seed={seed} set={name} attack={s['attack_success_rate']:.3f} "
                    f"target={s['target_success_rate']:.3f} all_hit={s['all_hit_rate']:.3f} "
                    f"sync={s['sync_success_rate']:.3f} all_sync={s['all_sync_rate']:.3f} "
                    f"max_spread={s['max_sync_spread']} mean_spread={s['mean_sync_spread']}",
                    flush=True,
                )
                if s["all_hit_rate"] >= 1.0 and s["all_sync_rate"] >= 1.0:
                    print(f"FOUND seed={seed} set={name} params={params}", flush=True)
                    return
            except Exception as exc:
                print(f"ERR seed={seed} set={name} {exc}", flush=True)


if __name__ == "__main__":
    main()
