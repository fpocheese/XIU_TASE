#!/usr/bin/env python3
"""Project-specific wrapper for the frozen-policy reviewer evaluation.

The reusable evaluator was originally written against an older direct-action
profile.  This wrapper pins it to the verified residual-guidance chain used by
the isolated xiu_onpolicy_3d_fix ablation clone.  It performs evaluation only;
no optimizer or backward operation is created.
"""

from __future__ import annotations

import eval_art_mappo_ablation_3d as evaluator


evaluator.PAPER_GUIDANCE_PROFILES.clear()
evaluator.PAPER_GUIDANCE_PROFILES.update(
    {
        "case1": {
            "defender_guidance_base_gain": 2.0,
            "defender_guidance_tau": 0.25,
            "defender_guidance_lead": 1.60,
            "defender_command_lag_tau": 0.25,
            "defender_residual_scale": 0.20,
            "defender_sync_speed_gain": 0.14,
            "defender_sync_tgo_ref": "min",
            "defender_speed_target": 40.0,
            "defender_speed_gain": 0.016,
            "attacker_load_limit": 1.75,
            "attacker_yaw_scale": 1.55,
            "attacker_pitch_scale": 1.55,
        },
        "case2": {
            "defender_guidance_base_gain": 2.6,
            "defender_guidance_tau": 0.35,
            "defender_guidance_lead": 1.70,
            "defender_command_lag_tau": 0.40,
            "defender_residual_scale": 0.20,
            "defender_sync_speed_gain": 1.40,
            "defender_sync_tgo_ref": "min",
            "defender_speed_target": 40.0,
            "defender_speed_gain": 0.008,
            "attacker_load_limit": 1.75,
            "attacker_yaw_scale": 1.55,
            "attacker_pitch_scale": 1.55,
        },
    }
)

from run_reviewer_supplementary_experiments_base import evaluate, parse_args


if __name__ == "__main__":
    arguments = parse_args()
    # The reusable evaluator exposes command lag as a robustness variable.
    # For the nominal ablation matrix it must remain at the verified
    # case-specific value used during training.
    if arguments.case == "case1":
        arguments.command_lag_tau = 0.25
    elif arguments.case == "case2":
        arguments.command_lag_tau = 0.40
    evaluate(arguments)
