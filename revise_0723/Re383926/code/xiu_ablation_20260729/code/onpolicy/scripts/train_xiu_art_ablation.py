#!/usr/bin/env python
"""Launch ART-MAPPO ablations on the verified xiu_onpolicy_3d_fix dynamics.

The implementation is deliberately a thin wrapper around the audited
paper-aligned trainer.  It changes only the case-specific deterministic
guidance profiles to those recovered from the stable_V2 success trajectories.
All architecture, trust, PPO, logging, checkpoint and manifest code remains
shared across the four controlled variants.
"""

from onpolicy.scripts import train_art_mappo_ablation_3d as implementation


implementation.PAPER_GUIDANCE_PROFILES = {
    "case1": {
        "defender_guidance_base_gain": 2.0,
        "defender_guidance_tau": 0.25,
        "defender_guidance_lead": 1.60,
        "defender_command_lag_tau": 0.25,
    },
    "case2": {
        "defender_guidance_base_gain": 2.6,
        "defender_guidance_tau": 0.35,
        "defender_guidance_lead": 1.70,
        "defender_command_lag_tau": 0.40,
    },
}


if __name__ == "__main__":
    implementation.main()
