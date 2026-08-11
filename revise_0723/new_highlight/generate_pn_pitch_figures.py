#!/usr/bin/env python3
"""Reconstruct PN pitch histories from the selected 3-D trajectory data."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
DATA_ROOT = (
    SCRIPT_DIR.parent.parent
    / "new_sim_fig"
    / "temp_pn_case12"
    / "final_legacy_3d_tgo_reasonable_nxny"
)
OUTPUT_DIR = SCRIPT_DIR / "generated_pn_pitch"
DT = 0.05


def generate(case: str, output_name: str) -> None:
    case_dir = DATA_ROOT / case
    with np.load(case_dir / f"{case}_selected_episode.npz") as data:
        positions = data["rep_def"]

    velocity = np.empty_like(positions)
    velocity[1:] = np.diff(positions, axis=0) / DT
    velocity[0] = velocity[1]
    pitch = np.arctan2(
        velocity[:, :, 2],
        np.hypot(velocity[:, :, 0], velocity[:, :, 1]),
    )

    with (case_dir / f"{case}_hit_events.csv").open(newline="") as stream:
        hit_steps = {
            int(row["defender_id"]): int(row["step"])
            for row in csv.DictReader(stream)
        }

    plt.rcParams.update(
        {
            "font.family": "serif",
            "mathtext.fontset": "cm",
            "font.size": 22,
            "axes.labelsize": 34,
            "xtick.labelsize": 28,
            "ytick.labelsize": 28,
            "axes.linewidth": 2.2,
        }
    )
    fig, ax = plt.subplots(figsize=(10.17, 5.88), dpi=100)
    colors = plt.get_cmap("tab20").colors

    for defender in range(positions.shape[1]):
        count = min(hit_steps[defender], positions.shape[0])
        time = np.arange(count) * DT
        ax.plot(time, pitch[:count, defender], color=colors[defender], linewidth=2.2)

    max_time = max(hit_steps.values()) * DT
    ax.set_xlim(0.0, np.ceil(max_time / 5.0) * 5.0)
    ax.set_ylim(-0.75, 0.75)
    ax.set_yticks([-0.5, 0.0, 0.5])
    ax.set_xlabel(r"$t$ (s)", labelpad=12)
    ax.set_ylabel(r"$\theta_D$ (rad)", labelpad=12)
    ax.grid(True, linestyle="--", linewidth=0.7, alpha=0.35)
    ax.tick_params(direction="in", length=10, width=2.0, pad=10)
    fig.tight_layout()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_DIR / output_name, dpi=100)
    plt.close(fig)


if __name__ == "__main__":
    generate("case1", "pn_nopn_pitch.png")
    generate("case2", "pn_sin_pitch.png")
