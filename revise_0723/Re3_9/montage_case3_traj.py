#!/usr/bin/env python3
"""Compose the Case-3 representative-trial montage (seed 74001).

Reads the six pre-rendered V10 panels (3-D trajectory, horizontal trajectory,
normal load n_y, velocity, heading, and within-group synchronization) and lays
them out on a clean 2x3 grid with (a)-(f) tags. One self-contained figure that
drops into both the manuscript and the response letter.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
PANELS = [
    ("case3_s74001_trajectory_3d.png", "(a) 3-D engagement trajectories"),
    ("case3_s74001_trajectory.png", "(b) Horizontal projection"),
    ("case3_s74001_ny.png", "(c) Normal load $n_y$"),
    ("case3_s74001_velocity.png", "(d) Speed"),
    ("case3_s74001_heading.png", "(e) Heading $\\gamma_D$"),
    ("case3_s74001_time_sync.png", "(f) Within-group synchronization"),
]


def main():
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "font.size": 8.5,
        "pdf.fonttype": 42,
    })
    fig, axes = plt.subplots(2, 3, figsize=(7.16, 4.55))
    for ax, (fname, title) in zip(axes.flat, PANELS):
        img = mpimg.imread(HERE / fname)
        ax.imshow(img)
        ax.set_title(title, fontsize=8.5, pad=3)
        ax.axis("off")
    fig.subplots_adjust(left=0.005, right=0.995, top=0.955, bottom=0.005,
                        wspace=0.04, hspace=0.10)
    stem = HERE / "case3_traj_montage"
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", dpi=300)
    fig.savefig(stem.with_suffix(".png"), bbox_inches="tight", dpi=300)
    plt.close(fig)
    print("wrote", stem.with_suffix(".pdf"))


if __name__ == "__main__":
    main()
