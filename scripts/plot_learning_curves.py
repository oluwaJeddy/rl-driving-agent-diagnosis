#!/usr/bin/env python3
"""
plot_learning_curves.py — mean evaluation reward vs training timesteps, 16 agents.

Reads each agent's SB3 EvalCallback log,
results/checkpoints/<condition>/seed_<n>/eval_logs/evaluations.npz,
and plots the mean reward over the evaluation episodes at each evaluation point.
One panel per reward condition, one line per seed. Y axes are not shared:
each condition uses a different reward function, so absolute rewards are not
comparable across panels.

Also prints, per agent, the change in mean reward between the last two
evaluation points (last minus second-to-last).

Note: results/checkpoints/ is gitignored, so this script needs the local
checkpoints to run.

Run from repo root:
    python scripts/plot_learning_curves.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
CKPT = REPO / "results" / "checkpoints"
OUT = REPO / "results" / "analysis" / "figures" / "figS2_learning_curves.png"
FIG_WIDTH_IN = 6.0
FIG_HEIGHT_IN = 4.6
DPI = 300

# Same conditions and seeds as scripts/reclassify_matrix_v2.CONDITIONS (16 agents).
CONDITIONS = {
    "baseline": [0, 1, 2, 3, 4],
    "full_ablation": [0, 1, 2, 3, 4],
    "on_road_only": [0, 1, 2],
    "collision10_only": [0, 1, 2],
}
SEED_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]


def load(cond: str, seed: int) -> tuple[np.ndarray, np.ndarray]:
    d = np.load(CKPT / cond / f"seed_{seed}" / "eval_logs" / "evaluations.npz")
    return d["timesteps"], d["results"].mean(axis=1)


def main() -> None:
    plt.rcParams.update({"font.size": 7, "axes.titlesize": 8, "axes.labelsize": 7,
                         "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "legend.fontsize": 6})
    fig, axes = plt.subplots(2, 2, figsize=(FIG_WIDTH_IN, FIG_HEIGHT_IN), sharex=True,
                             facecolor="white", layout="constrained")

    rows = []
    for ax, (cond, seeds) in zip(axes.flat, CONDITIONS.items()):
        ax.set_facecolor("white")
        for seed in seeds:
            t, m = load(cond, seed)
            ax.plot(t / 1000, m, marker="o", markersize=2.5, linewidth=1.0,
                    color=SEED_COLORS[seed], label=f"seed {seed}")
            rows.append((cond, seed, t[-2], m[-2], t[-1], m[-1], m[-1] - m[-2]))
        ax.set_title(cond)
        ax.grid(True, linestyle=":", linewidth=0.5, alpha=0.6)
    for ax in axes[1]:
        ax.set_xlabel("Training timesteps (thousands)")
    for ax in axes[:, 0]:
        ax.set_ylabel("Mean evaluation reward")
    # Seed colours are the same in every panel; baseline has all five seeds.
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=len(labels), frameon=False)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    # No bbox_inches="tight": it would change the 6 in width.
    fig.savefig(OUT, dpi=DPI, facecolor="white")

    print(f"{'condition':<18}{'seed':<6}{'t[-2]':>8}{'mean[-2]':>10}{'t[-1]':>8}{'mean[-1]':>10}{'change':>9}")
    for cond, seed, t2, m2, t1, m1, delta in rows:
        print(f"{cond:<18}{seed:<6}{t2:>8}{m2:>10.2f}{t1:>8}{m1:>10.2f}{delta:>+9.2f}")
    print(f"Saved -> {OUT.relative_to(REPO).as_posix()}")


if __name__ == "__main__":
    main()
