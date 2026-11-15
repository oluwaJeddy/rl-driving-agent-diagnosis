#!/usr/bin/env python3
"""
plot_failure_trace_v2.py — value estimate and TTC for one Alignment episode.

Selects, from the committed baseline seed 0 lane_change_simple run
(results/analysis/runs/baseline_seed0_lane_change_simple/), the lowest-numbered
episode that scripts/reclassify_matrix_v2.py labels Alignment at its defaults
AND that ended in a collision (crashed == True in the run's failure_report.csv),
and plots its critic value estimate and time-to-collision per step. Output is a
3.3 in wide, 300 dpi, white-background figure: results/analysis/figures/
fig3_alignment_trace_ep1.png (episode 1; episode 0 is Alignment but a near miss).

Steps where TTC is infinite (no closing hazard; stored as null) are left as
gaps. One step is one policy decision (policy_frequency = 1 Hz, so 1 s).

Run from repo root:
    python scripts/plot_failure_trace_v2.py
"""
from __future__ import annotations

import csv
import json
import math
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
import reclassify_matrix_v2 as v2  # noqa: E402

RUN_DIR = REPO / "results" / "analysis" / "runs" / "baseline_seed0_lane_change_simple"
OUT = REPO / "results" / "analysis" / "figures" / "fig3_alignment_trace_ep1.png"
FIG_WIDTH_IN = 3.3
FIG_HEIGHT_IN = 3.6
DPI = 300
TITLE = "Value estimate and time-to-collision, one Alignment episode"
SAFE_TTC_THRESHOLD_S = 2.0  # evaluate.SAFE_TTC_THRESHOLD_S
ACTION_NAMES = {0: "LANE_LEFT", 1: "IDLE", 2: "LANE_RIGHT", 3: "FASTER", 4: "SLOWER"}


def crashed_episodes() -> set[int]:
    """Episode ids whose run failure_report.csv row has crashed == True."""
    with open(RUN_DIR / "failure_report.csv", newline="") as f:
        return {int(r["episode"]) for r in csv.DictReader(f) if r["crashed"] == "True"}


def first_alignment_episode() -> tuple[Path, dict]:
    """Lowest-numbered episode that v2 labels Alignment AND that ended in a collision
    (near misses such as episode 0 are skipped)."""
    crashed = crashed_episodes()
    files = sorted((RUN_DIR / "evidence").glob("*.json"),
                   key=lambda p: int(re.search(r"_ep(\d+)_", p.name).group(1)))
    for path in files:
        d = json.loads(path.read_text())
        if d["episode_id"] in crashed and v2.reclassify_episode(d["evidence_trace"])[0] == "Alignment":
            return path, d
    raise RuntimeError("no crashed Alignment episode found")


def main() -> None:
    path, d = first_alignment_episode()
    trace = d["evidence_trace"]
    steps = [e["step"] for e in trace]
    values = [e["value_estimate"] for e in trace]
    ttcs = [e["ttc"] if e["ttc"] is not None and math.isfinite(e["ttc"]) else math.nan for e in trace]
    actions = [ACTION_NAMES[e["action"]] for e in trace]

    plt.rcParams.update({"font.size": 6, "axes.titlesize": 6.5, "axes.labelsize": 6,
                         "xtick.labelsize": 5, "ytick.labelsize": 5.5})
    fig, (ax_v, ax_t) = plt.subplots(2, 1, figsize=(FIG_WIDTH_IN, FIG_HEIGHT_IN), sharex=True,
                                     facecolor="white", layout="constrained")
    for ax in (ax_v, ax_t):
        ax.set_facecolor("white")
        ax.grid(True, linestyle=":", linewidth=0.5, alpha=0.6)

    ax_v.plot(steps, values, color="#2c3e50", marker="o", markersize=2.5, linewidth=1.0)
    ax_v.set_ylabel("Value estimate V(s)")
    ax_v.set_title(TITLE)

    # TTC gaps = no closing hazard (stored as null); no legend, per figure spec.
    ax_t.plot(steps, ttcs, color="#2980b9", marker="o", markersize=2.5, linewidth=1.0)
    ax_t.axhline(SAFE_TTC_THRESHOLD_S, color="#e67e22", linestyle="--", linewidth=0.9)
    ax_t.set_yscale("log")
    ax_t.set_yticks([0.5, 1, 2, 5])
    ax_t.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%g"))
    ax_t.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax_t.set_ylabel("TTC (s, log)")
    ax_t.set_xlabel("Step / action")
    ax_t.set_xticks(steps)
    ax_t.set_xticklabels([f"{s}\n{a}" for s, a in zip(steps, actions)], rotation=90)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    # No bbox_inches="tight": it would change the 3.3 in width.
    fig.savefig(OUT, dpi=DPI, facecolor="white")
    print(f"evidence: {path.relative_to(REPO).as_posix()}")
    print(f"Saved -> {OUT.relative_to(REPO).as_posix()}")


if __name__ == "__main__":
    main()
