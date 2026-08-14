#!/usr/bin/env python3
"""
statistical_tests.py — Mann-Whitney U tests on the reward-intervention
reclassification, plus a second-scenario (pedestrian_static) summary.

Existing data only: no retraining, no new evaluation runs. Reads:
  results/analysis/reclassified_summary.csv — per-seed Alignment/Planning
    counts for lane_change_simple (post reward-hacking reclassification).
  results/analysis/matrix_summary.csv       — per-condition crash%/category
    mean ± std, as stored, for lane_change_simple and pedestrian_static.
  results/analysis/matrix_raw.csv           — per-seed counts underlying
    matrix_summary.csv. results/analysis/runs/ does not exist in this repo;
    matrix_raw.csv is the only per-seed source found for pedestrian_static.

Tests: scipy.stats.mannwhitneyu, two-sided, method="exact" (small n), plus
method="asymptotic" (normal approximation with the standard tie correction)
reported alongside it since several groups contain tied values and the
exact method's null distribution assumes no ties.
No multiple-comparison correction is applied; raw p-values are reported
alongside a count of how many tests were run, per the analysis brief.

Run from repo root:
    python scripts/statistical_tests.py
"""
from __future__ import annotations

import ast
import csv
import statistics
from pathlib import Path

import pandas as pd
from scipy.stats import mannwhitneyu

REPO = Path(__file__).resolve().parent.parent
ANALYSIS = REPO / "results" / "analysis"
OUT_CSV = ANALYSIS / "statistical_tests.csv"

FIELDNAMES = [
    "part", "metric", "comparison", "group_a", "n_a", "median_a",
    "group_b", "n_b", "median_b", "U", "p_exact", "p_asymptotic_tie_corrected",
]


# ---------------------------------------------------------------------------
# Part 1 — Mann-Whitney U tests on per-seed Alignment / Planning counts
# ---------------------------------------------------------------------------

def load_per_seed_counts() -> tuple[dict, dict]:
    df = pd.read_csv(ANALYSIS / "reclassified_summary.csv").set_index("condition")
    alignment = {c: ast.literal_eval(df.loc[c, "alignment_per_seed"]) for c in df.index}
    planning = {c: ast.literal_eval(df.loc[c, "planning_per_seed"]) for c in df.index}
    return alignment, planning


def mwu_row(part: str, metric: str, comparison: str,
            label_a: str, a: list[int], label_b: str, b: list[int]) -> dict:
    u, p_exact = mannwhitneyu(a, b, alternative="two-sided", method="exact")
    # Asymptotic (normal-approximation) method: scipy applies a tie correction
    # to the variance automatically when ties are present, unlike "exact",
    # whose null distribution assumes no ties. U is identical either way;
    # only the p-value computation differs.
    u_asym, p_asym = mannwhitneyu(a, b, alternative="two-sided", method="asymptotic")
    assert u == u_asym, f"U mismatch between exact and asymptotic: {u} vs {u_asym}"
    return {
        "part": part,
        "metric": metric,
        "comparison": comparison,
        "group_a": label_a,
        "n_a": len(a),
        "median_a": statistics.median(a),
        "group_b": label_b,
        "n_b": len(b),
        "median_b": statistics.median(b),
        "U": u,
        "p_exact": p_exact,
        "p_asymptotic_tie_corrected": p_asym,
    }


def run_part1(alignment: dict, planning: dict) -> list[dict]:
    rows = []
    for metric_name, counts in (("Alignment", alignment), ("Planning", planning)):
        baseline = counts["baseline"]
        full_ablation = counts["full_ablation"]
        on_road_only = counts["on_road_only"]
        collision10_only = counts["collision10_only"]

        # (a) Pooled by collision weight
        increased = full_ablation + collision10_only
        unchanged = baseline + on_road_only
        rows.append(mwu_row(
            "1a", metric_name, "increased_collision_weight vs unchanged_collision_weight",
            "full_ablation+collision10_only", increased,
            "baseline+on_road_only", unchanged,
        ))

        # (b) Pairwise against baseline
        for other_name, other_counts in (
            ("full_ablation", full_ablation),
            ("on_road_only", on_road_only),
            ("collision10_only", collision10_only),
        ):
            rows.append(mwu_row(
                "1b", metric_name, f"baseline vs {other_name}",
                "baseline", baseline, other_name, other_counts,
            ))
    return rows


# ---------------------------------------------------------------------------
# Part 2 — pedestrian_static: stored condition-level summary + per-seed counts
# ---------------------------------------------------------------------------

def run_part2() -> tuple[list[dict], list[dict]]:
    summary_df = pd.read_csv(ANALYSIS / "matrix_summary.csv")
    stored_rows = []
    for _, row in summary_df.iterrows():
        stored_rows.append({
            "condition": row["condition"],
            "n_seeds": row["n_seeds"],
            "crash_pct": row["pedestrian_static__crash%"],
            "Alignment": row["pedestrian_static__Alignment"],
            "Planning": row["pedestrian_static__Planning"],
            "Perception": row["pedestrian_static__Perception"],
        })

    runs_dir = ANALYSIS / "runs"
    print(f"results/analysis/runs/ exists: {runs_dir.exists()}")

    raw_path = ANALYSIS / "matrix_raw.csv"
    per_seed_rows = []
    if raw_path.exists():
        raw_df = pd.read_csv(raw_path)
        ped = raw_df[raw_df["scenario"] == "pedestrian_static"].sort_values(["condition", "seed"])
        if not ped.empty:
            print(f"Per-seed pedestrian_static counts found in {raw_path.relative_to(REPO)} "
                  f"({len(ped)} (condition, seed) rows) -- not in results/analysis/runs/.")
            for cond, grp in ped.groupby("condition", sort=False):
                grp = grp.sort_values("seed")
                per_seed_rows.append({
                    "condition": cond,
                    "n_seeds": len(grp),
                    "crash_rate_per_seed": grp["crash_rate"].round(4).tolist(),
                    "alignment_per_seed": grp["n_alignment"].tolist(),
                    "planning_per_seed": grp["n_planning"].tolist(),
                    "perception_per_seed": grp["n_perception"].tolist(),
                })
    else:
        print(f"No per-seed pedestrian_static source found ({raw_path} does not exist either).")

    return stored_rows, per_seed_rows


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def write_part1_csv(rows: list[dict]) -> None:
    ANALYSIS.mkdir(parents=True, exist_ok=True)
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def print_part1(rows: list[dict]) -> None:
    print(f"\n=== Part 1: Mann-Whitney U (two-sided), {len(rows)} tests run, no correction applied ===")
    header = (f"{'part':>4} {'metric':>9} {'comparison':>45} "
              f"{'n_a':>4} {'med_a':>6} {'n_b':>4} {'med_b':>6} {'U':>6} "
              f"{'p_exact':>10} {'p_asym_tiecorr':>15}")
    print(header)
    print("-" * len(header))
    for r in rows:
        print(f"{r['part']:>4} {r['metric']:>9} {r['comparison']:>45} "
              f"{r['n_a']:>4} {r['median_a']:>6} {r['n_b']:>4} {r['median_b']:>6} "
              f"{r['U']:>6.1f} {r['p_exact']:>10.6f} {r['p_asymptotic_tie_corrected']:>15.6f}")


def print_part2(stored_rows: list[dict], per_seed_rows: list[dict]) -> None:
    print("\n=== Part 2: pedestrian_static, as stored in matrix_summary.csv ===")
    header = f"{'condition':>18} {'n_seeds':>7} {'crash%':>12} {'Alignment':>12} {'Planning':>12} {'Perception':>12}"
    print(header)
    print("-" * len(header))
    for r in stored_rows:
        print(f"{r['condition']:>18} {r['n_seeds']:>7} {r['crash_pct']:>12} "
              f"{r['Alignment']:>12} {r['Planning']:>12} {r['Perception']:>12}")

    print("\n=== Part 2: pedestrian_static per-seed counts (from matrix_raw.csv) ===")
    if not per_seed_rows:
        print("(none found)")
        return
    for r in per_seed_rows:
        print(f"{r['condition']} (n={r['n_seeds']}):")
        print(f"  crash_rate_per_seed : {r['crash_rate_per_seed']}")
        print(f"  alignment_per_seed  : {r['alignment_per_seed']}")
        print(f"  planning_per_seed   : {r['planning_per_seed']}")
        print(f"  perception_per_seed : {r['perception_per_seed']}")


def append_part2_csv(stored_rows: list[dict], per_seed_rows: list[dict]) -> None:
    per_seed_by_cond = {r["condition"]: r for r in per_seed_rows}
    with open(OUT_CSV, "a", newline="", encoding="utf-8") as fh:
        fh.write("\n")
        writer = csv.writer(fh)
        writer.writerow([
            "part", "condition", "n_seeds", "crash_pct_stored", "alignment_stored",
            "planning_stored", "perception_stored", "crash_rate_per_seed",
            "alignment_per_seed", "planning_per_seed", "perception_per_seed",
        ])
        for r in stored_rows:
            ps = per_seed_by_cond.get(r["condition"], {})
            writer.writerow([
                "2_pedestrian_static", r["condition"], r["n_seeds"], r["crash_pct"],
                r["Alignment"], r["Planning"], r["Perception"],
                ps.get("crash_rate_per_seed", ""), ps.get("alignment_per_seed", ""),
                ps.get("planning_per_seed", ""), ps.get("perception_per_seed", ""),
            ])


def main() -> None:
    alignment, planning = load_per_seed_counts()

    print("Per-seed Alignment counts read from reclassified_summary.csv:")
    for cond, counts in alignment.items():
        print(f"  {cond:>18}: {counts}")
    print("Per-seed Planning counts read from reclassified_summary.csv:")
    for cond, counts in planning.items():
        print(f"  {cond:>18}: {counts}")

    expected_alignment = {
        "baseline": [23, 12, 4, 12, 23],
        "full_ablation": [5, 4, 3, 0, 24],
        "on_road_only": [23, 19, 0],
        "collision10_only": [0, 22, 0],
    }
    mismatches = [c for c in expected_alignment if alignment.get(c) != expected_alignment[c]]
    if mismatches:
        print(f"\nWARNING: Alignment counts in CSV differ from the message for: {mismatches}")
    else:
        print("\nAlignment counts in CSV match the message exactly.")

    rows1 = run_part1(alignment, planning)
    write_part1_csv(rows1)
    print_part1(rows1)
    print(f"\nNote: several conditions contain tied values (e.g. baseline Alignment has "
          f"12 and 23 repeated); scipy's exact-method null distribution assumes no ties, "
          f"so p_exact for comparisons involving ties is an approximation of the exact test. "
          f"p_asymptotic_tie_corrected (scipy method=\"asymptotic\") applies the standard "
          f"tie correction to the normal-approximation variance and is reported alongside it.")

    stored_rows, per_seed_rows = run_part2()
    append_part2_csv(stored_rows, per_seed_rows)
    print_part2(stored_rows, per_seed_rows)

    print(f"\nWrote -> {OUT_CSV}")


if __name__ == "__main__":
    main()
