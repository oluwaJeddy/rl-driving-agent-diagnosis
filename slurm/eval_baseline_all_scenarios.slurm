#!/bin/bash
# =============================================================================
# Baseline agents x remaining 8 probe scenarios: one array task per scenario.
#
# Extends the multi-seed matrix (analyze_matrix.py / reclassify_matrix.py,
# which cover only lane_change_simple + pedestrian_static) to the other 8
# scenarios in evaluate.SCENARIOS. Each array task evaluates all 5 baseline
# seeds (30 episodes each) on ONE scenario, loading that seed's own
# vec_normalize.pkl, using the current (unmodified) taxonomy classifier in
# src/evaluate.py. No retraining.
#
# Submit (8 tasks, one per scenario):
#   sbatch slurm/eval_baseline_all_scenarios.slurm
#   # then, once all 8 tasks have finished:
#   python scripts/eval_baseline_all_scenarios.py --merge
#
# Run a single scenario manually (no Slurm):
#   SLURM_ARRAY_TASK_ID=0 bash slurm/eval_baseline_all_scenarios.slurm
# =============================================================================
#SBATCH --job-name=rl_baseline_allscen
#SBATCH --array=0-7
#SBATCH --cpus-per-task=1        # CPU-bound, single env, no vectorization here
#SBATCH --time=02:00:00          # 5 seeds x 30 episodes; generous margin over lighter scenarios
#SBATCH --mem=4G
#SBATCH --output=slurm/logs/%x_%A_%a.out
#SBATCH --error=slurm/logs/%x_%A_%a.err

set -euo pipefail

# ---------------------------------------------------------------------------
# Environment setup — adjust to match your HPC module/conda config
# ---------------------------------------------------------------------------
# module load python/3.10
# source "$HOME/venvs/rl/bin/activate"

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT/src:${PYTHONPATH:-}"

SCENARIO_INDEX=${SLURM_ARRAY_TASK_ID:-0}   # fallback 0 for manual runs

echo "========================================================"
echo "Baseline all-scenarios eval: scenario_index=${SCENARIO_INDEX}"
echo "Node: $(hostname)"
echo "========================================================"

python -u scripts/eval_baseline_all_scenarios.py --scenario-index "$SCENARIO_INDEX" --episodes 30

echo ""
echo "Scenario index ${SCENARIO_INDEX} complete -> results/analysis/baseline_all_scenarios.<scenario>.csv"
