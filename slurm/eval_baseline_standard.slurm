#!/bin/bash
# =============================================================================
# Baseline standard-highway-env evaluation: one array task per seed.
#
# Measures the baseline agent's performance on the plain training env config
# (no hazard-probe overrides) — a competence check to report alongside the
# hazard-probe failure rates in results/evaluation/.
#
# Submit (5 tasks, one per baseline seed):
#   sbatch slurm/eval_baseline_standard.slurm
#   # then, once all 5 tasks have finished:
#   python scripts/eval_baseline_standard_env.py --merge
#
# Run a single seed manually (no Slurm):
#   SLURM_ARRAY_TASK_ID=0 bash slurm/eval_baseline_standard.slurm
# =============================================================================
#SBATCH --job-name=rl_baseline_std
#SBATCH --array=0-4
#SBATCH --cpus-per-task=1        # CPU-bound, single env, no vectorization here
#SBATCH --time=00:30:00          # ~minutes locally per seed/100 episodes; generous margin
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

SEED=${SLURM_ARRAY_TASK_ID:-0}   # fallback 0 for manual runs

echo "========================================================"
echo "Baseline standard-env eval: seed=${SEED}"
echo "Node: $(hostname)"
echo "========================================================"

python -u scripts/eval_baseline_standard_env.py --seeds "$SEED" --episodes 100

echo ""
echo "Seed ${SEED} complete -> results/analysis/baseline_standard_env.seed${SEED}.csv"
