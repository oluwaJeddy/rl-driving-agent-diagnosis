#!/bin/bash
# =============================================================================
# Experiment matrix: train + evaluate all (condition, seed) pairs.
#
# Job array mapping (16 jobs total):
#   Task  0-4  : baseline         seeds 0-4   (5 seeds)
#   Task  5-9  : full_ablation    seeds 0-4   (5 seeds)
#   Task 10-12 : on_road_only     seeds 0-2   (3 seeds)
#   Task 13-15 : collision10_only seeds 0-2   (3 seeds)
#
# Submit:
#   sbatch slurm/experiments.slurm
#
# Run a single condition manually (no Slurm):
#   SLURM_ARRAY_TASK_ID=0 bash slurm/experiments.slurm
# =============================================================================
#SBATCH --job-name=rl_exp
#SBATCH --array=0-15
#SBATCH --cpus-per-task=4        # matches N_ENVS=4 in train_experiment.py
#SBATCH --gres=gpu:1             # V100/A100; remove if submitting to CPU-only queue
#SBATCH --time=00:45:00          # ~5 min train + ~15 min eval; 45 min is safe
#SBATCH --mem=12G
#SBATCH --output=slurm/logs/%x_%A_%a.out
#SBATCH --error=slurm/logs/%x_%A_%a.err

set -euo pipefail

# ---------------------------------------------------------------------------
# Environment setup — adjust to match your HPC module/conda config
# ---------------------------------------------------------------------------
# module load python/3.10
# source "$HOME/venvs/rl/bin/activate"

# Ensure src/ is importable (evaluate.py uses bare "from taxonomy import ...")
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT/src:${PYTHONPATH:-}"

# ---------------------------------------------------------------------------
# Map array task ID → (condition, seed)
# ---------------------------------------------------------------------------
TID=${SLURM_ARRAY_TASK_ID:-0}   # fallback 0 for manual runs

if   [ "$TID" -lt  5 ]; then CONDITION="baseline";         SEED=$TID
elif [ "$TID" -lt 10 ]; then CONDITION="full_ablation";    SEED=$((TID - 5))
elif [ "$TID" -lt 13 ]; then CONDITION="on_road_only";     SEED=$((TID - 10))
else                          CONDITION="collision10_only"; SEED=$((TID - 13))
fi

MODEL_PATH="results/checkpoints/${CONDITION}/seed_${SEED}/ppo_${CONDITION}_final"
VECNORM_PATH="results/checkpoints/${CONDITION}/seed_${SEED}/vec_normalize.pkl"
EVAL_OUT="results/evaluation/${CONDITION}/seed_${SEED}"

echo "========================================================"
echo "Task ${TID}: condition=${CONDITION}  seed=${SEED}"
echo "Node: $(hostname)  GPUs: ${CUDA_VISIBLE_DEVICES:-none}"
echo "========================================================"

# ---------------------------------------------------------------------------
# Step 1 — Train
# ---------------------------------------------------------------------------
echo ""
echo "--- TRAINING ---"
python -u src/train_experiment.py \
    --condition "$CONDITION" \
    --seed      "$SEED"

# ---------------------------------------------------------------------------
# Step 2 — Evaluate (VecNormalize always loaded; auto-detected if needed)
# ---------------------------------------------------------------------------
echo ""
echo "--- EVALUATION ---"
python -u src/evaluate.py \
    --model     "$MODEL_PATH" \
    --vecnorm   "$VECNORM_PATH" \
    --episodes  30 \
    --scenarios all \
    --seed      "$SEED" \
    --output    "$EVAL_OUT"

echo ""
echo "Task ${TID} complete."
echo "  Model   : ${MODEL_PATH}.zip"
echo "  VecNorm : ${VECNORM_PATH}"
echo "  Results : ${EVAL_OUT}"
