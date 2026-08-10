#!/bin/bash
#SBATCH --job-name=rl-matrix
#SBATCH --cpus-per-task=4
#SBATCH --time=05:00:00
#SBATCH --output=slurm/logs/%A_%a.out
#SBATCH --error=slurm/logs/%A_%a.err
 
set -e
cd ~/rl-driving-agent-diagnosis
source ~/miniconda3/etc/profile.d/conda.sh
conda activate carla
 
CONDITIONS=(baseline baseline baseline baseline baseline full_ablation full_ablation full_ablation full_ablation full_ablation on_road_only on_road_only on_road_only collision10_only collision10_only collision10_only)
SEEDS=(0 1 2 3 4 0 1 2 3 4 0 1 2 0 1 2)
 
C=${CONDITIONS[$SLURM_ARRAY_TASK_ID]}
S=${SEEDS[$SLURM_ARRAY_TASK_ID]}
echo "Task $SLURM_ARRAY_TASK_ID -> condition=$C seed=$S"
python src/train_experiment.py --condition "$C" --seed "$S"
