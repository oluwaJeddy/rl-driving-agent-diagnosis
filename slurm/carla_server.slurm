#!/bin/bash
#SBATCH --job-name=carla-server
#SBATCH --partition=shareq
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --output=slurm/logs/carla_%j.out
#SBATCH --error=slurm/logs/carla_%j.err
 
echo "Node: $(hostname)"
echo "Started: $(date)"
nvidia-smi
 
cd ~/carla916
./CarlaUE4.sh -RenderOffScreen -nosound -quality-level=Low -carla-rpc-port=2000
