#!/bin/bash
#SBATCH --job-name=hlth-pre
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=40
#SBATCH --output=log/slurm-%j.out
#SBATCH --time=20:00:00
#SBATCH --error=log/slurm-%j.err
#SBATCH --partition=gpu_p5
#SBATCH --account=oha@a100
#SBATCH -C a100
#SBATCH --qos=qos_gpu_a100-t3

export HF_HUB_OFFLINE=1

module purge
module load miniforge/24.9.0
conda activate synth-kg
module load cuda

# Run your processing script
python datasets/preprocessing/health/run.py
