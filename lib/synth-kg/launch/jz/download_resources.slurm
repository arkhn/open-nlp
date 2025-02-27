#!/bin/bash
#SBATCH --job-name=dl_resources
#SBATCH --nodes=1
#SBATCH --cpus-per-task=10
#SBATCH --output=log/slurm-%j.out
#SBATCH --error=log/slurm-%j.err
#SBATCH --partition=prepost
#SBATCH --account=oha@v100

module purge
module load miniforge/24.9.0
conda activate synth-kg

python launch/jz/download_resources.py
