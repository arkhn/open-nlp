#!/bin/bash
#SBATCH --job-name=dl_resources
#SBATCH --nodes=1
#SBATCH --cpus-per-task=10
#SBATCH --output=log/slurm-%j.out
#SBATCH --time=01:00:00
#SBATCH --error=log/slurm-%j.err
#SBATCH --partition=prepost
#SBATCH --account=lch@h100

module purge
module load miniforge/24.9.0
conda activate synth-kg

bash ./eval/health/evaluation.sh "$(MODEL_ID)" "datasets/health/eval/model_outputs/model=$(MODEL_ID)_size=$(SIZE)_step=$(STEP)/evaluation.parquet"
