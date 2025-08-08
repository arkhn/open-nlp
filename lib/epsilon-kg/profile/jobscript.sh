#!/bin/bash
#SBATCH --job-name={rule}
#SBATCH --nodes=1
#SBATCH --partition=gpu_p6
#SBATCH --account=lch@h100
#SBATCH -C h100
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=40
#SBATCH --output=log/slurm-%j.out
#SBATCH --time=10:00:00
#SBATCH --error=log/slurm-%j.err
#SBATCH --qos=qos_gpu_h100-t3
#SBATCH --mail-type=END
#SBATCH --mail-user=simon.meoni@inria.fr

module purge
module load arch/h100
module load miniforge/24.9.0
conda activate synth-kg
module load cuda

{exec_job}
