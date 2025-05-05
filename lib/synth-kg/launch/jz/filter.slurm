#!/bin/bash
#SBATCH --job-name=filter
#SBATCH --nodes=1
#SBATCH --partition=gpu_p6
#SBATCH --account=lch@h100
#SBATCH -C h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=40
#SBATCH --output=log/slurm-%j.out
#SBATCH --time=00:05:00
#SBATCH --error=log/slurm-%j.err
#SBATCH --qos=qos_gpu_h100-dev
#SBATCH --mail-type=END
#SBATCH --mail-user=simon.meoni@inria.fr

while [[ "$#" -gt 0 ]]; do
  case $1 in
  --INPUT_FILE)
    INPUT_FILE="$2"
    shift
    ;;
  *)
    echo "Unknown parameter passed: $1"
    exit 1
    ;;
  esac
  shift
done

module purge
module load miniforge/24.9.0
conda activate synth-kg

python training_steps/filter/run.py $INPUT_FILE
