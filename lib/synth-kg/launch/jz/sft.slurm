#!/bin/bash
#SBATCH --job-name=sft
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=40
#SBATCH --output=log/slurm-%j.out
#SBATCH --time=02:00:00
#SBATCH --error=log/slurm-%j.err
#SBATCH --partition=gpu_p5
#SBATCH --account=oha@a100
#SBATCH -C a100
#SBATCH --qos=qos_gpu_a100-dev
#SBATCH --mail-type=END
#SBATCH --mail-user=simon.meoni@inria.fr

# Initialize variables with default values
HYDRA_CONFIG="default.yaml"

# Parse arguments
while [[ "$#" -gt 0 ]]; do
  case $1 in
  --HYDRA_CONFIG)
    HYDRA_CONFIG="$2"
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
python training_steps/sft/train.py --config-name="$HYDRA_CONFIG"
