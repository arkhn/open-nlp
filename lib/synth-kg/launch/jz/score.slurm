#!/bin/bash
#SBATCH --job-name=score
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=40
#SBATCH --output=log/slurm-%j.out
#SBATCH --time=01:00:00
#SBATCH --error=log/slurm-%j.err
#SBATCH --partition=gpu_p5
#SBATCH --account=oha@a100
#SBATCH -C a100
#SBATCH --qos=qos_gpu_a100-dev

# Initialize variables with default values
EVALUATOR_PATH=""
PRIVATE_DATASET=""
PUBLIC_DATASET=""
N=4
OUTPUT_PATH=""

# Parse arguments
while [[ "$#" -gt 0 ]]; do
  case $1 in
    --EVALUATOR_PATH) EVALUATOR_PATH="$2"; shift ;;
    --PRIVATE_DATASET) PRIVATE_DATASET="$2"; shift ;;
    --PUBLIC_DATASET) PUBLIC_DATASET="$2"; shift ;;
    --N) N="$2"; shift ;;
    --OUTPUT_PATH) OUTPUT_PATH="$2"; shift ;;
    *) echo "Unknown parameter passed: $1"; exit 1 ;;
  esac
  shift
done

module purge
module load miniforge/24.9.0
conda activate synth-kg

module load cuda     # Adjust to the appropriate CUDA version

export HF_HUB_OFFLINE=1

python training_steps/score/run.py \
  --evaluator_path "$EVALUATOR_PATH" \
  --public_dataset "$PUBLIC_DATASET" \
  --private_dataset "$PRIVATE_DATASET" \
  --output_path "$OUTPUT_PATH" \
  --n "$N"
