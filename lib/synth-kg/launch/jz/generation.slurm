#!/bin/bash
#SBATCH --job-name=gen
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=10
#SBATCH --output=log/slurm-%j.out
#SBATCH --time=20:00:00
#SBATCH --error=log/slurm-%j.err
#SBATCH --partition=gpu_p5
#SBATCH --account=oha@a100
#SBATCH -C a100
#SBATCH --qos=qos_gpu_a100-t3

# Initialize variables with default values
DATASET_PATH="default_name"
MODEL="meta-llama/llama-2-7b-hf"
ADAPTERS_PATHS="default_path"
OUTPUT_PATH="default_path"
GPUS=8

# Parse arguments
while [[ "$#" -gt 0 ]]; do
  case $1 in
    --DATASET_PATH) DATASET_PATH="$2"; shift ;;
    --MODEL) MODEL="$2"; shift ;;
    --ADAPTERS_PATHS) ADAPTERS_PATHS="$2"; shift ;;
    --OUTPUT_PATH) OUTPUT_PATH="$2"; shift ;;
    *) echo "Unknown parameter passed: $1"; exit 1 ;;
  esac
  shift
done
module purge
module load miniforge/24.9.0
conda activate synth-kg

module load cuda     # Adjust to the appropriate CUDA version
# Run your processing script
CUDA_VISIBLE_DEVICES=0 python training_steps/generation/merge_adapters.py \
                                                          --model "$MODEL" \
                                                          --adapters "$ADAPTERS_PATHS"

python training_steps/generation/run.py --dataset "$DATASET_PATH" \
                                        --model  "./lora/merged" \
                                        --output_path "$OUTPUT_PATH" \
                                        --gpus "$GPUS"
