#!/bin/bash
#SBATCH --job-name=gen
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

# Initialize variables with default values
DATASET_PATH="default_name"
MODEL="meta-llama/llama-2-7b-hf"
ADAPTERS_PATHS="default_path"
OUTPUT_PATH="default_path"

# Parse arguments
while [[ "$#" -gt 0 ]]; do
  case $1 in
  --DATASET_PATH)
    DATASET_PATH="$2"
    shift
    ;;
  --MODEL)
    MODEL="$2"
    shift
    ;;
  --ADAPTERS_PATHS)
    ADAPTERS_PATHS="$2"
    shift
    ;;
  --OUTPUT_PATH)
    OUTPUT_PATH="$2"
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
module load arch/h100
module load miniforge/24.9.0
conda activate synth-kg

module load cuda # Adjust to the appropriate CUDA version
# Run your processing script
MERGE_ID=$(printf "%06d" $((RANDOM % 1000000)))
MERGE_OUTPUT_PATH=./lora/merge-${MERGE_ID}
CUDA_VISIBLE_DEVICES=0 python training_steps/generation/merge_adapters.py \
  --model "$MODEL" \
  --adapters "$ADAPTERS_PATHS" \
  --output_path "$MERGE_OUTPUT_PATH"

python training_steps/generation/run.py --dataset "$DATASET_PATH" \
  --model "$MERGE_OUTPUT_PATH" \
  --output_path "$OUTPUT_PATH" \
  --tp "$SLURM_GPUS_ON_NODE" \
  --pp "$SLURM_NNODES"
