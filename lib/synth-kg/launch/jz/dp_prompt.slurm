#!/bin/bash
#SBATCH --job-name=gen
#SBATCH --nodes=1
#SBATCH --partition=gpu_p6
#SBATCH --account=lch@h100
#SBATCH -C h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=40
#SBATCH --output=log/slurm-%j.out
#SBATCH --time=10:00:00
#SBATCH --error=log/slurm-%j.err
#SBATCH --qos=qos_gpu_h100-t3
#SBATCH --mail-type=END
#SBATCH --mail-user=simon.meoni@inria.fr

# Initialize variables with default values
DATASET_PATH="datasets/health/dp_size=60000/private.parquet"
MODEL="meta-llama/llama-2-7b-hf"
OUTPUT_PATH="./test-dp-prompt.parquet"

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
python training_steps/generation/dp_prompt_generate.py --dataset "$DATASET_PATH" \
  --model "$MODEL" \
  --output_path "$OUTPUT_PATH" \
  --tp "$SLURM_GPUS_ON_NODE" \
  --pp "$SLURM_NNODES"
