#!/bin/bash
#SBATCH --job-name=health-eval
#SBATCH --nodes=1
#SBATCH --partition=gpu_p6
#SBATCH --account=oha@h100
#SBATCH -C h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=40
#SBATCH --output=log/slurm-%j.out
#SBATCH --time=00:05:00
#SBATCH --error=log/slurm-%j.err
#SBATCH --qos=qos_gpu_h100-dev

# initialize variables with default values
DATASET="default_name"
OUTPUT_PATH="default_path"
SEED_DATASET="default_path"
HYDRA_CONFIG="default"

# Parse arguments
while [[ "$#" -gt 0 ]]; do
  case $1 in
  --DATASET)
    DATASET="$2"
    shift
    ;;
  --SEED_DATASET)
    SEED_DATASET="$2"
    shift
    ;;
  --OUTPUT_PATH)
    OUTPUT_PATH="$2"
    shift
    ;;
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

module load cuda # Adjust to the appropriate CUDA version
# Run your processing script
python eval/health/alpacare_generate.py --output_path "$OUTPUT_PATH" --dataset "$DATASET" --seed-dataset "$SEED_DATASET"
python eval/sft/train.py --config-name="$HYDRA_CONFIG" dataset=./evaluation_alpacare_sft.parquet
python eval/health/generate.py --model ./sft/merged/ --output_path "$OUTPUT_PATH"
