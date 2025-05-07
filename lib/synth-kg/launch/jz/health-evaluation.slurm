#!/bin/bash
#SBATCH --job-name=health-eval
#SBATCH --nodes=1
#SBATCH --partition=gpu_p6
#SBATCH --account=lch@h100
#SBATCH -C h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=40
#SBATCH --output=log/slurm-%j.out
#SBATCH --time=02:00:00
#SBATCH --error=log/slurm-%j.err
#SBATCH --qos=qos_gpu_h100-dev
#SBATCH --mail-type=END
#SBATCH --mail-user=simon.meoni@inria.fr

# initialize variables with default values
OUTPUT_PATH="default_path"
DOWNSTREAM_DS_PATH=""
HYDRA_CONFIG="health"

# Parse arguments
while [[ "$#" -gt 0 ]]; do
  case $1 in
  --OUTPUT_PATH)
    OUTPUT_PATH="$2"
    shift
    ;;
  --HYDRA_CONFIG)
    HYDRA_CONFIG="$2"
    shift
    ;;
  --DOWNSTREAM_DS_PATH)
    DOWNSTREAM_DS_PATH="$2"
    shift
    ;;
  --GROUP_ID)
    GROUP_ID="$2"
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

module load cuda

MERGE_ID=$(printf "%06d" $((RANDOM % 1000000)))
MERGE_OUTPUT_PATH=./lora/merge-${MERGE_ID}
python eval/health/alpacare_generate.py --output_path "$OUTPUT_PATH" --dataset "$DOWNSTREAM_DS_PATH" --tp "$SLURM_GPUS_ON_NODE"
python eval/sft/train.py --config-name="$HYDRA_CONFIG" dataset=./evaluation_alpacare_sft.parquet merge_output_path="$MERGE_OUTPUT_PATH" group_id="$GROUP_ID"
python eval/health/generate.py --model "$MERGE_OUTPUT_PATH" --output_path "$OUTPUT_PATH" --tp "$SLURM_GPUS_ON_NODE"
