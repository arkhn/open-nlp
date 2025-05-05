#!/bin/bash
#SBATCH --job-name=score
#SBATCH --nodes=1
#SBATCH --partition=gpu_p6
#SBATCH --account=lch@h100
#SBATCH -C h100
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=80
#SBATCH --output=log/slurm-%j.out
#SBATCH --time=20:00:00
#SBATCH --error=log/slurm-%j.err
#SBATCH --qos=qos_gpu_h100-t3
#SBATCH --mail-type=END
#SBATCH --mail-user=simon.meoni@inria.fr

# Initialize variables with default values
STS_MODEL=""
PRIVATE_DATASET=""
N=4
OUTPUT_PATH=""

# Parse arguments
while [[ "$#" -gt 0 ]]; do
  case $1 in
  --STS_MODEL)
    STS_MODEL="$2"
    shift
    ;;
  --PRIVATE_DATASET)
    PRIVATE_DATASET="$2"
    shift
    ;;
  --N)
    N="$2"
    shift
    ;;
  --OUTPUT_PATH)
    OUTPUT_PATH="$2"
    shift
    ;;
  --WDB_ID)
    WDB_ID="$2"
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
module load arch/h100
module load miniforge/24.9.0
conda activate synth-kg

module load cuda # Adjust to the appropriate CUDA version

export HF_HUB_OFFLINE=1
echo "$OUTPUT_PATH"
echo "$PRIVATE_DATASET"
python training_steps/score/run.py \
  --sts_model "$STS_MODEL" \
  --public_dataset "$OUTPUT_PATH" \
  --private_dataset "$PRIVATE_DATASET" \
  --tp "$SLURM_GPUS_ON_NODE" \
  --output_path "$OUTPUT_PATH" \
  --n "$N" \
  --wdb_id "$WDB_ID" \
  --group_id "$GROUP_ID"
