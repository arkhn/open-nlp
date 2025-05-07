#!/bin/bash
#SBATCH --job-name=preference-eval
#SBATCH --nodes=1
#SBATCH --cpus-per-task=10
#SBATCH --output=log/slurm-%j.out
#SBATCH --time=01:00:00
#SBATCH --error=log/slurm-%j.err
#SBATCH --partition=compil
#SBATCH --account=lch@h100
#SBATCH --mail-type=END
#SBATCH --mail-user=simon.meoni@inria.fr

module purge
module load miniforge/24.9.0
conda activate synth-kg

# Parse arguments
while [[ "$#" -gt 0 ]]; do
  case $1 in
  --MODEL_ID)
    MODEL_ID="$2"
    shift
    ;;
  --GROUP_ID)
    GROUP_ID="$2"
    shift
    ;;
  --SIZE)
    SIZE="$2"
    shift
    ;;
  --STEP)
    STEP="$2"
    shift
    ;;
  --SUFFIX_RUN_NAME)
    SUFFIX_RUN_NAME="$2"
    shift
    ;;
  *)
    echo "Unknown parameter passed: $1"
    exit 1
    ;;
  esac
  shift
done

bash ./eval/health/evaluation.sh "$MODEL_ID" datasets/health/eval/model_outputs/model="$MODEL_ID"_size="$SIZE"_step="$STEP"/evaluation.parquet "$GROUP_ID" "$SUFFIX_RUN_NAME"
