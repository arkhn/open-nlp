#!/bin/bash
#SBATCH --job-name=kto
#SBATCH --nodes=1
#SBATCH --partition=gpu_p6
#SBATCH --account=lch@h100
#SBATCH -C h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=40
#SBATCH --output=log/slurm-%j.out
#SBATCH --time=04:00:00
#SBATCH --error=log/slurm-%j.err
#SBATCH --qos=qos_gpu_h100-t3
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
  --HYDRA_ARGS)
    HYDRA_ARGS="$2"
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
eval set -- "$HYDRA_ARGS"
python training_steps/kto/run.py --config-name="$HYDRA_CONFIG" "$@"
