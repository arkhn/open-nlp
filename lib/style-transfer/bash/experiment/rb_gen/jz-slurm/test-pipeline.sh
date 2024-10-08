#!/bin/bash
#SBATCH --job-name=test-gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --output=log/test.out
#SBATCH --partition=gpu_p5
#SBATCH --account=oha@a100
#SBATCH -C a100
#SBATCH --qos=qos_gpu_a100-dev

module purge
module load singularity

SCRIPT_PATH="style_transfer/run_rb_gen.py"
PYTHON="/opt/pysetup/.venv/bin/python"
LIB_PATH="$WORK/open-nlp/lib/style-transfer/"
singularity exec --bind $WORK/open-nlp,$HF_HOME,$WANDB_CACHE_DIR,$WANDB_DIR \
                 --env PYTHONPATH=$PYTHONPATH:$LIB_PATH,HF_HUB_OFFLINE=True \
                 --nv $SINGULARITY_ALLOWED_DIR/style-transfer.sif \
                 bash -c "cd $LIB_PATH && $PYTHON $SCRIPT_PATH"
