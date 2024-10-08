#!/bin/bash
#SBATCH --job-name=convert-docker-to-singularity
#SBATCH --nodes=1
#SBATCH --cpus-per-task=10
#SBATCH --output=log/conversion%j.out
#SBATCH --partition=prepost
#SBATCH --account=oha@v100

module purge
module load singularity

DOCKER_IMAGE="simon6789/style-transfer"
SINGULARITY_IMAGE="style-transfer.sif"

singularity build $SINGULARITY_IMAGE docker://$DOCKER_IMAGE
