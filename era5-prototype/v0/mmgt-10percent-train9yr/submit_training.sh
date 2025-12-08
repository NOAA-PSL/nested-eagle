#!/bin/bash

#SBATCH -J era5v0-train8yr-10percent
#SBATCH -o slurm/training.10percent.%j.out
#SBATCH -e slurm/training.10percent.%j.err
#SBATCH --nodes=8
#SBATCH --tasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --qos=regular
#SBATCH --account=m4718
#SBATCH --constraint=gpu
#SBATCH -t 12:00:00

source ~/.azure.sh
conda activate anemoi
srun --jobid $SLURM_JOB_ID ~/anemoi-house/slurm2ddp.sh anemoi-training train --config-name=config
