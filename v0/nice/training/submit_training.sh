#!/bin/bash

#SBATCH -J nested-eagle-training
#SBATCH -o slurm_training.%j.out
#SBATCH -e slurm_training.%j.err
#SBATCH --nodes=8
#SBATCH --tasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --qos=regular
#SBATCH --account=m4718
#SBATCH --constraint=gpu
#SBATCH -t 06:00:00

conda activate eagle
srun --jobid $SLURM_JOB_ID ~/anemoi-house/slurm2ddp.sh anemoi-training train --config-name=config
