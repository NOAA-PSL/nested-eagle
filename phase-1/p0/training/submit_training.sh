#!/bin/bash

#SBATCH -J p0-training
#SBATCH -o slurm/training.%j.out
#SBATCH -e slurm/training.%j.err
#SBATCH --nodes=4
#SBATCH --tasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --qos=regular
#SBATCH --account=m4718
#SBATCH --constraint=gpu&hbm80g
#SBATCH -t 36:00:00

conda activate anemoi
srun --jobid $SLURM_JOB_ID ~/anemoi-house/slurm2ddp.sh anemoi-training train --config-name=config
