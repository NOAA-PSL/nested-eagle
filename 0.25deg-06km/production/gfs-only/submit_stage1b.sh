#!/bin/bash

#SBATCH -J gfs-stage1b
#SBATCH -o slurm/stage1b.%j.out
#SBATCH -e slurm/stage1b.%j.err
#SBATCH --nodes=16
#SBATCH --tasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --qos=regular
#SBATCH --account=m4718
#SBATCH --constraint=gpu
#SBATCH -t 6:00:00

module load nccl/2.21.5
conda activate eagle

export LD_PRELOAD=$CONDA_PREFIX/lib/libjemalloc.so
export NCCL_DEBUG=WARN
export PYTHONFAULTHANDLER=1

srun ~/anemoi-house/slurm2ddp.sh anemoi-training train --config-name=stage1b
