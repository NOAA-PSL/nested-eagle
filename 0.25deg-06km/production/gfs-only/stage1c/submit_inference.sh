#!/bin/bash

#SBATCH -J global-inference
#SBATCH -o slurm/inference.%j.out
#SBATCH -e slurm/inference.%j.err
#SBATCH --nodes=8
#SBATCH --tasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --qos=regular
#SBATCH --account=m4718
#SBATCH --constraint='gpu&hbm80g'
#SBATCH -t 5:00:00

conda activate eagle
srun eagle-tools inference inference.validation.yaml
srun eagle-tools inference inference.testing.yaml
