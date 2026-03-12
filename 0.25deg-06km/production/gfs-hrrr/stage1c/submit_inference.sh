#!/bin/bash

#SBATCH -J nested-inference
#SBATCH -o slurm/inference.%j.out
#SBATCH -e slurm/inference.%j.err
#SBATCH --nodes=8
#SBATCH --tasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --qos=regular
#SBATCH --account=m4718
#SBATCH --constraint='gpu&hbm80g'
#SBATCH -t 3:00:00

# Note: I think this would work just fine with a 40GB A100, but this is what ran.

conda activate eagle
srun eagle-tools inference inference.validation.yaml
srun eagle-tools inference inference.testing.yaml
