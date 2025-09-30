#!/bin/bash

#SBATCH -J nested-eagle-inference
#SBATCH -o slurm_inference.%j.out
#SBATCH -e slurm_inference.%j.err
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --qos=regular
#SBATCH --account=m4718
#SBATCH --constraint=gpu
#SBATCH -t 01:00:00

conda activate eagle
eagle-tools inference inference.validation.yaml
