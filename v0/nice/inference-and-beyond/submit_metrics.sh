#!/bin/bash

#SBATCH -J nested-eagle-metrics
#SBATCH -o slurm_metrics.%j.out
#SBATCH -e slurm_metrics.%j.err
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --qos=regular
#SBATCH --account=m4718
#SBATCH --constraint=cpu
#SBATCH -t 01:00:00

conda activate eagle
eagle-tools metrics metrics.validation.yaml
