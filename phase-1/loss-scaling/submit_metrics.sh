#!/bin/bash

#SBATCH -J loss-scaling-metrics
#SBATCH -o slurm/metrics.%j.out
#SBATCH -e slurm/metrics.%j.err
#SBATCH --nodes=1
#SBATCH --tasks-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --qos=regular
#SBATCH --account=m4718
#SBATCH --constraint=cpu
#SBATCH -t 12:00:00

conda activate ufs2arco
python compute_metrics.py
