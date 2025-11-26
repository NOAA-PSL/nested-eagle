#!/bin/bash

#SBATCH -J hrrr-forecasts
#SBATCH -o hrrr-forecasts.%j.out
#SBATCH -e hrrr-forecasts.%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=64
#SBATCH --cpus-per-task=4
#SBATCH --qos=debug
#SBATCH --account=m4718
#SBATCH --constraint=cpu
#SBATCH -t 00:30:00

module load conda
conda activate ufs2arco

srun ufs2arco data.yaml
python compute_metrics.py metrics.yaml
