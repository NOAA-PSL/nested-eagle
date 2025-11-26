#!/bin/bash

#SBATCH -J gfs-forecasts
#SBATCH -o gfs-forecasts.%j.out
#SBATCH -e gfs-forecasts.%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=64
#SBATCH --cpus-per-task=4
#SBATCH --qos=regular
#SBATCH --account=m4718
#SBATCH --constraint=cpu
#SBATCH -t 02:00:00

module load conda
conda activate ufs2arco

srun ufs2arco data.yaml
