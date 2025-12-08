#!/bin/bash

#SBATCH -J era5-nest-movie
#SBATCH -o slurm/movie.%j.out
#SBATCH -e slurm/movie.%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=64
#SBATCH --cpus-per-task=4
#SBATCH --qos=regular
#SBATCH --account=m4718
#SBATCH --constraint=cpu
#SBATCH -t 12:00:00

conda activate graphufs-cpu
python plot_long.py
