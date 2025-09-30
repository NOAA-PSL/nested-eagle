#!/bin/bash

#SBATCH -J nested-eagle-data
#SBATCH -o slurm_preprocessing.%j.out
#SBATCH -e slurm_preprocessing.%j.err
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=64
#SBATCH --cpus-per-task=4
#SBATCH --qos=regular
#SBATCH --account=m4718
#SBATCH --constraint=cpu
#SBATCH -t 01:00:00

conda activate eagle
python create_grids.py

srun ufs2arco gfs.yaml --overwrite
echo "done with gfs data"
srun ufs2arco hrrr.yaml --overwrite
echo "done with hrrr"
