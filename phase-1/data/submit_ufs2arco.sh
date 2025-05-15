#!/bin/bash

#SBATCH -J nested-era5-data
#SBATCH -o /pscratch/sd/t/timothys/nested-eagle/phase-2/data/slurm/preprocessing.%j.out
#SBATCH -e /pscratch/sd/t/timothys/nested-eagle/phase-2/data/slurm/preprocessing.%j.err
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=64
#SBATCH --cpus-per-task=4
#SBATCH --qos=debug
#SBATCH --account=m4718
#SBATCH --constraint=cpu
#SBATCH -t 00:30:00

module load conda
conda activate ufs2arco
export PYTHONPATH=""
python create_global_grid.py

srun ufs2arco conus.training.yaml --overwrite
srun ufs2arco conus.validation.yaml --overwrite
srun ufs2arco global.training.yaml --overwrite
srun ufs2arco global.validation.yaml --overwrite
