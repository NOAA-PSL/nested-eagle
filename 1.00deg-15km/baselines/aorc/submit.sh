#!/bin/bash

#SBATCH -J aorc
#SBATCH -o aorc.%j.out
#SBATCH -e aorc.%j.err
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=64
#SBATCH --cpus-per-task=4
#SBATCH --qos=regular
#SBATCH --account=m4718
#SBATCH --constraint=cpu
#SBATCH -t 08:00:00

module load conda
conda activate ufs2arco

srun ufs2arco data.yaml --overwrite
