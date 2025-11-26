#!/bin/bash

#SBATCH -J nested-eagle-v0-data
#SBATCH -o slurm/preprocessing.%j.out
#SBATCH -e slurm/preprocessing.%j.err
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=64
#SBATCH --cpus-per-task=4
#SBATCH --qos=regular
#SBATCH --account=m4718
#SBATCH --constraint=cpu
#SBATCH -t 04:00:00

module load conda
conda activate ufs2arco
#python create_grids.py

srun ufs2arco gfs.yaml --overwrite
echo "done with gfs data"
srun ufs2arco hrrr.yaml --overwrite
echo "done with hrrr"

# Now copy some things over to community
mywork=$WORK/nested-eagle/1.00deg-15km/data
mycomm=$COMMUNITY/nested-eagle/1.00deg-15km/data
mkdir -p $mycomm
mkdir -p $mycomm/logs/gfs
mkdir -p $mycomm/logs/hrrr

#cp -v $mywork/*.nc $mycomm
cp -v $mywork/*.yaml $mycomm
cp -v $mywork/logs/gfs/log.0000.*.out $mycomm/logs/gfs
cp -v $mywork/logs/hrrr/log.0000.*.out $mycomm/logs/hrrr
