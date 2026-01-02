#!/bin/bash

#SBATCH -J globaleagle2obs
#SBATCH -o slurm/wxvx.0.2.1.%j.out
#SBATCH -e slurm/wxvx.0.2.1.%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=64
#SBATCH --cpus-per-task=4
#SBATCH --qos=regular
#SBATCH --account=m4718
#SBATCH --constraint=cpu
#SBATCH -t 1:00:00

conda activate wxvx

# note that we only use 10 threads during the obs task since
# this is the default "connection pool limit"
echo "Running: wxvx -t obs -n 10"
wxvx -c config.wxvx.yaml -t obs -n 10 > log.obs.out 2>&1

echo "Running: wxvx -t ncobs -n ${SLURM_NTASKS}"
wxvx -c config.wxvx.yaml -t ncobs -n ${SLURM_NTASKS} > log.ncobs.out 2>&1
