#!/bin/bash

#SBATCH -J g104
#SBATCH -o slurm/g104.dev.%j.out
#SBATCH -e slurm/g104.dev.%j.err
#SBATCH --exclusive
#SBATCH --nodes=1
#SBATCH --qos=regular
#SBATCH --account=m4718
#SBATCH --constraint=cpu
#SBATCH -t 1:00:00

n_procs=$SLURM_CPUS_ON_NODE

conda activate /global/homes/t/timothys/miniforge3/envs/DEV-wxvx

# note that we only use 10 threads during the obs task since
# this is the default "connection pool limit"
#echo "Running: wxvx -t obs -n ${n_procs}"
#wxvx -c config.wxvx.yaml -t obs -n ${n_procs} > log.obs.out 2>&1

echo "Running: wxvx -t ncobs -n ${n_procs}"
wxvx -c config.wxvx.yaml -t ncobs -n ${n_procs} > log.ncobs.out 2>&1
