#!/bin/bash

#SBATCH -J 1x16encoders-obs-hrrr
#SBATCH -o slurm/1x16encoders.obs.hrrr.%j.out
#SBATCH -e slurm/1x16encoders.obs.hrrr.%j.err
#SBATCH --nodes=1
#SBATCH --qos=regular
#SBATCH --account=m4718
#SBATCH --constraint=cpu
#SBATCH -t 12:00:00

experiment=1x16encoders
domain=hrrr
n_procs=256

conda activate eagle
srun -n 32 eagle-tools prewxvx "${experiment}/prewxvx.${domain}.validation.yaml"
conda deactivate

conda activate /global/homes/t/timothys/miniforge3/envs/DEV-wxvx
wxvx -c "${experiment}/wxvx.${domain}.validation.yaml" -t grids -n ${n_procs} > "${experiment}/log.wxvx.grids.out" 2>&1
wxvx -c "${experiment}/wxvx.${domain}.validation.yaml" -t stats -n ${n_procs} > "${experiment}/log.wxvx.stats.out" 2>&1
conda deactivate

conda activate eagle
eagle-tools postwxvx "${experiment}/postwxvx.${domain}.validation.yaml"
