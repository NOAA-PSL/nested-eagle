#!/bin/bash

#SBATCH -J 1x16encoders-obs-global
#SBATCH -o slurm/1x16encoders.obs.global.%j.out
#SBATCH -e slurm/1x16encoders.obs.global.%j.err
#SBATCH --nodes=1
#SBATCH --qos=regular
#SBATCH --account=m4718
#SBATCH --constraint=cpu
#SBATCH -t 08:00:00

experiment=1x16encoders
domain=global
n_procs=256

conda activate eagle
srun -n 32 eagle-tools prewxvx "${experiment}/prewxvx.${domain}.validation.yaml"
conda deactivate

conda activate /global/homes/t/timothys/miniforge3/envs/DEV-wxvx
wxvx -c "${experiment}/wxvx.${domain}.validation.yaml" -t grids -n ${n_procs} > "${experiment}/log.wxvx.grids.${domain}.out" 2>&1
wxvx -c "${experiment}/wxvx.${domain}.validation.yaml" -t stats -n ${n_procs} > "${experiment}/log.wxvx.stats.${domain}.out" 2>&1
conda deactivate

conda activate eagle
eagle-tools postwxvx "${experiment}/postwxvx.${domain}.validation.yaml"
