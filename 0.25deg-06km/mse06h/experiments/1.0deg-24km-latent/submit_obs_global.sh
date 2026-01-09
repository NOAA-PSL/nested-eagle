#!/bin/bash

#SBATCH -J 1x16encoders-obs-global
#SBATCH -o slurm/1x16encoders.obs.global.%j.out
#SBATCH -e slurm/1x16encoders.obs.global.%j.err
#SBATCH --nodes=1
#SBATCH --qos=regular
#SBATCH --account=m4718
#SBATCH --constraint=cpu
#SBATCH -t 12:00:00

set -e

experiment=1x16encoders
domain=global
n_procs=256
n_procs_eagle=16

# Step 1: prewxvx
echo "Starting prewxvx..."
conda activate eagle
srun -n ${n_procs_eagle} eagle-tools prewxvx "${experiment}/prewxvx.${domain}.validation.yaml"
conda deactivate
echo "prewxvx completed successfully."

# Step 2: wxvx (grids and stats)
echo "Starting wxvx grids..."
conda activate /global/homes/t/timothys/miniforge3/envs/DEV-wxvx
wxvx -c "${experiment}/wxvx.${domain}.validation.yaml" -t grids -n ${n_procs} > "${experiment}/log.wxvx.grids.${domain}.out" 2>&1
echo "wxvx grids completed successfully."

echo "Starting wxvx stats..."
wxvx -c "${experiment}/wxvx.${domain}.validation.yaml" -t stats -n ${n_procs} > "${experiment}/log.wxvx.stats.${domain}.out" 2>&1
conda deactivate
echo "wxvx stats completed successfully."

# Step 3: postwxvx
echo "Starting postwxvx..."
conda activate eagle
eagle-tools postwxvx "${experiment}/postwxvx.${domain}.validation.yaml"
echo "postwxvx completed successfully."
