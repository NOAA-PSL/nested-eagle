#!/bin/bash

#SBATCH -J gfs2obs
#SBATCH -o slurm/parallel_wxvx.%j.out
#SBATCH -e slurm/parallel_wxvx.%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --qos=regular
#SBATCH --account=m4718
#SBATCH --constraint=cpu
#SBATCH -t 12:00:00

# Preprocess forecast files for wxvx
#conda activate anemoi
#eagle-tools prewxvx obs.prewxvx.yaml
#conda deactivate

conda activate wxvx

#echo "Pulling obs"
#wxvx -c obs.wxvx.yaml -t obs -n 16 > log.wxvx.obs 2>&1

# Write a yaml for each cycle and process separately
echo "Starting grids and stats"
python write_wxvx_cycles.py
cycles_dir="cycles-wxvx"
find ${cycles_dir}  -maxdepth 1 -type f -name "*.yaml" > input.txt
parallel -j 128 ./parallel_wxvx.sh {} :::: input.txt

conda deactivate

# Postprocess into a single file for each variable
conda activate anemoi
eagle-tools postwxvx obs.postwxvx.yaml
conda deactivate
