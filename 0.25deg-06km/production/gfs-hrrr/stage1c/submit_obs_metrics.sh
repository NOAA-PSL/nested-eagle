#!/bin/bash

#SBATCH -J nested-obs-metrics
#SBATCH -o slurm/obs-metrics.%j.out
#SBATCH -e slurm/obs-metrics.%j.err
#SBATCH --nodes=5
#SBATCH --tasks-per-node=16
#SBATCH --cpus-per-task=16
#SBATCH --qos=regular
#SBATCH --account=m4718
#SBATCH --constraint=cpu
#SBATCH -t 3:00:00

conda activate eagle
echo "Running for HRRR over validation period"
srun eagle-tools obs-metrics obs-metrics.hrrr.validation.yaml
echo " ... Done"
echo "Running for global over validation period"
srun eagle-tools obs-metrics obs-metrics.global.validation.yaml
echo " ... Done"

echo "Running for HRRR over testing period"
srun eagle-tools obs-metrics obs-metrics.hrrr.testing.yaml
echo " ... Done"
echo "Running for global over testing period"
srun eagle-tools obs-metrics obs-metrics.global.testing.yaml
echo " ... Done"
