#!/bin/bash

#SBATCH -J global-obs-metrics
#SBATCH -o slurm/obs-metrics.%j.out
#SBATCH -e slurm/obs-metrics.%j.err
#SBATCH --nodes=5
#SBATCH --tasks-per-node=16
#SBATCH --cpus-per-task=16
#SBATCH --qos=regular
#SBATCH --account=m4718
#SBATCH --constraint=cpu
#SBATCH -t 1:30:00

conda activate eagle
echo "Running over validation"
srun eagle-tools obs-metrics obs-metrics.validation.yaml
echo "Done with validation"
echo "Running over testing"
srun eagle-tools obs-metrics obs-metrics.testing.yaml
echo "Done with testing"
