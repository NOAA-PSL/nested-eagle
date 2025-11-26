#!/bin/bash

#SBATCH -J inference
#SBATCH -o slurm/inference.%j.out
#SBATCH -e slurm/inference.%j.err
#SBATCH --nodes=1
#SBATCH --tasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --qos=regular
#SBATCH --account=m4718
#SBATCH --constraint=gpu
#SBATCH -t 01:00:00

conda activate anemoi

# Define the directories to loop through
directories=("ll10-nc0256-win4320")
yaml_files=("inference.validation.yaml" "ar2023.yaml")

# Loop through each directory
for dir in "${directories[@]}"; do
  # Loop through each YAML file
  for yaml_file in "${yaml_files[@]}"; do
    # Construct the full command to run
    command="python run_inference.py ${dir}/${yaml_file}"

    # Print the command being executed
    echo "Running command: $command"

    # Execute the command
    eval "$command"

    # Check the exit status of the command
    if [ $? -eq 0 ]; then
      echo "Command for directory '$dir' with file '$yaml_file' completed successfully."
    else
      echo "Error: Command for directory '$dir' with file '$yaml_file' failed."
    fi

    echo "----------------------------------------"
  done
done
