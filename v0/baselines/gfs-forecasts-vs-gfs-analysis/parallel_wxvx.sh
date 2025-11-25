#!/bin/bash

# --- Define the work to be done for a single file ---
yamlfile=$1 # Use a local variable for the input path

# Extract the date string (e.g., 20230201T060000) from the filename
base_filename=$(basename "$yamlfile")
file_date_str=${base_filename%%.*}

echo "-> [PID: $$] Running wxvx for ${file_date_str}"

# Run the commands, redirecting output.
# The 'cycles_dir' variable is available within the function.
wxvx -c "$yamlfile" -t grids > "${cycles_dir}/log.wxvx.grids.${file_date_str}" 2>&1
wxvx -c "$yamlfile" -t stats > "${cycles_dir}/log.wxvx.stats.${file_date_str}" 2>&1

echo "-> [PID: $$] Finished ${file_date_str}"
