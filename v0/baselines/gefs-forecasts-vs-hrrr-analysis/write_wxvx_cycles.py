import yaml
from datetime import datetime, timedelta
import os

def generate_cycle_files(config_path='obs.wxvx.yaml', output_dir='cycles-wxvx'):
    """
    Reads a YAML configuration file, generates a series of dates based on the
    'cycles' section, and creates a new YAML file for each date.

    Args:
        config_path (str): The path to the input YAML configuration file.
        output_dir (str): The name of the directory to save the new YAML files.
    """
    # 1. Create the output directory if it doesn't exist
    try:
        os.makedirs(output_dir, exist_ok=True)
        print(f"Output directory '{output_dir}' created or already exists.")
    except OSError as e:
        print(f"Error creating directory {output_dir}: {e}")
        return

    # 2. Read the base YAML file
    try:
        with open(config_path, 'r') as file:
            base_config = yaml.safe_load(file)
            print(f"Successfully loaded '{config_path}'.")
    except FileNotFoundError:
        print(f"Error: The file '{config_path}' was not found.")
        return
    except yaml.YAMLError as e:
        print(f"Error parsing YAML file '{config_path}': {e}")
        return

    # 3. Parse the cycle information
    cycle_info = base_config.get('cycles', {})
    start_date = cycle_info.get('start')
    end_date = cycle_info.get('stop') # Use 'stop' key instead of 'end'
    step_hours = cycle_info.get('step')

    if not all([start_date, end_date, isinstance(step_hours, int)]):
        print("Error: 'cycles' section must contain 'start', 'stop', and 'step' (as an integer).")
        return

    print(type(start_date), type(end_date), type(step_hours))

    if not isinstance(start_date, datetime):
        try:
            start_date = datetime.fromisoformat(start_date)
            end_date = datetime.fromisoformat(end_date)
        except ValueError as e:
            print(f"Error parsing date format in 'cycles' section: {e}")
            return

    time_step = timedelta(hours=step_hours)

    # 4. Generate the list of dates
    dates = []
    current_date = start_date
    while current_date <= end_date:
        dates.append(current_date)
        current_date += time_step

    print(f"Generated {len(dates)} dates to process.")

    # 5. Loop through dates and create new YAML files
    for date_obj in dates:
        # Create a deep copy to avoid modifying the original config in memory
        new_config = yaml.safe_load(yaml.dump(base_config))

        # Format the date as required: YYYY-mm-ddThh:MM:ss
        formatted_date = date_obj.strftime('%Y-%m-%dT%H:%M:%S')

        # Update the 'cycles' section for the new file
        new_config['cycles'] = [formatted_date]

        # Define the new filename
        file_date_str = date_obj.strftime('%Y%m%dT%H%M%S')
        output_filename = f"{file_date_str}.{config_path}"
        output_filepath = os.path.join(output_dir, output_filename)

        # Write the new YAML file
        try:
            with open(output_filepath, 'w') as file:
                yaml.dump(new_config, file, default_flow_style=False, sort_keys=False)
            print(f"  -> Created '{output_filepath}'")
        except IOError as e:
            print(f"Error writing to file '{output_filepath}': {e}")

    print("\nProcessing complete.")

if __name__ == '__main__':
    generate_cycle_files()
