import os
import logging
import yaml

logger = logging.getLogger("eagle")

def open_yaml_config(config_filename: str):
    with open(config_filename, "r") as f:
        config = yaml.safe_load(f)

    # expand any environment variables
    for key, val in config.items():
        if "path" in key:
            config[key] = os.path.expandvars(val)

    # if output_path is not created, make it here
    if not os.path.isdir(config["output_path"]):
        logger.info(f"Creating output_path: {config['output_path']}")
        os.makedirs(config["output_path"])

    return config
