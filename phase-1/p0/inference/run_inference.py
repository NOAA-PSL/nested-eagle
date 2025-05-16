import sys
from datetime import datetime
import pandas as pd
from anemoi.inference.config.run import RunConfiguration
from anemoi.inference.runners import create_runner
from inference_globals import (
    CHECKPOINT,
    LAM,
    LAM_PATH,
    GLOBAL_PATH,
    LEAD_TIME,
    OUTPUT_PATH,
    INPUT_DATA_PATH,
)


def date_to_str(
    date: str,
) -> str:
    """
    Format initializtion date to a nicely formatted string that is used for writing out files.

    Args:
        date (str): date of initialization.

    Returns:
        str -- date of initialization in str format.
    """
    dt = datetime.fromisoformat(date)
    return dt.strftime("%Y%m%dT%H%M%SZ")


def create_config(
    init_date: str,
    extract_lam: bool,
    lead_time: int = LEAD_TIME,
    lam: bool = LAM,
    checkpoint: str = CHECKPOINT,
    input_data_path: str = INPUT_DATA_PATH,
    lam_path: str = LAM_PATH,
    global_path: str = GLOBAL_PATH,
    output_path: str = OUTPUT_PATH,
) -> dict:
    """
    Create the config that will be passed to anemoi-inference.
    As of right now the "extract_lam" functionality is really only used to save out a static lam file when wanted.
    This could be easily updated if we think we would ever be interested in running inference over just CONUS.

    Args:
        init_date (str): date of initialization.
        extract_lam (bool): logic to extract lam domain, or run whole nested domain.
        lead_time (int): desired lead time to save out. Default=LEAD_TIME.
        lam (bool): true/false indication if LAM. Default=LAM.
        checkpoint (str): path to checkpoint. Default=CHECKPOINT.
        input_data_path (str): path to input data when not using LAM (e.g. you trained on 1 source). Default=INPUT_DATA_PATH.
        lam_path (str): path to regional data when using LAM. Default=LAM_PATH.
        global_path (str: path to global data when using LAM. Default=GLOBAL_PATH.
        output_path (str): path for saving files. Default=OUTPUT_PATH.

    Returns:
        dict -- config for anemoi-inference.
    """
    date_str = date_to_str(date=init_date)

    if lam:
        if extract_lam:
            config = {
                "checkpoint": checkpoint,
                "date": init_date,
                "lead_time": lead_time,
                "input": {
                    "cutout": {
                        "lam_0": {"dataset": lam_path},
                        "global": {"dataset": global_path},
                    }
                },
                "output": {
                    "extract_lam": {
                        "output": {"netcdf": {"path": f"{output_path}/lam.nc"}}
                    }
                },
            }
        else:
            config = {
                "checkpoint": checkpoint,
                "date": init_date,
                "lead_time": lead_time,
                "input": {
                    "cutout": {
                        "lam_0": {"dataset": lam_path},
                        "global": {"dataset": global_path},
                    }
                },
                "output": {"netcdf": f"{output_path}/{date_str}.nc"},
            }

    else:
        config = {
            "checkpoint": checkpoint,
            "date": init_date,
            "lead_time": lead_time,
            "input": {
                "xarray-zarr": input_data_path,
            },
            "output": {"netcdf": f"{output_path}/{date_str}.nc"},
        }

    return config


def run_inference(
    init_date: str,
    extract_lam: bool,
) -> None:
    """
    Inference pipeline.

    Args:
        init_date (str): date of initialization.
        extract_lam (bool): true will run inference only on LAM domain.

    Returns:
        None -- files saved out to output path.
    """
    config_dict = create_config(
        init_date=init_date,
        extract_lam=extract_lam,
    )
    config = RunConfiguration.load(config_dict)
    runner = create_runner(config)
    runner.execute()

    return


if __name__ == "__main__":
    if len(sys.argv) != 5:
        print(
            "Usage: python run_inference.py <start_date>  <end_date>  <freq>  <extract_lam>"
        )
        print(
            "Example: python run_inference.py '2018-01-06T00:00:00' '2018-01-08T00:00:00' '12h' 'False'"
        )
        sys.exit(1)

    start_date = sys.argv[1]
    end_date = sys.argv[2]
    freq = sys.argv[3]
    extract_lam = sys.argv[4]

    dates = pd.date_range(start=start_date, end=end_date, freq=freq)

    nc_files = []
    for d in dates:
        run_inference(
            init_date=str(d),
            extract_lam=extract_lam,
        )
