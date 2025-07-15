import logging
import sys

import numpy as np
from scipy.spatial import SphericalVoronoi

import xarray as xr
import pandas as pd

import ufs2arco.utils

from .log import setup_simple_log
from .utils import open_yaml_config

logger = logging.getLogger("eagle")


def drop_forcing_vars(xds):
    for key in [
        "cos_julian_day",
        "sin_julian_day",
        "cos_local_time",
        "sin_local_time",
        "cos_latitude",
        "sin_latitude",
        "cos_longitude",
        "sin_longitude",
        "orog",
        "orography",
        "geopotential_at_surface",
        "land_sea_mask",
        "lsm",
        "insolation",
        "cos_solar_zenith_angle",
    ]:
        if key in xds:
            xds = xds.drop_vars(key)
    return xds


def get_gridcell_area_weights(xds, unit_mean=True, radius=1, center=np.array([0,0,0]), threshold=1e-12):
    """This is a nice code block copied from anemoi-graphs"""

    x = radius * np.cos(np.deg2rad(xds["latitude"])) * np.cos(np.deg2rad(xds["longitude"]))
    y = radius * np.cos(np.deg2rad(xds["latitude"])) * np.sin(np.deg2rad(xds["longitude"]))
    z = radius * np.sin(np.deg2rad(xds["latitude"]))
    sv = SphericalVoronoi(
        points=np.stack([x,y,z], -1),
        radius=radius,
        center=center,
        threshold=threshold,
    )
    area_weight = sv.calculate_areas()
    if unit_mean:
        area_weight /= area_weight.mean()

    return area_weight


def open_anemoi_dataset(path):
    xds = xr.open_zarr(path)
    vds = ufs2arco.utils.expand_anemoi_dataset(xds, "data", xds.attrs["variables"])
    vds = drop_forcing_vars(vds)
    return vds


def rmse(target, prediction, weights=1.):
    result = {}
    for key in prediction.data_vars:
        se = (target[key] - prediction[key])**2
        se = weights*se
        mse = se.mean(["cell", "ensemble"])
        result[key] = np.sqrt(mse).compute()

    xds = xr.Dataset(result)
    xds["lead_time"] = xds["time"] - xds["time"][0]
    xds = xds.swap_dims({"time": "lead_time"}).drop_vars("time")
    return xds


def mae(target, prediction, weights=1.):
    result = {}
    for key in prediction.data_vars:
        ae = np.abs(target[key] - prediction[key])
        ae = weights*ae
        mae = ae.mean(["cell", "ensemble"])
        result[key] = mae.compute()

    xds = xr.Dataset(result)
    xds["lead_time"] = xds["time"] - xds["time"][0]
    xds = xds.swap_dims({"time": "lead_time"}).drop_vars("time")
    return xds



def compute_error_metrics():
    """Compute grid cell area weighted RMSE and MAE

    Note that the arguments documented here are passed via a config yaml as in

    Example:
        >>> python compute_error_metrics.py recipe.yaml

    Args:
        forecast_path (str): directory containing forecast datasets to compare against a verification dataset. For now, the convention is that, within this directory, each forecast is in a separate netcdf file named as "<initial_date>.<lead_time>.nc", where initial_date = "%Y-%m-%dT%H" and lead_time is defined below
        lead_time (str): a string indicating length, e.g. 240h or 90d, it doesn't matter what format, just make it the same as what was saved during forecast time
        verification_dataset_path (str): path to the zarr verification dataset
        output_path (str): directory to save rmse.nc and mae.nc
        start_date (str): date of first last IC to grab, in %Y-%m-%dTH format
        end_date (str): date of last last IC to grab, in %Y-%m-%dTH format
        freq (str): frequency over which to grab initial condition dates, passed to pandas.date_range
    """

    if len(sys.argv) != 2:
        raise Exception("Did not get an argument. Usage is:\npython compute_error_metrics.py recipe.yaml")

    config = open_yaml_config(sys.argv[1])
    setup_simple_log()

    vds = open_anemoi_dataset(config["verification_dataset_path"])
    latlon_weights = get_gridcell_area_weights(vds)

    dates = pd.date_range(config["start_date"], config["end_date"], freq=config["freq"])

    rmse_container = None
    mae_container = None
    logger.info(f" --- Starting Metrics Computation --- ")
    for t0 in dates:
        st0 = t0.strftime("%Y-%m-%dT%H")
        fds = xr.open_dataset(
            f"{config['forecast_path']}/{st0}.{config['lead_time']}.nc",
        )

        fds = ufs2arco.utils.convert_anemoi_inference_dataset(fds)
        fds = drop_forcing_vars(fds)

        fds = fds.load()
        tds = vds.sel(time=fds.time.values).load()

        this_rmse = rmse(target=tds, prediction=fds, weights=latlon_weights)
        this_mae = mae(target=tds, prediction=fds, weights=latlon_weights)

        if rmse_container is None:
            rmse_container = this_rmse / len(dates)
            mae_container = this_mae / len(dates)
        else:
            rmse_container += this_rmse/len(dates)
            mae_container += this_mae/len(dates)

        logger.info(f"\tDone with {st0}")
    logger.info(f" --- Done Computing Metrics --- \n")

    logger.info(f" --- Storing Results --- ")
    rmse_container.to_netcdf(f"{config['output_path']}/rmse.nc")
    mae_container.to_netcdf(f"{config['output_path']}/mae.nc")
    logger.info(f" --- Done Storing Results at {config['output_path']} --- \n")
