import numpy as np
from scipy.spatial import SphericalVoronoi

import xarray as xr
import pandas as pd

from ufs2arco import utils


def get_gridcell_area_weights(xds, radius=1, center=np.array([0,0,0]), threshold=1e-12):
    """copying this code from anemoi-graphs"""

    x = radius * np.cos(np.deg2rad(xds["latitude"])) * np.cos(np.deg2rad(xds["longitude"]))
    y = radius * np.cos(np.deg2rad(xds["latitude"])) * np.sin(np.deg2rad(xds["longitude"]))
    z = radius * np.sin(np.deg2rad(xds["latitude"]))
    sv = SphericalVoronoi(
        points=np.stack([x,y,z], -1),
        radius=radius,
        center=center,
        threshold=threshold,
    )
    return sv.calculate_areas()


def open_anemoi_dataset(path):

    xds = xr.open_zarr(path)
    vds = utils.expand_anemoi_to_dataset(xds["data"], xds.attrs["variables"])

    for key in ["dates", "latitudes", "longitudes"]:
        vds[key] = xds[key]
    vds = vds.swap_dims({"time": "dates"})
    vds = vds.drop_vars("time")
    vds = vds.rename({"dates": "time"})
    vds = vds.rename({"latitudes": "latitude", "longitudes": "longitude"})
    vds = vds.set_coords(["latitude", "longitude"])
    return vds


def rmse(target, prediction, weights=None):
    result = {}
    for key in prediction.data_vars:
        se = (target[key] - prediction[key])**2
        se = se if weights is None else weights/weights.mean()*se
        mse = se.mean(["cell", "ensemble"])
        result[key] = np.sqrt(mse).compute()

    xds = xr.Dataset(result)
    xds["lead_time"] = xds["time"] - xds["time"][0]
    xds = xds.swap_dims({"time": "lead_time"}).drop_vars("time")
    return xds


def mae(target, prediction, weights=None):
    result = {}
    for key in prediction.data_vars:
        ae = np.abs(target[key] - prediction[key])
        ae = ae if weights is None else weights/weights.mean()*ae
        mae = ae.mean(["cell", "ensemble"])
        result[key] = mae.compute()

    xds = xr.Dataset(result)
    xds["lead_time"] = xds["time"] - xds["time"][0]
    xds = xds.swap_dims({"time": "lead_time"}).drop_vars("time")
    return xds



if __name__ == "__main__":

    project_dir = "/pscratch/sd/t/timothys/nested-eagle/phase-1"

    vds = open_anemoi_dataset(f"{project_dir}/data/global.validation.zarr")
    latlon_weights = get_gridcell_area_weights(vds)

    lead_time = "240h"
    dates = pd.date_range("2018-01-01T06", "2018-12-31T18", freq="54h")
    run_ids = {
        "default": "84ea2581-ccc7-4d18-a18e-69102153cfa0",
        "gmean-residual-stdev": "2d0ee47b-81ca-4898-b2cf-1b6ffa2bb9e9",
        "ones": "9ec834aa-7c3a-4251-b00b-f60028783667",
    }

    drop_these = [
        "cos_julian_day",
        "sin_julian_day",
        "cos_local_time",
        "sin_local_time",
        "cos_latitude",
        "sin_latitude",
        "cos_longitude",
        "sin_longitude",
        "orography",
        "land_sea_mask",
    ]
    for key in drop_these:
        if key in vds:
            vds = vds.drop_vars(key)

    for experiment, rid in run_ids.items():
        rmse_container = None
        mae_container = None
        print(f" --- Starting {experiment} --- ")
        for t0 in dates:
            st0 = t0.strftime("%Y-%m-%dT%H")
            fds = xr.open_dataset(
                f"{project_dir}/loss-scaling/{experiment}/inference-validation/{rid}/{st0}.{lead_time}.nc",
            )

            fds = utils.convert_anemoi_inference(fds)
            for key in drop_these:
                if key in fds:
                    fds = fds.drop_vars(key)

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

            print(f"\tDone with {st0}")
        print(f" --- Done with {experiment} --- ")


        rmse_container.to_netcdf(f"{project_dir}/loss-scaling/{experiment}/validation.rmse.nc")
        mae_container.to_netcdf(f"{project_dir}/loss-scaling/{experiment}/validation.mae.nc")
