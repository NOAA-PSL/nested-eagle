import sys
import logging

import numpy as np
import xarray as xr
import pandas as pd
import dask.array

sys.path.append("/global/homes/t/timothys/nested-eagle/")
from eagle.log import setup_simple_log

logger = logging.getLogger("eagle")

_n_y = 211 - 10 - 11
_n_x = 359 - 10 - 11

def reshape_dataset(xds):

    nds = xr.Dataset()
    nds["time"] = xds.time
    nds["y"] = xr.DataArray(
        np.arange(_n_y),
        coords={"y": np.arange(_n_y)},
    )
    nds["x"] = xr.DataArray(
        np.arange(_n_x),
        coords={"x": np.arange(_n_x)},
    )
    nds["latitude"] = xr.DataArray(
        xds.latitude.values.reshape((_n_y, _n_x)),
        dims=("y", "x"),
    )
    nds["longitude"] = xr.DataArray(
        xds.longitude.values.reshape((_n_y, _n_x)),
        dims=("y", "x"),
    )
    for key in xds.data_vars:
        nds[key] = xr.DataArray(
            xds[key].values.reshape((len(xds.time),_n_y,_n_x)),
            dims=("time", "y", "x"),
        )
    nds = nds.set_coords(["latitude", "longitude"])
    return nds


def open_dataset(t0: pd.Timestamp):

    st0 = t0.strftime("%Y-%m-%dT%H")
    data_dir = "/pscratch/sd/t/timothys/nested-eagle/1.00deg-15km/mse06h/experiments/training-steps/steps030k/inference-precip"
    xds = xr.open_dataset(
        f"{data_dir}/{st0}.48h.lam.nc",
        decode_timedelta=True,
        chunks="auto",
    )
    xds = xds.set_coords(["latitude", "longitude"])
    xds = xds[["accum_tp"]].load()
    xds = reshape_dataset(xds)


    lead_time = xds["time"] - xds["time"][0]
    xds["fhr"] = xr.DataArray(
        lead_time.values.astype("timedelta64[h]").astype(int),
        coords=xds.time.coords,
        attrs={"description": "forecast hour, lead time in hours"},
    )
    xds = xds.swap_dims({"time": "fhr"}).drop_vars("time")

    xds = xds.expand_dims({"t0": [t0]})

    # chop off initial condition which is zero for diagnostics like total precip
    xds = xds.isel(fhr=slice(1,None))

    # make sure chunks get overwritten
    xds.encoding = {}
    for key in xds.data_vars:
        xds[key].encoding = {}
    return xds


def create_container(xds, t0):

    nds = xr.Dataset(attrs=xds.attrs.copy())

    # dims and coord
    nds["t0"] = xr.DataArray(
        t0,
        coords={"t0": t0},
        dims="t0",
    )
    for key in xds.dims:
        if key != "t0":
            nds[key] = xds[key].copy()
    for key in xds.coords:
        if key not in nds:
            nds = nds.assign_coords({key: xds[key].copy()})

    # empty data vars
    for varname in xds.data_vars:
        dims = xds[varname].dims
        shape = tuple(len(nds[key]) for key in dims)
        chunks = {"t0": 1, "fhr": 1, "y": -1, "x": -1}
        nds[varname] = xr.DataArray(
            data=dask.array.zeros(
                shape=shape,
                chunks=tuple(chunks[key] for key in dims),
                dtype=xds[varname].dtype,
            ),
            dims=dims,
            attrs=xds[varname].attrs.copy(),
        )
    return nds


if __name__ == "__main__":

    setup_simple_log()
    store_path = "/pscratch/sd/t/timothys/nested-eagle/1.00deg-15km/mse06h/experiments/training-steps/steps030k/inference-precip/nested-eagle.conus15km.precip.zarr"


    all_t0 = pd.date_range("2023-02-01T00", "2024-01-30T00", freq="6h")

    # create a container
    template = open_dataset(all_t0[0])
    container = create_container(template, all_t0)
    container.to_zarr(store_path, compute=False, mode="w")
    logger.info(f"Created Container at {store_path}")

    # loop and fill region
    for t0 in all_t0:
        xds = open_dataset(t0)
        region = {}
        for key in xds.dims:
            if key in ("fhr", "y", "x"):
                region[key] = slice(None, None)
            elif key in ("t0",):
                indices = [list(container[key].values).index(value) for value in xds[key].values]
                region[key] = slice(indices[0], indices[-1]+1)
            else:
                raise KeyError("Unrecognized dimension name")

        xds.to_zarr(store_path, region=region)
        logger.info(f"Done with {t0}")
