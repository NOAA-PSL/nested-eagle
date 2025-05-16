"""
Script to preprocess and regrid nested anemoi output for use with weatherbench.
Produces a zarr file in the desired wbx format and resolution (e.g. 1.5 degrees)
"""

from typing import Tuple
import xarray as xr
import pandas as pd
import xesmf as xe
import numpy as np
from datetime import datetime
import sys
from inference_globals import (
    WBX_TARGET_PATH,
    VARS_OF_INTEREST,
    PATH_TO_LAM_FILE,
    PATH_TO_OUTPUT_ZARR,
    LAM_TARGET_PATH,
)


def clip_to_vars_of_interest(
    ds_nested: xr.Dataset,
    vars_of_interest: list[str] = VARS_OF_INTEREST,
) -> xr.Dataset:
    """
    Clip to variables of interest (plus latitude and longitude).

    Args:
        ds_nested (xr.Dataset): The input nested anemoi dataset from an inference run.
        vars_of_interest (list[str]): List of variable names to keep.

    Returns:
        xr.Dataset -- Filtered dataset with only selected variables.
    """
    var_list = vars_of_interest + ["latitude", "longitude"]
    return ds_nested[var_list]


def extract_data(
    ds_lam: xr.Dataset,
    ds_nested: xr.Dataset,
) -> Tuple[xr.Dataset, xr.Dataset]:
    """
    Extract LAM and global regions from the nested dataset (as two separate datasets) based on LAM spatial bounds.
    TODO - when we switch to GFS/HRRR it wont be a simple bounding box.
    We will probably want to add an option to straight up mask the ds instead of clipping to a box in that case.

    Args:
        ds_lam (xr.Dataset): Static dataset defining LAM region bounds.
        ds_nested (xr.Dataset): Full nested anemoi-dataset.

    Returns:
        Tuple[xr.Dataset, xr.Dataset] -- Separated LAM-only and global-only datasets.
    """

    lat_min, lat_max = ds_lam.latitude.min(), ds_lam.latitude.max()
    lon_min, lon_max = ds_lam.longitude.min(), ds_lam.longitude.max()

    mask = (
        (ds_nested.latitude >= lat_min)
        & (ds_nested.latitude <= lat_max)
        & (ds_nested.longitude >= lon_min)
        & (ds_nested.longitude <= lon_max)
    )

    ds_lam_only = ds_nested.copy()
    for var in ds_lam_only.data_vars:
        ds_lam_only[var] = ds_lam_only[var].where(mask, drop=False)

    ds_global_only = ds_nested.copy()
    for var in ds_global_only.data_vars:
        ds_global_only[var] = ds_global_only[var].where(~mask, drop=False)

    return ds_lam_only.dropna(dim="values"), ds_global_only.dropna(dim="values")


def create_2D_grid(
    ds: xr.Dataset,
    vars_of_interest: list[str] = VARS_OF_INTEREST,
) -> xr.Dataset:
    """
    Reshape dataset from 1D 'values' dimension to 2D latitude and longitude.
    Xesmf isn't able to regrid the 1D "values" (at least I couldnt' get it to), so this creates 2D arrays.

    Args:
        ds (xr.Dataset): Anemoi dataset with a flattened "values" dimension.
        vars_of_interest (list): Variables to reshape.

    Returns:
        xr.Dataset -- Dataset with shape (time, latitude, longitude).
    """
    ds_to_reshape = ds.copy()

    lats = ds_to_reshape.latitude.values
    lons = ds_to_reshape.longitude.values
    sort_index = np.lexsort((lons, lats))
    ds_to_reshape = ds_to_reshape.isel(values=sort_index)

    lat_length = len(np.unique(ds_to_reshape.latitude.values))
    lon_length = len(np.unique(ds_to_reshape.longitude.values))
    time_length = len(ds["time"].values)

    lats = ds_to_reshape["latitude"][:].values.reshape((lat_length, lon_length))
    lons = ds_to_reshape["longitude"][:].values.reshape((lat_length, lon_length))
    lat_1d = lats[:, 0]
    lon_1d = lons[0, :]

    data_vars = {}
    for v in vars_of_interest:
        reshaped_var = ds_to_reshape[v].values.reshape(
            (time_length, lat_length, lon_length)
        )
        data_vars[v] = (["time", "latitude", "longitude"], reshaped_var)

    reshaped = xr.Dataset(
        data_vars=data_vars, coords={"latitude": lat_1d, "longitude": lon_1d}
    )

    # xesmf was complaining about array not being in C format?
    # apparently just a performance issue - but was tired of getting the warnings.
    for var in reshaped.data_vars:
        reshaped[var].data = np.ascontiguousarray(reshaped[var].values)
    for coord in reshaped.coords:
        if coord not in reshaped.dims:
            reshaped = reshaped.assign_coords(
                {coord: np.ascontiguousarray(reshaped[coord].values)}
            )

    return reshaped


def regrid_ds(
    ds_to_regrid: xr.Dataset,
    ds_out: xr.Dataset,
    lam=False,
    vars_of_interest: list[str] = VARS_OF_INTEREST,
) -> xr.Dataset:
    """
    Regrid a dataset.

    Args:
        ds_to_regrid (xr.Dataset): Input dataset to regrid.
        ds_out (xr.Dataset): Target grid.
        lam (bool): If LAM then nans are dropped outside of LAM domain. Default=False.

    Returns:
        xr.Dataset -- Regridded dataset.
    """
    regridder = xe.Regridder(
        ds_to_regrid,
        ds_out,
        method="bilinear",
        unmapped_to_nan=True,
    )
    ds_regridded = regridder(ds_to_regrid)

    if lam:
        return ds_regridded.where(
            ~np.isnan(ds_regridded[vars_of_interest[0]]), drop=True
        )

    else:
        return ds_regridded


def flatten_grid(
    ds_to_flatten: xr.Dataset,
    vars_of_interest: list[str] = VARS_OF_INTEREST,
) -> xr.Dataset:
    """
    Flatten a 2D lat-lon gridded dataset to a 1D 'values' coordinate (aenmoi-format)
    We need to flatten back to 1D "values" at points to combine nested dataset back together easily,
        once on the same resolution.

    Args:
        ds_to_flatten (xr.Dataset): Dataset with 2D lat/lon grid.
        vars_of_interest (list): Variables to flatten.

    Returns:
        xr.Dataset -- Flattened dataset with 'values' dimension like an anemoi-dataset.
    """
    reshaped_lat_lon = np.meshgrid(
        ds_to_flatten["latitude"].values,
        ds_to_flatten["longitude"].values,
    )

    lats = reshaped_lat_lon[0].transpose().reshape(-1)
    lons = reshaped_lat_lon[1].transpose().reshape(-1)

    data_vars = {}
    for v in vars_of_interest:
        reshaped_array = []
        for t in np.arange(len(ds_to_flatten["time"].values)):
            arr = ds_to_flatten[v][t, :].values.reshape(-1)
            reshaped_array.append(arr)
        reshaped_array = np.array(reshaped_array)
        data_vars[v] = (["time", "values"], reshaped_array)

    data_vars["latitude"] = ("values", lats)
    data_vars["longitude"] = ("values", lons)

    ds_flattened = xr.Dataset(
        data_vars=data_vars,
    )

    return ds_flattened


def combine_lam_w_global(
    ds_nested_w_lam_cutout: xr.Dataset,
    ds_lam_w_global_res: xr.Dataset,
) -> xr.Dataset:
    """
    Concatenate LAM and global regions into a single dataset,
    since they are now on same grid.

    Args:
        ds_nested_w_lam_cutout (xr.Dataset): Global portion of dataset.
        ds_lam_w_global_res (xr.Dataset): Regridded LAM portion of dataset.

    Returns:
        xr.Dataset -- Combined dataset along the 'values' dimension.
    """
    ds = xr.concat([ds_nested_w_lam_cutout, ds_lam_w_global_res], dim="values")
    return ds


def open_target_ds_for_regridding(
    path_to_file: str,
) -> xr.Dataset:
    """
    Open target grid that will become ds_out for xesmf.

    Args:
        path_to_file (str): Path to open grid.

    Returns:
        xr.Dataset -- Dataset used as ds_out for xesmf.
    """
    if path_to_file.endswith(".zarr"):
        ds_out = xr.open_zarr(
            path_to_file,
            chunks=None,
            storage_options=dict(token="anon"),
        )
        try:
            ds_out = ds_out.rename(
                {
                    "lat": "latitude",
                    "lon": "longitude",
                }
            )
        except:
            pass
        ds_out = ds_out[["latitude", "longitude"]]

    elif path_to_file.endswith(".nc"):
        ds_out = xr.open_dataset(path_to_file)
        try:
            ds_out = ds_out.rename(
                {
                    "lat": "latitude",
                    "lon": "longitude",
                }
            )
        except:
            pass
        ds_out = ds_out[["latitude", "longitude"]]

    else:
        print("file type not implemented")  # e.g. grid will not work.

    return ds_out


def regrid_for_wbx(
    ds_lam_grid: xr.Dataset,
    ds_nested: xr.Dataset,
    wbx_target_path=WBX_TARGET_PATH,
    lam_target_path=LAM_TARGET_PATH,
) -> xr.Dataset:
    """
    Full regridding pipeline: regrid a nested anemoi dataset to match weatherbench target grid.

    Args:
        ds_lam_grid (xr.Dataset): Static LAM file to define the domain.
        ds_nested (xr.Dataset): Forecast dataset to regrid.
        wbx_target_path (str): Path to weatherbench grid.
        lam_target_path (str): Path to a global resolution to regrid the lam to global res.

    Returns:
        xr.Dataset -- Regridded dataset aligned to weatherbench grid.
    """
    ds_nested = clip_to_vars_of_interest(ds_nested=ds_nested)

    ds_lam, ds_global = extract_data(
        ds_lam=ds_lam_grid,
        ds_nested=ds_nested,
    )

    ds_lam_2d = create_2D_grid(
        ds=ds_lam,
    )

    ds_out_lam = open_target_ds_for_regridding(lam_target_path)

    ds_lam_regridded = regrid_ds(
        ds_to_regrid=ds_lam_2d,
        ds_out=ds_out_lam,
        lam=True,
    )

    ds_lam = flatten_grid(ds_to_flatten=ds_lam_regridded)

    ds_combined = combine_lam_w_global(
        ds_nested_w_lam_cutout=ds_global, ds_lam_w_global_res=ds_lam
    )

    ds_combined = create_2D_grid(
        ds=ds_combined,
    )

    ds_out_wbx = open_target_ds_for_regridding(wbx_target_path)

    ds_regridded = regrid_ds(
        ds_to_regrid=ds_combined,
        ds_out=ds_out_wbx,
    )

    return ds_regridded


def get_lam_grid(
    path_to_lam_file: str,
) -> xr.Dataset:
    """
    Open a static LAM file.
    This will be used to clip nested files into lam/global.

    Args:
        path_to_lam_file (str): Path to a static LAM file.

    Returns:
        xr.Dataset -- Static grid file for LAM domain.
    """
    if path_to_lam_file:
        ds = xr.open_dataset(path_to_lam_file)
    else:
        from run_inference import run_inference

        # create a lam.nc file on the fly.
        # this will probably need to be updated in the future, but works for now if you haven't created your own
        #      static lam file. Wanted some functionality if someone did not save out a static lam file yet.
        run_inference(
            init_date="2018-01-10",  # this needs to not be hard-coded (TODO). works rn because our val set is in this.
            extract_lam="True",
        )
        ds = xr.open_dataset("lam.nc")

    return ds


def create_zarr_for_wbx(
    dates: pd.date_range,
    path_to_lam_file: str = PATH_TO_LAM_FILE,
    path_to_output_zarr: str = PATH_TO_OUTPUT_ZARR,
) -> None:
    """
    Main function: read, regrid, and write to Zarr format ready for weatherbench.

    Args:
        dates (pd.date_range): List of dates to run inference for.
        path_to_lam_file (str): Path to LAM static file.
        path_to_output_zarr (str): Path to store zarr output.

    Returns:
        None
    """
    ds_lam_grid = get_lam_grid(path_to_lam_file=path_to_lam_file)

    for idx, date in enumerate(dates):
        dt = datetime.fromisoformat(str(date))
        date_str = dt.strftime("%Y%m%dT%H%M%SZ")

        ds_nested = xr.open_dataset(f"{date_str}.nc")
        ds = regrid_for_wbx(ds_lam_grid=ds_lam_grid, ds_nested=ds_nested)
        ds = ds.rename({"time": "fhr"})

        time_value = np.datetime64(date) + np.timedelta64(idx, "h")
        ds = ds.expand_dims({"time": [time_value]})
        ds = ds.transpose("time", "fhr", "latitude", "longitude")

        encoding = {
            "time": {
                "units": f"hours since {dates[0]}",
                "calendar": "standard",
            }
        }

        if idx == 0:
            print("saving container")
            ds.to_zarr(path_to_output_zarr, mode="w", encoding=encoding)
        elif idx > 0:
            print(f"saving timestep for {date}")
            ds.to_zarr(path_to_output_zarr, mode="a", append_dim="time")


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: create_wbx_zarr.py <start_date>  <end_date>  <freq>")
        print(
            "Example: create_wbx_zarr.py '2018-01-06T00:00:00' '2018-01-08T00:00:00' '12h'"
        )
        sys.exit(1)

    start_date = sys.argv[1]
    end_date = sys.argv[2]
    freq = sys.argv[3]

    dates = pd.date_range(start=start_date, end=end_date, freq=freq)
    create_zarr_for_wbx(dates)
