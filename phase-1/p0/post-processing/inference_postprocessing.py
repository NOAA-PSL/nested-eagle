import xarray as xr
import numpy as np
import pandas as pd
import xesmf as xe
from typing import List, Optional, Tuple, Union
import yaml
from pathlib import Path
import sys
from datetime import datetime


def open_raw_inference(path_to_raw_inference: str) -> xr.Dataset:
    """
    Open one Anemoi-Inference run.

    Args:
        path_to_raw_inference (str): Path to an Anemoi-Inference run.

    Returns:
        xr.Dataset: One initialization of an Anemoi-Inference run.
    """
    return xr.open_dataset(path_to_raw_inference)


def mask_values(
    area_to_return: str, ds_nested: xr.Dataset, lam_index: int
) -> xr.Dataset:
    """
    Mask dataset values based on LAM coordinates.

    Args:
        area_to_return (str): Either "lam" or "global" to specify which area to return.
        ds (xr.Dataset): Input nested dataset to mask.
        lam_index (int): Index where nested ds transitions from LAM->global.

    Returns:
        xr.Dataset: Masked dataset containing either LAM only ds or global only (lam missing) ds.
    """
    if area_to_return == "lam":
        return ds_nested.where(ds_nested["values"] < lam_index, drop=True)
    elif area_to_return == "global":
        return ds_nested.where(ds_nested["values"] >= lam_index, drop=True)
    else:
        raise ValueError("area_to_return must be either 'lam' or 'global'")


def create_2D_grid(
    ds: xr.Dataset,
    vars_of_interest: List[str],
    lcc_info: dict = None,
) -> xr.Dataset:
    """
    Reshape dataset from 1D 'values' dimension to 2D latitude and longitude.

    Args:
        ds (xr.Dataset): Anemoi dataset with a flattened "values" dimension.
        vars_of_interest (List[str]): Variables to reshape.
        lcc_info (dict): Necesary info about LCC configuation.

    Returns:
        xr.Dataset: Dataset with shape (time, latitude, longitude).
    """
    ds_to_reshape = ds.copy()

    if lcc_info:
        lat_length = lcc_info["lat_length"]
        lon_length = lcc_info["lon_length"]

        time_length = len(ds_to_reshape["time"].values)

        ds_to_reshape["x"] = np.arange(0, lon_length)
        ds_to_reshape["y"] = np.arange(0, lat_length)

        lats = ds_to_reshape["latitude"][:].values.reshape((lat_length, lon_length))
        lons = ds_to_reshape["longitude"][:].values.reshape((lat_length, lon_length))

        data_vars = {}
        for v in vars_of_interest:
            reshaped_var = ds_to_reshape[v].values.reshape(
                (time_length, lat_length, lon_length)
            )
            data_vars[v] = (["time", "y", "x"], reshaped_var)

        reshaped = xr.Dataset(
            data_vars=data_vars, coords={"time": ds_to_reshape["time"].values}
        )
        reshaped["latitude"] = (("y", "x"), lats)
        reshaped["longitude"] = (("y", "x"), lons)

    else:
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

    return make_contiguous(reshaped)


def make_contiguous(
    reshaped,
):
    """
    xesmf was complaining about array not being in C format?
    apparently just a performance issue - but was tired of getting the warnings :)
    """
    for var in reshaped.data_vars:
        reshaped[var].data = np.ascontiguousarray(reshaped[var].values)
    for coord in reshaped.coords:
        if coord not in reshaped.dims:
            reshaped = reshaped.assign_coords(
                {coord: np.ascontiguousarray(reshaped[coord].values)}
            )
    return reshaped


def add_level_dim_for_individual_var(
    ds: xr.Dataset, var: str, levels: List[int]
) -> xr.Dataset:
    """
    Add level dimensions instead of flattened variables (e.g. geopotential_500, geopotential_800)

    Args:
        ds (xr.Dataset): Input dataset.
        var (str): Variable name to process.
        levels (List[int]): List of levels to process.

    Returns:
        xr.Dataset: Dataset with added level dimension for the specified variables.
    """
    var_level_list = []
    names_to_drop = []

    for level in levels:
        var_name = f"{var}_{str(level)}"
        var_level_list.append(ds[var_name])
        names_to_drop.append(var_name)

    stacked = xr.concat(var_level_list, dim="level")
    stacked = stacked.assign_coords(level=levels)
    ds[var] = stacked

    return ds.drop_vars(names_to_drop)


def add_level_dim(
    ds: xr.Dataset, level_variables: List[str], levels: List[int]
) -> xr.Dataset:
    """
    Wrapper function to add level dimension for all relevant variables.

    Args:
        ds (xr.Dataset): Input dataset.
        level_variables (List[str]): List of variables that have levels.
        levels (List[int]): List of levels to process.

    Returns:
        xr.Dataset: Dataset with added level dimensions for all variables.
    """
    for var in level_variables:
        ds = add_level_dim_for_individual_var(ds=ds, var=var, levels=levels)
    return ds


def final_steps(ds: xr.Dataset, time: xr.DataArray) -> xr.Dataset:
    """
    Add helpful attributes and reorganize dimensions for verification pipelines.

    Args:
        ds (xr.Dataset): Input dataset.
        time (xr.DataArray): Time coordinate.

    Returns:
        xr.Dataset: Dataset with necessary attributes for verification pipelines.
    """
    ds = ds.assign_coords(time=time)
    ds.attrs["forecast_reference_time"] = str(ds["time"][0].values)
    if {"x", "y"}.issubset(ds.dims):
        return ds.transpose("time", "level", "y", "x").rename(
            {"x": "longitude", "y": "latitude"}
        )
    elif {"latitude", "longitude"}.issubset(ds.dims):
        return ds.transpose("time", "level", "latitude", "longitude")


def regrid_ds(ds_to_regrid: xr.Dataset, ds_out: xr.Dataset) -> xr.Dataset:
    """
    Regrid a dataset.

    Args:
        ds_to_regrid (xr.Dataset): Input dataset to regrid.
        ds_out (xr.Dataset): Target grid.

    Returns:
        xr.Dataset: Regridded dataset.
    """
    regridder = xe.Regridder(
        ds_to_regrid,
        ds_out,
        method="bilinear",
        unmapped_to_nan=True,  # this makes sure anything out of conus is nan instead of zero when regridding conus only
    )
    return regridder(ds_to_regrid)


def get_conus_ds_out(
    global_ds: xr.Dataset,
    conus_ds: xr.Dataset,
    global_info: dict,
) -> xr.Dataset:
    """
    Create conus dataset on global grid.
    This will then be used for regridding high-res conus to global res.
    That will then be inserted into global domain so it's all the same resolution for verification.

    Args:
        global_ds (xr.Dataset): Global dataset.
        conus_ds (xr.Dataset): CONUS dataset.
        global_info (dict): Necessary information for global grid.

    Returns:
        xr.Dataset: Output dataset with CONUS grid.
    """
    res = global_info["res"]
    lat_min = global_info["lat_min"]
    lon_min = global_info["lon_min"]
    lat_max = global_info["lat_max"]
    lon_max = global_info["lon_max"]

    return xr.Dataset(
        {
            "latitude": (
                ["latitude"],
                np.arange(lat_min, lat_max, res),
                {"units": "degrees_north"},
            ),
            "longitude": (
                ["longitude"],
                np.arange(lon_min, lon_max, res),
                {"units": "degrees_east"},
            ),
        }
    )


def flatten_grid(ds_to_flatten: xr.Dataset, vars_of_interest: List[str]) -> xr.Dataset:
    """
    Flatten a 2D lat-lon gridded dataset back to a 1D 'values' coordinate.
    This is necessary to eventually combine global and conus back together
        after high-res conus has been regridded to global res.

    Args:
        ds_to_flatten (xr.Dataset): Dataset with 2D lat/lon grid.
        vars_of_interest (List[str]): Variables to flatten.

    Returns:
        xr.Dataset: Flattened dataset with 'values' dimension.
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
        for t in range(len(ds_to_flatten["time"].values)):
            arr = ds_to_flatten[v][t, :].values.reshape(-1)
            reshaped_array.append(arr)
        reshaped_array = np.array(reshaped_array)
        data_vars[v] = (["time", "values"], reshaped_array)

    data_vars["latitude"] = ("values", lats)
    data_vars["longitude"] = ("values", lons)

    ds = xr.Dataset(data_vars=data_vars)

    return ds.dropna(dim="values", subset=[vars_of_interest[0]], how="any")


def combine_lam_w_global(
    ds_nested_w_lam_cutout: xr.Dataset, ds_lam_w_global_res: xr.Dataset
) -> xr.Dataset:
    """
    Combine LAM (regridded to global res) and global regions into a single dataset.

    Args:
        ds_nested_w_lam_cutout (xr.Dataset): Global portion of dataset.
        ds_lam_w_global_res (xr.Dataset): Regridded LAM portion of dataset.

    Returns:
        xr.Dataset: Combined dataset.
    """
    return xr.concat([ds_nested_w_lam_cutout, ds_lam_w_global_res], dim="values")


def postprocess_lam_only(
    ds_nested: xr.Dataset,
    lam_index: int,
    vars_of_interest: List[str],
    level_variables: List[str],
    levels: List[int],
    lcc_info: bool,
) -> xr.Dataset:
    """
    Postprocess LAM-only data.

    Args:
        ds_nested (xr.Dataset): Nested dataset.
        lam_index (int): Index where nested ds transitions from LAM->global.
        vars_of_interest (List[str]): All variables to process.
        level_variables (List[str]): Variables that have levels.
        levels (List[int]): List of levels to process.
        lcc_info (bool): Flag if lcc_flag grid (e.g. HRRR).

    Returns:
        xr.Dataset: Processed LAM dataset ready for verification :)
    """
    time = ds_nested["time"]

    ds_lam = mask_values(area_to_return="lam", ds_nested=ds_nested, lam_index=lam_index)
    ds_lam = create_2D_grid(
        ds=ds_lam, vars_of_interest=vars_of_interest, lcc_info=lcc_info
    )
    ds_lam = add_level_dim(ds=ds_lam, level_variables=level_variables, levels=levels)
    ds_lam = final_steps(ds=ds_lam, time=time)

    return ds_lam


def postprocess_global(
    ds_nested: xr.Dataset,
    lam_index: int,
    vars_of_interest: List[str],
    level_variables: List[str],
    levels: List[int],
    lcc_info: dict,
    global_info: dict,
) -> xr.Dataset:
    """
    Postprocess global data.
    This will output a global ds, and the LAM region has been regridded to global res within it.

    Args:
        ds_nested (xr.Dataset): Nested dataset.
        lam_index (int): Index where nested ds transitions from LAM->global.
        vars_of_interest (List[str]): All variables to process.
        level_variables (List[str]): Variables that have levels.
        levels (List[int]): List of levels to process.
        lcc_info (dict): Necessary information for a LCC grid.
        lcc_info (dict): Necessary information for the global grid.

    Returns:
        xr.Dataset: Post-processed global dataset.
    """
    time = ds_nested["time"]

    # create lam only ds and global only ds (lam has been cut out)
    lam_ds = mask_values(area_to_return="lam", ds_nested=ds_nested, lam_index=lam_index)
    global_ds = mask_values(
        area_to_return="global", ds_nested=ds_nested, lam_index=lam_index
    )

    # take lam from 1D to 2D (values dim -> lat/lon or x/y dims)
    lam_ds = create_2D_grid(
        ds=lam_ds, vars_of_interest=vars_of_interest, lcc_info=lcc_info
    )

    # create blank grid over conus that matches global resolution
    ds_out_conus = get_conus_ds_out(global_ds, lam_ds, global_info=global_info)

    # regrid lam to match global resolution
    lam_ds_regridded = regrid_ds(ds_to_regrid=lam_ds, ds_out=ds_out_conus)

    # flatten regridded lam back to 1D (lat/lon dims -> values dim)
    # necessary to concat it back to global grid
    ds_lam_regridded_flattened = flatten_grid(
        ds_to_flatten=lam_ds_regridded, vars_of_interest=vars_of_interest
    )

    # combine global ds and regridded lam ds together
    ds_combined = combine_lam_w_global(
        ds_nested_w_lam_cutout=global_ds, ds_lam_w_global_res=ds_lam_regridded_flattened
    )

    # go back to 2D again (lots of gynmastics here!!)
    ds_combined = create_2D_grid(ds=ds_combined, vars_of_interest=vars_of_interest)

    # some final postprocessing steps to make the file more user friendly for users/verification
    ds_combined = add_level_dim(
        ds=ds_combined, level_variables=level_variables, levels=levels
    )

    ds_combined = final_steps(ds=ds_combined, time=time)

    return ds_combined


def run(
    initialization: pd.Timestamp,
    config,
):
    """
    Run full pipeline.

    """
    vars_of_interest = config["vars_of_interest"]
    level_variables = config["level_variables"]
    levels = config["levels"]
    lam_index = config["lam_index"]
    lcc_info = config["lcc_info"]
    global_info = config["global_info"]
    raw_inference_files_base_path = config["raw_inference_files_base_path"]

    file_date = datetime.fromisoformat(initialization).strftime("%Y-%m-%dT%H")
    file_name = f"{file_date}.240h"

    ds_nested = open_raw_inference(
        path_to_raw_inference=f"{raw_inference_files_base_path}/{file_name}.nc"
    )

    lam_ds = postprocess_lam_only(
        ds_nested=ds_nested,
        lam_index=lam_index,
        vars_of_interest=vars_of_interest,
        level_variables=level_variables,
        levels=levels,
        lcc_info=lcc_info,
    )

    lam_ds.to_netcdf(f"lam_{file_name}.nc")

    global_ds = postprocess_global(
        ds_nested=ds_nested,
        lam_index=lam_index,
        vars_of_interest=vars_of_interest,
        level_variables=level_variables,
        levels=levels,
        lcc_info=lcc_info,
        global_info=global_info,
    )

    global_ds.to_netcdf(f"global_{file_name}.nc")

    # TODO - revisit if this is how we want to be saving files out?

    return


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python inference_postprocessing.py <config>")
        print("Example: python inference_postprocessing.py config.yaml")
        sys.exit(1)

    config_path = sys.argv[1]

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    start_date = config["initializations_to_run"]["start"]
    end_date = config["initializations_to_run"]["end"]
    freq = config["initializations_to_run"]["freq"]

    dates = pd.date_range(start=start_date, end=end_date, freq=freq)
    for i in dates:
        run(
            initialization=str(i),
            config=config,
        )
