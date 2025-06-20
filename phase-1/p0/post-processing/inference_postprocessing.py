import xarray as xr
import numpy as np
import pandas as pd
import xesmf as xe
from typing import List, Optional
import yaml
import sys
from datetime import datetime


def load_config(config_path: str) -> dict:
    """
    Load configuration from YAML file.

    Args:
        config_path (str): Path to config file.

    Returns:
        dict: Config.
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    return config


def open_raw_inference(path_to_raw_inference: str) -> xr.Dataset:
    """
    Open one Anemoi-Inference run.

    Args:
        path_to_raw_inference (str): Path to an Anemoi-Inference run.

    Returns:
        xr.Dataset: One initialization of an Anemoi-Inference run.
    """
    return xr.open_dataset(path_to_raw_inference)


def open_static_lam(path_to_lam_file: Optional[str] = None) -> xr.Dataset:
    """
    Open a static LAM file used to mask nested files into lam/global.

    Args:
        path_to_lam_file (Optional[str]): Path to a static LAM file.

    Returns:
        xr.Dataset: Static grid file representing LAM domain.

    TODO - run inference on the fly with "extract_lam" to create it if user has not created it themselves yet.
    """
    if path_to_lam_file:
        return xr.open_dataset(path_to_lam_file)
    raise ValueError("No LAM File Found - Automatic generation not yet implemented")


def mask_values(
    area_to_return: str, ds_nested: xr.Dataset, ds_static_lam: xr.Dataset
) -> xr.Dataset:
    """
    Mask dataset values based on LAM coordinates.

    Args:
        area_to_return (str): Either "lam" or "global" to specify which area to return.
        ds (xr.Dataset): Input nested dataset to mask.
        static_lam (xr.Dataset): Static LAM dataset containing LAM coordinates.

    Returns:
        xr.Dataset: Masked dataset containing either LAM only ds or global only (lam missing) ds.
    """
    lam_coords = set(
        zip(ds_static_lam["latitude"].values, ds_static_lam["longitude"].values)
    )
    ds_coords = list(zip(ds_nested["latitude"].values, ds_nested["longitude"].values))
    mask = np.array([(lat, lon) in lam_coords for lat, lon in ds_coords])

    if area_to_return == "lam":
        masked_indices = ds_nested["values"].where(mask).values
        masked_indices = masked_indices[~np.isnan(masked_indices)].astype(int)
        return ds_nested.sel(values=masked_indices)
    elif area_to_return == "global":
        masked_indices = ds_nested["values"].where(~mask).values
        masked_indices = masked_indices[~np.isnan(masked_indices)].astype(int)
        return ds_nested.sel(values=masked_indices)
    else:
        raise ValueError("area_to_return must be either 'lam' or 'global'")


def create_2D_grid(
    ds: xr.Dataset,
    vars_of_interest: List[str],
    curvilinear: bool = False,
) -> xr.Dataset:
    """
    Reshape dataset from 1D 'values' dimension to 2D latitude and longitude.

    Args:
        ds (xr.Dataset): Anemoi dataset with a flattened "values" dimension.
        vars_of_interest (List[str]): Variables to reshape.
        curvilinear (bool): Flag for curvilinear grid or not.

    Returns:
        xr.Dataset: Dataset with shape (time, latitude, longitude).
    """
    ds_to_reshape = ds.copy()

    if curvilinear:
        # hard coding hrrr dimensions in for the time being :)
        # TODO -- definitely do not do that ^^^ :)
        lat_length = 1059
        lon_length = 1799
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


def add_attrs(ds: xr.Dataset, time: xr.DataArray) -> xr.Dataset:
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
    return ds.transpose("time", "level", "latitude", "longitude")


def determine_global_resolution(global_ds: xr.Dataset) -> float:
    """
    Determine the resolution of the global dataset.

    Args:
        global_ds (xr.Dataset): Global dataset.

    Returns:
        float: Resolution in degrees. This should typically be 0.25 or 1.00.
    """
    # an attempt to determine global res on the fly.
    # TODO - can probably come up with a more robust way to do this.
    # but this should work for any grid that is a simple rectilinear 1 or 0.25 degree.
    lon = np.unique(global_ds["longitude"])
    res = np.abs(np.diff(lon)).min()
    return res


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


def get_conus_ds_out(global_ds: xr.Dataset, conus_ds: xr.Dataset) -> xr.Dataset:
    """
    Create conus dataset on global grid.
    This will then be used for regridding high-res conus to global res.
    That will then be inserted into global domain so it's all the same resolution for verification.

    Args:
        global_ds (xr.Dataset): Global dataset.
        conus_ds (xr.Dataset): CONUS dataset.

    Returns:
        xr.Dataset: Output dataset with CONUS grid.
    """
    res = determine_global_resolution(global_ds)
    shift = res / 2

    # Okay the "shift" thing is wonky.
    # I need to come back to this. I think it is an artifact of how we regriddeed ERA5 in p1 but am unsure.
    # It actually may be necessary to make sure we don't get overlapping pixels on boundaries of CONUS/global.
    # Either way, it works right now. I need to dig into it a little more.
    lat_min = conus_ds["latitude"].min() + shift
    lon_min = conus_ds["longitude"].min() + shift
    lat_max = conus_ds["latitude"].max() + shift
    lon_max = conus_ds["longitude"].max() + shift

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

    return xr.Dataset(data_vars=data_vars)


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
    ds_lam: xr.Dataset,
    vars_of_interest: List[str],
    level_variables: List[str],
    levels: List[int],
    curvilinear: bool,
) -> xr.Dataset:
    """
    Postprocess LAM-only data.

    Args:
        ds_nested (xr.Dataset): Nested dataset.
        ds_lam (xr.Dataset): LAM dataset.
        vars_of_interest (List[str]): All variables to process.
        level_variables (List[str]): Variables that have levels.
        levels (List[int]): List of levels to process.
        curvilinear (bool): Flag if curvilinear grid (e.g. HRRR).

    Returns:
        xr.Dataset: Processed LAM dataset ready for verification :)
    """
    time = ds_nested["time"]

    # mask global and return conus only values
    ds_lam = mask_values(
        area_to_return="lam", ds_nested=ds_nested, ds_static_lam=ds_lam
    )

    # go from 1D values to 2D lat/lon dimensions
    ds_lam = create_2D_grid(
        ds=ds_lam, vars_of_interest=vars_of_interest, curvilinear=curvilinear
    )

    # add a few formatting steps to make file more user friendly and ready for verification.
    ds_lam = add_level_dim(ds=ds_lam, level_variables=level_variables, levels=levels)
    ds_lam = add_attrs(ds=ds_lam, time=time)

    return ds_lam


def postprocess_global(
    ds_nested: xr.Dataset,
    static_lam: xr.Dataset,
    vars_of_interest: List[str],
    level_variables: List[str],
    levels: List[int],
    curvilinear: bool,
) -> xr.Dataset:
    """
    Postprocess global data.
    This will output a global ds, and the LAM region has been regridded to global res within it.

    Args:
        ds_nested (xr.Dataset): Nested dataset.
        static_lam (xr.Dataset): Static LAM dataset.
        vars_of_interest (List[str]): All variables to process.
        level_variables (List[str]): Variables that have levels.
        levels (List[int]): List of levels to process.
        curvilinear (bool): Flag if curvilinear grid (e.g. HRRR).

    Returns:
        xr.Dataset: Post-processed global dataset.
    """
    time = ds_nested["time"]

    # return lam only ds
    lam_ds = mask_values(
        area_to_return="lam", ds_nested=ds_nested, ds_static_lam=static_lam
    )
    # return global only ds (lam has been cut out)
    global_ds = mask_values(
        area_to_return="global", ds_nested=ds_nested, ds_static_lam=static_lam
    )

    # take lam from 1D to 2D (values dim -> lat/lon dims)
    lam_ds = create_2D_grid(
        ds=lam_ds, vars_of_interest=vars_of_interest, curvilinear=curvilinear
    )
    # create blank grid over conus that matches global resolution
    ds_out_conus = get_conus_ds_out(global_ds, lam_ds)

    # regrid lam to match global resolution
    lam_ds_regridded = regrid_ds(ds_to_regrid=lam_ds, ds_out=ds_out_conus)

    # flatten regridded lam back to 1D (lat/lon dims -> values dim)
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
    ds_combined = add_attrs(ds=ds_combined, time=time)

    return ds_combined


def run(
    initialization: pd.Timestamp,
    vars_of_interest: list[str],
    level_variables: list[str],
    levels: list[int],
    path_to_static_lam: str,
    raw_inference_files_base_path: str,
    curvilinear: bool,
):
    """
    Run full pipeline.

    """
    i = datetime.fromisoformat(initialization).strftime("%Y%m%dT%H%M%SZ")

    ds_nested = open_raw_inference(
        path_to_raw_inference=f"{raw_inference_files_base_path}/{i}.nc"
    )
    static_lam = open_static_lam(path_to_lam_file=path_to_static_lam)

    lam_ds = postprocess_lam_only(
        ds_nested=ds_nested,
        ds_lam=static_lam,
        vars_of_interest=vars_of_interest,
        level_variables=level_variables,
        levels=levels,
        curvilinear=curvilinear,
    )

    global_ds = postprocess_global(
        ds_nested=ds_nested,
        static_lam=static_lam,
        vars_of_interest=vars_of_interest,
        level_variables=level_variables,
        levels=levels,
        curvilinear=curvilinear,
    )

    lam_ds.to_netcdf(f"lam_{i}.nc")
    global_ds.to_netcdf(f"global_{i}.nc")

    # TODO - revisit if this is how we want to be saving files out.

    return


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python inference_postprocessing.py <config>")
        print("Example: python inference_postprocessing.py 'config.yaml")
        sys.exit(1)

    config_path = sys.argv[1]

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    vars_of_interest = config["vars_of_interest"]
    level_variables = config["level_variables"]
    levels = config["levels"]
    path_to_static_lam = config["path_to_static_lam"]
    raw_inference_files_base_path = config["raw_inference_files_base_path"]
    start_date = config["initializations_to_run"]["start"]
    end_date = config["initializations_to_run"]["end"]
    freq = config["initializations_to_run"]["freq"]
    curvilinear_flag = config["curvilinear"]

    dates = pd.date_range(start=start_date, end=end_date, freq=freq)
    for i in dates:
        run(
            initialization=str(i),
            vars_of_interest=vars_of_interest,
            level_variables=level_variables,
            levels=levels,
            path_to_static_lam=path_to_static_lam,
            raw_inference_files_base_path=raw_inference_files_base_path,
            curvilinear=curvilinear_flag,
        )
