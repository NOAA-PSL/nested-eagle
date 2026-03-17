import numpy as np
import matplotlib.pyplot as plt
import xarray as xr

from ufs2arco.transforms.horizontal_regrid import horizontal_regrid

if __name__ == "__main__":

    a2020 = xr.open_zarr(
        "s3://noaa-nws-aorc-v1-1-1km/2020.zarr",
        storage_options={"s3":{"anon":True}},
        decode_timedelta=True,
    )

    xds = a2020.isel(time=0)[["APCP_surface"]].load()

    result = horizontal_regrid(
        xds,
        target_grid_path="/pscratch/sd/t/timothys/nested-eagle/0.25deg-06km/data/hrrr_06km.nc",
        regridder_kwargs={
            "method": "conservative",
            "unmapped_to_nan": True,
            "reuse_weights": True,
            "filename": "/pscratch/sd/t/timothys/nested-eagle/0.25deg-06km/baselines/aorc/conservative_weights.nc",
        },
    )

    mask = ~np.isnan(apcp.APCP_surface)
    mask.attrs = {
        "long_name": "AORC Mask, True where data is present",
        "description": "Obtained by regridding one time slice, and grabbing NaNs",
        "note": "We probably don't have NaNs in the temporally accumulated result because of the accumulation operation",
    }
    mask = mask.drop_vars("time")
    mask.name = "aorc_mask"
    mask = mask.to_dataset()
    mask["x"] = xr.DataArray(
        np.arange(len(mask.x)),
        dims=("x",),
    )
    mask["y"] = xr.DataArray(
        np.arange(len(mask.y)),
        dims=("y",),
    )

    mask.to_netcdf("/pscratch/sd/t/timothys/nested-eagle/0.25deg-06km/baselines/aorc/mask.nc")
