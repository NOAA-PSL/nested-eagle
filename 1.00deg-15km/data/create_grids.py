"""
Create a 1 degree global grid to coarsen the 1/4 degree data to
"""

import os
import xesmf
import cf_xarray as cfxr

from ufs2arco import sources

if __name__ == "__main__":


    # global
    store_dir = f"{os.environ['SCRATCH']}/nested-eagle/v0/data"
    if not os.path.isdir(store_dir):
        os.makedirs(store_dir)

    ds = xesmf.util.grid_global(1, 1, cf=True, lon1=360)
    ds = ds.drop_vars("latitude_longitude")

    # GFS goes north -> south
    ds = ds.sortby("lat", ascending=False)

    ds.to_netcdf(f"{store_dir}/global_one_degree.nc")

    # hrrr
    hrrr = sources.AWSHRRRArchive(
        t0={"start": "2015-01-15T00", "end": "2015-01-15T06", "freq": "6h"},
        fhr={"start": 0, "end": 0, "step": 6},
        variables=["orog"],
    )
    hds = hrrr.open_sample_dataset(
        dims={"t0": hrrr.t0[0], "fhr": hrrr.fhr[0]},
        open_static_vars=True,
        cache_dir=f"{store_dir}/cache/grid-creation",
    )
    hds = hds.rename({"latitude": "lat", "longitude": "lon"})

    # Get bounds as vertices
    hds = hds.cf.add_bounds(["lat", "lon"])

    for key in ["lat", "lon"]:
        corners = cfxr.bounds_to_vertices(
            bounds=hds[f"{key}_bounds"],
            bounds_dim="bounds",
            order=None,
        )
        hds = hds.assign_coords({f"{key}_b": corners})
        hds = hds.drop_vars(f"{key}_bounds")

    hds = hds.rename({"x_vertices": "x_b", "y_vertices": "y_b"})

    # Get the nodes and bounds by subsampling carefully...
    # see notebooks
    hds = hds.isel(
        x=slice(None, -4, None),
        y=slice(None, -4, None),
        x_b=slice(None, -4, None),
        y_b=slice(None, -4, None),
    )
    chds = hds.isel(
        x=slice(2, None, 5),
        y=slice(2, None, 5),
        x_b=slice(0, None, 5),
        y_b=slice(0, None, 5),
    )
    chds = chds.drop_vars("orog")

    chds.to_netcdf(f"{store_dir}/hrrr_15km.nc")

