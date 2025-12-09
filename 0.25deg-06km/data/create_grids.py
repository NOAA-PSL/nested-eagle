import os
import xarray as xr

from ufs2arco import sources
from ufs2arco.transforms.horizontal_regrid import get_bounds

if __name__ == "__main__":

    store_dir = f"{os.getenv('SCRATCH')}/nested-eagle/0.25deg-06km/data"
    if not os.path.isdir(store_dir):
        os.makedirs(store_dir)

    # Get the parent grid
    src = sources.AWSHRRRArchive(
        t0={"start": "2015-03-15T00", "end": "2015-03-15T06", "freq": "6h"},
        fhr={"start": 0, "end": 0, "step": 6},
        variables=["orog"],
    )
    uds = src.open_sample_dataset(
        dims={"t0": src.t0[0], "fhr": src.fhr[0]},
        open_static_vars=True,
        cache_dir=f"./cache/grid-creation",
    )
    uds = uds.isel(t0=0, drop=True)
    uds = uds.rename({"latitude": "lat", "longitude": "lon"})
    uds = get_bounds(uds)
    uds = uds.rename({"x_vertices": "x_b", "y_vertices": "y_b"})

    # Get 6km grid
    centers = uds.isel(x_b=slice(1,-1,2), y_b=slice(1,-1,2))
    centers = centers.drop_vars(["lat", "lon", "orog"])
    centers = centers.rename({"lat_b": "lat", "lon_b": "lon", "x_b": "x", "y_b": "y"})

    bounds = uds.isel(x_b=slice(0,-1,2), y_b=slice(0,-1,2))
    bounds = bounds.drop_vars(["lat", "lon", "orog"])


    cds = xr.merge([centers, bounds])
    cds.to_netcdf(f"{store_dir}/hrrr_06km.nc")

