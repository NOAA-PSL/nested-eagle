"""
Create a 1 degree global grid to coarsen the 1/4 degree data to
"""

import os
import xesmf

if __name__ == "__main__":


    store_dir = f"{os.environ['SCRATCH']}/nested-eagle/phase-1/data"
    if not os.path.isdir(store_dir):
        os.makedirs(store_dir)

    ds = xesmf.util.grid_global(1, 1, cf=True, lon1=360)
    ds = ds.drop_vars("latitude_longitude")

    # the era5 we're using goes north -> south
    # may as well have same ordering in global / nest ???
    ds = ds.sortby("lat", ascending=False)

    ds.to_netcdf(f"{store_dir}/global_one_degree.nc")
