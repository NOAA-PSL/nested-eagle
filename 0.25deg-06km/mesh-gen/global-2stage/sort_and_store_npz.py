"""
Read the unsorted nc file, sort is we would with anemoi, store it as an npz file
"""
import xarray as xr
import numpy as np

from anemoi.graphs.generate.utils import get_coordinates_ordering

if __name__ == "__main__":

    for stage in [1, 2]:

        xds = xr.open_dataset(f"latent.stage{stage}.global_quarter_degree.unsorted.nc")
        glon, glat = np.meshgrid(xds["lon"], xds["lat"])

        coords = np.stack([glon.flatten(), glat.flatten()], axis=-1)
        order = get_coordinates_ordering(coords)
        glon = coords[order, 0]
        glat = coords[order, 1]

        np.savez(f"latent.stage{stage}.global_quarter_degree.sorted.npz", lon=glon, lat=glat)
