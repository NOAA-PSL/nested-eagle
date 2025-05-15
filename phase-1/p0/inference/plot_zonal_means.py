import os
import logging

import numpy as np
import matplotlib.pyplot as plt
import xarray as xr

import pandas as pd

from matplotlib.colorbar import ColorbarBase
import cartopy.crs as ccrs
import cmocean

import xesmf

from graphufs.log import setup_simple_log

def plot_t2m(pda, tda):

    fig, axs = plt.subplots(
        2, 1,
        figsize=(10,7),
        constrained_layout=True,
        sharex=True,
    )

    for xda, title, ax in zip(
        [pda, tda],
        ["Prediction: Nested-ERA5", "ERA5"],
        axs,
    ):
        if "values" in xda.dims:
            plotme = nested_zonal_mean(xda)
        else:
            plotme = xda.mean("lon").squeeze()
        plotme = plotme - 273.15
        plotme.plot.contourf(
            ax=ax,
            x="time",
            cmap="cmo.thermal",
            vmin=-10, vmax=30,
            levels=11,
            cbar_kwargs={"label": r"2m Temperature ($^\circ$C)"},
        )
        ax.set(xlabel="", ylabel="Latitude", title=title)
        ax.set(yticks=np.arange(-75,76,15))
        ax.yaxis.grid(True, color='gray', linestyle=':', linewidth=2, alpha=0.5)
    return fig, axs

def plot_z500(pda, tda):

    fig, axs = plt.subplots(
        2, 1,
        figsize=(10,7),
        constrained_layout=True,
        sharex=True,
    )

    for xda, title, ax in zip(
        [pda, tda],
        ["Prediction: Nested-ERA5", "ERA5"],
        axs,
    ):
        if "values" in xda.dims:
            plotme = nested_zonal_mean(xda)
        else:
            plotme = xda.mean("lon").squeeze()
        plotme = plotme / 9.80665 / 1000
        plotme.plot.contourf(
            ax=ax,
            x="time",
            cmap="Spectral_r",
            vmin=4.9, vmax=5.9,
            levels=11,
            cbar_kwargs={"label": r"Z500 (km)"},
        )
        ax.set(xlabel="", ylabel="Latitude", title=title)
        ax.set(yticks=np.arange(-75,76,15))
        ax.yaxis.grid(True, color='gray', linestyle=':', linewidth=2, alpha=0.5)
    return fig, axs

def regrid(pds, tds):
    dsout = xr.Dataset({"lat": tds.lat, "lon": tds.lon})
    regridder = xesmf.Regridder(ds_in=pds, ds_out=dsout, method="bilinear", periodic=False,extrap_method='nearest_s2d')
    ds_out = regridder(pds[["2m_temperature"]].isel(time=0), keep_attrs=True)
    return ds_out

def nested_zonal_mean(xds):


    nds = xr.Dataset()
    lat = np.arange(89.5, -90, -1)
    nds["lat"] = xr.DataArray(
        lat,
        coords={"lat": lat},
        dims=("lat",),
    )

    dslist = []
    for thislat in lat:

        tmp = xds.where(xds["latitude"] == thislat).mean("values")
        tmp = tmp.expand_dims({"lat": [thislat]})
        dslist.append(tmp)

    return xr.concat(dslist, dim="lat")



def main(
    read_path,
    store_dir,
    t0="2019-01-01T00",
    tf="2019-12-31T18",
):

    setup_simple_log()
    fig_dir = f"{store_dir}/figures/zonal-means.{t0}.{tf}/"

    if not os.path.isdir(fig_dir):
        os.makedirs(fig_dir)

    pds = xr.open_dataset(read_path)
    pds = pds.set_coords(["latitude", "longitude"])

    era = xr.open_zarr(
        "gs://weatherbench2/datasets/era5/1959-2023_01_10-wb13-6h-1440x721_with_derived_variables.zarr",
        storage_options={"token": "anon"},
    )
    era = era.sel(
        time=pds.time.values,
    )
    era = era.rename({"latitude": "lat", "longitude": "lon"})

    # 2m temperature
    logging.info(f"Loading true t2m\n{era['2m_temperature']}\n")
    era["2m_temperature"] = era["2m_temperature"].load()
    logging.info(" ... done")

    logging.info(f"Loading predicted t2m\n{pds['2m_temperature']}\n")
    pds["2m_temperature"] = pds["2m_temperature"].load()
    logging.info(" ... done")

    fig, axs = plot_t2m(pds["2m_temperature"], era["2m_temperature"])
    fig.savefig(f"{fig_dir}/2m_temperature.jpeg", bbox_inches="tight", dpi=300)

    # geopotential
    tz500 = era["geopotential"].sel(level=500)
    logging.info(f"Loading true z500\n{tz500}\n")
    tz500 = tz500.load()
    logging.info(" ... done")

    pz500 = pds["geopotential_500"]
    logging.info(f"Loading predicted z500\n{pz500}\n")
    pz500 = pz500.load();
    logging.info(" ... done")

    fig, axs = plot_z500(pds["geopotential_500"], era["geopotential"].sel(level=500))
    fig.savefig(f"{fig_dir}/z500.jpeg", bbox_inches="tight", dpi=300)

if __name__ == "__main__":
    inference_dir = "/pscratch/sd/t/timothys/aneml/nested-conus/era5-nest/inference/c080c4bf-7c5a-4f8d-ae86-b070d4e432e1"
    main(
        read_path=f"{inference_dir}/forecast.4320hr.2019-01-01T00.nc",
        store_dir=inference_dir,
        t0="2019-01-01T00",
        tf="2019-06-30T00",
    )
