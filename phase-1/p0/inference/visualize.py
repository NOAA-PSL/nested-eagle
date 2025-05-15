import os
import sys
import logging

import numpy as np
import xarray as xr
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
import cartopy.crs as ccrs
import cmocean

try:
    import xmovie
except ImportError:
    print("Could not import xmovie, can't use mode='movie'")


_projection = ccrs.Orthographic(
    central_longitude = -120,
    central_latitude = 20,
)

class SimpleFormatter(logging.Formatter):
    def format(self, record):
        record.relativeCreated = record.relativeCreated // 1000
        return super().format(record)

def setup_simple_log(level=logging.INFO):

    logging.basicConfig(
        stream=sys.stdout,
        level=level,
    )
    logger = logging.getLogger()
    formatter = SimpleFormatter(fmt="[%(relativeCreated)d s] [%(levelname)s] %(message)s")
    for handler in logger.handlers:
        handler.setFormatter(formatter)

def get_extend(xds, vmin=None, vmax=None):
    minval = []
    maxval = []
    for key in xds.data_vars:
        minval.append(xds[key].min().values)
        maxval.append(xds[key].max().values)
    minval = np.min(minval)
    maxval = np.max(maxval)
    vmin = minval if vmin is None else vmin
    vmax = maxval if vmax is None else vmax

    extend = "neither"
    if minval < vmin:
        extend = "min"
    if maxval > vmax:
        extend = "max" if extend == "neither" else "both"
    return extend, vmin, vmax

def get_precip_kwargs():
    n = 1
    levels = np.concatenate(
        [
            np.linspace(0, .1, 2*n),
            np.linspace(.1, 1, 5*n),
            np.linspace(1, 10, 5*n),
            np.linspace(10, 50, 3*n),
            #np.linspace(50, 80, 2),
        ],
    )
    norm = BoundaryNorm(levels, len(levels)+1)
    cmap = plt.get_cmap("cmo.rain", len(levels)+1)
    return {"norm": norm, "cmap": cmap, "cbar_kwargs": {"ticks": [0, 1, 10, 50]}}

def nested_scatter(ax, xds, varname, **kwargs):
    n_conus = 38_829
    mappables = []
    for slc, s in zip(
        [slice(None, n_conus), slice(n_conus, None)],
        [1/2, 12],
    ):

        p = ax.scatter(
            xds.longitudes.isel(values=slc),
            xds.latitudes.isel(values=slc),
            c=xds[varname].isel(values=slc),
            s=s,
            transform=ccrs.PlateCarree(),
            **kwargs
        )
        mappables.append(p)

    # Define bounding box corners
    lons = 225, 300
    lats = 21, 53

    kw = {
        "c": "gray",
        "transform": ccrs.PlateCarree(),
        "s": 1,
        "alpha": .3,
    }

    for lon in lons:
        yL = np.arange(*lats, .25)
        xL = np.full_like(yL, lon)
        ax.scatter(xL, yL, **kw)
    for lat in lats:
        xL = np.arange(*lons, .25)
        yL = np.full_like(xL, lat)
        ax.scatter(xL, yL, **kw)
    return mappables

def plot_single_timestamp(xds, fig, time, *args, **kwargs):

    axs = []

    truthname = [y for y in list(xds.data_vars) if y in ("ERA5", "Replay")][0]
    vtime = xds["time"].isel(time=time).values
    stime = str(vtime)[:13]

    # get these extra options
    cbar_kwargs = kwargs.pop("cbar_kwargs", {})
    extend = kwargs.pop("extend", None)
    t0 = kwargs.pop("t0", "")
    truth_x = kwargs.pop("truth_x", None)
    truth_y = kwargs.pop("truth_y", None)


    # Create axes
    ax = fig.add_subplot(1, 2, 1, projection=_projection)

    # Plot Truth
    # Note that this scatter is very slow for movies
    # But... it is the truest comparison to the other plot
    # We could move to datashader eventually
    p = ax.scatter(
        truth_x,
        truth_y,
        c=xds[truthname].isel(time=time),
        s=.5,
        transform=ccrs.PlateCarree(),
        **kwargs,
    )
    # pcolormesh option
    #p = xds[truthname].isel(time=time).plot(
    #    ax=ax,
    #    transform=ccrs.PlateCarree(),
    #    add_colorbar=False,
    #    **kwargs,
    #)
    ax.set(title="ERA5")
    axs.append(ax)

    # Plot model
    ax = fig.add_subplot(1, 2, 2, projection=_projection)

    pp = nested_scatter(ax, xds.isel(time=time), "Prediction: Nested-ERA5", **kwargs)
    ax.set(title="Prediction: Nested-ERA5")
    axs.append(ax)

    # now the colorbar
    [ax.set(xlabel="", ylabel="") for ax in axs]
    [ax.coastlines("50m") for ax in axs]

    label = xds.attrs.get("label", "")
    label += f"\nt0: {t0}"
    label += f"\nvalid: {stime}"
    fig.colorbar(
        p,
        ax=axs,
        orientation="horizontal",
        shrink=.8,
        pad=0.05,
        aspect=35,
        label=label,
        extend=extend,
        **cbar_kwargs,
    )
    fig.set_constrained_layout(True)

    return None, None

def calc_wind_speed(xds):
    if "ugrd10m" in xds:
        ws = np.sqrt(xds["ugrd10m"]**2 + xds["vgrd10m"]**2)
    else:
        ws = np.sqrt(xds["10m_u_component_of_wind"]**2 + xds["10m_v_component_of_wind"]**2)
    ws.attrs["units"] = "m/sec"
    ws.attrs["long_name"] = "10m Wind Speed"
    return ws

def get_truth(name):
    if name.lower() == "era5":
        url = "gs://weatherbench2/datasets/era5/1959-2023_01_10-wb13-6h-1440x721_with_derived_variables.zarr"
        rename = {}
    elif name.lower() == "replay":
        url = "gs://noaa-ufs-gefsv13replay/ufs-hr1/0.25-degree/03h-freq/zarr/fv3.zarr"
        rename = {"pfull": "level", "grid_yt": "lat", "grid_xt": "lon"}

    truth = xr.open_zarr(
        url,
        storage_options={"token": "anon"},
    )
    truth = truth.rename(rename)
    truth.attrs["name"] = name
    return truth


def main(
    read_path,
    store_dir,
    t0,
    tf,
    ifreq=1,
    mode="figure", # or movie
):
    """A note about t0
    In the inference yaml, I think this means "the very first initial condition"... makes sense

    But when I'm visualizing the data, I think about t0 as the last initial condition...
    as in the last data given to the model before making a forecast.
    So... the t0 given here is that one... the last IC.
    """

    setup_simple_log()

    assert mode in ["figure", "movie"]


    logging.info(f"Time Bounds:\n\tt0 = {t0}\n\ttf = {tf}\n")
    psl = xr.open_dataset(read_path)
    psl = psl.sel(time=slice(t0, tf))
    psl = psl.isel(time=slice(None, None, ifreq))
    psl = psl.rename({"longitude": "longitudes", "latitude": "latitudes"})
    psl = psl.set_coords(["longitudes", "latitudes"])

    logging.info(f"Ready to make {mode}s with dataset:\n{psl}\n")

    for tname in ["ERA5"]:

        truth = get_truth(tname)
        truth_x, truth_y = np.meshgrid(truth.longitude, truth.latitude)
        logging.info(f"Retrieved truth = {tname}\n{truth}\n")
        fig_dir = os.path.join(store_dir, f"{mode}s", f"{truth.name.lower()}-vs-nested")
        if not os.path.isdir(fig_dir):
            os.makedirs(fig_dir)
            logging.info(f"Created fig_dir: {fig_dir}")

        # Compute this
        psl["10m_wind_speed"] = calc_wind_speed(psl)
        truth["10m_wind_speed"] = calc_wind_speed(truth)

        # setup for each variable
        plot_options = {
            "total_precipitation_6hr": get_precip_kwargs(),
            "2m_temperature": {
                "cmap": "cmo.thermal",
                "vmin": -10,
                "vmax": 30,
            },
            "10m_wind_speed": {
                "cmap": "cmo.tempo_r",
                "vmin": 0,
                "vmax": 25,
            },
            "total_column_water": {
                "cmap": "cmo.rain",
                "vmin": 0,
                "vmax": 60,
            },

        }


        for varname, options in plot_options.items():

            logging.info(f"Plotting {varname} with options")
            for key, val in options.items():
                logging.info(f"\t{key}: {val}")

            ds = xr.Dataset({
                "Prediction: Nested-ERA5": psl[varname].load(),
            })

            ds[truth.name] = truth[varname].sel(
                time=ds["Prediction: Nested-ERA5"].time.values,
            ).load()

            # Convert to degC
            if varname[:3] == "tmp" or "temperature" in varname:
                for key in ds.data_vars:
                    ds[key] -= 273.15
                    ds[key].attrs["units"] = "degC"

                logging.info(f"\tconverted {varname} K -> degC")

            # Convert to mm->m
            if "total_precipitation" in varname:
                for key in ds.data_vars:
                    ds[key] *= 1000
                    ds[key].attrs["units"] = "m"

                logging.info(f"\tconverted {varname} mm -> m")

            label = " ".join([x.capitalize() for x in varname.split("_")])
            ds.attrs["label"] = f"{label} ({ds[truth.name].units})"

            # colorbar extension options
            options["extend"], vmin, vmax = get_extend(
                ds,
                vmin=options.get("vmin", None),
                vmax=options.get("vmax", None),
            )
            logging.info(f"\tcolorbar extend = {options['extend']}")

            # precip is weird, since we don't do vmin/vmax, we do BoundaryNorm colorbar map blah blah
            # since we know it's bounded to be positive in anemoi... at least in this model..
            # then just worry about max
            if "total_precipitation" in varname:
                options["extend"] = "max" if vmax > 50 else "neither"
                logging.info(f"\ttotal_precipitation hack: setting extend based on upper limit of 50")

            options["t0"] = t0
            options["truth_x"] = truth_x
            options["truth_y"] = truth_y

            dpi = 300
            width = 10
            height = 6.5
            pixelwidth = width*dpi
            pixelheight = height*dpi

            if mode == "figure":

                fig = plt.figure(figsize=(width, height))
                itime = list(pd.Timestamp(x) for x in ds["time"].values).index(pd.Timestamp(tf))
                plot_single_timestamp(
                    xds=ds,
                    fig=fig,
                    time=itime,
                    **options,
                )
                fname = f"{fig_dir}/{varname}.{t0}.{tf}.jpeg"
                fig.savefig(fname, dpi=dpi, bbox_inches="tight")
                logging.info(f"Stored figure at: {fname}\n")

            else:
                mov = xmovie.Movie(
                    ds,
                    plot_single_timestamp,
                    framedim="time",
                    input_check=False,
                    pixelwidth=pixelwidth,
                    pixelheight=pixelheight,
                    dpi=dpi,
                    **options
                )
                fname = f"{fig_dir}/{varname}.{t0}.{tf}.mp4"
                mov.save(fname, progress=True, overwrite_existing=True)
                logging.info(f"Stored movie at: {fname}\n")
