import string

import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import seaborn as sns


def nice_names(name):
    return string.capwords(name.replace("_", " "))


def get_units(name):
    if "geopotential" in name and "height" in name:
        return "m"
    if "temperature" in name:
        return "K"
    if "humidity" in name:
        return "kg kg$^{-1}$"
    if "pressure" in name:
        return "Pa"
    if "wind" in name:
        return "m s$^{-1}$"


def get_color(label):
    color = None
    if "Nested" in label:
        color = "C0"
    elif "HRRR" in label:
        color = "C1"
    elif "GFS" in label:
        color = "C2"
    elif "Global-EAGLE" in label:
        color = "C5"
    return color

def make_one_legend(fig, axs):
    handles, labels = axs.flatten()[0].get_legend_handles_labels()
    legend = fig.legend(
        handles,
        labels,
        loc='lower center',
        bbox_to_anchor=(0.5, -0.05),
        ncols=len(labels),
        frameon=False,
    )
    for line in legend.get_lines():
        line.set_linewidth(16)
    if len(axs.shape) == 1:
        plt.tight_layout(pad=0, rect=[0, 0.1, 1, 1])
    [ax.legend().remove() for ax in axs.flatten()];
    return legend

def single_plot(ax, dsdict, metric_name, varname, sel=None, **kwargs):

    estimator = kwargs.pop("estimator", "median")
    for label, xds in dsdict.items():

        plotme = xds[varname] if sel is None else xds[varname].sel(**sel)
        df = plotme.to_dataframe().reset_index()
        sns.lineplot(
            data=df,
            x="fhr",
            y=varname,
            ax=ax,
            label=label,
            color=get_color(label),
            estimator=estimator,
            **kwargs,
        )
    xticks = plotme.fhr.values
    xticklabels = [str(xx) for xx in xticks]
    xlabel = "Lead Time (hours)"
    if len(xticks) > 10:
        xticks = np.concatenate([ [xticks[0]], xticks[4::4]])
        xticklabels = [str(int(xx/24)) for xx in xticks]
        xlabel = "Lead Time (days)"

    title = f"{nice_names(varname)}  ({get_units(varname)})"
    ax.set(
        ylabel=metric_name if ax.get_subplotspec().is_first_col() else "",
        xlabel=xlabel if ax.get_subplotspec().is_last_row() else "",
        title=title if ax.get_subplotspec().is_first_row() else "",
        xticks=xticks,
        xticklabels=xticklabels,
    )
    ax.legend(frameon=False)


def plot_surface_vars(
    dsdict,
    metric_name,
    surface_vars=("surface_pressure", "10m_wind_speed", "2m_temperature", "2m_specific_humidity"),
    one_legend=True,
    **kwargs,
):
    ncols = len(surface_vars)
    fig, axs = plt.subplots(1, ncols, figsize=(5.25*ncols, 4.1), constrained_layout=True)

    for varname, ax in zip(surface_vars, axs):
        single_plot(ax=ax, dsdict=dsdict, metric_name=metric_name, varname=varname, **kwargs)
    if one_legend:
        make_one_legend(fig, axs)
    return fig, axs


def plot_level_vars(
    dsdict,
    metric_name,
    level_vars=("geopotential_height", "wind_speed", "temperature", "specific_humidity"),
    one_legend=True,
    **kwargs,
):

    levels = next(iter(dsdict.values())).level.values
    ncols = len(level_vars)
    nrows = len(levels)
    fig, axs = plt.subplots(nrows, ncols, figsize=(5.25*ncols, 4.5*nrows), constrained_layout=True, sharex=True)

    if ncols == 1:
        axs = [axs]
    if nrows == 1:
        axs = [axs]


    sel = kwargs.pop("sel", {})
    for level, axr in zip(levels, axs):
        for varname, ax in zip(level_vars, axr):

            sel["level"] = level
            single_plot(ax=ax, dsdict=dsdict, metric_name=metric_name, varname=varname, sel=sel, **kwargs)
            ax.legend(title=f"{level} hPa", frameon=False)

    if one_legend:
        make_one_legend(fig, axs)
    return fig, axs
