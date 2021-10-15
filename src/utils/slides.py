import sys
import os
import numpy as np
import pandas as pd

pd.options.mode.chained_assignment = None  # default='warn'
import xarray as xr
import math
import matplotlib.colors
import uncertainpy as un
import statistics as st
from datetime import datetime, timedelta
from matplotlib.offsetbox import AnchoredText
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import logging, coloredlogs

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
)

from src.utils.settings import config
from src.models.icestupaClass import Icestupa
from src.models.methods.metadata import get_parameter_metadata

if __name__ == "__main__":
    # Main logger
    logger = logging.getLogger(__name__)
    logger.setLevel("WARNING")

    blue = "#0a4a97"
    red = "#e23028"
    purple = "#9673b9"
    green = "#28a745"
    orange = "#ffc107"
    pink = "#ce507a"
    skyblue = "#9bc4f0"
    grey = "#ced4da"
    CB91_Blue = "#2CBDFE"
    CB91_Green = "#47DBCD"
    CB91_Pink = "#F3A0F2"
    CB91_Purple = "#9D2EC5"
    CB91_Violet = "#661D98"
    CB91_Amber = "#F5B14C"

    locations = ["guttannen21", "gangles21"]
    sims = ["normal", "R_F", "tcc", "RH", "T", "R_F+tcc+RH+T"]
    sims_mean = [
        "Estimate",
        "Equal spray radius",
        "Half cloudy days",
        "Twice more humidity",
        "Similar temperature",
        "All the above",
    ]
    label_dict = dict(zip(sims, sims_mean))

    compile = False
    layout = 1
    layout = 2

    if compile:
        time = pd.date_range("2020-11-01", freq="H", periods=365 * 24)
        # put data into a dataset
        ds = xr.DataArray(
            dims=["time", "locs", "sims"],
            coords={"time": time, "locs": locations, "sims": sims},
            attrs=dict(description="coords with matrices"),
        )

        for i, loc in enumerate(locations):
            for sim in sims:
                SITE, FOLDER = config(loc)
                icestupa_sim = Icestupa(loc)
                df = pd.DataFrame()
                print(loc, sim)
                if sim == "normal":
                    icestupa_sim.read_output()
                    icestupa_sim.self_attributes()
                    df = icestupa_sim.df[["time", "iceV"]]

                else:
                    if loc == "gangles21":
                        if sim == "T":
                            icestupa_sim.df["temp"] += 2
                        if sim == "RH":
                            icestupa_sim.df["RH"] *= 2
                            icestupa_sim.df.loc[icestupa_sim.df.RH > 100, "RH"] = 100
                        if sim == "R_F":
                            icestupa_sim.R_F = 6.9
                        if sim == "tcc":
                            icestupa_sim.tcc = 0.5
                        if sim == "R_F+tcc+RH+T":
                            icestupa_sim.df["temp"] += 2
                            icestupa_sim.df["RH"] *= 2
                            icestupa_sim.df.loc[icestupa_sim.df.RH > 100, "RH"] = 100
                            icestupa_sim.R_F = 6.9
                            icestupa_sim.tcc = 0.5
                        icestupa_sim.derive_parameters()
                        icestupa_sim.melt_freeze()
                        df = icestupa_sim.df[["time", "iceV"]]
                    else:
                        continue

                ds.loc[
                    dict(time=df.time.values[1:], locs=loc, sims=sim)
                ] = df.iceV.values[1:]
        ds.to_netcdf("data/slides/sims.nc")
    elif layout == 1:
        ds = xr.open_dataarray("data/slides/sims.nc")

        locations = ["guttannen21", "gangles21"]

        fig, ax = plt.subplots(len(locations), 1, sharex="col")
        for i, loc in enumerate(locations):
            SITE, FOLDER = config(loc)
            icestupa = Icestupa(loc)
            icestupa.self_attributes()
            df_c = pd.read_hdf(FOLDER["input"] + "model_input.h5", "df_c")
            df_c = df_c[1:]
            df_c = df_c.set_index("time").resample("D").mean().reset_index()
            dfv = df_c[["time", "DroneV", "DroneVError"]]
            x2 = dfv.time
            y2 = dfv.DroneV
            yerr = dfv.DroneVError

            for sim in sims:
                ds.sel(locs=loc, sims=sim).plot(
                    label=sim,
                    linewidth=1,
                    # color=CB91_Blue,
                    alpha=1,
                    zorder=10,
                    ax=ax[i],
                )
                ax[i].set_title(label="")
            maxV = round(
                ds.sel(locs=loc, sims="normal").dropna(dim="time").data.max(), 0
            )
            ax[i].scatter(x2, y2, s=5, color=CB91_Violet, zorder=7, label="UAV Volume")
            ax[i].errorbar(
                x2,
                y2,
                yerr=df_c.DroneVError,
                color=CB91_Violet,
                lw=1,
                alpha=0.5,
                zorder=8,
            )
            ax[i].set_ylim(
                round(icestupa.V_dome, 0) - 1,
                maxV,
            )
            v = get_parameter_metadata(loc)
            at = AnchoredText(
                v["slidename"], prop=dict(size=10), frameon=True, loc="upper left"
            )
            at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
            if i != len(locations) - 1:
                x_axis = ax[i].axes.get_xaxis()
                x_axis.set_visible(False)
                ax[i].spines["bottom"].set_visible(False)

            ax[i].yaxis.set_ticks(
                [
                    round(icestupa.V_dome, 0) - 1,
                    maxV,
                ]
            )
            ax[i].add_artist(at)
            # Hide the right and top spines
            ax[i].spines["right"].set_visible(False)
            ax[i].spines["top"].set_visible(False)
            ax[i].spines["left"].set_color("grey")
            ax[i].spines["bottom"].set_color("grey")
            [t.set_color("grey") for t in ax[i].xaxis.get_ticklines()]
            [t.set_color("grey") for t in ax[i].yaxis.get_ticklines()]
            # Only show ticks on the left and bottom spines
            ax[i].yaxis.set_ticks_position("left")
            ax[i].xaxis.set_ticks_position("bottom")
            # ax[i].yaxis.set_major_locator(plt.LinearLocator(numticks=2))
            ax[i].xaxis.set_major_locator(mdates.MonthLocator())
            ax[i].xaxis.set_major_formatter(mdates.DateFormatter("%b"))
            # ax[i].xaxis.grid(color="gray", linestyle="dashed")
            fig.autofmt_xdate()
        fig.text(0.04, 0.5, "Ice Volume[$m^3$]", va="center", rotation="vertical")
        handles, labels = ax[1].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper right", prop={"size": 8})
        plt.savefig(
            "data/slides/icev_sims.jpg",
            bbox_inches="tight",
            dpi=300,
        )
        plt.clf()
    elif layout == 2:
        ds = xr.open_dataarray("data/slides/sims.nc")
        fig, ax = plt.subplots(1, 1)
        locations = ["gangles21"]
        sims = ["normal", "R_F", "tcc", "T", "R_F+tcc+RH+T"]
        style = ["--", "-"]
        for i, loc in enumerate(locations):
            SITE, FOLDER = config(loc)
            icestupa = Icestupa(loc)
            icestupa.self_attributes()
            df_c = pd.read_hdf(FOLDER["input"] + "model_input.h5", "df_c")
            df_c = df_c[1:]
            df_c = df_c.set_index("time").resample("D").mean().reset_index()
            dfv = df_c[["time", "DroneV", "DroneVError"]]
            x2 = dfv.time
            y2 = dfv.DroneV
            yerr = dfv.DroneVError
            Vol = []

            for sim in sims:
                ds.loc[dict(locs=loc, sims=sim)] -= icestupa.V_dome
                y2 -= icestupa.V_dome
                yerr -= icestupa.V_dome
                ds.sel(locs=loc, sims=sim).plot(
                    label=label_dict[sim],
                    linewidth=1,
                    # linestyle=style[i],
                    # color=CB91_Blue,
                    alpha=1,
                    zorder=10,
                    ax=ax,
                )
                # ax.set_title(label="")
                ax.set(xlabel=None, title=None)
                maxV = round(
                    ds.sel(locs=loc, sims=sim).dropna(dim="time").data.max(), 0
                )
                Vol.append(maxV)

            Vol = np.array(Vol)
            maxV = round(
                ds.sel(locs=loc, sims="normal").dropna(dim="time").data.max(), 0
            )
            ax.set_ylim(0, maxV)
            ax.set_yticks(Vol)
            Vol = np.around(Vol / 100, decimals=0).astype(int)
            ax.set_yticklabels(Vol)

            # ax.yaxis.set_ticks(
            #     Vol
            # )

            # Hide the right and top spines
            ax.spines["right"].set_visible(False)
            ax.spines["top"].set_visible(False)
            ax.spines["left"].set_color("grey")
            ax.spines["bottom"].set_color("grey")
            [t.set_color("grey") for t in ax.xaxis.get_ticklines()]
            [t.set_color("grey") for t in ax.yaxis.get_ticklines()]
            # Only show ticks on the left and bottom spines
            ax.yaxis.set_ticks_position("left")
            ax.xaxis.set_ticks_position("bottom")
            # ax[i].yaxis.set_major_locator(plt.LinearLocator(numticks=2))
            ax.xaxis.set_major_locator(mdates.MonthLocator())
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
            fig.autofmt_xdate()
        fig.text(
            0.04,
            0.5,
            "Number of Swiss AIRs",
            va="center",
            rotation="vertical",
        )
        handles, labels = ax.get_legend_handles_labels()
        fig.legend(
            handles, labels, loc="upper right", prop={"size": 8}, title="What if?"
        )
        plt.savefig(
            "data/slides/icev_sims2.jpg",
            bbox_inches="tight",
            dpi=300,
        )
        plt.clf()
