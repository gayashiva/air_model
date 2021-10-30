import sys
import os
import numpy as np
import pandas as pd

pd.options.mode.chained_assignment = None  # default='warn'
import xarray as xr
import math
import matplotlib.colors
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

    locations = ["guttannen21", "gangles21"]
    sims = [
        "normal",
        "ppt",
        "T",
        "RH",
        "p",
        "v",
        "T+RH+p+v",
        "SW",
        "tcc",
        "SW+tcc",
        "R_F",
        "all",
        "T+RH+p+v+SW+tcc",
    ]
    sims_mean = [
        "Reference",
        "Remove snowfall",
        "Temperature + 2 $\\degree C$",
        "Rel. Hum. + 44 %",
        "Pressure + 171 $hPa$",
        "Wind - 1 $m\\,s^{-1}$",
        "Similar latent heat",
        "Shortwave - 171 $W\\,m^{-2}$",
        "Cloudiness + 0.5",
        "Similar day melt",
        # "Spray radius - 3 $m$",
        "Similar fountain",
        "All the above",
        "Similar Weather",
    ]
    label_dict = dict(zip(sims, sims_mean))

    # compile = True
    compile = False
    # layout = 1
    layout = 2

    if compile:
        time = pd.date_range("2020-11-01", freq="H", periods=365 * 24)
        ds = xr.DataArray(
            dims=["time", "locs", "sims"],
            coords={"time": time, "locs": locations, "sims": sims},
            attrs=dict(description="coords with matrices"),
        )

        for i, loc in enumerate(locations):
            for sim in sims:
                SITE, FOLDER = config(loc)
                df = pd.DataFrame()
                print(loc, sim)
                if sim == "normal":
                    icestupa_sim = Icestupa(loc)
                    icestupa_sim.read_output()
                    icestupa_sim.self_attributes()
                    df = icestupa_sim.df[["time", "iceV"]]
                elif sim == "ppt" and loc == "guttannen21":
                    icestupa_sim = Icestupa(loc)
                    icestupa_sim.df["ppt"] = 0
                    icestupa_sim.derive_parameters()
                    icestupa_sim.melt_freeze()
                    df = icestupa_sim.df[["time", "iceV"]]

                else:
                    if loc == "gangles21":
                        icestupa_sim = Icestupa(loc)
                        if sim == "T":
                            icestupa_sim.df["temp"] += 2
                        if sim == "v":
                            icestupa_sim.df["wind"] -= 1
                        if sim == "p":
                            icestupa_sim.df["press"] += 794 - 623
                        if sim == "SW":
                            icestupa_sim.df["SW_global"] -= 246 - 138
                        if sim == "RH":
                            icestupa_sim.df["RH"] += 44
                            icestupa_sim.df.loc[icestupa_sim.df.RH > 100, "RH"] = 100
                        if sim == "R_F":
                            icestupa_sim.R_F = 6.9
                        if sim == "tcc":
                            icestupa_sim.tcc = 0.5
                        if sim == "SW+tcc":
                            icestupa_sim.df["SW_global"] -= 246 - 138
                            icestupa_sim.tcc = 0.5
                        if sim == "T+RH+p+v":
                            icestupa_sim.df["temp"] += 2
                            icestupa_sim.df["wind"] -= 1
                            icestupa_sim.df["press"] += 794 - 623
                            icestupa_sim.df["RH"] += 44
                            icestupa_sim.df.loc[icestupa_sim.df.RH > 100, "RH"] = 100
                        if sim == "T+RH+p+v+SW+tcc":
                            icestupa_sim.df["temp"] += 2
                            icestupa_sim.df["wind"] -= 1
                            icestupa_sim.df["press"] += 794 - 623
                            icestupa_sim.df["RH"] += 44
                            icestupa_sim.df.loc[icestupa_sim.df.RH > 100, "RH"] = 100
                            icestupa_sim.df["SW_global"] -= 246 - 138
                            icestupa_sim.tcc = 0.5
                        if sim == "all":
                            icestupa_sim.df["temp"] += 2
                            icestupa_sim.df["wind"] -= 1
                            icestupa_sim.df["SW_global"] -= 246 - 138
                            icestupa_sim.df["press"] += 794 - 623
                            icestupa_sim.df["RH"] += 44
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
        # ds.to_netcdf("data/slides/sims_try.nc")
    elif layout == 1:
        ds = xr.open_dataarray("data/slides/sims.nc")

        locations = ["guttannen21", "gangles21"]
        # locations = ["guttannen21"]
        sims = ["normal", "ppt"]
        Vols = []

        # plt.figure()
        # ax = plt.gca()
        # ds.sel(locs="guttannen21", sims="normal").plot()
        # ds.sel(locs="guttannen21", sims="ppt").plot()
        # plt.legend()
        # plt.grid()
        # plt.savefig("data/slides/try.jpg")
        # sys.exit()

        for slide in range(4):
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

                if slide == 3:
                    ds.loc[dict(locs=loc, sims="normal")] -= icestupa.V_dome
                    ds.loc[dict(locs=loc, sims="ppt")] -= icestupa.V_dome
                    icestupa.V_dome = 0

                if slide <= 1:
                    ax[i].scatter(
                        x2, y2, s=5, color=CB91_Violet, zorder=7, label="UAV Volume"
                    )
                    ax[i].errorbar(
                        x2,
                        y2,
                        yerr=df_c.DroneVError,
                        color=CB91_Violet,
                        lw=1,
                        alpha=0.5,
                        zorder=8,
                    )
                    ds.sel(locs=loc, sims="normal").plot(
                        # label=sim,
                        linewidth=1,
                        color=CB91_Blue,
                        alpha=0,
                        zorder=10,
                        ax=ax[i],
                    )
                    ax[i].set_title(label="")

                if slide >= 1:
                    ds.sel(locs=loc, sims="normal").plot(
                        # label=sim,
                        linewidth=1,
                        color=CB91_Blue,
                        alpha=1,
                        zorder=10,
                        ax=ax[i],
                    )
                    ax[i].set_title(label="")

                if slide >= 2 and loc == "guttannen21":
                    ds.sel(locs=loc, sims="ppt").plot(
                        # label=sim,
                        linestyle="--",
                        linewidth=1,
                        color=CB91_Blue,
                        alpha=1,
                        zorder=10,
                        ax=ax[i],
                    )
                    ax[i].set_title(label="")

                maxV = round(
                    ds.sel(locs=loc, sims="normal").dropna(dim="time").data.max(), 0
                )
                Vols = [
                    round(icestupa.V_dome, 0),
                    maxV,
                ]
                ax[i].set_ylim(
                    round(icestupa.V_dome, 0),
                    maxV,
                )
                if i != len(locations) - 1:
                    x_axis = ax[i].axes.get_xaxis()
                    x_axis.set_visible(False)
                    ax[i].spines["bottom"].set_visible(False)

                if slide >= 2 and loc == "guttannen21":
                    Vols.append(
                        round(
                            ds.sel(locs=loc, sims="ppt").dropna(dim="time").data.max(),
                            0,
                        )
                    )

                ax[i].yaxis.set_ticks(Vols)
                v = get_parameter_metadata(loc)
                at = AnchoredText(
                    v["slidename"], prop=dict(size=10), frameon=True, loc="upper left"
                )
                at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
                ax[i].add_artist(at)

                ax[i].set(xlabel=None)
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
            plt.savefig(
                "data/slides/icev_slides_" + str(slide) + ".jpg",
                bbox_inches="tight",
                dpi=300,
            )
            plt.clf()
    elif layout == 2:
        ds = xr.open_dataarray("data/slides/sims.nc")
        icestupa = Icestupa()
        icestupa.self_attributes()
        CH_Vol = (
            round(
                ds.sel(locs="guttannen21", sims="ppt").dropna(dim="time").data.max(),
                0,
            )
            - icestupa.V_dome
        )
        loc = "gangles21"
        # sims = ["normal", "SW", "v", "T", "tcc", "R_F", "all"]
        # sims = ["normal", "T", "RH", "p", "SW", "tcc", "v", "R_F", "all"]
        sims_total = [
            "normal",
            "ppt",
            "T",
            "RH",
            "p",
            "v",
            "T+RH+p+v",
            "SW",
            "tcc",
            "SW+tcc",
            "R_F",
            "all",
            "T+RH+p+v+SW+tcc",
        ]
        sims1 = ["normal", "T", "RH", "p", "v", "T+RH+p+v"]
        sims2 = ["normal", "T+RH+p+v", "SW", "tcc", "SW+tcc", "R_F", "all"]
        # sims3 = ["normal", "T+RH+p+v", "SW+tcc", "R_F", "all"]
        sims3 = ["normal", "T+RH+p+v+SW+tcc", "R_F", "all"]
        # style = ["--", "-"]
        # for slide in range(3, 6):
        sims_list = [sims1, sims2, sims3]
        for sim1 in sims_total:
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
            ds.loc[dict(locs=loc, sims=sim1)] -= icestupa.V_dome
            y2 -= icestupa.V_dome
            yerr -= icestupa.V_dome
            if sim1 in sims3:
                print(sim1)
                print(ds.sel(locs=loc, sims=sim1).dropna(dim="time").data.max())
        for sims in sims_list:
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
            Vols = np.array([])
            res = []
            y_pos = np.array([])
            # slide = 4

            for sim1 in sims:
                # ds.loc[dict(locs=loc, sims=sim1)] -= icestupa.V_dome
                # y2 -= icestupa.V_dome
                # yerr -= icestupa.V_dome
                fig, ax = plt.subplots(1, 1)
                print(loc, sim1)
                for sim2 in sims:
                    if sims.index(sim2) < sims.index(sim1):
                        ds.sel(locs=loc, sims=sim2).plot(
                            label=label_dict[sim2],
                            linewidth=1,
                            # linestyle=style[i],
                            # color=CB91_Blue,
                            alpha=0.3,
                            zorder=10,
                            ax=ax,
                        )
                        V = round(
                            ds.sel(locs=loc, sims=sim2).dropna(dim="time").data.max(), 0
                        )
                        Vols = np.append(Vols, V)
                    if sims.index(sim2) == sims.index(sim1):
                        ds.sel(locs=loc, sims=sim2).plot(
                            label=label_dict[sim2],
                            linewidth=1,
                            # linestyle=style[i],
                            # color=CB91_Blue,
                            alpha=1,
                            zorder=10,
                            ax=ax,
                        )
                        V = ds.sel(locs=loc, sims=sim2).dropna(dim="time").data.max(), 0
                        Vols = np.append(Vols, V)
                CH_Vols = np.around(Vols / CH_Vol, decimals=1)
                [res.append(x.astype(int)) for x in CH_Vols if x.astype(int) not in res]
                y_pos = np.array([x * 87 for x in res])
                # [
                #     np.append(y_pos, x * 83)
                #     for x in CH_Vols
                #     if x.astype(int) not in y_pos
                # ]
                ax.set_yticks(y_pos)
                ax.set_yticklabels(res)

                maxV = y_pos.max()
                # ds.sel(locs=loc, sims=sim1).plot(
                #     label=label_dict[sim1],
                #     linewidth=1,
                #     # linestyle=style[i],
                #     # color=CB91_Blue,
                #     alpha=1,
                #     zorder=10,
                #     ax=ax,
                # )
                # maxV = round(
                #     ds.sel(locs=loc, sims="normal").dropna(dim="time").data.max(), 0
                # )
                ax.set_ylim(0, maxV)
                ax.set(xlabel=None, ylabel="Number of Swiss AIRs", title=None)
                ax.legend(loc="upper right", prop={"size": 8}, title="Simulations")

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
                plt.savefig(
                    "data/slides/icev_slides_" + sim1 + ".jpg",
                    bbox_inches="tight",
                    dpi=300,
                )
                plt.clf()
            # slide += 1
