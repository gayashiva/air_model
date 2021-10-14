import sys
import os
import numpy as np
import pandas as pd
import math
import matplotlib.colors
import uncertainpy as un
import statistics as st
from datetime import datetime, timedelta
from matplotlib.offsetbox import AnchoredText
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
)

from src.utils.settings import config
from src.models.icestupaClass import Icestupa
from src.models.methods.metadata import get_parameter_metadata

if __name__ == "__main__":

    locations = ["guttannen21", "gangles21"]

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

    fig, ax = plt.subplots(len(locations), 1, sharex="col")

    for i, location in enumerate(locations):
        SITE, FOLDER = config(location)
        icestupa_sim = Icestupa(location)

        # Sim settings

        # if location == "gangles21":
        #     icestupa_sim.cld = 0.68
        #     # icestupa_sim.df["RH"] = 79
        # if location == "guttannen21":
        #     sims = ["cld", "RH"]
        #     icestupa_sim.R_F = 10.2

        # icestupa_sim.derive_parameters()
        # icestupa_sim.melt_freeze()
        # df_sim = icestupa_sim.df[["time", "iceV"]]

        icestupa = Icestupa(location)
        icestupa.read_output()
        icestupa.self_attributes()

        df = icestupa.df[["time", "iceV"]]
        df_c = pd.read_hdf(FOLDER["input"] + "model_input.h5", "df_c")
        if icestupa.name in ["guttannen21", "guttannen20", "gangles21"]:
            df_c = df_c[1:]
        df_c = df_c.set_index("time").resample("D").mean().reset_index()
        dfv = df_c[["time", "DroneV", "DroneVError"]]
        df = df.reset_index()

        x = df.time[1:]
        y1 = df.iceV[1:]
        x2 = dfv.time
        y2 = dfv.DroneV
        yerr = dfv.DroneVError
        ax[i].plot(
            x,
            y1,
            label="Simulated Volume",
            linewidth=1,
            color=CB91_Blue,
            alpha=1,
            zorder=10,
        )

        # x_sim = df_sim.time[1:]
        # y1_sim = df_sim.iceV[1:]
        # ax[i].plot(
        #     x_sim,
        #     y1_sim,
        #     label="Simulated Volume",
        #     linewidth=1,
        #     linestyle="--",
        #     color=CB91_Blue,
        #     alpha=1,
        #     zorder=10,
        # )
        ax[i].errorbar(
            x2,
            y2,
            yerr=df_c.DroneVError,
            color=CB91_Violet,
            lw=1,
            alpha=0.5,
            zorder=8,
        )
        ax[i].scatter(x2, y2, s=5, color=CB91_Violet, zorder=7, label="UAV Volume")
        ax[i].set_ylim(round(icestupa.V_dome, 0) - 1, round(df.iceV.max(), 0))
        v = get_parameter_metadata(location)
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
                # round(df_sim.iceV.max(), 0),
                round(df.iceV.max(), 0),
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
        # "data/slides/icev_slide_" + str(number) + ".jpg",
        "data/slides/icev_slide1.jpg",
        bbox_inches="tight",
        dpi=300,
    )
    plt.clf()
