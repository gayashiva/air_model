""" Plot comparing auto and manual discharge at guttannen"""

import sys, json
import os
import seaborn as sns
import numpy as np
import pandas as pd
import math
import matplotlib.colors
from datetime import datetime, timedelta
from matplotlib.offsetbox import AnchoredText
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import logging, coloredlogs

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
)
from src.utils.settings import config
from src.models.icestupaClass import Icestupa
# from src.models.methods.metadata import get_parameter_metadata

if __name__ == "__main__":
    # Main logger
    logger = logging.getLogger(__name__)
    logger.setLevel("ERROR")
    location = 'guttannen22'
    sprays = ['dynamic_field', 'manual']

    mypal = sns.color_palette("Set1", 2)

    # fig, ax = plt.subplots(2, 1, sharex="col")
    fig, ax = plt.subplots(2, 1, gridspec_kw={'height_ratios': [1, 3]}, sharex="col")

    for i, spray in enumerate(sprays):
        SITE, FOLDER = config(location, spray)
        icestupa = Icestupa(location, spray)
        icestupa.read_output()
        df=icestupa.df

        if spray == "dynamic_field":
            spray = "Dynamic"
        else:
            spray = "Manual"

        x = df.time[1:]
        y1 = df.Discharge[1:]
        y2 = df.iceV[1:]
        ax[0].plot(
            x,
            y1,
            linewidth=1,
            color=mypal[i],
        )
        ax[0].spines["right"].set_visible(False)
        ax[0].spines["top"].set_visible(False)
        ax[0].spines["left"].set_color("grey")
        ax[0].spines["bottom"].set_color("grey")
        ax[0].set_ylabel("Discharge [$l/min$]", size=6)

        ax[1].plot(
            x,
            y2,
            label= spray,
            linewidth=1,
            color=mypal[i],
        )
        ax[1].spines["right"].set_visible(False)
        ax[1].spines["top"].set_visible(False)
        ax[1].spines["left"].set_color("grey")
        ax[1].spines["bottom"].set_color("grey")
        ax[1].set_ylabel("Ice Volume [$m^3$]")


    ax[1].xaxis.set_major_locator(mdates.WeekdayLocator())
    ax[1].xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    # plt.subplots_adjust(wspace=None, hspace=None)
    fig.subplots_adjust(hspace=None, wspace=None)
    fig.autofmt_xdate()
    handles, labels = ax[1].get_legend_handles_labels()
    ax[1].legend(handles, labels, loc="upper left", prop={"size": 8}, title="Method")
    plt.savefig("data/figs/paper3/autovsman.png", bbox_inches="tight", dpi=300)

    fig, ax = plt.subplots()
    for i, spray in enumerate(sprays):
        SITE, FOLDER = config(location, spray)
        icestupa = Icestupa(location, spray)
        icestupa.read_output()
        df=icestupa.df

        df_c = pd.read_hdf(FOLDER["input"] + spray + "/input.h5", "df_c")
        df_c = df_c[["time", "DroneV", "DroneVError"]]

        tol = pd.Timedelta("15T")
        df_c = df_c.set_index("time")
        df = df.set_index("time")
        df_c = pd.merge_asof(
            left=df,
            right=df_c,
            right_index=True,
            left_index=True,
            direction="nearest",
            tolerance=tol,
        )
        df_c = df_c[["DroneV", "DroneVError", "iceV"]]
        df = df.reset_index()

        if spray == "dynamic_field":
            spray = "Dynamic"
        else:
            spray = "Manual"

        x = df.time
        y1 = df.iceV
        ax.set_ylabel("Ice Volume[$m^3$]")
        ax.plot(
            x,
            y1,
            label="Modelled Volume",
            linewidth=1,
            color=mypal[i],
        )
        y2 = df_c.DroneV
        yerr = df_c.DroneVError
        # ax.fill_between(x, y1=V_dome, y2=0, color=grey, label="Dome Volume")
        ax.scatter(x, y2, color=mypal[i], label="Measured Volume")
        ax.errorbar(x, y2, yerr=df_c.DroneVError, color=mypal[i])
        ax.set_ylim(bottom=0)
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.spines["left"].set_color("grey")
        ax.spines["bottom"].set_color("grey")
        ax.xaxis.set_major_locator(mdates.WeekdayLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
        ax.xaxis.set_minor_locator(mdates.DayLocator())
        fig.autofmt_xdate()
    ax.legend(handles, labels, prop={"size": 8}, title="Method")
    plt.savefig("data/figs/slides/autovsman_vol.png", bbox_inches="tight", dpi=300)
    plt.clf()

