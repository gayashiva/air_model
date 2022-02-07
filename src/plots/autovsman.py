""" Plot comparing auto and manual discharge at guttannen"""

import sys
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
from src.models.methods.metadata import get_parameter_metadata

if __name__ == "__main__":
    # Main logger
    logger = logging.getLogger(__name__)
    logger.setLevel("ERROR")
    locations = ['guttannen22_auto', 'guttannen22_man']

    mypal = sns.color_palette("Set1", 2)
    fig, ax = plt.subplots(2, 1, sharex="col")

    for i, location in enumerate(locations):
        CONSTANTS, SITE, FOLDER = config(location)
        icestupa = Icestupa(location)
        icestupa.read_output()
        df=icestupa.df

        if location == "guttannen22_auto":
            spray = "Automatic "
        else:
            spray = "Manual "

        x = df.time[1:]
        y1 = df.Discharge[1:]
        y2 = df.iceV[1:]
        ax[0].plot(
            x,
            y1,
            # label= spray + "Discharge",
            linewidth=1,
            color=mypal[i],
        )
        ax[0].spines["right"].set_visible(False)
        ax[0].spines["top"].set_visible(False)
        ax[0].spines["left"].set_color("grey")
        ax[0].spines["bottom"].set_color("grey")
        ax[0].set_ylabel("Discharge [$l/min$]")

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
    # ax[1].xaxis.set_minor_locator(mdates.DayLocator())
    fig.autofmt_xdate()
    # fig.text(0.04, 0.5, "Ice Volume[$m^3$]", va="center", rotation="vertical")
    handles, labels = ax[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", prop={"size": 8})
    # plt.legend()
    plt.savefig("data/paper3/autovsmanual.jpg", bbox_inches="tight", dpi=300)
