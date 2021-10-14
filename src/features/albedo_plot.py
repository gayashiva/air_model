"""Icestupa class function that generates figures for web app
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os, sys
import matplotlib.dates as mdates
import matplotlib as mpl
from matplotlib.colors import ListedColormap
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.offsetbox import AnchoredText
from mpl_toolkits.axisartist.axislines import Axes
from mpl_toolkits import axisartist
import matplotlib.ticker as ticker

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
)

from src.utils.settings import config
from src.models.methods.metadata import get_parameter_metadata
from src.models.icestupaClass import Icestupa

if __name__ == "__main__":
    locations = ["gangles21", "guttannen21"]
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

    fig = plt.figure(figsize=(15, 12))
    subfigs = fig.subfigures(len(locations), 1)
    for ctr, location in enumerate(locations):
        SITE, FOLDER = config(location)
        icestupa = Icestupa(location)
        icestupa.read_output()

        ax = subfigs[ctr].subplots(2, 1, sharex="col", sharey="row") 
        x = icestupa.df.When
        y1 = icestupa.df.a
        y2 = icestupa.df.f_cone
        ax[0].plot(x, y1, color=CB91_Purple)
        ax[0].set_ylabel("Albedo")
        axt = ax[0].twinx()
        axt.plot(x, y2, color=CB91_Amber, linewidth=0.5)
        axt.set_ylabel("$f_{cone}$", color=CB91_Amber)
        for tl in axt.get_yticklabels():
            tl.set_color(CB91_Amber)

        y3 = icestupa.df.T_s

        ax[1].plot(
            x,
            y3,
            # label="Modelled",
            linewidth=1,
            # color=CB91_Amber,
            # color=CB91_Pink,
            # alpha=0.8,
            # zorder=0,
        )
        # if self.name in ["guttannen21", "guttannen20"]:
        #     y2 = df_cam.cam_temp
        #     ax[1].scatter(
        #         x,
        #         y2,
        #         color=CB91_Violet,
        #         s=1,
        #         label="Measured",
        #         zorder=1,
        #     )
        ax[1].set_ylabel("Surface Temperature [$\\degree C$]")
        # ax[1].legend()
        ax[0].set_ylim([0, 1])
        axt.set_ylim([0, 1])
        ax[0].xaxis.set_major_locator(mdates.WeekdayLocator())
        ax[0].xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
        ax[0].xaxis.set_minor_locator(mdates.DayLocator())
        fig.autofmt_xdate()
        subfigs[ctr].text(
            0.04,
            0.5,
            get_parameter_metadata(location)["shortname"],
            va="center",
            rotation="vertical",
            fontsize="x-large",
        )
        subfigs[ctr].subplots_adjust(hspace=0.05, wspace=0.025)

    plt.savefig(
        "data/paper/albedo.jpg",
        dpi=300,
        bbox_inches="tight",
    )
