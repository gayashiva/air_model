"""Icestupa class function that generates data plots
"""

import numpy as np
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText
from matplotlib.ticker import AutoMinorLocator
import pandas as pd
import logging
from codetiming import Timer

logger = logging.getLogger("__main__")


@Timer(text="%s executed in {:.2f} seconds" % __name__, logger=logging.warning)
def plot_input(df, folder, name):

    np.warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)
    blue = "#0a4a97"
    red = "#e23028"
    purple = "#9673b9"
    green = "#28a745"
    orange = "#ffc107"
    pink = "#ce507a"
    skyblue = "#9bc4f0"
    grey = "#ced4da"
    default = "#284D58"

    CB91_Blue = "#2CBDFE"
    CB91_Green = "#47DBCD"
    CB91_Pink = "#F3A0F2"
    CB91_Purple = "#9D2EC5"
    CB91_Violet = "#661D98"
    CB91_Amber = "#F5B14C"

    fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(
        nrows=5, ncols=1, sharex="col", figsize=(10, 14))

    x = df.time

    # y1 = df.ppt
    # ax1.plot(x, y1, linestyle="-", color=default, linewidth=1)
    # ax1.set_ylabel("Precipitation [$mm$]")
    y1 = df.tcc
    ax1.plot(x, y1, linestyle="-", color=default, linewidth=1)
    ax1.set_ylabel("Cloud")

    y2 = df.temp
    ax2.plot(x, y2, linestyle="-", color=default, linewidth=1)
    ax2.set_ylabel("Temperature [$\\degree C$]")

    y3 = df.RH
    ax3.plot(x, y3, linestyle="-", color=default, linewidth=1)
    ax3.set_ylabel("Humidity [$\\%$]")

    y4 = df.SW_global
    # y4 = df.ghi
    ax4.plot(x, y4, linestyle="-", color=default, linewidth=1)
    ax4.set_ylabel("Shortwave Radiation [$W\\,m^{-2}$]")

    y5 = df.wind
    ax5.plot(x, y5, linestyle="-", color=default, linewidth=1)
    ax5.set_ylabel("Wind speed [$m\\,s^{-1}$]")

    # ax1.spines[["top", "right"]].set_visible(False)
    # ax2.spines[["top", "right"]].set_visible(False)
    # ax3.spines[["top", "right"]].set_visible(False)
    # ax4.spines[["top", "right"]].set_visible(False)
    # ax5.spines[["top", "right"]].set_visible(False)
    ax1.xaxis.set_major_locator(mdates.MonthLocator())
    # ax1.xaxis.set_major_locator(mdates.WeekdayLocator())
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    # ax1.xaxis.set_minor_locator(mdates.DayLocator())
    fig.autofmt_xdate()
    plt.subplots_adjust(wspace=None, hspace=None)
    plt.savefig(
        folder + "input.png",
        bbox_inches="tight",
    )
    # plt.savefig(
    #     "data/figs/paper3/input.png",
    #     bbox_inches="tight",
    # )
    plt.clf()

    plt.close("all")
