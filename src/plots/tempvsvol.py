""" Plot comparing temp and volume"""

import sys, json
import os
import numpy as np
import pandas as pd
import math
import matplotlib.colors
from datetime import datetime, timedelta
from matplotlib.offsetbox import AnchoredText
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.lines import Line2D
import logging, coloredlogs

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
)
from src.utils.settings import config
from src.models.icestupaClass import Icestupa
from src.automate.projectile import get_projectile

if __name__ == "__main__":
    # Main logger
    logger = logging.getLogger(__name__)
    logger.setLevel("ERROR")
    location = 'altiplano20'
    location = 'gangles21'
    # sprays = ['scheduled_field', 'unscheduled_field']
    sprays = 'unscheduled_field'

    mypal = sns.color_palette("Set1", 2)
    default = "#284D58"
    grey = "#ced4da"
    legend_elements = [Line2D([0], [0], color=mypal[0], lw=4, label='Scheduled'),
                        Line2D([0], [0], color=mypal[1], lw=4, label='Non-scheduled'),
                        Line2D([0], [0], color=default, lw=4, label='Measured Temperature'),
                       Line2D([0], [0], marker='.', color='w', label='Measured Volume',
                              markerfacecolor='k', markersize=15),
                        Line2D([0], [0], color=grey, lw=4, label='Dome Volume'),
                       ]

    fig, ax = plt.subplots(2, 1, gridspec_kw={'height_ratios': [1,1]}, sharex="col")

    # SITE, FOLDER = config(location)
    icestupa = Icestupa(location, sprays)
    icestupa.read_output()
    # df=icestupa.df
    df = icestupa.df.set_index("time").resample("D").mean().reset_index()

    x = df.time[1:]

    y1 = df.temp[1:]
    ax[0].plot(
        x,
        y1,
        linewidth=0.8,
        color=default,
    )
    ax[0].axhline(y=0, color = 'k', linestyle = '--', alpha = 0.5, linewidth=0.9)
    ax[0].spines["right"].set_visible(False)
    ax[0].spines["top"].set_visible(False)
    ax[0].spines["left"].set_color("grey")
    ax[0].spines["bottom"].set_color("grey")
    ax[0].set_ylabel("Temperature [$\degree C$]")
    ax[0].set_ylim([-15,15])

    y2 = df.iceV[1:]
    ax[1].plot(
        x,
        y2,
        linewidth=1,
        color=mypal[1],
    )
    ax[1].spines["right"].set_visible(False)
    ax[1].spines["top"].set_visible(False)
    ax[1].spines["left"].set_color("grey")
    ax[1].spines["bottom"].set_color("grey")
    ax[1].set_ylabel("Ice Volume [$m^3$]")

    # ax[1].set_ylim([0,15])
    # ax[0].set_ylim([-13,10])

    # ax[1].xaxis.set_major_locator(mdates.WeekdayLocator())
    ax[1].xaxis.set_major_locator(mdates.MonthLocator())
    ax[1].xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    fig.subplots_adjust(hspace=None, wspace=None)
    fig.autofmt_xdate()
    # handles, labels = ax[1].get_legend_handles_labels()
    # ax[1].legend(handles, labels, loc="upper right", prop={"size": 8}, title="Fountain spray")
    plt.savefig(
        icestupa.fig + "/tempvsvol.png",
        bbox_inches="tight",
        dpi=300,
    )
    plt.close()

