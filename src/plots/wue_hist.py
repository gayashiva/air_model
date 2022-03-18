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
from matplotlib.lines import Line2D
import matplotlib.dates as mdates
import logging, coloredlogs

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
)
from src.utils.settings import config
from src.models.icestupaClass import Icestupa
# from src.models.methods.metadata import get_parameter_metadata

def keystoint(x):
    return {k: float(v) for k, v in x.items()}

if __name__ == "__main__":
    # Main logger
    logger = logging.getLogger(__name__)
    logger.setLevel("INFO")

    locations = ["guttannen22", "guttannen21", "gangles21"]
    sprays = ['manual', 'static', 'dynamic']
    styles=['.', 'x' , '*']

    mypal = sns.color_palette("Set1", 3)
    legend_elements = [Line2D([0], [0], color=mypal[0], lw=4, label='CH22'),
                        Line2D([0], [0], color=mypal[1], lw=4, label='CH21'),
                        Line2D([0], [0], color=mypal[2], lw=4, label='IN21'),
                       Line2D([0], [0], marker='.', color='w', label='Unscheduled',
                              markerfacecolor='k', markersize=15),
                       Line2D([0], [0], marker='X', color='w', label='ICV Scheduled',
                              markerfacecolor='k', markersize=10),
                       Line2D([0], [0], marker='*', color='w', label='WUE Scheduled',
                              markerfacecolor='k', markersize=15),
                       ]
    fig, ax = plt.subplots(1, 1, sharex="col")

    for i, loc in enumerate(locations):
        for j, spray in enumerate(sprays):

            SITE, FOLDER = config(loc)

            with open(FOLDER["output"] + spray +  "/results.json") as f:
                results = json.load(f, object_hook=keystoint)
            print(loc,spray, results["WUE"], results["iceV_max"])
            ax.scatter(results["WUE"], results["iceV_max"], label=loc+spray, color=mypal[i], marker=styles[j])

            if loc == 'guttannen22' and spray == "dynamic":
                with open(FOLDER["output"] + "dynamic_field/results.json") as f:
                    results = json.load(f, object_hook=keystoint)
                print(loc,"dynamic_field", results["WUE"], results["iceV_max"])
                ax.scatter(results["WUE"], results["iceV_max"], color=mypal[i], marker=styles[j])

        # ax = df_l.set_index('x')['y'].plot(style='.', color='k', ms=10)

    ax.set_ylabel("Max Ice Volume [$m^3$]")
    ax.set_xlabel("Water Use Efficiency [%]")
    ax.set_xlim([0,100])
    ax.set_ylim([0,1200])
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.legend(handles=legend_elements)
    plt.savefig("data/figs/paper3/wue.png", bbox_inches="tight", dpi=300)


    #     ax[0].spines["left"].set_color("grey")
    #     ax[0].spines["bottom"].set_color("grey")
    #     ax[0].set_ylabel("Discharge [$l/min$]")

    #     ax[1].plot(
    #         x,
    #         y2,
    #         label= spray,
    #         linewidth=1,
    #         color=mypal[i],
    #     )
    #     ax[1].spines["right"].set_visible(False)
    #     ax[1].spines["top"].set_visible(False)
    #     ax[1].spines["left"].set_color("grey")
    #     ax[1].spines["bottom"].set_color("grey")
    #     ax[1].set_ylabel("Ice Volume [$m^3$]")


    # ax[1].xaxis.set_major_locator(mdates.WeekdayLocator())
    # ax[1].xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    # # ax[1].xaxis.set_minor_locator(mdates.DayLocator())
    # fig.autofmt_xdate()
    # # fig.text(0.04, 0.5, "Ice Volume[$m^3$]", va="center", rotation="vertical")
    # handles, labels = ax[1].get_legend_handles_labels()
    # fig.legend(handles, labels, loc="upper right", prop={"size": 8})
    # # plt.legend()
    # plt.savefig("data/figs/paper3/autovsmanual.png", bbox_inches="tight", dpi=300)
