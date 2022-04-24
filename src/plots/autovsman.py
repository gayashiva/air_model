""" Plot comparing auto and manual discharge at guttannen"""

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
# from src.models.methods.metadata import get_parameter_metadata

if __name__ == "__main__":
    # Main logger
    logger = logging.getLogger(__name__)
    logger.setLevel("ERROR")
    location = 'guttannen22'
    sprays = ['scheduled_field', 'unscheduled_field']
    # sprays = ['dynamic_field']

    mypal = sns.color_palette("Set1", 2)
    default = "#284D58"
    grey = "#ced4da"
    legend_elements = [Line2D([0], [0], color=mypal[0], lw=4, label='Scheduled'),
                        Line2D([0], [0], color=mypal[1], lw=4, label='Unscheduled'),
                        Line2D([0], [0], color=default, lw=4, label='Measured Temperature'),
                       Line2D([0], [0], marker='.', color='w', label='Measured Volume',
                              markerfacecolor='k', markersize=15),
                        Line2D([0], [0], color=grey, lw=4, label='Dome Volume'),
                       ]

    fig, ax = plt.subplots(3, 1, gridspec_kw={'height_ratios': [1,1,2]}, sharex="col")
    for i, spray in enumerate(sprays):
        SITE, FOLDER = config(location, spray)
        icestupa = Icestupa(location, spray)
        icestupa.read_output()
        df=icestupa.df

        if spray == "scheduled_field":
            spray = "Scheduled"
        else:
            spray = "Unscheduled"

        x = df.time[1:]
        y1 = df.temp[1:]
        ax[0].plot(
            x,
            y1,
            linewidth=0.8,
            color=default,
        )
        ax[0].spines["right"].set_visible(False)
        ax[0].spines["top"].set_visible(False)
        ax[0].spines["left"].set_color("grey")
        ax[0].spines["bottom"].set_color("grey")
        ax[0].set_ylabel("Temperature [$\degree C$]", size=6)

        y2 = df.ppt[1:]
        ax[1].plot(
            x,
            y2,
            linewidth=0.8,
            color=default,
        )
        ax[1].spines["right"].set_visible(False)
        ax[1].spines["top"].set_visible(False)
        ax[1].spines["left"].set_color("grey")
        ax[1].spines["bottom"].set_color("grey")
        ax[1].set_ylabel("Precipitation [$mm\, w.e.$]", size=6)

        y3 = df.Discharge[1:]
        ax[2].plot(
            x,
            y3,
            label= spray,
            linewidth=1,
            color=mypal[i],
        )
        ax[2].spines["right"].set_visible(False)
        ax[2].spines["top"].set_visible(False)
        ax[2].spines["left"].set_color("grey")
        ax[2].spines["bottom"].set_color("grey")
        ax[2].set_ylabel("Discharge [$l/min$]")

        # ax[1].set_ylim([0,14])
        # ax[0].set_ylim([-13,10])

    ax[2].xaxis.set_major_locator(mdates.WeekdayLocator())
    ax[2].xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    fig.subplots_adjust(hspace=None, wspace=None)
    fig.autofmt_xdate()
    handles, labels = ax[2].get_legend_handles_labels()
    ax[2].legend(handles, labels, loc="upper right", prop={"size": 8}, title="Fountain spray")
    plt.savefig("data/figs/paper3/data.png", bbox_inches="tight", dpi=300)
    plt.close()

    fig, ax = plt.subplots(2, 1, gridspec_kw={'height_ratios': [1,1]}, sharex="col")
    for i, spray in enumerate(sprays):
        SITE, FOLDER = config(location, spray)
        icestupa = Icestupa(location, spray)
        icestupa.read_output()
        df=icestupa.df
        dfd = df.set_index("time").resample("D").mean().reset_index()

        # dfd["time"] = dfd["time"].dt.strftime("%b %d")
        # z = dfd[
        #     [
        #         "alb",
        #     ]
        # ]

        if spray == "scheduled_field":
            spray = "Scheduled"
        else:
            spray = "Unscheduled"

        x = dfd.time[1:]
        y1 = dfd.alb[1:]
        y2 = dfd.Qf[1:]

        # z.plot.bar(
        #     # stacked=True,
        #     # edgecolor="black",
        #     linewidth=0.5,
        #     alpha = (i+1)*0.5,
        #     color=mypal[i],
        #     ax=ax[0],
        # )
        ax[0].plot(
            x,
            y1,
            linewidth=0.8,
            color=mypal[i],
        )
        ax[0].spines["right"].set_visible(False)
        ax[0].spines["top"].set_visible(False)
        ax[0].spines["left"].set_color("grey")
        ax[0].spines["bottom"].set_color("grey")
        ax[0].set_ylabel("Albedo", size=8)
        ax[0].set_ylim([0,1])
        at = AnchoredText("(a)", prop=dict(size=10), frameon=True, loc="upper left")
        at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
        ax[0].add_artist(at)

        ax[1].plot(
            x,
            y2,
            label= spray,
            linewidth=0.8,
            color=mypal[i],
        )
        ax[1].spines["right"].set_visible(False)
        ax[1].spines["top"].set_visible(False)
        ax[1].spines["left"].set_color("grey")
        ax[1].spines["bottom"].set_color("grey")
        ax[1].set_ylabel("Fountain heat flux [$W\\,m^{-2}$]", size=8)
        # ax[1].set_ylim([0,100])
        at = AnchoredText("(b)", prop=dict(size=10), frameon=True, loc="upper left")
        at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
        ax[1].add_artist(at)

    ax[1].xaxis.set_major_locator(mdates.WeekdayLocator())
    ax[1].xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    fig.subplots_adjust(hspace=None, wspace=None)
    fig.autofmt_xdate()
    handles, labels = ax[1].get_legend_handles_labels()
    ax[1].legend(handles, labels, loc="upper right", prop={"size": 8}, title="Fountain spray")
    plt.savefig("data/figs/paper3/dis_processes.png", bbox_inches="tight", dpi=300)
    plt.close()

    fig, ax = plt.subplots()
    for i, spray in enumerate(sprays):
        SITE, FOLDER = config(location, spray)
        icestupa = Icestupa(location, spray)
        icestupa.read_output()
        df=icestupa.df

        df_c = pd.read_hdf(FOLDER["input_sim"]  + "/input.h5", "df_c")
        df_c = df_c[["time", "DroneV", "DroneVError"]]

        # tol = pd.Timedelta("15T")
        # df_c = df_c.set_index("time")
        # df = df.set_index("time")
        # df_c = pd.merge_asof(
        #     left=df,
        #     right=df_c,
        #     right_index=True,
        #     left_index=True,
        #     direction="nearest",
        #     tolerance=tol,
        # )
        # df_c = df_c[["DroneV", "DroneVError", "iceV"]]
        # df = df.reset_index()

        if spray == "scheduled_field":
            spray = "Scheduled"
        else:
            spray = "Unscheduled"

        x = df.time[1:]
        y1 = df.iceV[1:]
        x2 = df_c.time
        y2 = df_c.DroneV
        yerr = df_c.DroneVError
        ax.set_ylabel("Ice Volume[$m^3$]")
        ax.plot(
            x,
            y1,
            label=spray,
            linewidth=1,
            color=mypal[i],
        )
        # ax.fill_between(x, y1=icestupa.V_dome, y2=0, color=grey, label="Dome Volume")
        ax.scatter(x2, y2, color=mypal[i], label="Measured Volume")
        ax.errorbar(x2, y2, yerr, color=mypal[i], linewidth=0, elinewidth=1)
        ax.set_ylim([0,70])
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.spines["left"].set_color("grey")
        ax.spines["bottom"].set_color("grey")
        ax.xaxis.set_major_locator(mdates.WeekdayLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
        ax.xaxis.set_minor_locator(mdates.DayLocator())
        fig.autofmt_xdate()

    legend_elements = [Line2D([0], [0], color=mypal[0], lw=4, label='Automatic'),
                        Line2D([0], [0], color=mypal[1], lw=4, label='Traditional'),
                       Line2D([0], [0], marker='.', color='w', label='Measured',
                              markerfacecolor='k', markersize=15),
                       ]
    ax.legend(handles=legend_elements, prop={"size": 8}, title='AIR Volume')
    plt.savefig("data/figs/paper3/validation.png", bbox_inches="tight", dpi=300)
    plt.clf()

    sprays = ['scheduled_field', 'scheduled_icv']
    # sprays = ['scheduled_icv']
    fig, ax = plt.subplots()
    ax.axhline(y=2, color='k', linestyle='--', alpha=0.5)
    # location = 'gangles21'
    for i, spray in enumerate(sprays):
        SITE, FOLDER = config(location, spray)
        icestupa = Icestupa(location, spray)
        icestupa.read_output()
        df=icestupa.df

        # df["Discharge"] = np.where(df.Discharge== 0, np.nan, df.Discharge)
        # print(f'Median discharge of {spray} is {df.Discharge.median()}')

        if spray == "scheduled_field":
            spray = "Automation system"
        else:
            spray = "Model simulated"

        x = df.time[1:]
        y1 = df.Discharge[1:]
        ax.set_ylabel("Scheduled discharge rate [$l/min$]")
        ax.plot(
            x,
            y1,
            label=spray,
            linewidth=1,
            color=mypal[i],
        )
        ax.set_ylim([0,15])
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.spines["left"].set_color("grey")
        ax.spines["bottom"].set_color("grey")
        ax.xaxis.set_major_locator(mdates.WeekdayLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
        ax.xaxis.set_minor_locator(mdates.DayLocator())
        fig.autofmt_xdate()

    legend_elements = [Line2D([0], [0], color=mypal[0], lw=2, label='Measured'),
                        Line2D([0], [0], color=mypal[1], lw=2, label='Modelled'),
                        Line2D([0], [0], color='k', lw=2, ls='--', alpha=0.5, label='Minimum'),
                       ]
    ax.legend(handles=legend_elements, prop={"size": 8}, title='Type')
    plt.savefig("data/figs/paper3/simvsreal.png", bbox_inches="tight", dpi=300)
    plt.clf()
