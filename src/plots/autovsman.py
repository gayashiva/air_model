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

    # fig, ax = plt.subplots(2, 1, gridspec_kw={'height_ratios': [1, 3]}, sharex="col")

    # for i, spray in enumerate(sprays):
    #     SITE, FOLDER = config(location, spray)
    #     icestupa = Icestupa(location, spray)
    #     icestupa.read_output()
    #     df=icestupa.df

    #     df_c = pd.read_hdf(FOLDER["input_sim"] + "/input.h5", "df_c")
    #     df_c = df_c[["time", "DroneV", "DroneVError"]]

    #     tol = pd.Timedelta("15T")
    #     df_c = df_c.set_index("time")
    #     df = df.set_index("time")
    #     df_c = pd.merge_asof(
    #         left=df,
    #         right=df_c,
    #         right_index=True,
    #         left_index=True,
    #         direction="nearest",
    #         tolerance=tol,
    #     )
    #     df_c = df_c[["DroneV", "DroneVError", "iceV"]]
    #     df = df.reset_index()

    #     if spray == "scheduled_field":
    #         spray = "Scheduled"
    #     else:
    #         spray = "Unscheduled"

    #     df["T_ice_mean"] = (df["T_s"] + df["T_bulk"])/2
    #     df["T_bulk_meas"] = np.where(df.T_bulk_meas < 0,df.T_bulk_meas, np.NaN)
    #     df["T_ice_mean"] = np.where(df.T_ice_mean < 0,df.T_ice_mean, np.NaN)
    #     x = df.time[1:]
    #     y1 = (df.T_s[1:] + df.T_bulk[1:])/2
    #     y1t = df.T_bulk_meas[1:]

    #     column_1 = "T_ice_mean"
    #     column_2 = "T_bulk_meas"
    #     correlation = df[column_1].corr(icestupa.df[column_2])
    #     print("Correlation between %s and %s is %0.2f"%(column_1, column_2, correlation))

    #     if spray != "Unscheduled":
    #         ax[0].plot(
    #             x,
    #             y1,
    #             linewidth=0.5,
    #             color=mypal[i],
    #             zorder = i+1,
    #             alpha=0.8,
    #         )
    #         ax[0].plot(
    #             x,
    #             y1t,
    #             linewidth=1,
    #             color=default,
    #             zorder=0
    #         )
    #         ax[0].spines["right"].set_visible(False)
    #         ax[0].spines["top"].set_visible(False)
    #         ax[0].spines["left"].set_color("grey")
    #         ax[0].spines["bottom"].set_color("grey")
    #         ax[0].set_ylabel("Bulk Temperature [$\degree C$]", size=6)
    #         ax[0].set_ylim([-20,0])

    #     y2 = df.iceV[1:]
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

    #     x = df_c.index
    #     y2 = df_c.DroneV
    #     yerr = df_c.DroneVError
    #     ax[1].fill_between(x, y1=icestupa.V_dome, y2=0, color=grey, label="Dome Volume")
    #     ax[1].scatter(x, y2, color=mypal[i], s=8)
    #     ax[1].errorbar(x, y2, yerr=df_c.DroneVError, color=mypal[i], linewidth=1)
    #     ax[1].set_ylim([0,80])


    # ax[1].xaxis.set_major_locator(mdates.WeekdayLocator())
    # ax[1].xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    # fig.subplots_adjust(hspace=None, wspace=None)
    # fig.autofmt_xdate()
    # ax[1].legend(handles=legend_elements, prop={"size": 8})
    # plt.savefig("data/figs/paper3/validation.png", bbox_inches="tight", dpi=300)

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

    legend_elements = [Line2D([0], [0], color=mypal[0], lw=4, label='Automatic AIR Volume'),
                        Line2D([0], [0], color=mypal[1], lw=4, label='Traditional AIR Volume'),
                       Line2D([0], [0], marker='.', color='w', label='Measured AIR Volume',
                              markerfacecolor='k', markersize=15),
                       ]
    ax.legend(handles=legend_elements, prop={"size": 8})
    plt.savefig("data/figs/paper3/validation.png", bbox_inches="tight", dpi=300)
    plt.clf()

