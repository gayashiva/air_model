""" Plots for presentations"""
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
from src.models.methods.metadata import get_parameter_metadata

if __name__ == "__main__":
    # Main logger
    logger = logging.getLogger(__name__)
    logger.setLevel("ERROR")


    mypal = sns.color_palette("Set1", 2)
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

    for fig in range(0,3):
        if fig == 0:

            location= 'guttannen21'
            SITE, FOLDER = config('guttannen21', 'man')
            icestupa = Icestupa('guttannen21', spray="man")
            icestupa.read_output()
            df = icestupa.df

            df_c = pd.read_hdf(FOLDER["input"] + "input.h5", "df_c")
            if icestupa.name in ["guttannen21", "guttannen20", "gangles21"]:
                df_c = df_c[1:]
            # df_c = df_c.set_index("time").resample("D").mean().reset_index()
            dfv = df_c[["time", "DroneV", "DroneVError"]]

            fig, ax = plt.subplots()
            x = df.time[1:]
            y1 = df.iceV[1:]
            x2 = dfv.time
            y2 = dfv.DroneV
            ax.plot(
                x,
                y1,
                linewidth=1,
                color=CB91_Blue,
            )
            ax.scatter(x2, y2, s=10, color=CB91_Violet, zorder=7, label="UAV Volume")

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
            ax.xaxis.set_major_locator(mdates.MonthLocator())
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
            fig.autofmt_xdate()

            ax.set_ylabel("Ice Volume[$m^3$]")
            ax.legend(loc="upper right", prop={"size": 8})
            plt.savefig("data/figs/slides/guttannen21_1.png", bbox_inches="tight", dpi=300)


            df= df[df.time <= SITE["fountain_off_date"]]
            fig, ax = plt.subplots()
            x = df.time[1:]
            y1 = df.iceV[1:]
            ax.plot(
                x,
                y1,
                linewidth=1,
                color=CB91_Blue,
            )

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
            ax.xaxis.set_major_locator(mdates.WeekdayLocator())
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
            fig.autofmt_xdate()

            ax.set_ylabel("Ice Volume[$m^3$]")
            ax.legend(loc="upper right", prop={"size": 8})
            plt.savefig("data/figs/slides/guttannen21_2.png", bbox_inches="tight", dpi=300)

        if fig == 1:
            """Compare freezing and discharge rate"""
            SITE, FOLDER = config('guttannen21', 'man')
            icestupa = Icestupa('guttannen21', spray="man")
            icestupa.read_output()
            df = icestupa.df
            df= df[df.time <= SITE["fountain_off_date"]]

            fig, ax1 = plt.subplots()
            x = df.time
            y1 = df.Discharge
            y2 = df.fountain_froze /60
            ax1.plot(
                x,
                y1,
                color=mypal[0],
            )
            ax1.set_ylim(0, 20)

            ax2 = ax1.twinx() 
            ax2.set_ylabel('Freezing rate [$l/min$]', color = mypal[1]) 
            ax2.plot(x, y2, color = mypal[1]) 
            ax2.tick_params(axis ='y', labelcolor = mypal[1]) 
            ax2.set_ylim(0, 20)

            ax1.spines["top"].set_visible(False)
            ax2.spines["top"].set_visible(False)
            ax1.set_ylabel("Discharge rate [$l/min$]")
            ax1.xaxis.set_major_locator(mdates.WeekdayLocator())
            ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
            fig.autofmt_xdate()
            plt.savefig("data/figs/slides/guttannen21_freezingvsdischarge.png", bbox_inches="tight", dpi=300)

        if fig == 2:
            location = 'guttannen22'
            sprays = ['auto_field', 'man']

            fig, ax = plt.subplots(2, 1, sharex="col")

            with open("data/common/constants.json") as f:
                CONSTANTS = json.load(f)


            """Compare Auto vs Man Icestupa fig"""
            for i, spray in enumerate(sprays):
                SITE, FOLDER = config(location, spray)
                df = pd.read_hdf(FOLDER["output"] + spray + "/output.h5", "df")
                df= df[df.time <= SITE["fountain_off_date"]]

                if spray == "auto_field":
                    spray = "Automatic"
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
                handles, labels = ax[1].get_legend_handles_labels()
                ax[0].legend(handles, labels, loc="upper right", prop={"size": 8}, title="Method")


            ax[1].xaxis.set_major_locator(mdates.WeekdayLocator())
            ax[1].xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
            fig.autofmt_xdate()
            plt.savefig("data/figs/slides/guttannen22_icev_discharge.png", bbox_inches="tight", dpi=300)
