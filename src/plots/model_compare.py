""" Plot comparing cosipy model"""

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
from matplotlib.ticker import AutoMinorLocator, FormatStrFormatter
import matplotlib.dates as mdates
import logging, coloredlogs

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
)
from src.utils.settings import config
from src.models.icestupaClass import Icestupa
from src.automate.projectile import get_projectile
# from src.models.methods.metadata import get_parameter_metadata

if __name__ == "__main__":
    # Main logger
    logger = logging.getLogger(__name__)
    logger.setLevel("ERROR")

    locations = ['gangles21', 'guttannen21', 'guttannen22' ]
    shortname = ['IN21', 'CH21', 'CH22']
    spray = 'unscheduled_field'

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
        print(location)
        SITE, FOLDER = config(location, spray)
        icestupa = Icestupa(location, spray)
        icestupa.read_output()
        df = icestupa.df[["time", "iceV"]]
        df2 = pd.read_csv('/home/suryab/work/air_model/data/cosipy/' + location + '.csv')
        df2['time'] = pd.to_datetime(df2['time'])
        if location in ['guttannen21', 'guttannen22']:
            df2 = df2[1:]
        print(df2.head(1), df.head(1))

        if location == "guttannen20":
            SITE["start_date"] += pd.offsets.DateOffset(year=2023)
            SITE["expiry_date"] += pd.offsets.DateOffset(year=2023)
        if location == "guttannen22":
            SITE["start_date"] += pd.offsets.DateOffset(year=2022)
            SITE["expiry_date"] += pd.offsets.DateOffset(year=2023)
        if location == "guttannen21":
            SITE["start_date"] += pd.offsets.DateOffset(year=2022)
            SITE["expiry_date"] += pd.offsets.DateOffset(year=2023)
        if location == "gangles21":
            SITE["start_date"] += pd.offsets.DateOffset(year=2023)
            # SITE["end_date"] =SITE['expiry_date'] + pd.offsets.DateOffset(year=2023)

        if location == 'guttannen22':
            df_c = pd.read_hdf(FOLDER["input_sim"]  + "/input.h5", "df_c")
        else:
            df_c = pd.read_hdf(FOLDER["input"]  + "/input.h5", "df_c")

        df_c = df_c[["time", "DroneV", "DroneVError"]]

        if icestupa.name in ["guttannen21", "guttannen20", "gangles21"]:
            df_c = df_c[1:]

        df_c = df_c.set_index("time").resample("D").mean().reset_index()
        dfv = df_c[["time", "DroneV", "DroneVError"]]

        if location == "guttannen20":
            df["time"] = df["time"].mask(
                icestupa.df["time"].dt.year == 2019,
                icestupa.df["time"] + pd.offsets.DateOffset(year=2022),
            )
            df["time"] = df["time"].mask(
                icestupa.df["time"].dt.year == 2020,
                icestupa.df["time"] + pd.offsets.DateOffset(year=2023),
            )
            df2["time"] = df2["time"].mask(
                df2["time"].dt.year == 2019,
                df2["time"] + pd.offsets.DateOffset(year=2022),
            )
            df2["time"] = df2["time"].mask(
                df2["time"].dt.year == 2020,
                df2["time"] + pd.offsets.DateOffset(year=2023),
            )
            dfv["time"] = dfv["time"].mask(
                df_c["time"].dt.year == 2019,
                df_c["time"] + pd.offsets.DateOffset(year=2022),
            )
            dfv["time"] = dfv["time"].mask(
                df_c["time"].dt.year == 2020,
                df_c["time"] + pd.offsets.DateOffset(year=2023),
            )


        if location == "guttannen21":
            df["time"] = df["time"].mask(
                icestupa.df["time"].dt.year == 2020,
                icestupa.df["time"] + pd.offsets.DateOffset(year=2022),
            )
            df2["time"] = df2["time"].mask(
                df2["time"].dt.year == 2020,
                df2["time"] + pd.offsets.DateOffset(year=2022),
            )
            dfv["time"] = dfv["time"].mask(
                df_c["time"].dt.year == 2020,
                df_c["time"] + pd.offsets.DateOffset(year=2022),
            )

        if location == "guttannen22":
            df["time"] = df["time"].mask(
                icestupa.df["time"].dt.year == 2021,
                icestupa.df["time"] + pd.offsets.DateOffset(year=2022),
            )
            df["time"] = df["time"].mask(
                icestupa.df["time"].dt.year == 2022,
                icestupa.df["time"] + pd.offsets.DateOffset(year=2023),
            )
            df2["time"] = df2["time"].mask(
                df2["time"].dt.year == 2021,
                df2["time"] + pd.offsets.DateOffset(year=2022),
            )
            df2["time"] = df2["time"].mask(
                df2["time"].dt.year == 2022,
                df2["time"] + pd.offsets.DateOffset(year=2023),
            )
            dfv["time"] = dfv["time"].mask(
                df_c["time"].dt.year == 2021,
                df_c["time"] + pd.offsets.DateOffset(year=2022),
            )
            dfv["time"] = dfv["time"].mask(
                df_c["time"].dt.year == 2022,
                df_c["time"] + pd.offsets.DateOffset(year=2023),
            )

        else:
            df["time"] = df["time"].mask(
                icestupa.df["time"].dt.year == 2021,
                icestupa.df["time"] + pd.offsets.DateOffset(year=2023),
            )
            df2["time"] = df2["time"].mask(
                df2["time"].dt.year == 2021,
                df2["time"] + pd.offsets.DateOffset(year=2023),
            )
            dfv["time"] = dfv["time"].mask(
                df_c["time"].dt.year == 2021,
                df_c["time"] + pd.offsets.DateOffset(year=2023),
            )

        # df_out[location] = icestupa.df["iceV"]
        df = df.reset_index()
        df2 = df2.reset_index()

        x = df.time[1:]
        y1 = df.iceV[1:]
        y12 = df2.iceV[1:]
        x2 = dfv.time
        y2 = dfv.DroneV
        yerr = dfv.DroneVError
        ax[i].plot(
            x,
            y1,
            label="AIR Model",
            linewidth=1,
            color=CB91_Blue,
            # color=CB91_Green,
            zorder=10,
        )
        ax[i].plot(
            x,
            y12,
            label="CosiStupa Model",
            linewidth=1,
            # color=CB91_Blue,
                linestyle="--",
            color=CB91_Amber,
            zorder=10,
        )
        ax[i].errorbar(
            x2, y2, yerr=df_c.DroneVError, color=CB91_Violet, lw=1, alpha=0.5, zorder=8
        )
        ax[i].scatter(x2, y2, s=5, color=CB91_Violet, zorder=7, label="UAV Volume")

        # if location != "gangles21":
        #     ax[i].axvline(
        #         SITE["expiry_date"],
        #         color="black",
        #         alpha=0.5,
        #         linestyle="--",
        #         zorder=2,
        #         label="Expiry date",
        #     )

        at = AnchoredText(
            shortname[i], prop=dict(size=10), frameon=True, loc="upper left"
        )
        at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
        if i != 2:
            x_axis = ax[i].axes.get_xaxis()
            x_axis.set_visible(False)
            ax[i].spines["bottom"].set_visible(False)
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
        ax[i].yaxis.set_major_locator(plt.LinearLocator(numticks=2))
        ax[i].yaxis.set_major_formatter(FormatStrFormatter("%i"))
        ax[i].xaxis.set_major_locator(mdates.MonthLocator())
        ax[i].xaxis.set_major_formatter(mdates.DateFormatter("%b"))
        fig.autofmt_xdate()

    fig.text(0.04, 0.5, "Ice Volume[$m^3$]", va="center", rotation="vertical")
    handles, labels = ax[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", prop={"size": 8})
    plt.savefig("data/figs/thesis/model_compare.jpg", bbox_inches="tight", dpi=300)


    locations = ['gangles21', 'guttannen21']
    shortname = ['IN21', 'CH21']
    spray = 'unscheduled_field'

    fig, ax = plt.subplots(len(locations), 1, sharex="col")

    for i, location in enumerate(locations):
        print(location)
        SITE, FOLDER = config(location, spray)
        icestupa = Icestupa(location, spray)
        icestupa.read_output()
        df = icestupa.df[["time", "iceV"]]

        if location == "guttannen20":
            SITE["start_date"] += pd.offsets.DateOffset(year=2023)
            SITE["expiry_date"] += pd.offsets.DateOffset(year=2023)
        if location == "guttannen22":
            SITE["start_date"] += pd.offsets.DateOffset(year=2022)
            SITE["expiry_date"] += pd.offsets.DateOffset(year=2023)
        if location == "guttannen21":
            SITE["start_date"] += pd.offsets.DateOffset(year=2022)
            SITE["expiry_date"] += pd.offsets.DateOffset(year=2023)
        if location == "gangles21":
            SITE["start_date"] += pd.offsets.DateOffset(year=2023)
            # SITE["end_date"] =SITE['expiry_date'] + pd.offsets.DateOffset(year=2023)

        if location == 'guttannen22':
            df_c = pd.read_hdf(FOLDER["input_sim"]  + "/input.h5", "df_c")
        else:
            df_c = pd.read_hdf(FOLDER["input"]  + "/input.h5", "df_c")

        df_c = df_c[["time", "DroneV", "DroneVError"]]

        if icestupa.name in ["guttannen21", "guttannen20", "gangles21"]:
            df_c = df_c[1:]

        df_c = df_c.set_index("time").resample("D").mean().reset_index()
        dfv = df_c[["time", "DroneV", "DroneVError"]]

        if location == "guttannen20":
            df["time"] = df["time"].mask(
                icestupa.df["time"].dt.year == 2019,
                icestupa.df["time"] + pd.offsets.DateOffset(year=2022),
            )
            df["time"] = df["time"].mask(
                icestupa.df["time"].dt.year == 2020,
                icestupa.df["time"] + pd.offsets.DateOffset(year=2023),
            )
            dfv["time"] = dfv["time"].mask(
                df_c["time"].dt.year == 2019,
                df_c["time"] + pd.offsets.DateOffset(year=2022),
            )
            dfv["time"] = dfv["time"].mask(
                df_c["time"].dt.year == 2020,
                df_c["time"] + pd.offsets.DateOffset(year=2023),
            )

        if location == "guttannen21":
            df["time"] = df["time"].mask(
                icestupa.df["time"].dt.year == 2020,
                icestupa.df["time"] + pd.offsets.DateOffset(year=2022),
            )
            dfv["time"] = dfv["time"].mask(
                df_c["time"].dt.year == 2020,
                df_c["time"] + pd.offsets.DateOffset(year=2022),
            )

        if location == "guttannen22":
            df["time"] = df["time"].mask(
                icestupa.df["time"].dt.year == 2021,
                icestupa.df["time"] + pd.offsets.DateOffset(year=2022),
            )
            df["time"] = df["time"].mask(
                icestupa.df["time"].dt.year == 2022,
                icestupa.df["time"] + pd.offsets.DateOffset(year=2023),
            )
            dfv["time"] = dfv["time"].mask(
                df_c["time"].dt.year == 2021,
                df_c["time"] + pd.offsets.DateOffset(year=2022),
            )
            dfv["time"] = dfv["time"].mask(
                df_c["time"].dt.year == 2022,
                df_c["time"] + pd.offsets.DateOffset(year=2023),
            )

        else:
            df["time"] = df["time"].mask(
                icestupa.df["time"].dt.year == 2021,
                icestupa.df["time"] + pd.offsets.DateOffset(year=2023),
            )
            dfv["time"] = dfv["time"].mask(
                df_c["time"].dt.year == 2021,
                df_c["time"] + pd.offsets.DateOffset(year=2023),
            )

        df = df.reset_index()

        x = df.time[1:]
        y1 = df.iceV[1:]
        x2 = dfv.time
        y2 = dfv.DroneV
        yerr = dfv.DroneVError
        ax[i].plot(
            x,
            y1,
            label="AIR Model",
            linewidth=1,
            color=CB91_Blue,
            # color=CB91_Green,
            zorder=10,
        )
        ax[i].errorbar(
            x2, y2, yerr=df_c.DroneVError, color=CB91_Violet, lw=1, alpha=0.5, zorder=8
        )
        ax[i].scatter(x2, y2, s=5, color=CB91_Violet, zorder=7, label="UAV Volume")

        at = AnchoredText(
            shortname[i], prop=dict(size=10), frameon=True, loc="upper left"
        )
        at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
        if i != len(locations) - 1:
            x_axis = ax[i].axes.get_xaxis()
            x_axis.set_visible(False)
            ax[i].spines["bottom"].set_visible(False)
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
        ax[i].yaxis.set_major_locator(plt.LinearLocator(numticks=2))
        ax[i].yaxis.set_major_formatter(FormatStrFormatter("%i"))
        ax[i].xaxis.set_major_locator(mdates.MonthLocator())
        ax[i].xaxis.set_major_formatter(mdates.DateFormatter("%b"))
        fig.autofmt_xdate()

    fig.text(0.04, 0.5, "Ice Volume[$m^3$]", va="center", rotation="vertical")
    handles, labels = ax[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", prop={"size": 8})
    plt.savefig("data/figs/thesis/IN21vsCH21.jpg", bbox_inches="tight", dpi=300)
