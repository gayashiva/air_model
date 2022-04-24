""" Plot comparing simulated and real discharge of automated guttannen22"""

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
from sklearn.metrics import mean_squared_error

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
)
from src.utils.settings import config
from src.models.icestupaClass import Icestupa

if __name__ == "__main__":
    # Main logger
    logger = logging.getLogger(__name__)
    logger.setLevel("WARNING")

    loc= "Guttannen 2022"
    icestupa = Icestupa(loc, spray="unscheduled_field")
    SITE, FOLDER = config(loc, spray="unscheduled_field")
    icestupa.read_output()
    df = icestupa.df


    df_f = pd.read_csv(FOLDER['input'] + "discharge_types.csv", sep=",", header=0, parse_dates=["time"])
    df_f = df_f[["time", "scheduled_wue", "scheduled_icv"]]

    objs = ["wue", "icv"]
    styles=['.', 'x']

    default = "#284D58"
    fig = plt.figure()
    ax1 = fig.add_subplot(111)


    for j, obj in enumerate(objs):
        print(j,obj)

        column_1 = "fountain_froze"
        column_2 = "Discharge"
        icestupa = Icestupa(loc, spray="scheduled_"+obj)
        df_f = icestupa.df
        corr= df[column_1].corr(df_f[column_2])
        print("Correlation between %s and %s is %0.2f"%(column_1, column_2, corr))
        rmse = mean_squared_error(df_f[column_2], df[column_1]/60, squared=False)
        print(f"Calculated correlation {corr} and RMSE {rmse}")

        # df["fountain_froze"] = np.where(df.fountain_froze == 0, np.nan, df.fountain_froze)
        # df_f["Discharge"] = np.where(df_f.Discharge == 0, np.nan, df_f.Discharge)


        # fig, ax = plt.subplots(2, 1, sharex="col")

        x = df.time[1:]
        y1 = df.fountain_froze[1:]/60
        y2 = df_f.Discharge[1:]
        ax1.scatter(y1, y2, s=10, marker=styles[j], color=default, label = obj)
        ax1.set_xlabel("Validated freezing rate [$l/min$]")
        ax1.set_ylabel("Scheduled discharge rate [$l/min$]")
        ax1.grid()

        lims = [
            np.min([ax1.get_xlim(), ax1.get_ylim()]),  # min of both axes
            np.max([ax1.get_xlim(), ax1.get_ylim()]),  # max of both axes
        ]
        lims = [0,2.5]

        # now plot both limits against eachother
        ax1.plot(lims, lims, "--k", alpha=0.25, zorder=0)
        ax1.set_aspect("equal")
        ax1.set_xlim(lims)
        ax1.set_ylim(lims)

    ax1.legend(prop={"size": 8}, title="Objective", loc="upper right")
    plt.savefig(
        "data/figs/paper3/freezing_rate_corr.png",
        bbox_inches="tight",
        dpi=300,
    )

        # ax[0].plot(
        #     x,
        #     y1,
        #     # label= spray + "Discharge",
        #     linewidth=1,
        #     # color=mypal[i],
        # )
        # ax[0].spines["right"].set_visible(False)
        # ax[0].spines["top"].set_visible(False)
        # ax[0].spines["left"].set_color("grey")
        # ax[0].spines["bottom"].set_color("grey")
        # ax[0].set_ylabel("Freezing rate[$l/min$]")
        # ax[0].set_ylim([0,2])

        # ax[1].plot(
        #     x,
        #     y2,
        #     # label= spray,
        #     linewidth=1,
        #     # color=mypal[i],
        # )
        # ax[1].spines["right"].set_visible(False)
        # ax[1].spines["top"].set_visible(False)
        # ax[1].spines["left"].set_color("grey")
        # ax[1].spines["bottom"].set_color("grey")
        # ax[1].set_ylabel("Sim Discharge [$l/min$]")
        # ax[1].set_ylim([0,2])


        # ax[1].xaxis.set_major_locator(mdates.WeekdayLocator())
        # ax[1].xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
        # fig.autofmt_xdate()
        # handles, labels = ax[1].get_legend_handles_labels()
        # fig.legend(handles, labels, loc="upper right", prop={"size": 8})
        # plt.savefig("data/figs/paper3/simvsreal_dis.jpg", bbox_inches="tight", dpi=300)
        # plt.clf()
