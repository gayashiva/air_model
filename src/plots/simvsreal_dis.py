
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

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
)
from src.utils.settings import config
from src.models.icestupaClass import Icestupa
from src.models.methods.metadata import get_parameter_metadata
from src.automate.gen_coeffs import autoDis
from src.automate.autoDischarge import TempFreeze
from lmfit.models import GaussianModel

if __name__ == "__main__":
    # Main logger
    logger = logging.getLogger(__name__)
    logger.setLevel("WARNING")

    loc= "Guttannen 2022"
    icestupa = Icestupa(loc, spray="manual")
    SITE, FOLDER = config(loc, spray="manual")
    icestupa.read_output()
    df = icestupa.df
    # df = df[df.time<datetime(2022,1,26)] #automation error

    objs = ["WUE", "ICV"]
    styles=['.', 'x']
    # objs = ["WUE"]
    # objs = ["ICV"]

    default = "#284D58"
    fig = plt.figure()
    ax1 = fig.add_subplot(111)

    for j,obj in enumerate(objs):
        print(j,obj)
        with open(FOLDER["input"] + "dynamic/coeffs_" + obj + ".json") as f:
            params = json.load(f)
        sun_params = {"amplitude": params["amplitude"], "center": params["center"], "sigma": params["sigma"]}

        for i in range(0,df.shape[0]):
            # df.loc[i, "Discharge_sim"] = autoDis(**params, time=df.time.dt.hour[i], temp=df.temp[i],rh=df.RH[i],wind=df.wind[i])
            data=dict()
            data["temp"] = df.temp[i]
            data["rh"] = df.RH[i]
            data["wind"] = df.wind[i]
            data["obj"] = obj
            data["alt"] = SITE["alt"]/1000
            model = GaussianModel()

            df.loc[i, "Discharge_sim"] = TempFreeze(data) + model.eval(x=df.time.dt.hour[i], **sun_params)
            
            if obj == "ICV":
                df.loc[i, "Discharge_sim"]*= math.sqrt(2) * math.pi * icestupa.R_F **2
            if obj == "WUE":
                df.loc[i, "Discharge_sim"]*= math.pi * icestupa.R_F **2
            # df.loc[i, "Discharge_sim"]*= math.pi * 7 **2

            # TODO correct with params
            if df.Discharge_sim[i] < 0:
                df.loc[i, "Discharge_sim"] = 0
            if df.Discharge_sim[i] >= 11:
                df.loc[i, "Discharge_sim"] = 11
            # if df.wind[i] >= 8 or df.temp[i] > -2 or df.temp[i] < -8:
            #     df.loc[i, "Discharge_sim"] = 0

        df["fountain_froze"] = np.where(df.fountain_froze == 0, np.nan, df.fountain_froze)
        df["Discharge_sim"] = np.where(df.Discharge_sim == 0, np.nan, df.Discharge_sim)
        column_1 = "fountain_froze"
        column_2 = "Discharge_sim"
        correlation = df[column_1].corr(icestupa.df[column_2])
        print("Correlation between %s and %s is %0.2f"%(column_1, column_2, correlation))

        # fig, ax = plt.subplots(2, 1, sharex="col")

        x = df.time[1:]
        y1 = df.fountain_froze[1:]/60
        y2 = df.Discharge_sim[1:]
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

        ax1.scatter(y1, y2, s=10, marker=styles[j], color=default, label = obj)
        ax1.set_xlabel("Validated freezing rate [$l/min$]")
        ax1.set_ylabel("Scheduled discharge rate [$l/min$]")
        ax1.grid()

        # lims = [
        #     np.min([ax1.get_xlim(), ax1.get_ylim()]),  # min of both axes
        #     np.max([ax1.get_xlim(), ax1.get_ylim()]),  # max of both axes
        # ]
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
