
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

# def autoDis(a, b, c, d, amplitude, center, sigma, temp, time, rh, v):
#     model = GaussianModel()
#     params = {"amplitude": amplitude, "center": center, "sigma": sigma}
#     return a * temp + b * rh + c * v + d + model.eval(x=time, **params)


if __name__ == "__main__":
    # Main logger
    logger = logging.getLogger(__name__)
    logger.setLevel("ERROR")

    location = "Guttannen 2022"
    icestupa = Icestupa(location, spray="auto")
    icestupa.read_output()
    df = icestupa.df

    with open(icestupa.sim + "coeffs.json") as f:
        param_values = json.load(f)

    for i in range(0,df.shape[0]):
        df.loc[i, "Discharge_sim"] = autoDis(**param_values, time=df.time.dt.hour[i], temp=df.temp[i],rh=df.RH[i], v=df.wind[i])
        # TODO correct with params
        if df.Discharge_sim[i] < 2:
            df.loc[i, "Discharge_sim"] = 0
        if df.Discharge_sim[i] >= 13:
            df.loc[i, "Discharge_sim"] = 13
        if df.wind[i] >= 8 or df.temp[i] > -2 or df.temp[i] < -8:
            df.loc[i, "Discharge_sim"] = 0
    logger.warning(df.Discharge_sim.describe())
    logger.warning(df.Discharge.describe())

    fig, ax = plt.subplots(2, 1, sharex="col")
    x = df.time[1:]
    y1 = df.Discharge[1:]
    y2 = df.Discharge_sim[1:]
    ax[0].plot(
        x,
        y1,
        # label= spray + "Discharge",
        linewidth=1,
        # color=mypal[i],
    )
    ax[0].spines["right"].set_visible(False)
    ax[0].spines["top"].set_visible(False)
    ax[0].spines["left"].set_color("grey")
    ax[0].spines["bottom"].set_color("grey")
    ax[0].set_ylabel("Discharge [$l/min$]")

    ax[1].plot(
        x,
        y2,
        # label= spray,
        linewidth=1,
        # color=mypal[i],
    )
    ax[1].spines["right"].set_visible(False)
    ax[1].spines["top"].set_visible(False)
    ax[1].spines["left"].set_color("grey")
    ax[1].spines["bottom"].set_color("grey")
    ax[1].set_ylabel("Sim Discharge [$l/min$]")


    ax[1].xaxis.set_major_locator(mdates.WeekdayLocator())
    ax[1].xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
# ax[1].xaxis.set_minor_locator(mdates.DayLocator())
    fig.autofmt_xdate()
# fig.text(0.04, 0.5, "Ice Volume[$m^3$]", va="center", rotation="vertical")
    handles, labels = ax[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", prop={"size": 8})
# plt.legend()
    plt.savefig("data/figs/paper3/simvsreal_dis.jpg", bbox_inches="tight", dpi=300)
