"""Plot calibration data for DX"""
import sys
import os
import numpy as np
import pandas as pd
import math
import matplotlib.colors
import statistics as st
from datetime import datetime, timedelta
import seaborn as sns
from matplotlib.offsetbox import AnchoredText
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import logging, coloredlogs
import multiprocessing
from time import sleep
import os, sys, time
from ast import literal_eval
import matplotlib.patches as mpatches

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

    locations = ["gangles21", "guttannen21"]

    dfx = pd.read_csv("data/paper1/dx_calibrate.csv")

    df = pd.read_csv("data/paper1/GSA.csv")

    pal = sns.color_palette("Set1", n_colors=2)
    pal_res = pal[::-1]

    fig, ax = plt.subplots(1, 2, figsize=(9, 4))
    sns.barplot(
        y="param", x="value", hue="AIR", data=df, palette=pal_res, ax=ax[0]
    )
    ax[0].set_ylabel("Parameter")
    ax[0].set_xlabel("Sensitivity of Net Water Loss")
    handles, labels = ax[0].get_legend_handles_labels()
    ax[0].get_legend().remove()

    at = AnchoredText("(a)", prop=dict(size=10), frameon=True, loc="upper right")
    at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
    ax[0].add_artist(at)

    sns.lineplot(
        x="DX", y="rmse", data=dfx[(dfx.AIR == "CH21")], color=pal[1], ax=ax[1]
    )
    ax1t = ax[1].twinx()
    sns.lineplot(
        x="DX", y="rmse", data=dfx[(dfx.AIR == "IN21")], color=pal[0], ax=ax1t
    )
    ax[1].plot(dfx[(dfx.AIR == "CH21")].loc[(dfx.rmse == dfx.loc[dfx.AIR == "CH21", "rmse"].min()), "DX"], dfx.loc[dfx.AIR == "CH21", "rmse"].min(), 'bo')
    ax1t.plot(dfx[(dfx.AIR == "IN21")].loc[(dfx.rmse == dfx.loc[dfx.AIR == "IN21", "rmse"].min()), "DX"], dfx.loc[dfx.AIR == "IN21", "rmse"].min(), 'ro')
    ax[1].set_xlabel(get_parameter_metadata("DX")["latex"] + " " + get_parameter_metadata("DX")["units"])
    ax[1].set_ylabel(" ")
    ax1t.set_ylabel("RMSE [$m^3$]")
    for tl in ax1t.get_yticklabels():
        tl.set_color(pal[0])
    for tl in ax[1].get_yticklabels():
        tl.set_color(pal[1])
        # tl.set_color('b')
    at = AnchoredText("(b)", prop=dict(size=10), frameon=True, loc="upper right")
    at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
    ax[1].add_artist(at)
    ax[1].legend(loc = 'upper left', title= 'AIR', handles=handles)

    # x1 = dfx[(dfx.AIR == "CH21")].DX
    # x2 = dfx[(dfx.AIR == "IN21")].DX
    # y1 = dfx[(dfx.AIR == "CH21")].rmse
    # y2 = dfx[(dfx.AIR == "IN21")].rmse
    # ax[1].plot(x1, y1, linestyle="-", color="#284D58", linewidth=1)
    # ax[1].set_ylabel("Discharge [$l\\, min^{-1}$]")

    # ax1t = ax[1].twinx()
    # ax1t.plot(
    #     x2,
    #     y2,
    #     linestyle="-",
    #     color=pal[1],
    #     # label="Plaffeien",
    # )
    # ax1t.set_ylabel("Precipitation [$mm$]", color=pal[1])
    # for tl in ax1t.get_yticklabels():
    #     tl.set_color(pal[1])

    plt.savefig(
        "data/paper1/Figure_5.jpg",
        dpi=300,
        bbox_inches="tight",
    )

    # fig, ax = plt.subplots()
    # plt.savefig("data/paper1/sensitivities.jpg", bbox_inches="tight", dpi=300)
    # plt.clf()
