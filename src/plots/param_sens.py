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
    ax[0].get_legend().remove()

    at = AnchoredText("(a)", prop=dict(size=10), frameon=True, loc="upper right")
    at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
    ax[0].add_artist(at)

    sns.lineplot(
        x="DX", y="rmse", hue="AIR", data=dfx, palette=pal, ax=ax[1]
    )
    ax[1].plot(dfx[(dfx.AIR == "CH21")].loc[(dfx.rmse == dfx.loc[dfx.AIR == "CH21", "rmse"].min()), "DX"], dfx.loc[dfx.AIR == "CH21", "rmse"].min(), 'bo')
    ax[1].plot(dfx[(dfx.AIR == "IN21")].loc[(dfx.rmse == dfx.loc[dfx.AIR == "IN21", "rmse"].min()), "DX"], dfx.loc[dfx.AIR == "IN21", "rmse"].min(), 'ro')
    ax[1].set_xlabel(get_parameter_metadata("DX")["latex"] + " " + get_parameter_metadata("DX")["units"])
    ax[1].set_ylabel("Normalized RMSE")
    at = AnchoredText("(b)", prop=dict(size=10), frameon=True, loc="upper right")
    at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
    ax[1].add_artist(at)
    ax[1].legend(loc = 'upper left', title= 'AIR')

    plt.savefig(
        "data/paper1/Figure_5.jpg",
        dpi=300,
        bbox_inches="tight",
    )

    # fig, ax = plt.subplots()
    # plt.savefig("data/paper1/sensitivities.jpg", bbox_inches="tight", dpi=300)
    # plt.clf()
