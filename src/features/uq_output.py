""" Uncertainty Quantification figures of Icestupa class
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.colors
import uncertainpy as un
import statistics as st
import sys
import os
from datetime import datetime, timedelta
import matplotlib.dates as mdates

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
)

from src.utils.settings import config
from src.models.icestupaClass import Icestupa


def draw_plot(data, edge_color, fill_color, labels):
    bp = ax.boxplot(data, patch_artist=True, labels=labels, sym="o")

    for element in ["boxes", "whiskers", "fliers", "means", "medians", "caps"]:
        plt.setp(bp[element], color=edge_color)

    for patch in bp["boxes"]:
        patch.set(facecolor=fill_color)


if __name__ == "__main__":
    # location="guttannen20"
    location="gangles21"
    # location="schwarzsee19"

    # Get settings for given location and trigger
    SITE, FOLDER = config(location)
    icestupa = Icestupa(location)
    icestupa.read_output()
    icestupa.self_attributes()

    if location == "guttannen21":
        total_days = 170
    if location == "schwarzsee19":
        total_days = 50
    if location == "guttannen20":
        total_days = 100
    if location == "gangles21":
        total_days = 150

    names = [
        "T_PPT",
        "H_PPT",
        "IE",
        "A_I",
        "A_S",
        "A_DECAY",
        "T_W",
        "DX",
    ]
    variance = []
    mean = []
    evaluations = []

    for name in names:
        data = un.Data()
        filename1 = FOLDER["sim"] + name + ".h5"
        data.load(filename1)
        variance.append(data["max_volume"].variance)
        mean.append(data["max_volume"].mean)
        evaluations.append(data["max_volume"].evaluations)

        eval = data["max_volume"].evaluations
        print(
            f"95 percent confidence interval caused by {name} is {round(st.mean(eval),2)} and {round(2 * st.stdev(eval),2)}"
        )

    names_label = [
        "$T_{ppt}$",
        "$H_{ppt}$",
        "$\\epsilon_{ice}$",
        r"$\alpha_{ice}$",
        r"$\alpha_{snow}$",
        "$\\tau$",
        "$T_{water}$",
        "$\\Delta x$",
    ]

    fig, ax = plt.subplots()
    print(len(evaluations), len(names_label))
    draw_plot(evaluations, "k", "xkcd:grey", names_label)
    ax.set_xlabel("Parameter")
    ax.set_ylabel("Sensitivity of Maximum Ice Volume [$m^3$]")
    ax.grid(axis="y")
    plt.savefig(FOLDER["sim"]+ "sensitivities.jpg", bbox_inches="tight", dpi=300)

