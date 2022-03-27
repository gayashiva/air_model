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
from matplotlib.lines import Line2D
import logging, coloredlogs

if __name__ == "__main__":
    # Main logger
    logger = logging.getLogger(__name__)
    logger.setLevel("ERROR")
    df = pd.read_csv("/home/suryab/work/air_model/data/common/Ladakh_drone_surveys.csv", sep=";")
    df = df[["Name", "Icestupa height (fountain altitude)", "Calculated Volume", "Date of flight"]]
    df.rename(
        columns={
            "Icestupa height (fountain altitude)": "Altitude",
            "Calculated Volume": "Volume",
            "Date of flight": "Winter",
        },
        inplace=True,
    )
    df["Volume"] /= 1000
    for i in range(0, df.shape[0]):
        df.Winter.loc[i] = df.Winter[i].split(".")[-1]

    winters = ["2019", "2020", "2021"]
    styles=['*', '.', 'x']
    dot_dict = dict(zip(winters, styles))

    #sort dataframe
    df = df.sort_values(by='Altitude').reset_index(drop=True)
    print(df)

    mypal = sns.color_palette("Set1", 2)
    legend_elements = [
                       Line2D([0], [0], marker='*', color='w', label='Winter 2019',
                              markerfacecolor='k', markersize=15),
                       Line2D([0], [0], marker='.', color='w', label='Winter 2020',
                              markerfacecolor='k', markersize=15),
                       Line2D([0], [0], marker='X', color='w', label='Winter 2021',
                              markerfacecolor='k', markersize=10),
                       ]

    fig, ax = plt.subplots()

    for i in range(0, df.shape[0]):
        if df.Name.loc[i].split(" ")[-1] != str(2):
            ax.scatter(df.Altitude[i],df.Volume[i],s=20, color=mypal[1], marker=dot_dict[df.Winter[i]])
        else:
            print(df.Name[i])
        # if df.Name[i] in ["Igoo", "Shara"]:
        # else:
            # ax.scatter(df.Altitude[i],df.Volume[i],s=20, color=mypal[1], marker=dot_dict[df.Winter[i]])

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
    ax.set_ylabel("Ice Volume [$million\,litres$]")
    ax.set_xlabel("Altitude [$m$]")
    ax.legend(handles=legend_elements, prop={"size": 8})
    plt.savefig("data/figs/slides/ladakh_surveys_0.png", bbox_inches="tight", dpi=300)
    plt.axvline(x=4200, color = 'k', linestyle = '--', alpha = 0.5, linewidth=0.9)
    plt.savefig("data/figs/slides/ladakh_surveys_1.png", bbox_inches="tight", dpi=300)
