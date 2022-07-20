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
from matplotlib.lines import Line2D
import matplotlib.dates as mdates
import logging, coloredlogs

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
)
from src.utils.settings import config
from src.models.icestupaClass import Icestupa
# from src.models.methods.metadata import get_parameter_metadata

def keystoint(x):
    return {k: float(v) for k, v in x.items()}

if __name__ == "__main__":
    # Main logger
    logger = logging.getLogger(__name__)
    logger.setLevel("ERROR")

    locations = ["guttannen22", "guttannen21", "gangles21"]
    sprays = ['unscheduled_field', 'scheduled_icv', 'scheduled_wue']
    styles=['.', 'x' , '*']

    mypal = sns.color_palette("Set1", len(locations))
    legend_elements = [
                       Line2D([0], [0], marker='.', color='w', label='CH21',
                              markerfacecolor='k', markersize=15),
                       Line2D([0], [0], marker='X', color='w', label='CH22',
                              markerfacecolor='k', markersize=10),
                       Line2D([0], [0], marker='*', color='w', label='IN21',
                              markerfacecolor='k', markersize=15),
            Line2D([0], [0], color=mypal[0], lw=4, label='Unscheduled'),
            Line2D([0], [0], color=mypal[1], lw=4, label='Weather-sensitive'),
            Line2D([0], [0], color=mypal[2], lw=4, label='Water-sensitive'),
                       Line2D([0], [0], marker='o', color='k', label='Experiment',
                              markerfacecolor='w', markersize=10, lw=0),
                       Line2D([0], [0], marker='s', color='k', label='Simulation',
                              markerfacecolor='w', markersize=10, lw=0),
                       ]
    fig, (ax, ax1) = plt.subplots(ncols=2, figsize=(15,6))

    SITE, FOLDER = config('guttannen22', 'scheduled_field')
    with open(FOLDER["output"] +  "/results.json") as f:
        results = json.load(f, object_hook=keystoint)
    ax.scatter(results["WUE"], results["iceV_max"], color='k', marker='.', s=500, facecolors='none')
    ax.scatter(results["WUE"], results["iceV_max"], color=mypal[1], marker=styles[1])

    for i, loc in enumerate(locations):
        for j, spray in enumerate(sprays):
            SITE, FOLDER = config(loc, spray)
            try: 
                with open(FOLDER["output"] +  "/results.json") as f:
                    results = json.load(f, object_hook=keystoint)
                print(loc,spray, results["WUE"], results["iceV_max"])
                if j == 0:
                    ax.scatter(results["WUE"], results["iceV_max"], color='k', marker='.', s=500, facecolors='none')
                else:
                    ax.scatter(results["WUE"], results["iceV_max"], color='k', marker='s', s=100,
                               facecolors='none')
                ax.scatter(results["WUE"], results["iceV_max"], color=mypal[j], marker=styles[i])


            except FileNotFoundError:
                logger.error("No simulation exists")

    ax.set_ylabel("Max Ice Volume [$m^3$]")
    ax.set_xlabel("Water Use Efficiency [%]")
    ax.set_xlim([0,100])
    ax.set_ylim([0,1400])
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["left"].set_color("grey")
    ax.spines["bottom"].set_color("grey")
    ax.legend(handles=legend_elements)
    at = AnchoredText("(a)", prop=dict(size=15), frameon=True, loc="upper left")
    at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
    ax.add_artist(at)


    loc= "Gangles 2021"
    SITE, FOLDER = config(loc, spray="unscheduled_field")

    objs = ["unscheduled_field", "scheduled_icv", "scheduled_wue"]
    labels = ["Unscheduled", "Weather-sensitive", "Water-sensitive"]
    styles=['.', 'x']

    mypal = sns.color_palette("Set1", 3)
    default = "#284D58"

    df1 = Icestupa(loc, spray=objs[0]).df
    df2 = Icestupa(loc, spray=objs[1]).df
    df3 = Icestupa(loc, spray=objs[2]).df

    df1 = df1.set_index("time")
    df1 = df1[SITE["start_date"] : datetime(2021, 4, 13)]
    df1 = df1.reset_index()
    df2 = df2.set_index("time")
    df2 = df2[SITE["start_date"] : datetime(2021, 4, 13)]
    df2 = df2.reset_index()
    df3 = df3.set_index("time")
    df3 = df3[SITE["start_date"] : datetime(2021, 4, 13)]
    df3 = df3.reset_index()

    mask = (df1.Discharge == 0) & (df1.time < SITE["fountain_off_date"])
    mask2 = (df2.Discharge > df2.Discharge.quantile(0.0)) & (df2.time < SITE["fountain_off_date"]) 
    # print(df2.time[mask2].isin(df2.time[mask]).value_counts())
    print(df2.Discharge[mask].mean())
    print(df2.Discharge.quantile(0.5))
    print(df3.Discharge[mask].mean())
    print(df3.Discharge.quantile(0.5))
    # print(df2.Discharge[mask].size)
    # print(df2.Discharge[mask2].size)
    # print(df2.Discharge[mask2].isin(df2.Discharge[mask]).count())

    x = df1.time[1:]
    y1 = df1.Discharge[1:]
    y2 = df2.Discharge[1:]
    y3 = df3.Discharge[1:]

    ax1.plot(
        x,
        y1,
        label= labels[0],
        linewidth=1,
        color=mypal[0],
    )
    ax1.spines["right"].set_visible(False)
    ax1.spines["top"].set_visible(False)
    ax1.spines["left"].set_color("grey")
    ax1.spines["bottom"].set_color("grey")

    ax1.plot(
        x,
        y2,
        label= labels[1],
        linewidth=1,
        color=mypal[1],
    )
    ax1.spines["right"].set_visible(False)
    ax1.spines["top"].set_visible(False)
    ax1.spines["left"].set_color("grey")
    ax1.spines["bottom"].set_color("grey")
    ax1.set_ylabel("IN21 Discharge rate [$l/min$]")
    at = AnchoredText("(b)", prop=dict(size=15), frameon=True, loc="upper left")
    at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
    ax1.add_artist(at)

    ax1.plot(
        x,
        y3,
        label= labels[2],
        linewidth=1,
        color=mypal[2],
    )


    # ax1.legend(prop={"size": 8}, title="IN21 Fountain type", loc="upper right")
    ax1.xaxis.set_major_locator(mdates.WeekdayLocator())
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    fig.autofmt_xdate()

    plt.savefig("data/figs/paper3/wue.png", bbox_inches="tight", dpi=300)
    plt.clf()


