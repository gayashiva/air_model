"""Function that returns returns new discharge after height increase by dh
"""
import sys, os
import pandas as pd
import math
import numpy as np
import logging
import coloredlogs
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.lines import Line2D

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
)
from src.utils.settings import config
from src.models.icestupaClass import Icestupa


def get_dis_new(
    dia=0.016, dh=1, dis=0
): 
    Area = math.pi * math.pow(dia, 2) / 4
    G = 9.8
    v= dis/ (60 * 1000 * Area)
    if v**2 - 2 * G * dh < 0 :
        dis_new = 0
    else:
        v_new = math.sqrt( v**2 - 2 * G * dh )
        dis_new = v_new * (60 * 1000 * Area)
    # logger.warning("Discharge calculated is %s" % (dis))
    return dis_new

def get_projectile(
    dia=0, h_f=3, dis=0, r=0, theta_f = 45
):  # returns discharge or spray radius using projectile motion

    Area = math.pi * math.pow(dia, 2) / 4
    theta_f = math.radians(theta_f)
    G = 9.8
    if r == 0:
        data_ry = []
        v = dis / (60 * 1000 * Area)
        t = 0.0
        while True:
            # now calculate the height y
            y = h_f + (t * v * math.sin(theta_f)) - (G * t * t) / 2
            # projectile has hit ground level
            if y < 0:
                break
            r = v * math.cos(theta_f) * t
            data_ry.append((r, y))
            t += 0.01
        # logger.warning("Spray radius is %s" % (r))
        return r
    else:
        v = math.sqrt(
            G ** 2
            * r ** 2
            / (math.cos(theta_f) ** 2 * 2 * G * h_f + math.sin(2 * theta_f) * G * r)
        )
        dis = v * (60 * 1000 * Area)
        # logger.warning("Discharge calculated is %s" % (dis))
        return dis


if __name__ == "__main__":
    # Main logger
    logger = logging.getLogger(__name__)
    logger.setLevel("ERROR")

    # print(get_dis_new(dh=4, dis_old=60))
    # print(get_dis_new(dh=1, dis=13))
    # print(get_projectile(h_f=3, dia=0.006, r=3, theta_f=60))
    # print(get_projectile(h_f=4.7, dia=0.005, dis=5))

    location = 'guttannen22'
    sprays = ['scheduled_field', 'unscheduled_field']
    # sprays = ['unscheduled_field']
    mypal = sns.color_palette("Set1", 2)

    fig, ax = plt.subplots()
    for i, spray in enumerate(sprays):
        SITE, FOLDER = config(location, spray)
        icestupa = Icestupa(location, spray)
        icestupa.read_output()
        if spray == "scheduled_field":
            spray = "Scheduled"
        else:
            spray = "Unscheduled"

        df=icestupa.df
        for j in range(0,df.shape[0]):
            if df.Discharge[j] !=0:
                df.loc[j,'radf'] = get_projectile(h_f=4, dia=0.005, dis=df.Discharge[j])
            else:
                df.loc[j,'radf'] = np.nan
        print(df.radf.mean())

        x = df.time[1:]
        y1 = df.radf[1:]
        ax.plot(
            x,
            y1,
            linewidth=0.8,
            color=mypal[i],
            label=spray,
        )
        ax.axhline(y=icestupa.R_F, linewidth=0.8, linestyle='--',color=mypal[i])
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.spines["left"].set_color("grey")
        ax.spines["bottom"].set_color("grey")
        ax.set_ylabel("Spray Radius[$m$]")
    ax.xaxis.set_major_locator(mdates.WeekdayLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    fig.subplots_adjust(hspace=None, wspace=None)
    fig.autofmt_xdate()
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, loc="upper right", prop={"size": 8}, title="Fountain spray")
    plt.savefig("data/figs/paper3/radf.png", bbox_inches="tight", dpi=300)
    plt.close()
