""" Plot comparing auto and manual fountain characteristics at guttannen"""

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
import matplotlib.dates as mdates
from matplotlib.lines import Line2D
import logging, coloredlogs

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
)
from src.utils.settings import config
from src.models.icestupaClass import Icestupa
# from src.models.methods.metadata import get_parameter_metadata

if __name__ == "__main__":
    # Main logger
    logger = logging.getLogger(__name__)
    logger.setLevel("ERROR")

    mypal = sns.color_palette("Set1", 2)
    default = "#284D58"
    grey = "#ced4da"
    sprays = ['scheduled_field', 'unscheduled_field']
    dias = [5,7]

    legend_elements = [Line2D([0], [0], color=mypal[0], lw=4, label='Scheduled'),
                        Line2D([0], [0], color=mypal[1], lw=4, label='Unscheduled'),
                       ]

    axd = plt.figure(constrained_layout=True).subplot_mosaic(
        """
        AB
        CC
        """
    )

    df = pd.read_csv('/home/suryab/work/air_model/data/guttannen22/interim/dis_height.csv')
    df2 = pd.read_csv('/home/suryab/work/air_model/data/guttannen22/interim/dia_pressure.csv')
    print(df2)

    df['Discharge'] = np.where(df.Discharge== 0, np.nan, df.Discharge)

    axd['A'].plot(df2.Diameter, df2.Pressure, color=default)

    for i,spray in enumerate(sprays):
        SITE, FOLDER = config('guttannen22', spray)
        icestupa = Icestupa('guttannen22', spray)
        icestupa.read_output()

        axd['B'].plot(df[df.Diameter==dias[i]].Height, df[df.Diameter==dias[i]].Discharge, color = mypal[i])
        axd['B'].plot(df[df.Diameter==dias[i]].Height, df[df.Diameter==dias[i]].Discharge, color = mypal[i])
        axd['A'].scatter(dias[i], df2.Pressure[dias[i]-1], color = mypal[i], zorder=10)

        axd['C'].plot(icestupa.df.time[1:], icestupa.df.Discharge[1:], color = mypal[i])



    at = AnchoredText("(a) Height constant", prop=dict(size=10), frameon=True, loc="upper right")
    at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
    axd['A'].add_artist(at)
    at = AnchoredText("(b) Diameter constant", prop=dict(size=10), frameon=True, loc="upper right")
    at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
    axd['B'].add_artist(at)
    # at = AnchoredText("(c)", prop=dict(size=10), frameon=True, loc="upper right")
    # at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
    # axd['C'].add_artist(at)
    axd['B'].set_xlabel("Height [$m$]")
    axd['B'].set_ylabel("Discharge [$l/min$]")
    axd['C'].set_ylabel("Discharge [$l/min$]")
    axd['A'].set_xlabel("Aperture diameter [$mm$]")
    axd['A'].set_ylabel("Pressure [$m$]")
    axd['C'].xaxis.set_major_locator(mdates.WeekdayLocator())
    axd['C'].xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))

    axd['A'].set_ylim([0,6])
    axd['A'].set_xlim([0,20])
    axd['B'].set_xlim([0,10])
    axd['B'].set_ylim([0,25])
    axd['C'].set_ylim([0,15])


    for sec in ['A', 'B', 'C']:
        axd[sec].spines["right"].set_visible(False)
        axd[sec].spines["top"].set_visible(False)
        axd[sec].spines["left"].set_color("grey")
        axd[sec].spines["bottom"].set_color("grey")

    for label in axd['C'].get_xticklabels():
        label.set_ha('right')
        label.set_rotation(30)

    axd['C'].legend(handles=legend_elements, prop={"size": 8}, title='(c) Fountain')
    plt.savefig("data/figs/paper3/fountains.png", bbox_inches="tight", dpi=300)
    plt.close()
