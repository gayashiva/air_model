""" Plot comparing CH stupas"""

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
from matplotlib.ticker import AutoMinorLocator, FormatStrFormatter
import matplotlib.dates as mdates
from matplotlib.lines import Line2D
import logging, coloredlogs

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
)
from src.utils.settings import config
from src.models.icestupaClass import Icestupa

if __name__ == "__main__":
    # Main logger
    logger = logging.getLogger(__name__)
    logger.setLevel("ERROR")

    locations = ["guttannen22", "guttannen21", "guttannen20"]
    # shortnames = ["CH20", "CH21", "CH22"]
    shortnames = ["CH22", "CH21", "CH20"]
    # locations = ["guttannen20","guttannen21"]
    spray = 'unscheduled_field'
    data = []

    for i, location in enumerate(locations):
        print(location)
        SITE, FOLDER = config(location, spray)
        icestupa = Icestupa(location, spray)
        icestupa.read_output()
        df_jan = icestupa.df.loc[icestupa.df.time.dt.month == 1]
        iceV_diff = df_jan.iceV[df_jan.index[-1]] - df_jan.iceV[df_jan.index[0]]
        print(f'\tMedian temp {df_jan.temp.median()}')
        print(f'\tMedian wind {df_jan.wind.median()}')
        print(f'\tVolume diff {iceV_diff}')
        print(f'\n\tSpray radius {icestupa.R_F}\n')
        print(df_jan[df_jan.wind<0])
        df_jan['R_F'] = int(icestupa.R_F)
        df_jan['iceV'] = int(iceV_diff)
        if i == 0:
            df = df_jan
        else:
            df = pd.concat([df, df_jan])

    pal = sns.color_palette("Set1", n_colors=3)
    legend_elements = []

    for i in range(0,3):
        legend_elements.append(Line2D([0], [0], marker='.', lw=0, color=pal[i], label=shortnames[i],
                              markerfacecolor=pal[i], markersize=10))

    fig, ax = plt.subplots(1, 2, figsize=(9, 4))
    sns.violinplot(y = 'iceV', x="temp", data=df, palette=pal,orient="h", ax=ax[0])
    ax[0].invert_yaxis()
    ax[0].set_ylabel("Estimated Volume change [$m^3$]")
    ax[0].set_xlabel("January temperature [$\degree\,C$]")
    handles, labels = ax[0].get_legend_handles_labels()

    at = AnchoredText("(a)", prop=dict(size=10), frameon=True, loc="upper left")
    at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
    ax[0].add_artist(at)
    ax[1] = sns.violinplot(y = 'R_F', x="wind", data=df, palette=pal,orient="h", ax=ax[1])
    ax[1].invert_yaxis()
    ax[1].set_ylabel("Observed spray radius [$m$]")
    ax[1].set_xlabel("January wind speed [$m/s$]")
    # ax[1].set_ylim([0,10])
    # ax[1].set_xlim([0,1.4])

    at = AnchoredText("(b)", prop=dict(size=10), frameon=True, loc="upper left")
    at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
    ax[1].add_artist(at)

    for i in [0,1]:
        # Hide the right and top spines
        ax[i].spines["right"].set_visible(False)
        ax[i].spines["top"].set_visible(False)
        ax[i].spines["left"].set_color("grey")
        ax[i].spines["bottom"].set_color("grey")
        [t.set_color("grey") for t in ax[i].xaxis.get_ticklines()]
        [t.set_color("grey") for t in ax[i].yaxis.get_ticklines()]
        # Only show ticks on the left and bottom spines
        ax[i].yaxis.set_ticks_position("left")
        ax[i].xaxis.set_ticks_position("bottom")

    # ax = sns.violinplot(y = 'R_F', x="wind", data=df, palette=pal,orient="h")
    # ax.invert_yaxis()
    # # ax.set_xlim([0,8])
    # # ax.set_ylim([0,200])
    ax[1].legend(handles=legend_elements, loc="upper right", prop={"size": 8})
    plt.savefig(
        "data/figs/paper3/CH_diffs.jpg",
        dpi=300,
        bbox_inches="tight",
    )

        # data.append([shortnames[i], df_jan.temp.median(), df_jan.wind.median(), iceV_diff, icestupa.R_F ])

    # df = pd.DataFrame(data, columns = ['name','temp','wind', 'vol', 'r_F'])
    # print(df)

    # pal = sns.color_palette("Set1", n_colors=3)
    # pal_res = pal[::-1]
    # legend_elements = []

    # for i in range(0,3):
    #     legend_elements.append(Line2D([0], [0], marker='.', lw=0, color=pal[i], label=df.loc[i,'name'],
    #                           markerfacecolor=pal[i], markersize=10))

    # fig, ax = plt.subplots(1, 2, figsize=(9, 4))
    # # sns.violinplot(x="temp", y="wind", data=df_jan, axes=ax[0])
    # ax[0].scatter(df.temp, df.vol, s=20, color = pal, label=df.name)
    # ax[0].set_ylabel("Estimated Volume change [$m^3$]")
    # ax[0].set_xlabel("Median January temperature [$\degree\,C$]")
    # ax[0].set_ylim([0,160])
    # ax[0].set_xlim([-4,1])
    # handles, labels = ax[0].get_legend_handles_labels()
    # # ax[0].get_legend().remove()

    # at = AnchoredText("(a)", prop=dict(size=10), frameon=True, loc="upper left")
    # at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
    # ax[0].add_artist(at)
    # # sns.violinplot(x="wind", y="r_F", inner='quartile', data=df[df.name=='CH20'], axes = ax[1])
    # ax[1].scatter(df.wind, df.r_F, s=20, color = pal, label=df.name)
    # ax[1].set_ylabel("Observed spray radius [$m$]")
    # ax[1].set_xlabel("Median January wind speed [$m/s$]")
    # ax[1].set_ylim([0,10])
    # ax[1].set_xlim([0,1.4])

    # at = AnchoredText("(b)", prop=dict(size=10), frameon=True, loc="upper left")
    # at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
    # ax[1].add_artist(at)

    # for i in [0,1]:
    #     # Hide the right and top spines
    #     ax[i].spines["right"].set_visible(False)
    #     ax[i].spines["top"].set_visible(False)
    #     ax[i].spines["left"].set_color("grey")
    #     ax[i].spines["bottom"].set_color("grey")
    #     [t.set_color("grey") for t in ax[i].xaxis.get_ticklines()]
    #     [t.set_color("grey") for t in ax[i].yaxis.get_ticklines()]
    #     # Only show ticks on the left and bottom spines
    #     ax[i].yaxis.set_ticks_position("left")
    #     ax[i].xaxis.set_ticks_position("bottom")

    # ax[1].legend(handles=legend_elements, loc="upper right", prop={"size": 8})
    # # fig.legend(handles, labels, loc="upper right", prop={"size": 8})
    # plt.savefig(
    #     "data/figs/thesis/CH_diffs.jpg",
    #     dpi=300,
    #     bbox_inches="tight",
    # )
