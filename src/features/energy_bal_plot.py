
"""Icestupa class function that generates figures for web app
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os,sys
import matplotlib.dates as mdates
import matplotlib as mpl
from matplotlib.colors import ListedColormap
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.offsetbox import AnchoredText
from mpl_toolkits.axisartist.axislines import Axes
from mpl_toolkits import axisartist
import matplotlib.ticker as ticker

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
)

from src.utils.settings import config
from src.models.methods.metadata import get_parameter_metadata
from src.models.icestupaClass import Icestupa
from src.models.methods.metadata import get_parameter_metadata
import seaborn as sns
import matplotlib.pyplot as plt

if __name__ == "__main__":
    locations = ['gangles21', 'guttannen21', 'guttannen20']
    # locations = ['guttannen21',  'gangles21']
    # locations = ['guttannen21']

    blue = "#0a4a97"
    red = "#e23028"
    purple = "#9673b9"
    green = "#28a745"
    orange = "#ffc107"
    pink = "#ce507a"
    skyblue = "#9bc4f0"
    grey = '#ced4da'
    CB91_Blue = "#2CBDFE"
    CB91_Green = "#47DBCD"
    CB91_Pink = "#F3A0F2"
    CB91_Purple = "#9D2EC5"
    CB91_Violet = "#661D98"
    CB91_Amber = "#F5B14C"

    total_days = 101
    fig, ax = plt.subplots(ncols = len(locations), nrows = 4, figsize=(12,14))
    fig.subplots_adjust(hspace=0.05, wspace=0.05)
    for ctr, location in enumerate(locations):
        SITE, FOLDER = config(location)
        icestupa = Icestupa(location)
        icestupa.read_output()

        icestupa.df = icestupa.df.rename(
            {
                "SW": "$q_{SW}$",
                "LW": "$q_{LW}$",
                "Qs": "$q_S$",
                "Ql": "$q_L$",
                "Qf": "$q_{F}$",
                "Qg": "$q_{G}$",
                "Qsurf": "$q_{surf}$",
                "Qmelt": "$-q_{freeze/melt}$",
                "Qt": "$-q_{T}$",
            },
            axis=1,
        )

        dfds = icestupa.df[
            [
                "When",
                "ppt",
                "dep",
                "melted",
                "sub",
                "SA",
                "fountain_froze",
            ]
        ]

        with pd.option_context("mode.chained_assignment", None):
            for i in range(0, dfds.shape[0]):
                if icestupa.df.loc[i, "SA"] != 0:
                    dfds.loc[i, "Discharge"] = dfds.loc[i, "fountain_froze"] / (
                        icestupa.df.loc[i, "SA"] * icestupa.RHO_I
                    )
                    dfds.loc[i, "melted"] *= -1 / (icestupa.df.loc[i, "SA"] * icestupa.RHO_I)
                    dfds.loc[i, "sub"] *= -1 / (icestupa.df.loc[i, "SA"] * icestupa.RHO_I)
                    dfds.loc[i, "ppt"] *= 1 / (icestupa.df.loc[i, "SA"] * icestupa.RHO_I)
                    dfds.loc[i, "dep"] *= 1 / (icestupa.df.loc[i, "SA"] * icestupa.RHO_I)
                else:
                    dfds.loc[i, "Discharge"] = 0 
                    dfds.loc[i, "melted"] *= 0
                    dfds.loc[i, "sub"] *= 0
                    dfds.loc[i, "ppt"] *= 0
                    dfds.loc[i, "dep"] *= 0

        dfds = dfds.set_index("When").resample("D").sum().reset_index()

        dfds = dfds.rename(
            columns={
                "ppt": "Snowfall",
                "melted": "Melt",
                "sub": "Sublimation",
                "dep": "Deposition",
            }
        )


        y2 = dfds[
            [
                "Discharge",
                "Snowfall",
                "Deposition",
                "Melt",
                "Sublimation",
            ]
        ]

        dfd = icestupa.df.set_index("When").resample("D").mean().reset_index()
        dfd[["$-q_{freeze/melt}$", "$-q_{T}$"]] *=-1
        z = dfd[["$-q_{freeze/melt}$",  "$q_{SW}$", "$q_{LW}$", "$q_S$", "$q_L$", "$q_{F}$","$-q_{T}$", "$q_{G}$"]]

        # if y2.shape[0]<total_days:
        #     df2 = pd.DataFrame([[0]*y2.shape[1]],columns=y2.columns)
        #     for i in range(y2.shape[0], total_days):
        #         y2 = y2.append(df2, ignore_index=True)
        #         z = z.append(df2, ignore_index=True)
        # else:
        #     y2 = y2[:total_days]
        #     z= z[:total_days]
        y2.index = y2.index + 1
        z.index = z.index + 1

        idx_slice = y2.index < 20
        y2.loc[idx_slice].plot.bar(
            stacked=True,
            edgecolor="black",
            linewidth=0.5,
            color=[skyblue, "xkcd:azure", orange, "#0C70DE", green ],
            ax=ax[0, ctr],
        )
        ax[0,ctr].xaxis.set_label_text("")
        ax[0,ctr].grid(color="black", alpha=0.3, linewidth=0.5, which="major")
        ax[0,ctr].set_ylim(-0.06, 0.06)
        at = AnchoredText(get_parameter_metadata(location)['shortname'], prop=dict(size=10), frameon=True, loc="upper left")
        at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
        ax[0,ctr].add_artist(at)
        ax[0,ctr].legend(loc="upper center", ncol=3)
        # ax[0,ctr].xaxis.set_ticks(np.arange(0, 51, 10))
        # ax[0,ctr].xaxis.set_major_formatter(ticker.FormatStrFormatter('%i'))
        ax[0,ctr].set_ylabel('Thickness [$m$ w. e.]')

        idx_slice = z.index < 20
        z[idx_slice].plot.bar(
            stacked=True, 
            edgecolor="black", 
            linewidth=0.5, 
            color=[purple, red, orange, green, "xkcd:yellowgreen", "xkcd:azure", pink,blue ],
            ax=ax[1,ctr]
            )
        ax[1,ctr].grid(color="black", alpha=0.3, linewidth=0.5, which="major")
        ax[1,ctr].set_ylim(-300, 300)
        ax[1,ctr].legend(loc="upper center", ncol=4)
        # ax[1,ctr].xaxis.set_ticks(np.arange(0, 51, 10))
        # ax[1,ctr].xaxis.set_major_formatter(ticker.FormatStrFormatter('%i'))
        ax[1,ctr].set_ylabel("Energy Flux [$W\\,m^{-2}$]")
        ax[1,ctr].xaxis.set_label_text("Day number")

        idx_slice = y2.index > (y2.shape[0]) - 20
        y2[idx_slice].plot.bar(
            stacked=True,
            edgecolor="black",
            linewidth=0.5,
            color=[skyblue, "xkcd:azure", orange, "#0C70DE", green ],
            ax=ax[2, ctr],
        )
        ax[2,ctr].xaxis.set_label_text("")
        # ax[2,ctr].grid(color="black", alpha=0.3, linewidth=0.5, which="major")
        ax[2,ctr].set_ylim(-0.06, 0.06)
        at = AnchoredText(get_parameter_metadata(location)['shortname'], prop=dict(size=10), frameon=True, loc="upper left")
        at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
        ax[2,ctr].add_artist(at)
        ax[2,ctr].legend(loc="upper center", ncol=3)
        # ax[2,ctr].xaxis.set_ticks(np.arange(0, total_days, 10))
        # ax[2,ctr].xaxis.set_major_formatter(ticker.FormatStrFormatter('%i'))
        ax[2,ctr].set_ylabel('Thickness [$m$ w. e.]')

        idx_slice = z.index > (z.shape[0]) - 20
        z[idx_slice].plot.bar(
            stacked=True, 
            edgecolor="black", 
            linewidth=0.5, 
            color=[purple, red, orange, green, "xkcd:yellowgreen", "xkcd:azure", pink,blue ],
            ax=ax[3,ctr]
            )
        # ax[3,ctr].grid(color="black", alpha=0.3, linewidth=0.5, which="major")
        ax[3,ctr].set_ylim(-300, 300)
        ax[3,ctr].legend(loc="upper center", ncol=4)
        # ax[3,ctr].xaxis.set_ticks(np.arange(0, total_days, 10))
        # ax[3,ctr].xaxis.set_major_formatter(ticker.FormatStrFormatter('%i'))
        ax[3,ctr].set_ylabel("Energy Flux [$W\\,m^{-2}$]")
        ax[3,ctr].xaxis.set_label_text("Day number")

    for i in range(4):
        for j in range(len(locations)):
            ax[i,j].get_legend().remove()
            ax[i,j].axes.get_yaxis().set_visible(False)
        # if i != 3:
        #     ax[i,j].xaxis.set_label_text("")
        # if j != 0:
        #     ax[i,j].axes.get_yaxis().set_visible(False)

    # ax[0,1].legend(loc="upper center", bbox_to_anchor=(-0.1, 0.15), ncol=3)
    # ax[1,1].legend(loc="upper center", bbox_to_anchor=(-0.1, 1), ncol=4)
    plt.savefig(
        "data/paper/mass_energy_bal.jpg",
        dpi=300,
        bbox_inches="tight",
    )
