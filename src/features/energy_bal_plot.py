
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

    fig, ax = plt.subplots(ncols = 2, nrows = 2 * len(locations), figsize=(12,14))
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
        # dfd["When"] = dfd["When"].dt.strftime("%b %d")
        # dfd = dfd.set_index('When')

        dfd[["$-q_{freeze/melt}$", "$-q_{T}$"]] *=-1
        z = dfd[["$-q_{freeze/melt}$",  "$q_{SW}$", "$q_{LW}$", "$q_S$", "$q_L$", "$q_{F}$","$-q_{T}$", "$q_{G}$"]]

        y2.index = y2.index + 1
        z.index = z.index + 1
        days =19 
        xlim1=[-0.5,days+0.5]
        xlim2=[z.shape[0]-days-0.5,z.shape[0]+0.5]

        for j in range(2):
            y2.plot.bar(
                stacked=True,
                edgecolor="black",
                linewidth=0.5,
                color=[skyblue, "xkcd:azure", orange, "#0C70DE", green ],
                ax=ax[2*ctr,j],
            )
        ax[2*ctr,0].set_xlim(xlim1) # most of the data
        ax[2*ctr,1].set_xlim(xlim2)
        # at = AnchoredText(get_parameter_metadata(location)['shortname'], prop=dict(size=10), frameon=True, loc="upper left")
        # at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
        # ax[0,ctr].add_artist(at)

        for j in range(2):
            z.plot.bar(
                stacked=True, 
                edgecolor="black", 
                linewidth=0.5, 
                color=[purple, red, orange, green, "xkcd:yellowgreen", "xkcd:azure", pink,blue ],
                ax=ax[2*ctr+1, j]
                )
        ax[2*ctr+1,0].set_xlim(xlim1)
        ax[2*ctr+1,1].set_xlim(xlim2)

    for ctr in range(2*len(locations)):
        for j in range(2):
            ax[ctr,j].get_legend().remove()

            # # hide the spines between ax and ax2
            ax[ctr,j].spines['top'].set_visible(False)

            if ctr%2!=0:
                ax[ctr,j].set_ylim(-310, 310)
                ax[ctr,j].set_ylabel("Energy Flux [$W\\,m^{-2}$]")

                d = .015 # how big to make the diagonal lines in axes coordinates
                # arguments to pass plot, just so we don't keep repeating them
                kwargs = dict(transform=ax[ctr,j].transAxes, color='k', clip_on=False)
                if j == 0:
                    ax[ctr,j].plot((1-d,1+d),(-d,+d), **kwargs) # top-left diagonal
                    # ax[ctr,j].plot((1-d,1+d),(1-d,1+d), **kwargs) # bottom-left diagonal
                else:
                    ax[ctr,j].plot((-d,d),(-d,+d), **kwargs) # top-right diagonal
                    # ax[ctr,j].plot((-d,d),(1-d,1+d), **kwargs) # bottom-right diagonal
            else:
                ax[ctr,j].set_ylim(-0.065, 0.065)
                ax[ctr,j].set_ylabel('Thickness [$m$ w. e.]')
                ax[ctr,j].spines['bottom'].set_visible(False)
                ax[ctr,j].tick_params(bottom = False)
                ax[ctr,j].tick_params(labelbottom = False)


            if j == 0:
                ax[ctr,j].spines['right'].set_visible(False)
            else:
                ax[ctr,j].spines['left'].set_visible(False)
                ax[ctr,j].set_ylabel('')
                ax[ctr,j].tick_params(left=False, labelleft= False,)
            ax[ctr,j].tick_params(right = False)
            ax[ctr,j].grid(color="black", alpha=0.3, axis = 'y', linewidth=0.5, which="major")

    fig.subplots_adjust(hspace=0.25, wspace=0.025)
    ax[0,0].legend(loc="upper center", bbox_to_anchor=(0.5, 1.5), ncol=3)
    ax[1,1].legend(loc="upper center", bbox_to_anchor=(0.5, 2.6), ncol=4)
    plt.savefig(
        "data/paper/mass_energy_bal.jpg",
        dpi=300,
        bbox_inches="tight",
    )
