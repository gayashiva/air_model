"""Icestupa class function that generates figures for web app
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os, sys
import matplotlib.dates as mdates
import matplotlib as mpl
from matplotlib.colors import ListedColormap
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.offsetbox import AnchoredText
from mpl_toolkits.axisartist.axislines import Axes
from mpl_toolkits import axisartist
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
)

from src.utils.settings import config
from src.models.methods.metadata import get_parameter_metadata
from src.models.icestupaClass import Icestupa

def add_patch(legend, title = "Energy Balance Components", label="-$q_{surf}$", color = 'k'):
    from matplotlib.patches import Patch
    ax = legend.axes

    handles, labels = ax.get_legend_handles_labels()
    # handles.append(Patch(facecolor='k'))
    handles.append(Line2D([0], [0], color=color, linestyle='--'))
    labels.append(label)

    legend._legend_box = None
    legend._init_legend_box(handles, labels)
    legend._set_loc(legend._loc)
    # legend.set_title(legend.get_title().get_text())
    legend.set_title(title)

if __name__ == "__main__":
    locations = ["gangles21", "guttannen21"]

    blue = "#0a4a97"
    red = "#e23028"
    purple = "#9673b9"
    green = "#28a745"
    orange = "#ffc107"
    pink = "#ce507a"
    skyblue = "#9bc4f0"
    grey = "#ced4da"
    CB91_Blue = "#2CBDFE"
    CB91_Green = "#47DBCD"
    CB91_Pink = "#F3A0F2"
    CB91_Purple = "#9D2EC5"
    CB91_Violet = "#661D98"
    CB91_Amber = "#F5B14C"

    fig = plt.figure(figsize=(12, 14))
    subfigs = fig.subfigures(len(locations), 1, wspace=0.25)
    for ctr, location in enumerate(locations):
        SITE, FOLDER = config(location)
        icestupa = Icestupa(location)
        icestupa.read_output()
        icestupa.df = icestupa.df[:icestupa.last_hour]

        icestupa.df = icestupa.df.rename(
            {
                "SW": "$q_{SW}$",
                "LW": "$q_{LW}$",
                "Qs": "$q_S$",
                "Ql": "$q_L$",
                "Qf": "$q_{F}$",
                "Qg": "$q_{G}$",
                "Qsurf": "$-q_{surf}$",
                "Qmelt": "$-q_{melt}$",
                "Qfreeze": "$-q_{freeze}$",
                "Qt": "$-q_{T}$",
            },
            axis=1,
        )

        icestupa.df.loc[icestupa.df.Discharge !=0, "Discharge"] = 1

        dfds = icestupa.df[
            [
                "When",
                "ppt",
                "dep",
                "melted",
                "sub",
                "SA",
                "fountain_froze",
                "fountain_runoff",
                "Discharge",
                "mb",
            ]
        ]

        with pd.option_context("mode.chained_assignment", None):
            for i in range(0, dfds.shape[0]):
                if icestupa.df.loc[i, "SA"] != 0:
                    dfds.loc[i, "Ice"] = dfds.loc[i, "fountain_froze"] / (
                        icestupa.df.loc[i, "SA"] * 1000
                    )
                    dfds.loc[i, "melted"] *= -1 / (
                        icestupa.df.loc[i, "SA"] * 1000
                    )
                    dfds.loc[i, "sub"] *= -1 / (
                        icestupa.df.loc[i, "SA"] * 1000
                    )
                    dfds.loc[i, "ppt"] *= 1 / (
                        icestupa.df.loc[i, "SA"] * 1000
                    )
                    dfds.loc[i, "dep"] *= 1 / (
                        icestupa.df.loc[i, "SA"] * 1000
                    )
                    dfds.loc[i, "Runoff"] = dfds.loc[i, "fountain_runoff"] / (
                        icestupa.df.loc[i, "SA"] * 1000
                    )
                else:
                    dfds.loc[i, "Ice"] = 0
                    dfds.loc[i, "melted"] *= 0
                    dfds.loc[i, "sub"] *= 0
                    dfds.loc[i, "ppt"] *= 0
                    dfds.loc[i, "dep"] *= 0

        dfds["sub/dep"] = dfds["sub"] + dfds["dep"]
        dfds = dfds.set_index("When").resample("D").sum().reset_index()

        dfds = dfds.rename(
            columns={
                "ppt": "Snowfall",
                "melted": "Melt",
                "sub/dep": "Sublimation/Deposition",
            }
        )

        y2 = dfds[
            [
                "Ice",
                "Melt",
                "Snowfall",
                "Sublimation/Deposition",
                # "Runoff",
            ]
        ]
        y2 = y2.mul(1000)
        dfds["mb"] *= (1000)

        dfd = icestupa.df.set_index("When").resample("D").mean().reset_index()
        # dfd["When"] = dfd["When"].dt.strftime("%b %d")
        # dfd = dfd.set_index('When')


        dfd[["$-q_{freeze}$", "$-q_{melt}$", "$-q_{T}$", "$-q_{surf}$"]] *= -1
        z = dfd[
            [
                "$-q_{freeze}$",
                "$-q_{melt}$",
                "$-q_{T}$",
                "$q_{SW}$",
                "$q_{LW}$",
                "$q_S$",
                "$q_L$",
                "$q_{F}$",
                "$q_{G}$",
            ]
        ]

        y2.index = y2.index + 1
        z.index = z.index + 1
        days = 19
        xlim1 = [-0.5, days + 0.5]
        xlim2 = [z.shape[0] - 1.5- days -1, z.shape[0] - 1.5]

        ax = subfigs[ctr].subplots(2, 2)
        # subfigs[ctr].suptitle(get_parameter_metadata(location)['shortname'], fontsize='x-large')
        for j in range(2):
            y2.plot.bar(
                stacked=True,
                edgecolor="black",
                linewidth=0.5,
                color=["xkcd:azure", "#0C70DE", skyblue, "xkcd:yellowgreen", pink],
                ax=ax[0, j],
            )
            ax[0, j].plot(dfds["mb"],'--k.')
            z.plot.bar(
                stacked=True,
                edgecolor="black",
                linewidth=0.5,
                color=[
                    "xkcd:azure",
                    "#0C70DE",
                    CB91_Violet,
                    red,
                    orange,
                    green,
                    "xkcd:yellowgreen",
                    # purple,
                    pink,
                    blue,
                ],
                ax=ax[1, j],
            )
            ax[1, j].plot(dfd["$-q_{surf}$"],'--k.')

        for i in range(2):
            if ctr == 0:
                ax[0, 0].title.set_text("Accumulation period")
                ax[0, 1].title.set_text("Ablation Period")
                ax[0, 0].title.set_size("x-large")
                ax[0, 1].title.set_size("x-large")
            for j in range(2):
                ax[i, j].get_legend().remove()
                ax[i, j].spines["top"].set_visible(False)

                if i == 1:
                    ax[i, j].set_ylim(-310, 310)
                    ax[i, j].set_ylabel("Energy [$W\\,m^{-2}$]")

                    d = 0.015  # how big to make the diagonal lines in axes coordinates
                    kwargs = dict(
                        transform=ax[i, j].transAxes, color="k", clip_on=False
                    )
                    if j == 0:
                        ax[i, j].plot(
                            (1 - d, 1 + d), (-d, +d), **kwargs
                        )  # top-left diagonal
                    else:
                        ax[i, j].plot((-d, d), (-d, +d), **kwargs)  # top-right diagonal
                else:
                    ax[i, j].set_ylim(-95, 95)
                    ax[i, j].set_ylabel("Thickness [$mm$ w. e.]")
                    ax[i, j].spines["bottom"].set_visible(False)
                    ax[i, j].tick_params(bottom=False)
                    ax[i, j].tick_params(labelbottom=False)

                if j == 0:
                    ax[i, j].set_xlim(xlim1)  # most of the data
                    ax[i, j].spines["right"].set_visible(False)
                else:
                    ax[i, j].set_xlim(xlim2)  # most of the data
                    ax[i, j].spines["left"].set_visible(False)
                    ax[i, j].set_ylabel("")
                    ax[i, j].tick_params(
                        left=False,
                        labelleft=False,
                    )
                ax[i, j].tick_params(right=False)
                ax[i, j].grid(
                    color="black", alpha=0.3, axis="y", linewidth=0.5, which="major"
                )

        subfigs[ctr].text(
            0.04,
            0.5,
            get_parameter_metadata(location)["shortname"],
            va="center",
            rotation="vertical",
            fontsize="x-large",
        )
        subfigs[ctr].subplots_adjust(hspace=0.05, wspace=0.025)
    lgd1 = ax[0, 0].legend(
        loc="upper center", bbox_to_anchor=(1, 4), ncol=5, 
    )
    lgd2 = ax[1, 0].legend(
        loc="upper center", bbox_to_anchor=(1, 2.4), ncol=10
    )
    add_patch(lgd1, title="Thickness Components", label = 'Net Thickness', color='k')
    add_patch(lgd2)
    plt.savefig(
        "data/paper/mass_energy_bal.jpg",
        dpi=300,
        bbox_inches="tight",
    )
