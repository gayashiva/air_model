"""Icestupa class function that generates figures for web app
"""

import numpy as np
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText
from matplotlib.ticker import AutoMinorLocator
import pandas as pd
import logging
from codetiming import Timer

logger = logging.getLogger("__main__")


@Timer(text="%s executed in {:.2f} seconds" % __name__, logger=logging.warning)
def summary_figures(self):

    np.warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)
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

    """MB and EB fig"""
    self.df = self.df.rename(
        {
            "SW": "$q_{SW}$",
            "LW": "$q_{LW}$",
            "Qs": "$q_S$",
            "Ql": "$q_L$",
            "Qf": "$q_{F}$",
            "Qg": "$q_{G}$",
            "Qsurf": "$q_{surf}$",
            "Qmelt": "$-q_{melt}$",
            "Qfreeze": "$-q_{freeze}$",
            "Qt": "$-q_{T}$",
        },
        axis=1,
    )

    dfds = self.df[
        [
            "time",
            "fountain_froze",
            "ppt",
            "dep",
            "melted",
            "sub",
            "A_cone",
            "iceV",
            "Discharge",
            "wasted",
        ]
    ]

    with pd.option_context("mode.chained_assignment", None):
        for i in range(0, dfds.shape[0]):
            if self.df.loc[i, "A_cone"] != 0:
                dfds.loc[i, "Ice"] = dfds.loc[i, "fountain_froze"] / (
                    self.df.loc[i, "A_cone"] * self.RHO_I
                )
                dfds.loc[i, "melted"] *= -1 / (self.df.loc[i, "A_cone"] * self.RHO_I)
                dfds.loc[i, "sub"] *= -1 / (self.df.loc[i, "A_cone"] * self.RHO_I)
                dfds.loc[i, "ppt"] *= 1 / (self.df.loc[i, "A_cone"] * self.RHO_I)
                dfds.loc[i, "dep"] *= 1 / (self.df.loc[i, "A_cone"] * self.RHO_I)
            else:
                dfds.loc[i, "Ice"] = 0
                dfds.loc[i, "melted"] *= 0
                dfds.loc[i, "sub"] *= 0
                dfds.loc[i, "ppt"] *= 0
                dfds.loc[i, "dep"] *= 0

    dfds = dfds.set_index("time").resample("D").sum().reset_index()

    dfds = dfds.rename(
        columns={
            "ppt": "Snow",
            "melted": "Melt",
            "sub": "Sublimation",
            "dep": "Deposition",
        }
    )

    y2 = dfds[
        [
            "time",
            "Ice",
            "Snow",
            "Deposition",
            "Sublimation",
            "Melt",
        ]
    ]
    y2 = y2.set_index("time")

    dfds2 = self.df.set_index("time").resample("D").mean().reset_index()

    dfd = self.df.set_index("time").resample("D").mean().reset_index()
    dfd["time"] = dfd["time"].dt.strftime("%b %d")
    dfd[["$-q_{freeze}$", "$-q_{melt}$", "$-q_{T}$"]] *= -1
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

    fig = plt.figure(figsize=(12, 14))
    ax1 = fig.add_subplot(3, 1, 1)
    z.plot.bar(
        stacked=True,
        edgecolor="black",
        linewidth=0.5,
        color=[
            "xkcd:azure",
            purple,
            pink,
            red,
            orange,
            green,
            "xkcd:yellowgreen",
            CB91_Violet,
            blue,
        ],
        ax=ax1,
    )
    ax1.xaxis.set_label_text("")
    ax1.grid(color="black", alpha=0.3, linewidth=0.5, which="major")
    plt.ylabel("Energy Flux [$W\\,m^{-2}$]")
    plt.legend(loc="upper center", ncol=8)
    # plt.ylim(-125, 125)
    x_axis = ax1.axes.get_xaxis()
    x_axis.set_visible(False)
    at = AnchoredText("(a)", prop=dict(size=10), frameon=True, loc="upper left")
    at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
    ax1.add_artist(at)

    ax2 = fig.add_subplot(3, 1, 2)
    ax2 = y2.plot.bar(
        stacked=True,
        edgecolor="black",
        linewidth=0.5,
        color=[skyblue, "xkcd:azure", orange, green, "#0C70DE"],
        ax=ax2,
    )

    plt.ylabel("Thickness ($m$ w. e.)")
    plt.xticks(rotation=45)
    plt.legend(loc="upper center", ncol=6)
    if y2.sum(axis=1).max() > 0.1:
        ax2.set_ylim(-0.1, 0.1)  # Uneven thickness
    ax2.yaxis.set_minor_locator(AutoMinorLocator())
    ax2.grid(axis="y", color="black", alpha=0.3, linewidth=0.5, which="major")
    ax2.xaxis.set_label_text("")
    x_axis = ax2.axes.get_xaxis()
    x_axis.set_visible(False)
    at = AnchoredText("(b)", prop=dict(size=10), frameon=True, loc="upper left")
    at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
    ax2.add_artist(at)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax2.xaxis.set_major_locator(mdates.WeekdayLocator())
    ax2.xaxis.set_minor_locator(mdates.DayLocator())

    ax4 = fig.add_subplot(3, 1, 3)
    ax4.bar(
        x="time",
        height="iceV",
        linewidth=0.5,
        edgecolor="black",
        color=skyblue,
        data=dfds2,
    )
    ax4.xaxis.set_label_text("")
    ax4.set_ylabel("Ice Volume($m^3$)")
    ax4.grid(axis="y", color="black", alpha=0.3, linewidth=0.5, which="major")
    at = AnchoredText("(c)", prop=dict(size=10), frameon=True, loc="upper left")
    at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
    ax4.add_artist(at)
    ax4.xaxis.set_major_locator(mdates.WeekdayLocator())
    ax4.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax4.xaxis.set_minor_locator(mdates.DayLocator())
    fig.autofmt_xdate()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(
        self.fig + self.spray + "/output.png",
        bbox_inches="tight",
    )
    plt.clf()

    """Ice Volume Fig"""
    # if self.name in ["guttannen21", "guttannen20", "gangles21", "guttannen22"]:
    df_c = pd.read_hdf(self.input + "input.h5", "df_c")
    # df_c = df_c.rename(columns={"When": "time"})

    df_c = df_c[["time", "DroneV", "DroneVError"]]
    if self.name in ["guttannen21", "guttannen20", "gangles21"]:
        df_c = df_c[1:]

    tol = pd.Timedelta("15T")
    df_c = df_c.set_index("time")
    self.df = self.df.set_index("time")
    df_c = pd.merge_asof(
        left=self.df,
        right=df_c,
        right_index=True,
        left_index=True,
        direction="nearest",
        tolerance=tol,
    )
    df_c = df_c[["DroneV", "DroneVError", "iceV"]]
    self.df = self.df.reset_index()

    if self.name in ["guttannen21", "guttannen20"]:
        df_cam = pd.read_hdf(self.input + "input.h5", "df_cam")
        tol = pd.Timedelta("15T")
        self.df = self.df.set_index("time")
        df_cam = pd.merge_asof(
            left=self.df,
            right=df_cam,
            right_index=True,
            left_index=True,
            direction="nearest",
            tolerance=tol,
        )
        df_cam = df_cam[["cam_temp", "T_s", "T_bulk"]]
        self.df = self.df.reset_index()

    fig, ax = plt.subplots()
    x = self.df.time
    y1 = self.df.iceV
    ax.set_ylabel("Ice Volume[$m^3$]")
    ax.plot(
        x,
        y1,
        label="Modelled Volume",
        linewidth=1,
        color=CB91_Blue,
    )
    # if self.name in ["guttannen21", "guttannen22", "guttannen20", "schwarzsee19", "gangles21"]:
    y2 = df_c.DroneV
    yerr = df_c.DroneVError
    ax.fill_between(x, y1=self.V_dome, y2=0, color=grey, label="Dome Volume")
    ax.scatter(x, y2, color=CB91_Green, label="Measured Volume")
    ax.errorbar(x, y2, yerr=df_c.DroneVError, color=CB91_Green)

    ax.set_ylim(bottom=0)
    plt.legend()
    ax.xaxis.set_major_locator(mdates.WeekdayLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax.xaxis.set_minor_locator(mdates.DayLocator())
    fig.autofmt_xdate()
    plt.savefig(
        self.fig + self.spray + "/Vol_Validation.png",
        bbox_inches="tight",
    )
    plt.clf()

    fig, ax = plt.subplots()
    x = self.df.time
    y1 = self.df.Discharge
    ax.set_ylabel("Discharge [$l min-1$]")
    ax.plot(
        x,
        y1,
        linewidth=1,
        color=CB91_Blue,
    )
    plt.legend()
    ax.xaxis.set_major_locator(mdates.WeekdayLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax.xaxis.set_minor_locator(mdates.DayLocator())
    fig.autofmt_xdate()
    plt.savefig(
        self.fig + self.spray + "/Discharge.png",
        bbox_inches="tight",
    )
    plt.clf()

    if self.name in ["guttannen21", "guttannen20"]:
        fig, ax = plt.subplots()
        CB91_Purple = "#9D2EC5"
        CB91_Violet = "#661D98"
        CB91_Amber = "#F5B14C"
        x = self.df.time
        y1 = self.df.T_s
        y2 = df_cam.cam_temp
        ax.plot(
            x,
            y1,
            label="Modelled Temperature",
            linewidth=1,
            color=CB91_Amber,
            zorder=0,
        )
        ax.scatter(
            x,
            y2,
            color=CB91_Violet,
            s=1,
            label="Measured Temperature",
            zorder=1,
        )
        plt.legend()
        ax.xaxis.set_major_locator(mdates.WeekdayLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
        ax.xaxis.set_minor_locator(mdates.DayLocator())
        fig.autofmt_xdate()
        plt.savefig(
            self.fig + self.spray + "/Temp_Validation.png",
            bbox_inches="tight",
        )
    plt.close("all")

    self.df = self.df.rename(
        {
            "$q_{SW}$": "SW",
            "$q_{LW}$": "LW",
            "$q_S$": "Qs",
            "$q_L$": "Ql",
            "$q_{F}$": "Qf",
            "$q_{G}$": "Qg",
            "$q_{surf}$": "Qsurf",
            "$-q_{freeze/melt}$": "Qmelt",
            "$-q_{T}$": "Qt",
        },
        axis=1,
    )
