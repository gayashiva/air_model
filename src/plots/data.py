"""Icestupa class function that generates data plots
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


def shade(df_in, col, name):
    if name not in ["guttannen20", "guttannen21", "gangles21"]:
        df_ERA5 = df_in.copy()
        df = df_in.copy()
        df.loc[
            :, ["time", "temp", "SW_direct", "SW_diffuse", "wind", "press", "RH"]
        ] = np.NaN

    else:
        mask = df_in.missing_type.str.contains(col, na=False)
        df_ERA5 = df_in.copy()
        df = df_in.copy()
        df.loc[
            mask, ["time", "temp", "SW_direct", "SW_diffuse", "wind", "press", "RH"]
        ] = np.NaN

        df_ERA5.loc[
            ~mask,
            [
                "time",
                "temp",
                "SW_direct",
                "SW_diffuse",
                "wind",
                "press",
                "RH",
                "missing_type",
            ],
        ] = np.NaN

    events = np.split(df_ERA5.time, np.where(np.isnan(df_ERA5.time.values))[0])
    # removing NaN entries
    events = [
        ev[~np.isnan(ev.values)] for ev in events if not isinstance(ev, np.ndarray)
    ]
    # removing empty DataFrames
    events = [ev for ev in events if not ev.empty]
    return df, df_ERA5, events


@Timer(text="%s executed in {:.2f} seconds" % __name__, logger=logging.warning)
def plot_input(df, folder, name):

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

    fig, (ax1, ax2, ax3, ax5, ax6, ax7) = plt.subplots(
        nrows=6, ncols=1, sharex="col", sharey="row", figsize=(12, 18)
    )

    x = df.time

    y1 = df.ppt
    ax1.plot(x, y1, linestyle="-", color=CB91_Blue, linewidth=1)
    ax1.set_ylabel("Precipitation [$mm$]", color=CB91_Blue)

    df_SZ, df_ERA5, events = shade(name=name, df_in=df, col="temp")
    y2 = df_SZ.temp
    y2_ERA5 = df_ERA5.temp
    ax2.plot(x, y2, linestyle="-", color="#284D58", linewidth=1)
    for ev in events:
        ax2.axvspan(
            ev.head(1).values, ev.tail(1).values, facecolor="xkcd:grey", alpha=0.25
        )
    ax2.plot(x, y2_ERA5, linestyle="-", color="#284D58")
    ax2.set_ylabel("Temperature [$\\degree C$]")

    y3 = df.SW_global
    lns2 = ax3.plot(x, y3, linestyle="-", label="Shortwave Global", color=red)
    # lns1 = ax3.plot(
    #     x,
    #     df.SW_diffuse,
    #     linestyle="-",
    #     label="Shortwave Diffuse",
    #     color=orange,
    #     alpha=0.6,
    # )
    ax3.axvspan(
        df.time.head(1).values,
        df.time.tail(1).values,
        facecolor="grey",
        alpha=0.25,
    )
    ax3.set_ylabel("Radiation [$W\\,m^{-2}$]")

    # lns = lns1 + lns2
    lns = lns2
    labs = [l.get_label() for l in lns]
    ax3.legend(lns, labs, ncol=2, loc="best")

    # y4 = df.LW_in
    # ax4.plot(x, y4, linestyle="-", color=green, alpha=0.6, label="Longwave")
    # ax4.axvspan(
    #     df.time.head(1).values,
    #     df.time.tail(1).values,
    #     facecolor="grey",
    #     alpha=0.25,
    # )
    # ax4.set_ylabel("Radiation [$W\\,m^{-2}$]")
    # ax4.legend(loc="best")

    df_SZ, df_ERA5, events = shade(name=name, df_in=df, col="RH")
    y5 = df_SZ.RH
    y5_ERA5 = df_ERA5.RH
    ax5.plot(x, y5, linestyle="-", color="#284D58", linewidth=1)
    ax5.plot(x, y5_ERA5, linestyle="-", color="#284D58")
    for ev in events:
        ax5.axvspan(ev.head(1).values, ev.tail(1).values, facecolor="grey", alpha=0.25)
    ax5.set_ylabel("Humidity [$\\%$]")

    df_SZ, df_ERA5, events = shade(name=name, df_in=df, col="press")
    y6 = df_SZ.press
    y6_ERA5 = df_ERA5.press
    ax6.plot(x, y6, linestyle="-", color="#264653", linewidth=1)
    ax6.plot(x, y6_ERA5, linestyle="-", color="#284D58")
    for ev in events:
        ax6.axvspan(ev.head(1).values, ev.tail(1).values, facecolor="grey", alpha=0.25)
    ax6.set_ylabel("Pressure [$hPa$]")

    df_SZ, df_ERA5, events = shade(name=name, df_in=df, col="wind")
    y7 = df_SZ.wind
    y7_ERA5 = df_ERA5.wind
    ax7.plot(x, y7, linestyle="-", color="#264653", linewidth=1, label="Schwarzsee")
    ax7.plot(x, y7_ERA5, linestyle="-", color="#284D58")
    for ev in events:  # Creates DeprecationWarning
        ax7.axvspan(ev.head(1).values, ev.tail(1).values, facecolor="grey", alpha=0.25)
    ax7.set_ylabel("Wind speed [$m\\,s^{-1}$]")

    ax1.xaxis.set_major_locator(mdates.WeekdayLocator())
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax1.xaxis.set_minor_locator(mdates.DayLocator())
    fig.autofmt_xdate()
    plt.savefig(
        folder + "Model_Input.png",
        bbox_inches="tight",
    )
    plt.clf()

    plt.close("all")
