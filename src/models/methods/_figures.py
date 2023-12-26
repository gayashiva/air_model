"""Icestupa class function that generates figures for web app
"""

import numpy as np
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText
from matplotlib.ticker import AutoMinorLocator
import pandas as pd
import logging, json
from codetiming import Timer

logger = logging.getLogger("__main__")

# Define a function to format numbers in a human-readable way
def format_number(value):
    if value >= 1e6:
        return f"{value / 1e6:.0f}M"
    elif value >= 1e3:
        return f"{value / 1e3:.0f}K"
    else:
        return f"{value:.0f}"


# @Timer(text="%s executed in {:.2f} seconds" % __name__, logger=logging.warning)
def summary_figures(self):

    # np.warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)

    CB91_Blue = "#2CBDFE"

    """Ice Volume Fig"""
    fig, ax = plt.subplots()
    x = self.df.time
    y1 = self.df.iceV
    ax.set_ylabel("Ice Reservoir Volume[$m^3$]")
    ax.plot(
        x,
        y1,
        # label=self.name,
        linewidth=1,
        color=CB91_Blue,
    )
    ax.set_ylim(bottom=0)
    # ax.legend()
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    ax.xaxis.set_minor_locator(mdates.WeekdayLocator())
    fig.autofmt_xdate()

    with open(self.output + "results.json", "r") as f:
        results_dict = json.load(f)

    # Add a text box below the legend with the title "Summary Metrics"
    textbox_content = "\n".join([f"{key}: {format_number(value)}" for key, value in results_dict.items()])
    # Define a string variable
    summary_title = self.name + " Summary"
    textbox_content = f"{summary_title}\n\n" + textbox_content

    # Set the coordinates for the right-top corner
    x_coord = 0.82
    y_coord = 0.98

    # Annotate the text box inside the figure
    ax.annotate(
        textbox_content,
        xy=(x_coord, y_coord),
        xycoords="axes fraction",
        ha='center',
        va='top',
        fontsize=8,
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.7),
    )

    # Save the modified figure
    plt.savefig(
        self.fig + "/IRVol_with_results.png",
        bbox_inches="tight",
        dpi=300,
    )
    plt.clf()

    fig, ax = plt.subplots()
    x = self.df.time
    y1 = self.df.tau_atm
    ax.set_ylabel("Transmittivity []")
    ax.plot(
        x,
        y1,
        label=self.name,
        linewidth=1,
        # s=1,
        color=CB91_Blue,
    )
    plt.legend()
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    ax.xaxis.set_minor_locator(mdates.WeekdayLocator())
    fig.autofmt_xdate()
    plt.savefig(
        self.fig + "/tau_atm.png",
        bbox_inches="tight",
        dpi=300,
    )

    fig, ax = plt.subplots()
    x = self.df.time
    y1 = self.df.SW_extra
    ax.set_ylabel("SW_extra []")
    ax.plot(
        x,
        y1,
        label=self.name,
        linewidth=1,
        # s=1,
        color=CB91_Blue,
    )
    plt.legend()
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    ax.xaxis.set_minor_locator(mdates.WeekdayLocator())
    fig.autofmt_xdate()
    plt.savefig(
        self.fig + "/SW_extra.png",
        bbox_inches="tight",
        dpi=300,
    )

    fig, ax = plt.subplots()
    x = self.df.time
    y1 = self.df.SW_global
    ax.set_ylabel("SW_global []")
    ax.plot(
        x,
        y1,
        label=self.name,
        linewidth=1,
        # s=1,
        color=CB91_Blue,
    )
    plt.legend()
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    ax.xaxis.set_minor_locator(mdates.WeekdayLocator())
    fig.autofmt_xdate()
    plt.savefig(
        self.fig + "/SW_global.png",
        bbox_inches="tight",
        dpi=300,
    )
    plt.clf()

    plt.close("all")
