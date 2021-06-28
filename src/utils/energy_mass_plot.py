# libraries
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
 
sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
)

from src.utils.settings import config
from src.models.icestupaClass import Icestupa

if __name__ == "__main__":
    location="guttannen21"
    # location="gangles21"
    # location="schwarzsee19"

    # Get settings for given location and trigger
    SITE, FOLDER = config(location)
    icestupa = Icestupa(location)
    icestupa.read_output()
    icestupa.self_attributes()

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

    dfd = icestupa.df.set_index("When").resample("D").mean().reset_index()
    dfd["ice"] -= icestupa.V_dome * icestupa.RHO_I
    # dfd["When"] = dfd["When"].dt.strftime("%b %d")

    dfd[["$-q_{freeze/melt}$", "$-q_{T}$"]] *=-1
    z = dfd[["$-q_{freeze/melt}$", "$-q_{T}$", "$q_{SW}$", "$q_{LW}$", "$q_S$", "$q_L$", "$q_{F}$", "$q_{G}$"]]

    # Make data
    data = pd.DataFrame({
        'ice':dfd.ice.values,'water':dfd.meltwater.values,'sub':dfd.vapour.values,'runoff':dfd.unfrozen_water.values,},index=dfd.index.values)
     
    # We need to transform the data from raw data to percentage (fraction)
    data_perc = data.divide(data.sum(axis=1), axis=0)

    fig = plt.figure(figsize=(12, 6))
    ax1 = fig.add_subplot(2, 1, 1)
    ax1 = z.plot.bar(
                stacked=True, 
                edgecolor="black", 
                linewidth=0.5, 
                color=[purple, pink, red, orange, green, "xkcd:yellowgreen", "xkcd:azure", blue ],
                ax=ax1
                )
    x_axis = ax1.axes.get_xaxis()
    x_axis.set_visible(False)
    ax1.xaxis.set_label_text("")
    ax1.set_ylabel("Energy Flux [$W\\,m^{-2}$]")
    ax1.grid(axis="y", color="black", alpha=0.3, linewidth=0.5, which="major")
    at = AnchoredText("(a)", prop=dict(size=10), frameon=True, loc="upper left")
    at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
    ax1.add_artist(at)
    plt.legend(loc="upper center", ncol=8)
    plt.ylim(-200, 200)

    ax2 = fig.add_subplot(2, 1, 2)
    ax2.stackplot(
        dfd.When,  data_perc["ice"],  data_perc["water"],
        data_perc["sub"],data_perc["runoff"],labels=['ice','water','sub','runoff'])
    ax2.set_ylabel("Mass fraction")
    at = AnchoredText("(b)", prop=dict(size=10), frameon=True, loc="upper left")
    at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
    ax2.add_artist(at)
    ax2.margins(0,0)
    ax2.xaxis.set_major_locator(mdates.WeekdayLocator())
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax2.xaxis.set_minor_locator(mdates.DayLocator())
    fig.autofmt_xdate()
    plt.legend(loc="upper center", ncol=4)
    plt.ylim(0, 0.5)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("data/paper/try2.jpg", bbox_inches="tight", dpi=300)
    plt.close()
