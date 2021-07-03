
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
from mpl_toolkits.axisartist.axislines import Axes
from mpl_toolkits import axisartist
import uncertainpy as un
 
sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
)

from src.utils.settings import config
from src.models.icestupaClass import Icestupa
from src.models.methods.metadata import get_parameter_metadata
import seaborn as sns
import matplotlib.pyplot as plt
from src.models.methods.metadata import get_parameter_metadata

if __name__ == "__main__":
    # locations = ['guttannen20',  'gangles21','guttannen21']
    locations = ['guttannen21']
    fig, ax = plt.subplots()
    current_palette = sns.color_palette('Set1')
    sns.set(style="darkgrid")
    ctr = 0

    for location in locations:
        SITE, FOLDER = config(location)
        icestupa = Icestupa(location)
        icestupa.read_output()
        for i in range(1,icestupa.df.shape[0]-1):
            icestupa.df.loc[i, "growth_rate"] = (
                icestupa.df.loc[i+1, "ice"]
                - icestupa.df.loc[i, "ice"]
            )
        icestupa.df.growth_rate = icestupa.df.growth_rate/60
        print(icestupa.df.growth_rate.describe())
        # sns.histplot(icestupa.df[icestupa.df.growth_rate!=0].growth_rate/icestupa.DT * 60, label = location)
        sns.histplot(icestupa.df[icestupa.df.melted!=0].melted/icestupa.DT * 60, label = location)
        # sns.histplot(icestupa.df[icestupa.df.fountain_froze!=0].fountain_froze/60, label = get_parameter_metadata(location)['shortname'], color = current_palette[ctr])
        # sns.kdeplot(icestupa.df[icestupa.df.fountain_froze!=0].fountain_froze/60, label = location, color = current_palette[ctr])
        plt.ylabel("Discharge duration [ $hours$ ]")
        plt.xlabel("Freezing rate [ $l\\, min^{-1}$ ]")
        plt.legend()
        plt.savefig("data/paper/try.jpg", bbox_inches="tight", dpi=300)
        ctr +=1
