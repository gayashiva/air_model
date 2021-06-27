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
# from src.utils.uq_output import draw_plot
from src.models.icestupaClass import Icestupa
from src.models.methods.metadata import get_parameter_metadata
import seaborn as sns
import matplotlib.pyplot as plt

def draw_plot(data, edge_color, fill_color, labels):
    bp = ax.boxplot(data, patch_artist=True, labels=labels, sym="o")

    for element in ["boxes", "whiskers", "fliers", "means", "medians", "caps"]:
        plt.setp(bp[element], color=edge_color)

    for patch in bp["boxes"]:
        patch.set(facecolor=fill_color)

if __name__ == "__main__":
    # sns.set(style="darkgrid")
#     df = sns.load_dataset('tips')
#     print(df.head())
#     print(df.describe())
# 
#     sns.boxplot(x="day", y="total_bill", hue="smoker", data=df, palette="Set1", width=0.5)
#     plt.savefig("data/paper/box.jpg", bbox_inches="tight", dpi=300)
    locations = ['guttannen21',  'gangles21','guttannen20', 'schwarzsee19']

    index = pd.date_range(start ='1-1-2022', 
         end ='1-1-2024', freq ='D', name= "When")
    df_out = pd.DataFrame(columns=locations,index=index)

    names = [
        "T_PPT",
        "H_PPT",
        "IE",
        "A_I",
        "A_S",
        "A_DECAY",
        "T_W",
        "DX",
    ]
    names_label = [
        "$T_{ppt}$",
        "$H_{ppt}$",
        "$\\epsilon_{ice}$",
        r"$\alpha_{ice}$",
        r"$\alpha_{snow}$",
        "$\\tau$",
        "$T_{water}$",
        "$\\Delta x$",
    ]

    # fig, ax = plt.subplots(4, 1, sharex='col', figsize=(12, 14))
    # fig.subplots_adjust(hspace=0.4, wspace=0.4)
    # i=0
    fig, ax = plt.subplots()
    for location in locations:
        # Get settings for given location and trigger
        SITE, FOLDER = config(location)
        icestupa = Icestupa(location)
        icestupa.read_output()
        icestupa.self_attributes()
        evaluations = []

        for name in names:
            data = un.Data()
            filename1 = FOLDER["sim"] + name + ".h5"
            data.load(filename1)
            evaluations.append(data["max_volume"].evaluations)

        draw_plot(evaluations, "k", "xkcd:grey", names_label)
        # ax.set_xlabel("Parameter")
        # ax.set_ylabel("Sensitivity of Maximum Ice Volume [$m^3$]")
        ax.grid(axis="y")
    plt.savefig("data/paper/sensitivities.jpg", bbox_inches="tight", dpi=300)
