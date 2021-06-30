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
    locations = ['guttannen21',  'gangles21','guttannen20']

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
        "DX",
        "T_W",
        "d_mean",
        "r_spray",
    ]
    names_label = [
        "$T_{ppt}$",
        "$H_{ppt}$",
        "$\\epsilon_{ice}$",
        r"$\alpha_{ice}$",
        r"$\alpha_{snow}$",
        "$\\tau$",
        "$\\Delta x$",
        "$T_{water}$",
        "$d_{mean}$",
        "$r_{spray}$",
    ]
    zip_iterator = zip(names, names_label)
    param_dictionary = dict(zip_iterator)

    evaluations = []
    percent_change= []
    site= []
    param= []
    result= []
    fig, ax = plt.subplots()
    for location in locations:
        SITE, FOLDER = config(location)
        icestupa = Icestupa(location)
        icestupa.read_output()
        print(icestupa.df.iceV.max())
        icestupa.self_attributes()

        for name in names:
            data = un.Data()
            filename1 = FOLDER["sim"] + name + ".h5"
            data.load(filename1)
            evaluations.append(data["max_volume"].evaluations)
            percent_change.append((data["max_volume"].evaluations - icestupa.df.iceV.max())/icestupa.df.iceV.max()*100)
            for i in range(0,len(data["max_volume"].evaluations)):
                result.append([get_parameter_metadata(location)['shortname'], param_dictionary[name], data["max_volume"].evaluations[i],(data["max_volume"].evaluations[i]-
    icestupa.df.iceV.max())/icestupa.df.iceV.max()*100])

    df = pd.DataFrame(result, columns=['Site', 'param', 'iceV', 'percent_change'])
    print(df.head())
    print(df.tail())

    sns.set(style="darkgrid")
    ax = sns.boxplot(x="param", y="percent_change", hue="Site", data=df, palette="Set1", width=0.5)
    ax.set_xlabel("Parameter")
    ax.set_ylabel("Sensitivity of Maximum Ice Volume [$\%$]")
    plt.savefig("data/paper/sensitivities.jpg", bbox_inches="tight", dpi=300)
