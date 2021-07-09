# libraries
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
import uncertainpy as un
import statistics as st

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
)

from src.utils.settings import config

# from src.utils.uq_output import draw_plot
from src.models.icestupaClass import Icestupa
from src.models.methods.metadata import get_parameter_metadata
import seaborn as sns
import matplotlib.pyplot as plt

if __name__ == "__main__":
    locations = ["gangles21", "guttannen21", "guttannen20"]

    index = pd.date_range(start="1-1-2022", end="1-1-2024", freq="D", name="When")
    df_out = pd.DataFrame(columns=locations, index=index)

    names = [
        "IE",
        "A_I",
        "A_S",
        "A_DECAY",
        "T_PPT",
        "Z",
        "DX",
        "T_W",
        # "D_MEAN",
        # "MU_CONE",
        # "r_spray",
    ]
    names_label = [
        "$\\epsilon_{ice}$",
        r"$\alpha_{ice}$",
        r"$\alpha_{snow}$",
        "$\\tau$",
        "$T_{ppt}$",
        "$z_{0}$",
        "$\\Delta x$",
        "$T_{water}$",
        # "$d_{mean}$",
        # r"$\mu_{cone}$",
        # "$r_{spray}$",
    ]
    zip_iterator = zip(names, names_label)
    param_dictionary = dict(zip_iterator)

    evaluations = []
    percent_change = []
    efficiency_change = []
    site = []
    param = []
    result = []
    freeze_rate = []
    melt_rate = []
    # growth_rate = []
    fig, ax = plt.subplots()
    for location in locations:
        SITE, FOLDER = config(location)
        icestupa = Icestupa(location)
        icestupa.read_output()
        icestupa.self_attributes()
        feature_name = "efficiency"

        M_input = round(icestupa.df["input"].iloc[-1], 1)
        M_water = round(icestupa.df["meltwater"].iloc[-1], 1)
        M_ice = round(icestupa.df["ice"].iloc[-1] - icestupa.V_dome * icestupa.RHO_I, 1)
        icestupa.se = (M_water + M_ice) / M_input * 100

        for j in range(0, icestupa.df.shape[0]):
            if icestupa.df.loc[j, "fountain_froze"] != 0:
                freeze_rate.append(
                    [
                        get_parameter_metadata(location)["shortname"],
                        j,
                        icestupa.df.loc[j, "fountain_froze"] / 60,
                    ]
                )
            if icestupa.df.loc[j, "melted"] != 0:
                melt_rate.append(
                    [
                        get_parameter_metadata(location)["shortname"],
                        j,
                        icestupa.df.loc[j, "melted"] / 60,
                    ]
                )
            # growth_rate.append([get_parameter_metadata(location)['shortname'],j,icestupa.df.loc[j,"growth"]])
            # growth_rate.append([get_parameter_metadata(location)['shortname'],j,icestupa.df.loc[j,"fountain_froze"]/60 - icestupa.df.loc[j,"melted"]/60])
        for name in names:
            data = un.Data()
            filename1 = FOLDER["sim"] + name + ".h5"
            data.load(filename1)
            evaluations.append(data[feature_name].evaluations)
            eval = data[feature_name].evaluations
            print(
                f"95 percent confidence interval caused by {name} is {round(st.mean(eval),2)} and {round(2 * st.stdev(eval),2)}"
            )
            # percent_change.append(
            #     (data[feature_name].evaluations - icestupa.df.iceV.max())
            #     / icestupa.df.iceV.max()
            #     * 100
            # )
            # efficiency_change.append((data[feature_name].evaluations - icestupa.se))
            for i in range(0, len(data[feature_name].evaluations)):
                result.append(
                    [
                        get_parameter_metadata(location)["shortname"],
                        param_dictionary[name],
                        data[feature_name].evaluations[i],
                        (data[feature_name].evaluations[i] - icestupa.se),
                        # (data[feature_name].evaluations[i] - icestupa.df.iceV.max())
                        # / icestupa.df.iceV.max()
                        # * 100,
                    ]
                )

    df = pd.DataFrame(result, columns=["Site", "param", "SE", "percent_change"])
    df2 = pd.DataFrame(freeze_rate, columns=["Site", "hour", "frozen"])
    df3 = pd.DataFrame(melt_rate, columns=["Site", "hour", "melted"])
    df4 = pd.DataFrame(freeze_rate, columns=["Site", "hour", "growth"])
    print(df2.head())
    print(df2.tail())

    ax = sns.boxplot(
        x="param", y="percent_change", hue="Site", data=df, palette="Set1", width=0.5
    )
    ax.set_xlabel("Parameter")
    ax.set_ylabel("Sensitivity of Storage Efficiency [$\%$]")
    plt.savefig("data/paper/sensitivities.jpg", bbox_inches="tight", dpi=300)
    plt.clf()

    ax = sns.histplot(
        df2, x="frozen", hue="Site", palette="Set1", element="step", fill=False
    )
    ax.set_ylabel("Discharge duration [ $hours$ ]")
    ax.set_xlabel("Freezing rate [ $l\\, min^{-1}$ ]")
    plt.savefig("data/paper/freeze_rate.jpg", bbox_inches="tight", dpi=300)
    plt.clf()

    ax = sns.histplot(
        df3, x="melted", hue="Site", palette="Set1", element="step", fill=False
    )
    ax.set_ylabel("Discharge duration [ $hours$ ]")
    ax.set_xlabel("Melting rate [ $l\\, min^{-1}$ ]")
    plt.savefig("data/paper/melt_rate.jpg", bbox_inches="tight", dpi=300)
    plt.clf()

    # ax = sns.histplot(df4, x="growth", hue="Site", palette="Set1", element="step", fill=False)
    # ax.set_ylabel("Discharge duration [ $hours$ ]")
    # ax.set_xlabel("Growth rate [ $l\\, min^{-1}$ ]")
    # plt.savefig("data/paper/growth_rate.jpg", bbox_inches="tight", dpi=300)
    # plt.clf()
