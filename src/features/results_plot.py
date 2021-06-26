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
 
sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
)

from src.utils.settings import config
from src.models.icestupaClass import Icestupa

if __name__ == "__main__":
    locations = ['guttannen20', 'guttannen21', 'gangles21', 'schwarzsee19']

    index = pd.date_range(start ='1-1-2022', 
         end ='1-1-2024', freq ='D', name= "When")
    df_out = pd.DataFrame(columns=locations,index=index)

    for location in locations:
        # Get settings for given location and trigger
        SITE, FOLDER = config(location)
        icestupa = Icestupa(location)
        icestupa.read_output()
        icestupa.self_attributes()

        df = icestupa.df[["When","iceV"]]
        if location == 'schwarzsee19':
            df['When'] = df['When'].mask(icestupa.df['When'].dt.year == 2019, 
                                         icestupa.df['When'] + pd.offsets.DateOffset(year=2023))
        if location == 'guttannen20':
            df['When'] = df['When'].mask(icestupa.df['When'].dt.year == 2019, 
                                         icestupa.df['When'] + pd.offsets.DateOffset(year=2022))
            df['When'] = df['When'].mask(icestupa.df['When'].dt.year == 2020, 
                                         icestupa.df['When'] + pd.offsets.DateOffset(year=2023))
        if location == 'guttannen21':
            df['When'] = df['When'].mask(icestupa.df['When'].dt.year == 2020, 
                                         icestupa.df['When'] + pd.offsets.DateOffset(year=2022))
        df['When'] = df['When'].mask(icestupa.df['When'].dt.year == 2021, 
                                     icestupa.df['When'] + pd.offsets.DateOffset(year=2023))

        dfd = df.set_index("When").resample("D").mean().reset_index()
        # dfd = df
        dfd = dfd.set_index("When")
        print(dfd.tail())
        df_out[location] = dfd["iceV"] 


    df_out.to_csv("data/paper/try.csv")
    df_out = df_out.reset_index()
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

    # location = "guttannen20"
    location = "gangles21"
    # Get settings for given location and trigger
    SITE, FOLDER = config(location)
    icestupa = Icestupa(location)
    icestupa.read_output()
    icestupa.self_attributes()
    df_c = pd.read_hdf(FOLDER["input"] + "model_input_" + icestupa.trigger + ".h5", "df_c")
    if icestupa.name in ["guttannen21", "guttannen20"]:
        df_c = df_c[1:]
    df_c = df_c.set_index("When").resample("D").mean().reset_index()
    dfv = df_c[["When", "DroneV", "DroneVError"]]
    if location == 'schwarzsee19':
        dfv['When'] = dfv['When'].mask(df_c['When'].dt.year == 2019, 
                                     df_c['When'] + pd.offsets.DateOffset(year=2023))
    if location == 'guttannen20':
        dfv['When'] = dfv['When'].mask(df_c['When'].dt.year == 2019, 
                                     df_c['When'] + pd.offsets.DateOffset(year=2022))
        dfv['When'] = dfv['When'].mask(df_c['When'].dt.year == 2020, 
                                     df_c['When'] + pd.offsets.DateOffset(year=2023))
    if location == 'guttannen21':
        dfv['When'] = dfv['When'].mask(df_c['When'].dt.year == 2020, 
                                     df_c['When'] + pd.offsets.DateOffset(year=2022))
    dfv['When'] = dfv['When'].mask(df_c['When'].dt.year == 2021, 
                                 df_c['When'] + pd.offsets.DateOffset(year=2023))

    fig, ax = plt.subplots()
    x = df_out.When
    y1 = df_out[location]
    x2 = dfv.When
    y2 = dfv.DroneV
    # yerr = df_c.DroneVError
    # ax.set_ylabel("Ice Volume[$m^3$]")
    ax.plot(
        x,
        y1,
        "b-",
        label="Modelled Volume",
        linewidth=1,
        color=CB91_Blue,
    )
    # ax.fill_between(x, y1=icestupa.V_dome, y2=0, color=grey, label = "Dome Volume")
    ax.scatter(x2, y2, color=CB91_Green, label="Measured Volume")
    # ax.errorbar(x, y2,yerr=df_c.DroneVError, color=CB91_Green)

    ax.set_ylim(0, round(df_out[location].max(),0))
    # plt.legend()

    # Hide the right and top spines
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    # Only show ticks on the left and bottom spines
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')

    ax.yaxis.set_major_locator(plt.LinearLocator(numticks=2))
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    # ax.xaxis.set_major_locator(mdates.WeekdayLocator("%b %d"))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    # ax.xaxis.set_minor_locator(mdates.WeekdayLocator())
    fig.autofmt_xdate()
    plt.savefig("data/paper/try.jpg", bbox_inches="tight", dpi=300)
    plt.clf()
