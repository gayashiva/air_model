""" Plots for meltwater"""
import sys, json
import os
import seaborn as sns
import numpy as np
import pandas as pd
import math
import matplotlib.colors
from datetime import datetime, timedelta
from matplotlib.offsetbox import AnchoredText
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.lines import Line2D
import logging, coloredlogs
from sklearn.metrics import r2_score

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
)
from src.utils.settings import config
from src.models.icestupaClass import Icestupa

if __name__ == "__main__":
    # Main logger
    logger = logging.getLogger(__name__)
    logger.setLevel("ERROR")

    df = pd.read_csv("/home/suryab/work/air_model/data/common/meltwater_2018.txt", sep=",")
    df = df[df.columns.drop(list(df.filter(regex="Unnamed")))]
    df = df[df.columns.drop(list(['Day']))]
    df = df[:-1]
    df.rename(
        columns={
            "Water Collected": "meltwater",
        },
        inplace=True,
    )

    start_date = datetime(2018, 4, 13)
    for i in range(0,df.shape[0]):
        df.loc[i,"time"] = start_date + pd.Timedelta(i, unit="d")

    df2 = pd.read_csv("/home/suryab/work/air_model/data/common/meltwater_2017.txt", sep=",")
    df2 = df2[df2.columns.drop(list(df2.filter(regex="Unnamed")))]
    df2['Days'] = df2['Days'].str.split('-').str.get(1)
    df2['Month'] = df2['Month'].ffill()
    df2['Month'] = df2['Month'].replace(to_replace=['April','May', 'June'], value=[4,5,6])
    start_date = datetime(2017, 4, 1)
    df2.rename(
        columns={
            "Water Collected": "meltwater",
        },
        inplace=True,
    )

    for i in range(0,df2.shape[0]):
        df2.loc[i,"time"] = datetime(year=2017, month=int(df2.loc[i,"Month"]), day=int(df2.loc[i,"Days"]))
    df2 = df2[df2.columns.drop(['Days', 'Month'])]
    # df2 = df2.set_index('time', drop=True)
    # df = df.set_index('time', drop=True)

    result = pd.concat([df, df2])
    result= result.set_index('time', drop=True).sort_index()
    result['meltwater'] /=1000
    df1 = result[result.index>datetime(2018, 1, 1)].reset_index()
    df2 = result[result.index<datetime(2018, 1, 1)].reset_index()
    df1['time'] = df1['time'].mask(df1['time'].dt.year == 2018, 
                                 df1['time'] + pd.offsets.DateOffset(year=2021))
    df2['time'] = df['time'].mask(df2['time'].dt.year == 2017, 
                                 df2['time'] + pd.offsets.DateOffset(year=2021))


    location = 'gangles21'
    spray = 'unscheduled_field'

    mypal = sns.color_palette("Set1", 2)
    default = "#284D58"
    grey = "#ced4da"

    SITE, FOLDER = config(location, spray)
    icestupa = Icestupa(location, spray)
    icestupa.read_output()

    dfd = icestupa.df.set_index("time").resample("D").sum().reset_index()
    dfd = dfd[dfd.time > datetime(2021, 4, 1)]

    fig, ax = plt.subplots(1, 1)
    x1 = df1.time
    y1 = df1.meltwater
    x2 = df2.time
    y2 = df2.meltwater
    x3 = dfd.time
    y3 = dfd.melted/1000
    ax.plot(x1,y1, label='IN18')
    ax.plot(x2,y2, label='IN17')
    ax.plot(x3,y3, label='IN21')
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["left"].set_color("grey")
    ax.spines["bottom"].set_color("grey")
    ax.set_ylabel("Daily melt [$m^3$]")
    ax.legend()

    ax.xaxis.set_major_locator(mdates.WeekdayLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    fig.autofmt_xdate()
    plt.savefig("data/figs/thesis/melt.png", bbox_inches="tight", dpi=300)
    plt.close()

