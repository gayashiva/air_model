"""Plot calibration data for DX"""
import sys
import os
import numpy as np
import pandas as pd
import math
import matplotlib.colors
import statistics as st
from datetime import datetime, timedelta
import seaborn as sns
from matplotlib.offsetbox import AnchoredText
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import logging, coloredlogs
import multiprocessing
from time import sleep
import os, sys, time
from ast import literal_eval

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
)

from src.utils.settings import config
from src.models.icestupaClass import Icestupa
from src.models.methods.metadata import get_parameter_metadata


if __name__ == "__main__":
    # Main logger
    logger = logging.getLogger(__name__)
    logger.setLevel("ERROR")

    locations = ["gangles21", "guttannen21"]

    params = ['DX']
    kind = ['volume']


    dfx = pd.DataFrame(columns=params)
    for ctr, location in enumerate(locations):
        icestupa = Icestupa(location)
        CONSTANTS, SITE, FOLDER = config(location)
        icestupa.read_output()
        df_c = pd.read_hdf(FOLDER["input"] + "model_input.h5", "df_c")

        file_path = 'loo-cv-'+'volume'+'-'
        file_path += '-'.join('{}'.format(key) for key in params)

        df = pd.read_csv(FOLDER['sim'] + file_path)
        df = df.set_index('rmse').sort_index().reset_index()
        df['params'] = df['params'].apply(literal_eval)

        df = pd.concat([df.drop(['params'], axis=1), df['params'].apply(pd.Series)], axis=1)

        print()

        df['AIR'] = get_parameter_metadata(location)['shortname']
        df[['DX']] *= 1000
        df = df.round(2)
        df.rmse /=df_c.DroneV.max() 
        dfx = dfx.append(df, ignore_index = True)

    print(dfx.head())
    print(dfx.tail())

    df1 = dfx.loc[dfx["AIR"] == "IN21"]
    df2 = dfx.loc[dfx["AIR"] == "CH21"]
    df3 = dfx.loc[dfx["rmse"] <= 0.1]

    IN_DX = (df1.loc[df1.rmse == df1.rmse.min()].DX.values[0])
    CH_DX = (df2.loc[df2.rmse == df2.rmse.min()].DX.values[0])

    print(f'\t\nThe recommended DX for IN is {IN_DX} and CH is {CH_DX} mm.')
    print(f'\t\nDX Range recommended is {df3.DX.min()} and CH is {df3.DX.max()} mm.')

    dfx.to_csv("data/paper1/dx_calibrate.csv")

    fig, ax = plt.subplots()
    ax = sns.lineplot(
        x="DX", y="rmse", hue="AIR", data=dfx, palette="Set1"
    )
    print(dfx.loc[dfx.AIR == "CH21", "rmse"].min())
    ax.plot(dfx[(dfx.AIR == "CH21")].loc[(dfx.rmse == dfx.loc[dfx.AIR == "CH21", "rmse"].min()), "DX"], dfx.loc[dfx.AIR == "CH21", "rmse"].min(), 'bo')
    ax.plot(dfx[(dfx.AIR == "IN21")].loc[(dfx.rmse == dfx.loc[dfx.AIR == "IN21", "rmse"].min()), "DX"], dfx.loc[dfx.AIR == "IN21", "rmse"].min(), 'ro')
    ax.set_xlabel(get_parameter_metadata("DX")["latex"])
    ax.set_ylabel("Normalized RMSE")

    plt.savefig(
        "data/paper1/dx.jpg",
        dpi=300,
        bbox_inches="tight",
    )

