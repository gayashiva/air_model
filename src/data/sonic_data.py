"""Convert raw sonic data to model input format
"""
# External modules
import sys, os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.backends.backend_pdf import PdfPages
from pandas.plotting import register_matplotlib_converters
import math
import time
import glob
from pathlib import Path
import logging
import coloredlogs
# Locals
dirname = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.append(dirname)
from src.utils.settings import config

if __name__ == "__main__":
    # Main logger
    logger = logging.getLogger(__name__)
    logger.setLevel("INFO")

    location="guttannen22"
    CONSTANTS, SITE, FOLDER = config(location)
    cols_old = [
        "TIMESTAMP",
        "T_probe_Avg",
        "RH_probe_Avg",
        "amb_press_Avg",
        "WS",
        "SnowHeight",
        "SW_IN",
        "SW_OUT",
        "LW_IN",
        "LW_OUT",
        "H",
        "Tice_Avg(1)",
        "Tice_Avg(2)",
        "Tice_Avg(3)",
        "Tice_Avg(4)",
        "Tice_Avg(5)",
        "Tice_Avg(6)",
        "Tice_Avg(7)",
        "Tice_Avg(8)",
    ]
    cols_new = ["time", "temp", "RH", "press", "wind", "snow_h", "SW_global", "SW_out", "LW_in", "LW_out",
        "Qs_meas", "T_ice_1", "T_ice_2", "T_ice_3", "T_ice_4", "T_ice_5","T_ice_6","T_ice_7","T_ice_8"]
    cols_dict = dict(zip(cols_old, cols_new))

    path = FOLDER["raw"] + "CardConvert/"
    all_files = glob.glob(path + "*.dat")
    print(all_files)
    li = []

    for file in all_files:

        df = pd.read_csv(
            file,
            sep=",",
            skiprows=[0,2,3],
            parse_dates=["TIMESTAMP"],
        )
        df = df[cols_old]
        df = df.rename(columns=cols_dict)

        for col in df.columns:
            if col != 'time':
                df[col] = df[col].astype(float)
        df = df.round(2)
        li.append(df)

    df = pd.concat(li, axis=0, ignore_index=True)
    df = df.set_index("time").sort_index()
    df = df[SITE["start_date"] :]
    df = df.reset_index()
    print(df.head())
    print(df.tail())
    col = "snow_h"
    print(df[col].describe())

    """Correct data errors"""
    df= df.replace("NAN", np.NaN)
    df = df.set_index("time").resample("H").mean().reset_index()
    df["ppt"] = 0
    df["missing_type"] = "-"

    df.to_csv(FOLDER["input"] + SITE["name"] + "_input_model.csv", index=False)

    fig, ax = plt.subplots()
    x = df.time
    y = df[col]
    ax.plot(x,y)
    # ax.set_ylim(0,10)

    # y1 = df["Tice_Avg(1)"]
    # y2 = df["Tice_Avg(3)"]
    # y3 = df["Tice_Avg(5)"]
    # ax.set_ylabel("Temp")
    # ax.plot(x,y1, label="1")
    # ax.plot(x,y2, label="2")
    # ax.plot(x,y3, label="3")
    # ax.legend()

    ax.xaxis.set_major_locator(mdates.WeekdayLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax.xaxis.set_minor_locator(mdates.DayLocator())
    fig.autofmt_xdate()
    plt.savefig(
        FOLDER['fig'] + "temp.jpg",
        bbox_inches="tight",
    )
    plt.clf()

