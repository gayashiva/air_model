"""Calibrate camera pixels with height info and extract radius evolution
"""
import sys
import os
import json
import numpy as np
import pandas as pd
import math
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import logging, coloredlogs
from tqdm import tqdm

dirname = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.append(dirname)
from src.utils.settings import config
from src.models.icestupaClass import Icestupa
from src.models.methods.metadata import get_parameter_metadata

if __name__ == "__main__":
    # Main logger
    logger = logging.getLogger(__name__)
    logger.setLevel("ERROR")

    loc = "guttannen21"
    CONSTANTS, SITE, FOLDER = config(loc)
    dfh = pd.read_csv(
        FOLDER['raw'] + 'height_evolution.csv',
        sep=",",
    )
    print(dfh.columns)

    dfh["Label"] = dfh["Label"].str.split("_").str[-1]
    dfh["time"] = pd.to_datetime(dfh["Label"], format="%b-%d %H:%M")
    dfh['time'] = dfh['time'].mask(dfh['time'].dt.year == 1900, dfh['time'] + pd.offsets.DateOffset(year=2020))
    dfh['time'] = dfh['time'].mask(dfh['time'].dt.month == 1, dfh['time'] + pd.offsets.DateOffset(year=2021))
    dfh = dfh[['time', 'Y']]
    dfh = dfh.reset_index(drop=True)
    dfh = dfh.groupby(['time'])['Y'].apply(lambda x: x.max() - x.min())
    dfh = dfh.reset_index()
    # dfh.groupby(['time'])['B'].apply(lambda x: ','.join(x.astype(str))).reset_index()
    # dfh = dfh.set_index('time')

    f_heights = [
        {"time": SITE["start_date"], "h_f": 2.68},
        {"time": datetime(2020, 12, 30, 16), "h_f": 3.75},
        {"time": datetime(2021, 1, 7, 16), "h_f": 4.68},
        {"time": datetime(2021, 1, 11, 16), "h_f": 5.68},
    ]
    df_h = pd.DataFrame(f_heights)
    dfh["h_f"] = df_h["h_f"]
    dfh["pixels/m"] = dfh["h_f"]/dfh["Y"]
    ppm = dfh["pixels/m"].mean()
    print("For guttannen21, pixels/m:%0.3f"%ppm)

    dfr = pd.read_csv(
        FOLDER['raw'] + 'rad_evolution.csv',
        sep=",",
    )
    dfr["Label"] = dfr["Label"].str.split("_").str[-1]
    dfr["time"] = pd.to_datetime(dfr["Label"], format="%b-%d %H:%M")
    dfr['time'] = dfr['time'].mask(dfr['time'].dt.year == 1900, dfr['time'] + pd.offsets.DateOffset(year=2020))
    dfr['time'] = dfr['time'].mask(dfr['time'].dt.month == 1, dfr['time'] + pd.offsets.DateOffset(year=2021))
    dfr['time'] = dfr['time'].mask(dfr['time'].dt.month == 2, dfr['time'] + pd.offsets.DateOffset(year=2021))
    dfr['time'] = dfr['time'].mask(dfr['time'].dt.month == 3, dfr['time'] + pd.offsets.DateOffset(year=2021))
    dfr['time'] = dfr['time'].mask(dfr['time'].dt.month == 4, dfr['time'] + pd.offsets.DateOffset(year=2021))
    dfr['time'] = dfr['time'].mask(dfr['time'].dt.month == 5, dfr['time'] + pd.offsets.DateOffset(year=2021))
    dfr = dfr[['time', 'X']]
    dfr = dfr.reset_index(drop=True)
    dfr = dfr.groupby(['time'])['X'].apply(lambda x: x.max() - x.min())
    dfr = dfr.reset_index()

    dfr["rad"] = dfr['X'] * ppm/2
    dfr = dfr[["time", 'rad']]
    dfr.to_csv(FOLDER['input']+ "rad_cam.csv")
    dfr['deltaT'] = (dfr['time']-dfr['time'].shift()).dt.total_seconds().div(60*60, fill_value=0)
    dfr['deltaR'] = (dfr['rad']-dfr['rad'].shift())
    dfr['rate[mm/hr]'] = dfr['deltaR']/dfr['deltaT'] * 1000
    dfr['constant'] = dfr["rad"] * dfr['rate[mm/hr]'] ** 2

        # dfr.time.diff().dt.seconds
    # dfr['delta'] = (dfr['time']-dfr['time'].shift()).fillna(0)

    print(dfr)

    fig, ax = plt.subplots()
    x = dfr.time
    y = dfr.rad
    ax.scatter(x,y)

    ax.xaxis.set_major_locator(mdates.WeekdayLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax.xaxis.set_minor_locator(mdates.DayLocator())
    fig.autofmt_xdate()
    plt.savefig(
        FOLDER['fig'] + "rad_cam.jpg",
        bbox_inches="tight",
    )
    plt.clf()
