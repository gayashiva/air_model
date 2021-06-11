
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.backends.backend_pdf import PdfPages
from pandas.plotting import register_matplotlib_converters
import math
import time
from pathlib import Path
from tqdm import tqdm
import os
import logging
import coloredlogs
import glob
dirname = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.append(dirname)
from src.utils.settings import config
from src.data.make_dataset import era5, linreg, meteoswiss, meteoswiss_parameter

if __name__ == "__main__":
    logger = logging.getLogger(__name__)
    coloredlogs.install(
        fmt="%(name)s %(levelname)s %(message)s",
        level=logging.INFO,
        logger=logger,
    )


    SITE, FOLDER, *args = config("ravat20")
    df_ERA5, df_in3 = era5(SITE["name"])
    print(df_ERA5.describe())

    df_ERA5 = df_ERA5.set_index("When")
    df = df_ERA5
    df['missing_type'] = 'NA'

    # Fill from ERA5
    logger.warning("Temperature NaN percent: %0.2f" %(df["T_a"].isna().sum()/df.shape[0]*100))
    logger.warning("wind NaN percent: %0.2f" %(df["v_a"].isna().sum()/df.shape[0]*100))

    df = df.reset_index()

    if SITE["name"] in ["diavolezza21"]:
        cols = [
            "When",
            "Discharge",
            "T_a",
            "RH",
            "v_a",
            "SW_direct",
            "SW_diffuse",
            "Prec",
            "p_a",
            "missing_type",
            "LW_in",
            # "a",
        ]

    if SITE["name"] in ["schwarzsee19", "ravat20"]:
        cols = [
            "When",
            "T_a",
            "RH",
            "v_a",
            "SW_direct",
            "SW_diffuse",
            "Prec",
            # "vp_a",
            "p_a",
            "missing_type",
            "LW_in",
        ]
    if SITE["name"] in ["guttannen20", "guttannen21"]:
        cols = [
            "When",
            "T_a",
            "RH",
            "v_a",
            "SW_direct",
            "SW_diffuse",
            "Prec",
            "vp_a",
            "p_a",
            "missing_type",
            "LW_in",
        ]

    df_out = df[cols]

    if df_out.isna().values.any():
        print(df_out[cols].isna().sum())
        for column in cols:
            if df_out[column].isna().sum() > 0 and column in ["a"]:
                albedo = df_out.a.replace(0, np.nan).mean()
                df_out.loc[df_out[column].isna(), column] = albedo
                logger.warning("Albedo Null values extrapolated in %s " %albedo)
            if df_out[column].isna().sum() > 0 and column in ["Discharge"]:
                discharge = df_out.Discharge.replace(0, np.nan).mean()

                df_out.loc[df_out[column].isna(), column] = discharge

                logger.warning(" Discharge Null values extrapolated in %s " %discharge)

            if df_out[column].isna().sum() > 0 and column not in ["missing_type", "Discharge"]:
                logger.warning(" Null values interpolated in %s" %column)
                df_out.loc[:, column] = df_out[column].interpolate()

    df_out = df_out.round(3)
    if len(df_out[df_out.index.duplicated()]):
        logger.error("Duplicate indexes")

    logger.info(df_out.tail())
    df_out.to_csv(FOLDER["input"]+ SITE["name"] + "_input_model.csv", index=False)

    fig, ax1 = plt.subplots()
    skyblue = "#9bc4f0"
    blue = "#0a4a97"
    x = df_out.When
    y = df_out.T_a
    ax1.plot(
        x,
        y,
        linestyle="-",
        color=blue,
    )
    ax1.set_ylabel("Discharge [$l\\, min^{-1}$]")
    ax1.xaxis.set_major_locator(mdates.WeekdayLocator())
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax1.xaxis.set_minor_locator(mdates.DayLocator())
    fig.autofmt_xdate()
    plt.savefig(FOLDER["input"]+ SITE["name"] + "test.png")


