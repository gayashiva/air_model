"""Compile raw data from the location, meteoswiss or ERA5
"""

# External modules
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
from scipy import stats
from sklearn.linear_model import LinearRegression

# Locals
dirname = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.append(dirname)
from src.utils.settings import config
from src.data.make_dataset import e_sat

def field(location="schwarzsee19"):
    SITE, FOLDER = config(location)
    if location == "gangles21":
        df_in = pd.read_csv(
            FOLDER["input"] + SITE["name"] + "_input_model.csv",
            header=0,
            parse_dates=["When"]
        )
        df = df_in.round(3)
        # CSV output
        logger.info(df_in.head())
        logger.info(df_in.tail())
        # df.to_csv(input_folder + SITE["name"] + "_input_field.csv")
        mask = (df["When"] >= SITE["start_date"]) & (df["When"] <= SITE["end_date"])
        df = df.loc[mask]
        df = df.reset_index()
        return df

def era5(location="schwarzsee19"):

    if location in ["gangles21"]:
        df_in3 = pd.read_csv(
            "/home/suryab/work/ERA5/outputs/leh_2021.csv",
            sep=",",
            header=0,
            parse_dates=["When"],
        )

        df_in3 = df_in3.set_index("When")
        df_in3 = df_in3.reset_index()

    SITE, FOLDER = config(location)

    mask = (df_in3["When"] >= SITE["start_date"]) & (df_in3["When"] <= SITE["end_date"])
    df_in3 = df_in3.loc[mask]
    df_in3 = df_in3.reset_index(drop="True")

    time_steps = 60 * 60
    df_in3["ssrd"] /= time_steps
    df_in3["strd"] /= time_steps
    df_in3["fdir"] /= time_steps
    df_in3["v_a"] = np.sqrt(df_in3["u10"] ** 2 + df_in3["v10"] ** 2)
    # Derive RH
    df_in3["t2m"] -= 273.15
    df_in3["d2m"] -= 273.15
    df_in3["t2m_RH"] = df_in3["t2m"]
    df_in3["d2m_RH"] = df_in3["d2m"]
    df_in3 = df_in3.apply(lambda x: e_sat(x) if x.name == "t2m_RH" else x)
    df_in3 = df_in3.apply(lambda x: e_sat(x) if x.name == "d2m_RH" else x)
    df_in3["RH"] = 100 * df_in3["d2m_RH"] / df_in3["t2m_RH"]
    df_in3["sp"] = df_in3["sp"] / 100
    df_in3["tp"] = df_in3["tp"] * 1000 / 3600  # mm/s
    df_in3["SW_diffuse"] = df_in3["ssrd"] - df_in3["fdir"]
    df_in3 = df_in3.set_index("When")

    # CSV output
    df_in3.rename(
        columns={
            "t2m": "T_a",
            "sp": "p_a",
            "tp": "Prec",
            "fdir": "SW_direct",
            "strd": "LW_in",
        },
        inplace=True,
    )

    df_in3 = df_in3[
        [
            "T_a",
            "RH",
            "Prec",
            "v_a",
            "SW_direct",
            "SW_diffuse",
            "LW_in",
            "p_a",
        ]
    ]

    df_in3 = df_in3.round(3)

    upsampled = df_in3.resample("15T")
    interpolated = upsampled.interpolate(method="linear")
    interpolated = interpolated.reset_index()

    df_in3 = interpolated[
        [
            "When",
            "T_a",
            "RH",
            "v_a",
            "SW_direct",
            "SW_diffuse",
            "LW_in",
            "p_a",
            "Prec",
        ]
    ]

    df_in3 = df_in3.reset_index()
    mask = (df_in3["When"] >= SITE["start_date"]) & (df_in3["When"] <= SITE["end_date"])
    df_in3 = df_in3.loc[mask]
    df_in3 = df_in3.reset_index()

    df_in3.to_csv(FOLDER["input"] + SITE["name"] + "_input_ERA5.csv")

    df_ERA5 = interpolated[
        [
            "When",
            "T_a",
            "v_a",
            "RH",
            "SW_direct",
            "SW_diffuse",
            "LW_in",
            "p_a",
            "Prec",
        ]
    ]

    # logger.info(df_ERA5.head())
    # logger.info(df_ERA5.tail())
    return df_ERA5, df_in3

if __name__ == "__main__":

    logger = logging.getLogger(__name__)
    coloredlogs.install(
        fmt="%(name)s %(levelname)s %(message)s",
        level=logging.INFO,
        logger=logger,
    )

    location = "gangles21"
    SITE, FOLDER = config("gangles21")

    df = field(location)
    df_ERA5, df_in3 = era5(SITE["name"])

    # print(df_ERA5.SW_direct.describe())
    # print(df.SW_direct.describe())

    pp = PdfPages(FOLDER["input"] + "compare.pdf")
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.scatter(df.SW_direct, df_ERA5.SW_direct, s=2)
    ax1.set_xlabel("Field SW")
    ax1.set_ylabel("ERA5 SW")
    ax1.grid()
    lims = [
    np.min([ax1.get_xlim(), ax1.get_ylim()]),  # min of both axes
    np.max([ax1.get_xlim(), ax1.get_ylim()]),  # max of both axes
    ]
    # now plot both limits against eachother
    ax1.plot(lims, lims, '--k', alpha=0.25, zorder=0)
    ax1.set_aspect('equal')
    ax1.set_xlim(lims)
    ax1.set_ylim(lims)
    # format the ticks
    pp.savefig(bbox_inches="tight")
    plt.clf()
    ax1 = fig.add_subplot(111)
    ax1.plot(df.SW_direct)
    ax1.set_ylabel("Field SW")
    pp.savefig(bbox_inches="tight")
    plt.clf()
    ax1 = fig.add_subplot(111)
    ax1.plot(df_ERA5.SW_direct + df_ERA5.SW_diffuse)
    ax1.set_ylabel("ERA5 SW")
    pp.savefig(bbox_inches="tight")
    plt.clf()
    pp.close()
