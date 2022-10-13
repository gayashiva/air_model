"""Compile raw data from the location, meteoswiss or ERA5
"""

# External modules
import sys, os, json
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
import logging
import coloredlogs
from scipy import stats
from sklearn.linear_model import LinearRegression
from pvlib import location, atmosphere, irradiance

# Locals
dirname = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.append(dirname)
from src.utils.settings import config
from src.data.field import get_field
from src.data.era5 import get_era5
from src.data.meteoswiss import get_meteoswiss
from src.plots.data import plot_input
from src.utils import setup_logger

if __name__ == "__main__":
    # Main logger
    logger = logging.getLogger(__name__)
    logger.setLevel("WARNING")
    # logger.setLevel("INFO")

    loc = "sibinacocha22"
    SITE, FOLDER = config(loc)
    file = FOLDER["raw"] + loc + ".csv"

    df = pd.read_csv(
        file,
        sep=",",
    )
    df["time"] = df["Date"] + " " + df["Hour"]
    df = df[df.columns.drop(['Date', 'Hour', 'Wind_direction'])]
# converting the string to datetime format
    df['time'] = pd.to_datetime(df['time'], format='%Y/%m/%d %H')
    df = df.replace(['S/D', np.nan]).fillna(method='ffill')

    df.rename(
        columns={
            "Wind_speed": "wind",
            "Temperature": "temp",
            "Precipitation": "ppt",
        },
        inplace=True,
    )
    types_dict = {
        "wind": float,
        "temp": float,
        "RH": float,
    }
    for col, col_type in types_dict.items():
        df[col] = df[col].astype(col_type)
    # Derived
    df["SW_global"] = 0
    df["Discharge"] = 30
    df["press"] = atmosphere.alt2pres(SITE["alt"]) / 100
    logger.info(df.head())

    if SITE["name"] in ["sibinacocha21", "sibinacocha22"]:
        cols = [
            "time",
            "Discharge",
            "temp",
            "RH",
            "wind",
            "SW_global",
            "ppt",
            # "vp_a",
            "press",
        ]
    df_out = df[cols]

    if df_out.isna().values.any():
        logger.warning(df_out[cols].isna().sum())
        df_out = df_out.interpolate(method='ffill', axis=0)
        df_out.loc[df_out.cld.isna(), "cld"] = 0

    df_out = df_out.round(3)
    if len(df_out[df_out.index.duplicated()]):
        logger.error("Duplicate indexes")

    plot_input(df_out, FOLDER['fig'], SITE["name"])
    logger.info(df_out.tail())
    df_out.to_csv(FOLDER["input"]  + "aws.csv", index=False)


    # with open("data/common/constants.json") as f:
    #     CONSTANTS = json.load(f)

    # for loc in locations:
    #     SITE, FOLDER = config(loc)

    # logger.warning("Temperature NaN percent: %0.2f" %(df["T_a"].isna().sum()/df.shape[0]*100))
    # logger.warning("wind NaN percent: %0.2f" %(df["v_a"].isna().sum()/df.shape[0]*100))

    # SITE, FOLDER, *args = config("ravat20")
    # df_ERA5, df_in3 = era5(SITE["name"])
    # print(df_ERA5.describe())
    #
    # df_ERA5 = df_ERA5.set_index("When")
    # df = df_ERA5
    # df['missing_type'] = 'NA'
    #
    # # Fill from ERA5
    # logger.warning("Temperature NaN percent: %0.2f" %(df["T_a"].isna().sum()/df.shape[0]*100))
    # logger.warning("wind NaN percent: %0.2f" %(df["v_a"].isna().sum()/df.shape[0]*100))
    #
    # df = df.reset_index()
    #
    # if SITE["name"] in ["diavolezza21"]:
    #     cols = [
    #         "When",
    #         "Discharge",
    #         "T_a",
    #         "RH",
    #         "v_a",
    #         "SW_direct",
    #         "SW_diffuse",
    #         "Prec",
    #         "p_a",
    #         "missing_type",
    #         "LW_in",
    #         # "a",
    #     ]
    #
    # if SITE["name"] in ["schwarzsee19", "ravat20"]:
    #     cols = [
    #         "When",
    #         "T_a",
    #         "RH",
    #         "v_a",
    #         "SW_direct",
    #         "SW_diffuse",
    #         "Prec",
    #         # "vp_a",
    #         "p_a",
    #         "missing_type",
    #         "LW_in",
    #     ]
    # if SITE["name"] in ["guttannen20", "guttannen21"]:
    #     cols = [
    #         "When",
    #         "T_a",
    #         "RH",
    #         "v_a",
    #         "SW_direct",
    #         "SW_diffuse",
    #         "Prec",
    #         "vp_a",
    #         "p_a",
    #         "missing_type",
    #         "LW_in",
    #     ]
    #
    # df_out = df[cols]
    #
    # if df_out.isna().values.any():
    #     print(df_out[cols].isna().sum())
    #     for column in cols:
    #         if df_out[column].isna().sum() > 0 and column in ["a"]:
    #             albedo = df_out.a.replace(0, np.nan).mean()
    #             df_out.loc[df_out[column].isna(), column] = albedo
    #             logger.warning("Albedo Null values extrapolated in %s " %albedo)
    #         if df_out[column].isna().sum() > 0 and column in ["Discharge"]:
    #             discharge = df_out.Discharge.replace(0, np.nan).mean()
    #
    #             df_out.loc[df_out[column].isna(), column] = discharge
    #
    #             logger.warning(" Discharge Null values extrapolated in %s " %discharge)
    #
    #         if df_out[column].isna().sum() > 0 and column not in ["missing_type", "Discharge"]:
    #             logger.warning(" Null values interpolated in %s" %column)
    #             df_out.loc[:, column] = df_out[column].interpolate()
    #
    # df_out = df_out.round(3)
    # if len(df_out[df_out.index.duplicated()]):
    #     logger.error("Duplicate indexes")
    #
    # logger.info(df_out.tail())
    # df_out.to_csv(FOLDER["input"]+ SITE["name"] + "_input_model.csv", index=False)
    #
    # fig, ax1 = plt.subplots()
    # skyblue = "#9bc4f0"
    # blue = "#0a4a97"
    # x = df_out.When
    # y = df_out.T_a
    # ax1.plot(
    #     x,
    #     y,
    #     linestyle="-",
    #     color=blue,
    # )
    # ax1.set_ylabel("Discharge [$l\\, min^{-1}$]")
    # ax1.xaxis.set_major_locator(mdates.WeekdayLocator())
    # ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    # ax1.xaxis.set_minor_locator(mdates.DayLocator())
    # fig.autofmt_xdate()
    # plt.savefig(FOLDER["input"]+ SITE["name"] + "test.png")
    #

