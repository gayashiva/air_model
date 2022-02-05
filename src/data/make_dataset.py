"""Compile raw data from the location, meteoswiss or ERA5
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
from pathlib import Path
from tqdm import tqdm
import logging
import coloredlogs
from scipy import stats
from sklearn.linear_model import LinearRegression

# Locals
dirname = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.append(dirname)
from src.utils.settings import config
from src.data.field import get_field
from src.data.era5 import get_era5
from src.data.meteoswiss import get_meteoswiss
from src.plots.data import plot_input
from src.utils import setup_logger

def linreg(X, Y):
    mask = ~np.isnan(X) & ~np.isnan(Y)
    slope, intercept, r_value, p_value, std_err = stats.linregress(X[mask], Y[mask])
    return slope, intercept, r_value


if __name__ == "__main__":
    # Main logger
    logger = logging.getLogger(__name__)
    logger.setLevel("INFO")

    # locations = ["gangles21", "guttannen20", "guttannen21"]
    locations = ["guttannen21"]

    for location in locations:
        CONSTANTS, SITE, FOLDER = config(location)

        if location in ["gangles21"]:
            df = get_field(location)
            df = df.set_index("time")
            df = df[SITE["start_date"] : SITE["expiry_date"]]
            df = df.reset_index()

            logger.info(df.missing_type.describe())
            logger.info(df.missing_type.unique())
        else:

            if location in ["schwarzsee19", "guttannen22"]:
                df = get_field(location)

            if location in ["guttannen21", "guttannen20"]:
                df = get_meteoswiss(location)

            df = df.set_index("time")
            df = df[SITE["start_date"] : SITE["expiry_date"]]
            df = df.reset_index()

            # Replace Wind zero values for 3 hours
            mask = df.wind.shift().eq(df.wind)
            for i in range(1, 3 * 4):
                mask &= df.wind.shift(-1 * i).eq(df.wind)
            mask &= df.wind == 0
            df.wind = df.wind.mask(mask)

            if location in ["guttannen22"]:
                df_swiss = get_meteoswiss(location)
                df_swiss = df_swiss.set_index("time")
                df_swiss = df_swiss[SITE["start_date"] : SITE["expiry_date"]]
                df_swiss = df_swiss.reset_index()

                df_swiss = df_swiss.set_index("time")
                df = df.set_index("time")

                for col in ["vp_a"]:
                    logger.info("%s from meteoswiss" % col)
                    df[col] = df_swiss[col]
                df_swiss = df_swiss.reset_index()
                df = df.reset_index()

            df_ERA5_full = get_era5(SITE["name"])

            df = df.set_index("time")

            df_ERA5_full = df_ERA5_full.set_index("time")
            df_ERA5 = df_ERA5_full[SITE["start_date"] : SITE["expiry_date"]]
            df_ERA5 = df_ERA5.reset_index()
            df_ERA5_full = df_ERA5_full.reset_index()

            # Fit ERA5 to field data
            if SITE["name"] in ["guttannen21", "guttannen20", "guttannen22"]:
                fit_list = ["temp", "RH", "wind"]

            if SITE["name"] in ["schwarzsee19"]:
                fit_list = ["temp", "RH", "wind", "press"]

            for column in fit_list:
                Y = df[column].values.reshape(-1, 1)
                X = df_ERA5[column].values.reshape(-1, 1)
                slope, intercept, r_value = linreg(X, Y)
                logger.info(f"Correlation of {column} in ERA5 is {r_value} at {location}")
                df_ERA5[column] = slope * df_ERA5[column] + intercept
                df_ERA5_full[column] = slope * df_ERA5_full[column] + intercept
                if column in ["wind"]:
                    # Correct negative wind
                    df_ERA5.loc[df_ERA5.wind < 0, 'wind'] = 0
                    df_ERA5_full.loc[df_ERA5_full.wind < 0, 'wind'] = 0

            df_ERA5 = df_ERA5.set_index("time")

            # Fill from ERA5
            df["missing_type"] = ""
            for col in [
                "temp",
                "RH",
                "wind",
                "ppt",
                "press",
                "SW_direct",
                "SW_diffuse",
                "LW_in",
            ]:
                try:
                    mask = df[col].isna()
                    percent_nan = df[col].isna().sum() / df.shape[0] * 100
                    logger.info(" %s has %s percent NaN values" % (col, percent_nan))
                    if percent_nan > 1:
                        logger.warning(" Null values filled with ERA5 in %s" % col)
                        df.loc[df[col].isna(), "missing_type"] = (
                            df.loc[df[col].isna(), "missing_type"] + col
                        )
                        df.loc[df[col].isna(), col] = df_ERA5[col]
                    else:
                        logger.warning(" Null values interpolated in %s" % col)
                        df.loc[:, col] = df[col].interpolate()
                except KeyError:
                    logger.warning("%s from ERA5" % col)
                    df[col] = df_ERA5[col]
                    df["missing_type"] = df["missing_type"] + col
            logger.info(df.missing_type.describe())
            logger.info(df.missing_type.unique())

            df = df.reset_index()

        if SITE["name"] in ["gangles21"]:
            cols = [
                "time",
                # "Discharge",
                "temp",
                "RH",
                "wind",
                # "SW_direct",
                # "SW_diffuse",
                "SW_global",
                "ppt",
                # "vp_a",
                "press",
                "missing_type",
                # "LW_in",
            ]

        if SITE["name"] in ["guttannen20", "guttannen21"]:
            cols = [
                "time",
                "temp",
                "RH",
                "wind",
                "SW_direct",
                "SW_diffuse",
                "ppt",
                "vp_a",
                "press",
                "missing_type",
                "LW_in",
            ]

        if SITE["name"] in ["guttannen22"]:
            cols = [
                "time",
                "temp",
                "RH",
                "wind",
                "SW_direct",
                "SW_diffuse",
                # "alb",
                "ppt",
                "vp_a",
                "press",
                "LW_in",
                "snow_h",
                "missing_type",
            ]


        df_out = df[cols]

            
        if df_out.isna().values.any():
            logger.warning(df_out[cols].isna().sum())
            df = df.interpolate(method='linear', limit_direction='forward', axis=0)
            df_out.loc[df_out.wind.isna(), "wind"] = 0
            df_out.loc[df_out.ppt.isna(), "ppt"] = 0
            if df_out.isna().values.any():
                logger.error(df_out[cols].isna().sum())
            # for column in cols:
            #     if df_out[column].isna().sum() > 0 and column not in ["missing_type"]:
            #         logger.warning(" Null values interpolated in %s" % column)
            #         df_out.loc[:,column] = df_out[column].interpolate()
                    # print(df_out.loc[df_out[column]==np.NaN, column])
                    # df_out.loc[df_out[column]==np.NaN, column] = 0

        df_out = df_out.round(3)
        if len(df_out[df_out.index.duplicated()]):
            logger.error("Duplicate indexes")

        if "SW_global" not in df_out.columns:
            logger.warning("SW global added")
            df_out["SW_global"] = df_out["SW_direct"] + df_out["SW_diffuse"]

        if "SW_direct" not in df_out.columns:
            logger.warning("SW direct added from global")
            df_out["SW_direct"] = df_out["SW_global"]
            df_out["SW_diffuse"] = 0

        logger.info(df_out.tail())
        plot_input(df_out, FOLDER['fig'], SITE["name"])
        df_out = df_out.drop(columns=['missing_type'])

        df_out.to_csv(FOLDER["input"]  + "input.csv", index=False)
