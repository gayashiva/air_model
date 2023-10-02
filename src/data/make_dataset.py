"""Compile raw data from the location, meteoswiss or ERA5
"""

# External modules
import sys, os, json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from scipy import stats

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
    logger.setLevel("INFO")

    # locations = ["leh20", "south_america20", "north_america20", "europe20", "central_asia20"]
    locations = ["south_america20"]
    # locations = ["leh20"]
    spray="ERA5_"

    with open("constants.json") as f:
        CONSTANTS = json.load(f)

    if spray == "ERA5_":
        for loc in locations:
            SITE, FOLDER = config(loc, spray)

            df1= pd.read_csv(
                "data/era5/" + loc[:-2] + "_2019.csv",
                sep=",",
                header=0,
                parse_dates=["time"],
            )
            df2= pd.read_csv(
                "data/era5/" + loc[:-2] + "_2020.csv",
                sep=",",
                header=0,
                parse_dates=["time"],
            )
            # Combine DataFrames
            df = pd.concat([df1, df2])
            df['time'] = pd.to_datetime(df['time'])
            # df = df.set_index("time")

            # # Resample to daily minimum temperature
            # daily_min_temps = df['temp'].resample('D').min()

            # # Find longest consecutive period
            # minimum_period = 7
            # current_period = 0
            # start_date_list = []
            # crit_temp = 0

            # for date, temp in daily_min_temps.iteritems():
            #     if temp < crit_temp:
            #         current_period += 1
            #         if current_period == minimum_period:
            #             # longest_period = current_period
            #             start_date_list.append(date - pd.DateOffset(days=current_period - 1))
            #     else:
            #         current_period = 0

            # print("Start Dates:", start_date_list)

            # df = df[start_date_list[0] : ]
            # df = df.reset_index()

            df["Discharge"] = 1000000

            cols = [
                "time",
                "temp",
                "RH",
                "wind",
                "ppt",
                "Discharge",
            ]
            df_out = df[cols]

            if df_out.isna().values.any():
                logger.warning(df_out[cols].isna().sum())
                df_out = df_out.interpolate(method='ffill', axis=0)

            df_out = df_out.round(3)
            if len(df_out[df_out.index.duplicated()]):
                logger.error("Duplicate indexes")

            logger.info(df_out.head())
            logger.info(df_out.tail())
            # plot_input(df_out, FOLDER['fig'], SITE["name"])

            if not os.path.exists(dirname + "/data/" + loc):
                logger.warning("Creating folders")
                os.mkdir(dirname + "/data/" + loc)
                os.mkdir(dirname + "/" + FOLDER["input"])
                os.mkdir(dirname + "/" + FOLDER["output"])
                os.mkdir(dirname + "/" + FOLDER["fig"])

            df_out.to_csv(FOLDER["input"]  + "aws.csv", index=False)
