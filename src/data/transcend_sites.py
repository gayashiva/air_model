"""Compile raw data from the location, meteoswiss or ERA5
"""

# External modules
import sys, os, json, glob
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
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

if __name__ == "__main__":
    # Main logger
    logger = logging.getLogger(__name__)
    logger.setLevel("INFO")

    with open("data/common/constants.json") as f:
        CONSTANTS = json.load(f)

    # loc = "chuapalca20"
    loc = "tarata20"
    SITE, FOLDER = config(loc)

    # Path to the folder containing CSV files
    folder_path = FOLDER["raw"]

    # List to store individual DataFrames
    dataframes = []

    # Iterate through all files in the folder
    for filename in os.listdir(folder_path):
        if filename.endswith('.csv'):
            file_path = os.path.join(folder_path, filename)
            df= pd.read_csv(
                file_path,
                sep=",",
                skiprows=11,
                header=None,
            )
            dataframes.append(df)

    # Concatenate all DataFrames into a single DataFrame
    df = pd.concat(dataframes, ignore_index=True)

    df.rename(columns = {0 : 'date', 1 : 'time', 2 : 'temp', 3:'ppt', 4:'RH', 6:'wind' }, inplace = True)
    df["Discharge"] = 1000000
    # Convert specific columns to desired data types
    columns_to_convert = ['temp', 'ppt', 'RH', 'wind']
    df[columns_to_convert] = df[columns_to_convert].apply(pd.to_numeric, errors='coerce')

    df['time'] = pd.to_datetime(df['date']+ " " + df['time'])
    df = df.drop(columns=['date', 5])
    df = df.set_index("time")
    df.sort_index(inplace=True)
    print(df.head())
    print(df.tail())

    # Resample to daily minimum temperature
    daily_min_temps = df['temp'].resample('D').min()

    # Find longest consecutive period
    longest_period = 0
    current_period = 0
    start_date = None
    crit_temp = 0

    for date, temp in daily_min_temps.iteritems():
        if temp < crit_temp:
            current_period += 1
            if current_period > longest_period:
                longest_period = current_period
                start_date = date - pd.DateOffset(days=current_period - 1)
        else:
            current_period = 0

    print("Start Date:", start_date)
    print("Longest Consecutive Days:", longest_period)
    expiry_date = start_date + timedelta(days=longest_period*2)
    # start_date=datetime(2022, 5, 1)
    # expiry_date = start_date + timedelta(days=365)

    df = df[start_date : expiry_date]
    df = df.reset_index()

    if df.isna().values.any():
        logger.warning(df.isna().sum())
        df = df.interpolate(method='ffill', axis=0)

    df = df.round(3)


    logger.info(df.head())
    logger.info(df.tail())


    # logger.warning("Creating folders")
    # os.mkdir(dirname + "/" + FOLDER["input"])
    # os.mkdir(dirname + "/" + FOLDER["output"])
    # os.mkdir(dirname + "/" + FOLDER["fig"])

    df.to_csv(FOLDER["input"]  + "aws.csv", index=False)

