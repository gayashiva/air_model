"""Function that returns data from field AWS
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

# Locals
dirname = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.append(dirname)
from src.utils.settings import config


def get_field(location="schwarzsee19"):
    SITE, FOLDER = config(location)
    if location == "gangles21":
        col_list = [
            "TIMESTAMP",
            "AirTC_Avg",
            "RH",
            "WS",
        ]
        cols = ["temp", "RH", "wind"]

        df_in = pd.read_csv(
            FOLDER["raw"] + "/Gangles_Table15Min.dat",
            sep=",",
            skiprows=[0, 2, 3, 4],
            parse_dates=["TIMESTAMP"],
        )
        df_in = df_in[col_list]

        df_in.rename(
            columns={
                "TIMESTAMP": "time",
                "AirTC_Avg": "temp",
                "RH_probe_Avg": "RH",
                "WS": "wind",
            },
            inplace=True,
        )

        df_in1 = pd.read_csv(
            FOLDER["raw"] + "/Gangles_Table60Min.dat",
            sep=",",
            skiprows=[0, 2, 3],
            parse_dates=["TIMESTAMP"],
        )
        df_in1.rename(
            columns={
                "TIMESTAMP": "time",
                "BP_mbar": "press",  # mbar same as hPa
            },
            inplace=True,
        )

        for col in df_in1:
            if col != "time":
                df_in1[col] = pd.to_numeric(df_in1[col], errors="coerce")

        df_in = df_in.set_index("time")
        df_in1 = df_in1.set_index("time")

        df_in1 = df_in1.reindex(
            pd.date_range(df_in1.index[0], df_in1.index[-1], freq="15Min"),
            fill_value=np.NaN,
        )

        df_in = df_in.replace("NAN", np.NaN)
        df_in1 = df_in1.replace("NAN", np.NaN)
        df_in1 = df_in1.resample("15Min").interpolate("linear")
        df_in.loc[:, "press"] = df_in1["press"]

        df_in = df_in.replace("NAN", np.NaN)
        if df_in.isnull().values.any():
            print("Warning: Null values present")
            print(df_in[cols].isnull().sum())
        df_in = df_in.round(3)
        df_in = df_in.reset_index()
        df_in.rename(
            columns={
                "index": "time",
            },
            inplace=True,
        )

        start_date = datetime(2020, 12, 14)
        df_in = df_in.set_index("time")
        df_in = df_in[start_date:]

        df1 = pd.read_csv(
            FOLDER["raw"] + "/HIAL_input_field.csv",
            sep=",",
            parse_dates=["When"],
        )
        df1 = df1.rename(columns={"When": "time"})

        df = df_in
        df1 = df1.set_index("time")
        cols = ["SW_global"]
        for col in cols:
            df.loc[:, col] = df1[col]

        df = df.reset_index()
        df = df[df.columns.drop(list(df.filter(regex="Unnamed")))]
        df = df.dropna()
        # df.to_csv("outputs/" + location + "_input_field.csv")

        mask = df["SW_global"] < 0
        mask_index = df[mask].index
        df.loc[mask_index, "SW_global"] = 0
        # diffuse_fraction = 0
        # df["SW_diffuse"] = diffuse_fraction * df.SW_global
        # df["SW_direct"] = (1-diffuse_fraction)* df.SW_global
        df = df.set_index("time").resample("H").mean().reset_index()

        df["ppt"] = 0
        df["missing_type"] = "-"
        # df["cld"] = 0

        df.to_csv(FOLDER["input"] + location + "_input_model.csv")
        return df

    if location == "guttannen20":
        df_in = pd.read_csv(
            FOLDER["raw"] + SITE["name"] + "_field.txt",
            header=None,
            encoding="latin-1",
            skiprows=7,
            sep="\\s+",
            index_col=False,
            names=[
                "Date",
                "Time",
                "Discharge",
                "Wind Direction",
                "Wind Speed",
                "Maximum Wind Speed",
                "Temperature",
                "Humidity",
                "Pressure",
                "Pluviometer",
            ],
        )
        types_dict = {
            "Date": str,
            "Time": str,
            "Discharge": float,
            "Wind Direction": float,
            "Wind Speed": float,
            "Temperature": float,
            "Humidity": float,
            "Pressure": float,
            "Pluviometer": float,
        }
        for col, col_type in types_dict.items():
            df_in[col] = df_in[col].astype(col_type)
        df_in["time"] = pd.to_datetime(df_in["Date"] + " " + df_in["Time"])
        df_in["time"] = pd.to_datetime(df_in["time"], format="%Y.%m.%d %H:%M:%S")
        df_in = df_in.drop(["Pluviometer", "Date", "Time"], axis=1)
        df_in = df_in.set_index("time").resample("H").mean().reset_index()

        mask = (df_in["time"] >= SITE["start_date"]) & (
            df_in["time"] <= SITE["end_date"]
        )
        df_in = df_in.loc[mask]
        df_in = df_in.reset_index()
        days = pd.date_range(start=SITE["start_date"], end=SITE["end_date"], freq="H")
        days = pd.DataFrame({"time": days})

        df = pd.merge(
            df_in[
                [
                    "time",
                    "Discharge",
                    "Wind Speed",
                    "Temperature",
                    "Humidity",
                    "Pressure",
                ]
            ],
            days,
            on="time",
        )

        df = df.round(3)
        # CSV output
        df.rename(
            columns={
                "Wind Speed": "wind",
                "Temperature": "temp",
                "Humidity": "RH",
                "Pressure": "press",
            },
            inplace=True,
        )
        logger.info(df_in.head())
        logger.info(df_in.tail())
        df.to_csv(FOLDER["input"] + SITE["name"] + "_input_field.csv")

    if location == "guttannen21":
        df_in = pd.read_csv(
            FOLDER["raw"] + SITE["name"] + "_field.txt",
            header=None,
            encoding="latin-1",
            skiprows=7,
            sep="\\s+",
            names=[
                "Date",
                "Time",
                "Wind Direction",
                "Wind Speed",
                "Maximum Wind Speed",
                "Temperature",
                "Humidity",
                "Pressure",
                "Pluviometer",
            ],
        )
        types_dict = {
            "Date": str,
            "Time": str,
            "Wind Direction": float,
            "Wind Speed": float,
            "Temperature": float,
            "Humidity": float,
            "Pressure": float,
            "Pluviometer": float,
        }
        for col, col_type in types_dict.items():
            df_in[col] = df_in[col].astype(col_type)
        df_in["time"] = pd.to_datetime(df_in["Date"] + " " + df_in["Time"])
        df_in["time"] = pd.to_datetime(df_in["time"], format="%Y.%m.%d %H:%M:%S")
        df_in = df_in.drop(["Pluviometer", "Date", "Time"], axis=1)
        logger.debug(df_in.head())
        logger.debug(df_in.tail())
        df_in = df_in.set_index("time").resample("H").mean().reset_index()

        mask = (df_in["time"] >= SITE["start_date"]) & (
            df_in["time"] <= SITE["end_date"]
        )
        df_in = df_in.loc[mask]
        df_in = df_in.reset_index()
        days = pd.date_range(start=SITE["start_date"], end=SITE["end_date"], freq="H")
        days = pd.DataFrame({"time": days})

        df = pd.merge(
            df_in[
                [
                    "time",
                    "Wind Speed",
                    "Temperature",
                    "Humidity",
                    "Pressure",
                ]
            ],
            days,
            on="time",
        )

        df = df.round(3)
        # CSV output
        df.rename(
            columns={
                "Wind Speed": "wind",
                "Temperature": "temp",
                "Humidity": "RH",
                "Pressure": "press",
            },
            inplace=True,
        )
        df.to_csv(FOLDER["input"] + SITE["name"] + "_input_field.csv")

    if location == "schwarzsee19":
        df_in = pd.read_csv(
            FOLDER["raw"] + SITE["name"][:-2] + "_aws.txt",
            header=None,
            encoding="latin-1",
            skiprows=7,
            sep="\\s+",
            names=[
                "Date",
                "Time",
                "Discharge",
                "Wind Direction",
                "Wind Speed",
                "Maximum Wind Speed",
                "Temperature",
                "Humidity",
                "Pressure",
                "Pluviometer",
            ],
        )

        df_in = df_in.drop(["Pluviometer"], axis=1)

        df_in["time"] = pd.to_datetime(df_in["Date"] + " " + df_in["Time"])
        df_in["time"] = pd.to_datetime(df_in["time"], format="%Y.%m.%d %H:%M:%S")

        # Correct datetime errors
        for i in tqdm(range(1, df_in.shape[0])):
            if str(df_in.loc[i, "time"].year) != "2019":
                df_in.loc[i, "time"] = df_in.loc[i - 1, "time"] + pd.Timedelta(
                    minutes=5
                )

        df_in = df_in.set_index("time").resample("H").last().reset_index()

        mask = (df_in["time"] >= SITE["start_date"]) & (
            df_in["time"] <= SITE["end_date"]
        )
        df_in = df_in.loc[mask]
        df_in = df_in.reset_index()

        days = pd.date_range(start=SITE["start_date"], end=SITE["end_date"], freq="H")
        days = pd.DataFrame({"time": days})

        df = pd.merge(
            days,
            df_in[
                [
                    "time",
                    "Discharge",
                    "Wind Speed",
                    "Maximum Wind Speed",
                    "Wind Direction",
                    "Temperature",
                    "Humidity",
                    "Pressure",
                ]
            ],
            on="time",
        )

        # Include Spray time
        df_nights = pd.read_csv(
            FOLDER["raw"] + "schwarzsee_fountain_time.txt",
            sep="\\s+",
        )

        df_nights["Start"] = pd.to_datetime(
            df_nights["Date"] + " " + df_nights["start"]
        )
        df_nights["End"] = pd.to_datetime(df_nights["Date"] + " " + df_nights["end"])
        df_nights["Start"] = pd.to_datetime(
            df_nights["Start"], format="%Y-%m-%d %H:%M:%S"
        )
        df_nights["End"] = pd.to_datetime(df_nights["End"], format="%Y-%m-%d %H:%M:%S")

        df_nights["Date"] = pd.to_datetime(df_nights["Date"], format="%Y-%m-%d")

        df["Fountain"] = 0

        for i in range(0, df_nights.shape[0]):
            df_nights.loc[i, "Start"] = df_nights.loc[i, "Start"] - pd.Timedelta(days=1)
            df.loc[
                (df["time"] >= df_nights.loc[i, "Start"])
                & (df["time"] <= df_nights.loc[i, "End"]),
                "Fountain",
            ] = 1

        # CSV output
        df.rename(
            columns={
                "Wind Speed": "wind",
                "Temperature": "temp",
                "Humidity": "RH",
                "Pressure": "press",
            },
            inplace=True,
        )

        df.Discharge = df.Fountain * df.Discharge
        df.to_csv(FOLDER["input"] + SITE["name"] + "_input_field.csv")
    df = df.set_index("time").resample("H").mean().reset_index()
    return df
