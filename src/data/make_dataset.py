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
from src.data.settings import config

def field(location="schwarzsee"):
    if location == "guttannen20":
        df_in = pd.read_csv(
            raw_folder + SITE["name"] + ".txt",
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
        df_in["When"] = pd.to_datetime(df_in["Date"] + " " + df_in["Time"])
        df_in["When"] = pd.to_datetime(df_in["When"], format="%Y.%m.%d %H:%M:%S")
        df_in = df_in.drop(["Pluviometer", "Date", "Time"], axis=1)
        df_in = df_in.set_index("When").resample("15T").mean().reset_index()

        mask = (df_in["When"] >= SITE["start_date"]) & (
            df_in["When"] <= SITE["end_date"]
        )
        df_in = df_in.loc[mask]
        df_in = df_in.reset_index()
        days = pd.date_range(start=SITE["start_date"], end=SITE["end_date"], freq="15T")
        days = pd.DataFrame({"When": days})

        df = pd.merge(
            df_in[
                [
                    "When",
                    "Discharge",
                    "Wind Speed",
                    "Temperature",
                    "Humidity",
                    "Pressure",
                ]
            ],
            days,
            on="When",
        )

        df = df.round(3)
        # CSV output
        df.rename(
            columns={
                "Wind Speed": "v_a",
                "Temperature": "T_a",
                "Humidity": "RH",
                "Pressure": "p_a",
            },
            inplace=True,
        )
        logger.info(df_in.head())
        logger.info(df_in.tail())
        df.to_csv(input_folder + SITE["name"] + "_input_field.csv")


    if location == "guttannen21":
        df_in = pd.read_csv(
            raw_folder + SITE["name"] + "_11Feb20.txt",
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
        df_in["When"] = pd.to_datetime(df_in["Date"] + " " + df_in["Time"])
        df_in["When"] = pd.to_datetime(df_in["When"], format="%Y.%m.%d %H:%M:%S")
        df_in = df_in.drop(["Pluviometer", "Date", "Time"], axis=1)
        logger.debug(df_in.head())
        logger.debug(df_in.tail())
        df_in = df_in.set_index("When").resample("15T").mean().reset_index()

        mask = (df_in["When"] >= SITE["start_date"]) & (
            df_in["When"] <= SITE["end_date"]
        )
        df_in = df_in.loc[mask]
        df_in = df_in.reset_index()
        days = pd.date_range(start=SITE["start_date"], end=SITE["end_date"], freq="15T")
        days = pd.DataFrame({"When": days})

        df = pd.merge(
            df_in[
                [
                    "When",
                    "Wind Speed",
                    "Temperature",
                    "Humidity",
                    "Pressure",
                ]
            ],
            days,
            on="When",
        )

        df = df.round(3)
        # CSV output
        df.rename(
            columns={
                "Wind Speed": "v_a",
                "Temperature": "T_a",
                "Humidity": "RH",
                "Pressure": "p_a",
            },
            inplace=True,
        )
        df.to_csv(input_folder + SITE["name"] + "_input_field.csv")

    if location == "schwarzsee19":
        df_in = pd.read_csv(
            raw_folder + SITE["name"][:-2] + "_aws.txt",
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

        df_in["When"] = pd.to_datetime(df_in["Date"] + " " + df_in["Time"])
        df_in["When"] = pd.to_datetime(df_in["When"], format="%Y.%m.%d %H:%M:%S")

        # Correct datetime errors
        for i in tqdm(range(1, df_in.shape[0])):
            if str(df_in.loc[i, "When"].year) != "2019":
                df_in.loc[i, "When"] = df_in.loc[i - 1, "When"] + pd.Timedelta(
                    minutes=5
                )

        df_in = df_in.set_index("When").resample("15T").last().reset_index()

        mask = (df_in["When"] >= SITE["start_date"]) & (
            df_in["When"] <= SITE["end_date"]
        )
        df_in = df_in.loc[mask]
        df_in = df_in.reset_index()

        days = pd.date_range(start=SITE["start_date"], end=SITE["end_date"], freq="15T")
        days = pd.DataFrame({"When": days})

        df = pd.merge(
            days,
            df_in[
                [
                    "When",
                    "Discharge",
                    "Wind Speed",
                    "Maximum Wind Speed",
                    "Wind Direction",
                    "Temperature",
                    "Humidity",
                    "Pressure",
                ]
            ],
            on="When",
        )

        # Include Spray time
        df_nights = pd.read_csv(
            raw_folder + "schwarzsee_fountain_time.txt",
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
                (df["When"] >= df_nights.loc[i, "Start"])
                & (df["When"] <= df_nights.loc[i, "End"]),
                "Fountain",
            ] = 1

        # CSV output
        df.rename(
            columns={
                "Wind Speed": "v_a",
                "Temperature": "T_a",
                "Humidity": "RH",
                "Pressure": "p_a",
            },
            inplace=True,
        )

        df.Discharge = df.Fountain * df.Discharge
        df.to_csv(input_folder + SITE["name"] + "_input_field.csv")
    df = df.set_index("When").resample("15T").mean().reset_index()
    # else:
    #     input_file = input_folder + SITE["name"] + "_input_field.csv"
    #     df = pd.read_csv(input_file, sep=",", header=0, parse_dates=['When'])
    return df


def era5(df, location="schwarzsee19"):

    if location in ["schwarzsee19"]:
        df_in3 = pd.read_csv(
            "/home/suryab/work/ERA5/outputs/" + location[:-2] + "_2019.csv",
            sep=",",
            header=0,
            parse_dates=["When"],
        )

    if location in ["guttannen20"]:
        df_in3 = pd.read_csv(
            "/home/suryab/work/ERA5/outputs/" + location[:-2] + "_2019.csv",
            sep=",",
            header=0,
            parse_dates=["When"],
        )
        df_in2 = pd.read_csv(
            "/home/suryab/work/ERA5/outputs/" + location[:-2] + "_2020.csv",
            sep=",",
            header=0,
            parse_dates=["When"],
        )
        df_in3 = df_in3.set_index("When")
        df_in2 = df_in2.set_index("When")
        df_in3 = pd.concat([df_in2, df_in3])
        df_in3 = df_in3.reset_index()
    if location in ["guttannen21"]:
        df_in3 = pd.read_csv(
            "/home/suryab/work/ERA5/outputs/" + location[:-2] + "_2021.csv",
            sep=",",
            header=0,
            parse_dates=["When"],
        )
        df_in2 = pd.read_csv(
            "/home/suryab/work/ERA5/outputs/" + location[:-2] + "_2020.csv",
            sep=",",
            header=0,
            parse_dates=["When"],
        )
        df_in3 = df_in3.set_index("When")
        df_in2 = df_in2.set_index("When")
        df_in3 = pd.concat([df_in2, df_in3])
        df_in3 = df_in3.reset_index()

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

    df_in3.to_csv(input_folder + SITE["name"] + "_input_ERA5.csv")

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

    logger.info(df_ERA5.head())
    logger.info(df_ERA5.tail())
    return df_ERA5, df_in3


def e_sat(T, surface="water", a1=611.21, a3=17.502, a4=32.19):
    T += 273.16
    if surface == "ice":
        a1 = 611.21  # Pa
        a3 = 22.587  # NA
        a4 = -0.7  # K
    return a1 * np.exp(a3 * (T - 273.16) / (T - a4))


def linreg(X, Y):
    mask = ~np.isnan(X) & ~np.isnan(Y)
    slope, intercept, r_value, p_value, std_err = stats.linregress(X[mask], Y[mask])
    return slope, intercept


def meteoswiss_parameter(parameter):
    # d = {"time":{"name":"When", "units":"()"}}
    d = {
        "time": {
            "name": "When",
            "units": "(  )",
        },
        "rre150z0": {
            "name": "Prec",
            "units": "($mm$)",
        },
        "dkl010z0": {
            "name": "Wind direction",
            "units": "($\\degree$)",
        },
        "fkl010z0": {
            "name": "v_a",
            "units": "($ms^{-1}$)",
        },
        "ure200s0": {
            "name": "RH",
            "units": "($%$)",
        },
        "prestas0": {
            "name": "p_a",
            "units": "($hPa$)",
        },
        "pva200s0": {
            "name": "vp_a",
            "units": "($hPa$)",
        },
        "tde200s0": {
            "name": "T_ad",
            "units": "($\\degree C$)",
        },
        "tre200s0": {
            "name": "T_a",
            "units": "($\\degree C$)",
        },
        "gre000z0": {
            "name": "SW_global",
            "units": "($W\\,m^{-2}$)",
        },
        "oli000z0": {
            "name": "LW_in",
            "units": "($W\\,m^{-2}$)",
        },
    }

    value = d.get(parameter)

    return value


def meteoswiss(location="schwarzsee19"):

    if location == "schwarzsee19":
        location = "plaffeien19"
    location = location[:-2]

    df = pd.read_csv(
        os.path.join(raw_folder, location + "_meteoswiss.txt"),
        # sep="\s+",
        sep=";",
        skiprows=2,
    )
    for col in df.columns:
        if meteoswiss_parameter(col):
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df = df.rename(columns={col: meteoswiss_parameter(col)["name"]})
            logger.info("%s from meteoswiss" % meteoswiss_parameter(col)["name"])
        else:
            df = df.drop(columns=col)
    df["When"] = pd.to_datetime(df["When"], format="%Y%m%d%H%M")

    df["Prec"] = df["Prec"] / (10 * 60)  # ppt rate mm/s
    df = (
        df.set_index("When")
        .resample("15T")
        # .interpolate(method="linear")
        .mean()
        .reset_index()
    )
    mask = (df["When"] >= SITE["start_date"]) & (df["When"] <= SITE["end_date"])
    df = df.loc[mask]
    logger.warning(df.columns)
    return df


if __name__ == "__main__":

    logger = logging.getLogger(__name__)
    coloredlogs.install(
        fmt="%(name)s %(levelname)s %(message)s",
        level=logging.INFO,
        logger=logger,
    )

    SITE, FOUNTAIN, FOLDER = config("Guttannen 2021")

    raw_folder = os.path.join(dirname, "data/" + SITE["name"] + "/raw/")
    input_folder = os.path.join(dirname, "data/" + SITE["name"] + "/interim/")

    if SITE["name"] in ["schwarzsee19", "guttannen20"]:
        df = field(location=SITE["name"])
    if SITE["name"] in ["guttannen21"]:
        df = meteoswiss(SITE["name"])

    df_ERA5, df_in3 = era5(df, SITE["name"])

    df_ERA5 = df_ERA5.set_index("When")
    df = df.set_index("When")
    df["missing"] = 0

    df_ERA5 = df_ERA5.reset_index()
    mask = (df_ERA5["When"] >= SITE["start_date"]) & (
        df_ERA5["When"] <= SITE["end_date"]
    )

    # Fit ERA5 to field data

    if SITE["name"] in ["guttannen21"]:
        fit_list = ["T_a", "RH", "v_a"]
    else:
        fit_list = ["T_a", "RH", "v_a", "p_a"]

    for column in fit_list:
        Y = df[column].values.reshape(-1, 1)
        X = df_ERA5[mask][column].values.reshape(-1, 1)
        slope, intercept = linreg(X, Y)
        df_ERA5[column] = slope * df_ERA5[column] + intercept

    df_ERA5 = df_ERA5.set_index("When")

    # Fill from ERA5
    logger.warning("Temperature NaN rows: %s" %df["T_a"].isna().sum())

    if SITE["name"] in ["guttannen20", "guttannen21"]:
        for col in ["T_a", "RH", "v_a"]:
            df.loc[df[col].isna(), "missing"] = 1
            df.loc[df[col].isna(), "missing_type"] = col
            df.loc[df[col].isna(), col] = df_ERA5[col]

        # for col in ["p_a", "SW_direct", "SW_diffuse", "LW_in"]:
        for col in ["p_a", "SW_direct", "SW_diffuse", "LW_in", "Prec"]:
            logger.info("%s from ERA5" % col)
            df[col] = df_ERA5[col]

    # if SITE["name"] in ["guttannen20"]:
    #     for col in ["Prec", "vp_a"]:
    #         logger.info("%s from meteoswiss" % col)
    #         df[col] = df_swiss[col]

    if SITE["name"] in ["schwarzsee19"]:
        df["v_a"] = df["v_a"].replace(0, np.NaN)
        for col in ["T_a", "RH", "v_a", "p_a"]:
            df.loc[df[col].isna(), "missing"] = 1
            df.loc[df[col].isna(), "missing_type"] = col
            df.loc[df[col].isna(), col] = df_ERA5[col]

        for col in ["SW_direct", "SW_diffuse", "LW_in"]:
            logger.info("%s from ERA5" % col)
            df[col] = df_ERA5[col]

    if SITE["name"] in ["schwarzsee19"]:
        df_swiss = meteoswiss(SITE["name"])

        df_swiss = df_swiss.set_index("When")

        for col in ["Prec"]:
            logger.info("%s from meteoswiss" % col)
            df[col] = df_swiss[col]
        df_swiss = df_swiss.reset_index()

    df = df.reset_index()

    # Compare ERA5 and field data
    # for column in ["T_a", "RH", "v_a", "p_a"]:

    #     slope, intercept, r_value, p_value, std_err = stats.linregress(
    #         df[column].values, df_in3[column].values
    #     )
    #     print("ERA5", column, r_value)
    # slope, intercept, r_value, p_value, std_err = stats.linregress(
    #     df[column].values, df_swiss[column].values
    # )
    # print("Plf", column, r_value)

    if SITE["name"] in ["schwarzsee19", "guttannen20"]:
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
            "missing",
            "LW_in",
        ]
    else:
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
            "missing",
            "LW_in",
        ]

    df_out = df[cols]

    if df_out.isna().values.any():
        print(df_out[cols].isna().sum())
        for column in cols:
            if df_out[column].isna().sum() > 0:
                logger.warning("Warning: Null values filled in %s" %column)
                df_out.loc[:, column] = df_out[column].interpolate()

    df_out = df_out.round(3)
    logger.warning(df_out[df_out.index.duplicated()])
    logger.info(df_out.tail())
    df_out.to_csv(input_folder + SITE["name"] + "_input_model.csv")

    if SITE["name"] in ['schwarzsee19']:
        df_ERA5["Prec"] = 0
        df_ERA5["missing"] = 1
        df_ERA5 = df_ERA5.reset_index()
        mask = (df_ERA5["When"] > df_out["When"].iloc[-1]) & (
            df_ERA5["When"] <= datetime(2019, 5, 30)
        )
        df_ERA5 = df_ERA5.loc[mask]
        mask = (df_swiss["When"] >df_out["When"].iloc[-1] ) & ( df_swiss["When"] <= SITE["end_date"]
                )
        df_swiss = df_swiss.loc[mask]
        df_swiss = df_swiss.set_index("When")

        df_out = df_out.set_index("When")
        df_ERA5 = df_ERA5.set_index("When")

        df_ERA5["Prec"] = df_swiss["Prec"]
        concat = pd.concat([df_out, df_ERA5])
        concat.loc[concat["Prec"].isnull(), "Prec"] = 0
        concat.loc[concat["v_a"] < 0, "v_a"] = 0
        logger.warning(concat[concat.index.duplicated()])
        logger.info(concat.tail())

        concat = concat.reset_index()

        if concat.isnull().values.any():
            print("Warning: Null values present")
            print(
                concat[
                    [
                        "When",
                        "T_a",
                        "RH",
                        "v_a",
                        # "Discharge",
                        "SW_direct",
                        "SW_diffuse",
                        "Prec",
                        "p_a",
                        "missing",
                    ]
                ]
                .isnull()
                .sum()
            )

        concat.to_csv(input_folder + SITE["name"] + "_input_model.csv")
        concat.to_hdf(
            input_folder + SITE["name"] + "_input_model.h5",
            key="df",
            mode="w",
        )
        fig = plt.figure()
        plt.plot(concat.v_a)
        plt.ylabel('some numbers')
        plt.savefig(input_folder + SITE["name"] + "test.png")

