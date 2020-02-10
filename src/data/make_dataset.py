import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.backends.backend_pdf import PdfPages
import math
from pathlib import Path
import os
import logging
from src.data.config import site, dates, option, folders, fountain, surface
from src.models.air_forecast import projectile_xy, albedo
from pandas.plotting import register_matplotlib_converters

register_matplotlib_converters()

# python -m src.data.make_dataset


if site == "schwarzsee":

    # read files
    df_in = pd.read_csv(
        folders["data_file"],
        header=None,
        encoding="latin-1",
        skiprows=7,
        sep="\s+",
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

    # Drop
    df_in = df_in.drop(["Pluviometer"], axis=1)

    # Add Radiation data
    df_in2 = pd.read_csv(
        os.path.join(folders["dirname"], "data/raw/plaffeien_rad.txt"),
        sep="\s+",
        skiprows=2,
    )
    df_in2["When"] = pd.to_datetime(df_in2["time"], format="%Y%m%d%H%M")
    df_in2["ods000z0"] = pd.to_numeric(df_in2["ods000z0"], errors="coerce")
    df_in2["gre000z0"] = pd.to_numeric(df_in2["gre000z0"], errors="coerce")
    df_in2["Rad"] = df_in2["gre000z0"] - df_in2["ods000z0"]
    df_in2["DRad"] = df_in2["ods000z0"]

    # Add Precipitation data
    df_in2["Prec"] = pd.to_numeric(df_in2["rre150z0"], errors="coerce")
    df_in2["Prec"] = df_in2["Prec"] / 2  # 5 minute sum
    df_in2 = df_in2.set_index("When").resample("5T").ffill().reset_index()

    # Datetime
    df_in["When"] = pd.to_datetime(df_in["Date"] + " " + df_in["Time"])
    df_in["When"] = pd.to_datetime(df_in["When"], format="%Y.%m.%d %H:%M:%S")

    # Correct data errors
    i = 1
    while df_in.loc[i, "When"] != datetime(2019, 2, 6, 16, 15):
        if str(df_in.loc[i, "When"].year) != "2019":
            df_in.loc[i, "When"] = df_in.loc[i - 1, "When"] + pd.Timedelta(minutes=5)
        i = i + 1

    while df_in.loc[i, "When"] != datetime(2019, 3, 2, 15):
        if str(df_in.loc[i, "When"].year) != "2019":
            df_in.loc[i, "When"] = df_in.loc[i - 1, "When"] + pd.Timedelta(minutes=5)
        i = i + 1

    while df_in.loc[i, "When"] != datetime(2019, 3, 6, 16, 25):
        if str(df_in.loc[i, "When"].year) != "2019":
            df_in.loc[i, "When"] = df_in.loc[i - 1, "When"] + pd.Timedelta(minutes=5)
        i = i + 1

    df_in = df_in.resample("5Min", on="When").first().drop("When", 1).reset_index()

    # Fill missing data
    for i in range(1, df_in.shape[0]):
        if np.isnan(df_in.loc[i, "Temperature"]):
            df_in.loc[i, "Temperature"] = df_in.loc[i - 288, "Temperature"]
            df_in.loc[i, "Humidity"] = df_in.loc[i - 288, "Humidity"]
            df_in.loc[i, "Wind Speed"] = df_in.loc[i - 288, "Wind Speed"]
            df_in.loc[i, "Maximum Wind Speed"] = df_in.loc[
                i - 288, "Maximum Wind Speed"
            ]
            df_in.loc[i, "Wind Direction"] = df_in.loc[i - 288, "Wind Direction"]
            df_in.loc[i, "Pressure"] = df_in.loc[i - 288, "Pressure"]
            df_in.loc[i, "Discharge"] = df_in.loc[i - 288, "Discharge"]

    mask = (df_in["When"] >= dates["start_date"]) & (df_in["When"] <= dates["end_date"])
    df_in = df_in.loc[mask]
    df_in = df_in.reset_index()

    mask = (df_in2["When"] >= dates["start_date"]) & (
        df_in2["When"] <= dates["end_date"]
    )
    df_in2 = df_in2.loc[mask]
    df_in2 = df_in2.reset_index()

    days = pd.date_range(start=dates["start_date"], end=dates["end_date"], freq="5T")
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

    # Add Radiation DataFrame
    df["Rad"] = df_in2["Rad"]
    df["DRad"] = df_in2["DRad"]
    df["Prec"] = df_in2["Prec"] / 1000

    mask = (df["When"] >= dates["start_date"]) & (df["When"] <= dates["end_date"])
    df = df.loc[mask]
    df = df.reset_index()

    for i in range(1, df.shape[0]):
        if np.isnan(df.loc[i, "Rad"]):
            df.loc[i, "Rad"] = df.loc[i - 1, "Rad"]
        if np.isnan(df.loc[i, "DRad"]):
            df.loc[i, "DRad"] = df.loc[i - 1, "DRad"]
        if np.isnan(df.loc[i, "Prec"]):
            df.loc[i, "Prec"] = df.loc[i - 1, "Prec"]

    # v_a mean
    v_a = df["Wind Speed"].replace(0, np.NaN).mean()  # m/s Average Humidity
    df["Wind Speed"] = df["Wind Speed"].replace(0, v_a)

    # CSV output
    df.rename(
        columns={
            "Wind Speed": "v_a",
            "Temperature": "T_a",
            "Humidity": "RH",
            "Volume": "V",
            "Pressure": "p_a",
        },
        inplace=True,
    )

    df_out = df[
        ["When", "T_a", "RH", "v_a", "Discharge", "Rad", "DRad", "Prec", "p_a"]
    ]

    df_out = df_out.round(5)


if site == "plaffeien":

    """
    Parameter
    ---------
              Unit                                 Description
    pva200s0  hPa                                  Vapour pressure 2 m above ground; current value
    prestas0  hPa                                  Pressure at station level (QFE); current value
    ods000z0  W/m²                                 diffuse radiation, average 10 minutes
    gre000z0  W/m²                                 Global radiation; ten minutes mean
    tre200s0  °C                                   Air temperature 2 m above ground; current value
    rre150z0  mm                                   Precipitation; ten minutes total
    ure200s0  %                                    Relative air humidity 2 m above ground; current value
    fkl010z0  m/s                                  Wind speed scalar; ten minutes mean
    """

    # read files
    df_in = pd.read_csv(folders["data_file"], encoding="latin-1", skiprows=2, sep=";")
    df_in["When"] = pd.to_datetime(df_in["time"], format="%Y%m%d%H%M")  # Datetime

    time_steps = 5 * 60  # s # Model time steps
    mask = (df_in["When"] >= dates["start_date"]) & (df_in["When"] <= dates["end_date"])
    df_in = df_in.loc[mask]
    df_in = df_in.reset_index()

    # Add Radiation data
    df_in["ods000z0"] = pd.to_numeric(df_in["ods000z0"], errors="coerce")
    df_in["gre000z0"] = pd.to_numeric(df_in["gre000z0"], errors="coerce")
    df_in["Rad"] = df_in["gre000z0"] - df_in["ods000z0"]
    df_in["DRad"] = df_in["ods000z0"]
    df_in["T_a"] = pd.to_numeric(
        df_in["tre200s0"], errors="coerce"
    )  # Add Temperature data
    df_in["Prec"] = pd.to_numeric(
        df_in["rre150z0"], errors="coerce"
    )  # Add Precipitation data
    df_in["RH"] = pd.to_numeric(df_in["ure200s0"], errors="coerce")  # Add Humidity data
    df_in["v_a"] = pd.to_numeric(
        df_in["fkl010z0"], errors="coerce"
    )  # Add wind speed data
    df_in["p_a"] = pd.to_numeric(df_in["prestas0"], errors="coerce")  # Air pressure
    df_in["vp_a"] = pd.to_numeric(
        df_in["pva200s0"], errors="coerce"
    )  # Vapour pressure over air

    df_in["Prec"] = df_in["Prec"] / 1000

    # Fill nans
    df_in = df_in.fillna(method="ffill")

    df_out = df_in[["When", "T_a", "RH", "v_a", "Rad", "DRad", "Prec", "p_a", "vp_a",]]

    # 5 minute sum
    cols = ["T_a", "RH", "v_a", "Rad", "DRad", "Prec", "p_a", "vp_a"]
    df_out[cols] = df_out[cols] / 2
    df_out = df_out.set_index("When").resample("5T").ffill().reset_index()


    cols = [
        "When",
        "T_a",
        "RH",
        "v_a",
        "Rad",
        "DRad",
        "Prec",
        "p_a",
        "vp_a",
    ]
    df_out = df_out[cols]
    df_out = df_out.round(5)

if site == "guttannen":

    """
    Parameter
    ---------
              Unit                                 Description
    pva200s0  hPa                                  Vapour pressure 2 m above ground; current value
    prestas0  hPa                                  Pressure at station level (QFE); current value
    gre000z0  W/m²                                 Global radiation; ten minutes mean
    oli000z0  W/m²                                 Longwave incoming radiation; ten minute average
    tre200s0  °C                                   Air temperature 2 m above ground; current value
    rre150z0  mm                                   Precipitation; ten minutes total
    ure200s0  %                                    Relative air humidity 2 m above ground; current value
    fkl010z0  m/s                                  Wind speed scalar; ten minutes mean
    """

    # read files
    df_in = pd.read_csv(folders["data_file"], encoding="latin-1", skiprows=2, sep=";")
    df_in["When"] = pd.to_datetime(df_in["time"], format="%Y%m%d%H%M")  # Datetime

    mask = (df_in["When"] >= dates["start_date"]) & (df_in["When"] <= dates["end_date"])
    df_in = df_in.loc[mask]
    df_in = df_in.reset_index()

    # Convert to int
    df_in["oli000z0"] = pd.to_numeric(
        df_in["oli000z0"], errors="coerce"
    )  # Add Longwave Radiation data
    df_in["gre000z0"] = pd.to_numeric(
        df_in["gre000z0"], errors="coerce"
    )  # Add Radiation data
    df_in["T_a"] = pd.to_numeric(
        df_in["tre200s0"], errors="coerce"
    )  # Add Temperature data
    df_in["Prec"] = pd.to_numeric(
        df_in["rre150z0"], errors="coerce"
    )  # Add Precipitation data
    df_in["RH"] = pd.to_numeric(df_in["ure200s0"], errors="coerce")  # Add Humidity data
    df_in["v_a"] = pd.to_numeric(
        df_in["fkl010z0"], errors="coerce"
    )  # Add wind speed data
    df_in["p_a"] = pd.to_numeric(df_in["prestas0"], errors="coerce")  # Air pressure
    df_in["vp_a"] = pd.to_numeric(
        df_in["pva200s0"], errors="coerce"
    )  # Vapour pressure over air

    df_in["Rad"] = df_in["gre000z0"] - df_in["gre000z0"] * 0.1
    df_in["DRad"] = df_in["gre000z0"] * 0.1
    df_in["LW"] = df_in["oli000z0"]
    df_in["Prec"] = df_in["Prec"] / 1000

    # Fill nans
    df_in = df_in.fillna(method="ffill")

    df_out = df_in[
        ["When", "T_a", "RH", "v_a", "Rad", "DRad", "oli000z0", "Prec", "p_a", "vp_a",]
    ]
    df_out = df_out.round(5)

    # 5 minute sum
    cols = ["T_a", "RH", "v_a", "Rad", "DRad", "Prec", "p_a", "vp_a", "oli000z0"]
    df_out[cols] = df_out[cols] / 2
    df_out = df_out.set_index("When").resample("5T").ffill().reset_index()

    cols = [
        "When",
        "T_a",
        "RH",
        "v_a",
        "Rad",
        "DRad",
        "Prec",
        "p_a",
        "vp_a",
    ]
    df_out = df_out[cols]
    df_out = df_out.round(5)


filename = folders["input_folder"] + site

df_out.to_csv(filename + "_input.csv")



