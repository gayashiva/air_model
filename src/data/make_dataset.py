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
from src.data.config import site, dates, option, folders, fountain
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

    df["Fountain"] = 0  # Fountain run time

    df_nights = pd.read_csv(
        os.path.join(folders["dirname"], "data/raw/schwarzsee_fountain_time.txt"),
        sep="\s+",
    )

    df_nights["Start"] = pd.to_datetime(df_nights["Date"] + " " + df_nights["start"])
    df_nights["End"] = pd.to_datetime(df_nights["Date"] + " " + df_nights["end"])
    df_nights["Start"] = pd.to_datetime(df_nights["Start"], format="%Y-%m-%d %H:%M:%S")
    df_nights["End"] = pd.to_datetime(df_nights["End"], format="%Y-%m-%d %H:%M:%S")

    df_nights["Date"] = pd.to_datetime(df_nights["Date"], format="%Y-%m-%d")
    mask = (df_nights["Date"] >= dates["start_date"]) & (
        df_nights["Date"] <= dates["end_date"]
    )
    df_nights = df_nights.loc[mask]
    df_nights = df_nights.reset_index()

    for i in range(0, df_nights.shape[0]):
        df_nights.loc[i, "Start"] = df_nights.loc[i, "Start"] - pd.Timedelta(days=1)
        df.loc[
            (df["When"] >= df_nights.loc[i, "Start"])
            & (df["When"] <= df_nights.loc[i, "End"]),
            "Fountain",
        ] = 1

    df.Discharge = df.Discharge * df.Fountain

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
    if option == "schwarzsee":
        df["Fountain"] = 0
        mask = df["Discharge"] > 0.1
        mask_index = df[mask].index
        df.loc[mask_index, "Fountain"] = 1

        df_out = df[
            [
                "When",
                "T_a",
                "RH",
                "v_a",
                "Discharge",
                "Fountain",
                "Rad",
                "DRad",
                "Prec",
                "p_a",
            ]
        ]

    else:
        df_out = df[
            ["When", "T_a", "RH", "v_a", "Fountain", "Rad", "DRad", "Prec", "p_a"]
        ]
        if option == "temperature":
            """ Use Temperature """
            mask = df_out["T_a"] < fountain["t_c"]
            mask_index = df_out[mask].index
            df_out.loc[mask_index, "Fountain"] = 1
            mask = df_out["When"] >= dates["fountain_off_date"]
            mask_index = df_out[mask].index
            df_out.loc[mask_index, "Fountain"] = 0

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

    df_out["Discharge"] = 0  # litres per minute

    """Fountain Discharge"""

    if option == "schwarzsee":

        """ Use Schwarzsee"""

        df_out["Fountain"] = 0  # Fountain run time
        df_out["Discharge"] = 0  # Fountain run time

        df_nights = pd.read_csv(
            os.path.join(folders["dirname"], "data/raw/schwarzsee_fountain_time.txt"),
            sep="\s+",
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
        mask = (df_nights["Date"] >= dates["start_date"]) & (
            df_nights["Date"] <= dates["end_date"]
        )
        df_nights = df_nights.loc[mask]
        df_nights = df_nights.reset_index()

        for i in range(0, df_nights.shape[0]):
            df_nights.loc[i, "Start"] = df_nights.loc[i, "Start"] - pd.Timedelta(days=1)
            df_out.loc[
                (df_out["When"] >= df_nights.loc[i, "Start"])
                & (df_out["When"] <= df_nights.loc[i, "End"]),
                "Fountain",
            ] = 1

        df_out.Discharge = fountain["discharge"] * df_out.Fountain  # Litres per minute

    if option == "energy":
        """ Use Energy Flux """
        """Settings"""
        z = 2  # m height of AWS

        """Material Properties"""
        a_w = 0.6
        we = 0.95
        z0mi = 0.001
        z0ms = 0.0015
        z0hi = 0.0001
        c = 0.5
        Lf = 334 * 1000  #  J/kg Fusion
        cw = 4.186 * 1000  # J/kg Specific heat water
        rho_w = 1000  # Density of water
        rho_a = 1.29  # kg/m3 air density at mean sea level
        p0 = 1013  # hPa
        k = 0.4  # Van Karman constant
        bc = 5.670367 * math.pow(10, -8)  # Stefan Boltzman constant

        """Initialise"""
        df_out["TotalE"] = 0
        df_out["SW"] = 0
        df_out["LW"] = 0
        df_out["Qs"] = 0

        """ Simulation """
        for i in range(1, df_out.shape[0]):

            # Vapor Pressure empirical relations
            if "vp_a" not in list(df_out.columns):
                Ea = (
                    6.11
                    * math.pow(
                        10,
                        7.5
                        * df_out.loc[i - 1, "T_a"]
                        / (df_out.loc[i - 1, "T_a"] + 237.3),
                    )
                    * df_out.loc[i, "RH"]
                    / 100
                )
            else:
                Ea = df_out.loc[i, "vp_a"]

            df_out.loc[i, "e_a"] = (
                1.24
                * math.pow(abs(Ea / (df_out.loc[i, "T_a"] + 273.15)), 1 / 7)
                * (1 + 0.22 * math.pow(c, 2))
            )

            # Short Wave Radiation SW
            df_out.loc[i, "SW"] = (1 - a_w) * (
                df_out.loc[i, "Rad"] + df_out.loc[i, "DRad"]
            )

            # Long Wave Radiation LW
            if "oli000z0" not in list(df_out.columns):
                df_out.loc[i, "LW"] = df_out.loc[i, "e_a"] * bc * math.pow(
                    df_out.loc[i, "T_a"] + 273.15, 4
                ) - we * bc * math.pow(0 + 273.15, 4)
            else:
                df_out.loc[i, "LW"] = df_out.loc[i, "oli000z0"] - we * bc * math.pow(
                    0 + 273.15, 4
                )

            # Sensible Heat Qs
            df_out.loc[i, "Qs"] = (
                cw
                * rho_a
                * df_out.loc[i, "p_a"]
                / p0
                * math.pow(k, 2)
                * df_out.loc[i, "v_a"]
                * (df_out.loc[i, "T_a"])
                / (np.log(z / z0mi) * np.log(z / z0hi))
            )

            # Total Energy W/m2
            df_out.loc[i, "TotalE"] = (
                df_out.loc[i, "SW"] + df_out.loc[i, "LW"] + df_out.loc[i, "Qs"]
            )

        df_out.Discharge[df_out.TotalE < 0] = fountain["discharge"]  # litres per minute

    if option == "temperature":
        """ Use Temperature """
        mask = df_out["T_a"] < fountain["t_c"]
        mask_index = df_out[mask].index
        df_out.loc[mask_index, "Fountain"] = 1
        mask = df_out["When"] >= dates["fountain_off_date"]
        mask_index = df_out[mask].index
        df_out.loc[mask_index, "Fountain"] = 0

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
        "Fountain",
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

    df_out["Fountain"] = 0  # Fountain run time

    """Fountain Discharge"""

    if option == "schwarzsee":

        """ Use Schwarzsee"""

        df_nights = pd.read_csv(
            os.path.join(folders["dirname"], "data/raw/schwarzsee_fountain_time.txt"),
            sep="\s+",
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
        mask = (df_nights["Date"] >= dates["start_date"]) & (
            df_nights["Date"] <= dates["end_date"]
        )
        df_nights = df_nights.loc[mask]
        df_nights = df_nights.reset_index()

        for i in range(0, df_nights.shape[0]):
            df_nights.loc[i, "Start"] = df_nights.loc[i, "Start"] - pd.Timedelta(days=1)
            df_out.loc[
                (df_out["When"] >= df_nights.loc[i, "Start"])
                & (df_out["When"] <= df_nights.loc[i, "End"]),
                "Fountain",
            ] = 1

    if option == "energy":  # todo examine again
        """ Use Energy Flux """
        """Settings"""
        z = 2  # m height of AWS

        """Material Properties"""
        a_w = 0.6
        we = 0.95
        z0mi = 0.001
        z0ms = 0.0015
        z0hi = 0.0001
        c = 0.5
        Lf = 334 * 1000  #  J/kg Fusion
        cw = 4.186 * 1000  # J/kg Specific heat water
        rho_w = 1000  # Density of water
        rho_a = 1.29  # kg/m3 air density at mean sea level
        p0 = 1013  # hPa
        k = 0.4  # Van Karman constant
        bc = 5.670367 * math.pow(10, -8)  # Stefan Boltzman constant

        """Initialise"""
        df_out["TotalE"] = 0
        df_out["SW"] = 0
        df_out["LW"] = 0
        df_out["Qs"] = 0

        """ Simulation """
        for i in range(1, df_out.shape[0]):

            # Vapor Pressure empirical relations
            if "vp_a" not in list(df_out.columns):
                Ea = (
                    6.11
                    * math.pow(
                        10,
                        7.5
                        * df_out.loc[i - 1, "T_a"]
                        / (df_out.loc[i - 1, "T_a"] + 237.3),
                    )
                    * df_out.loc[i, "RH"]
                    / 100
                )
            else:
                Ea = df_out.loc[i, "vp_a"]

            df_out.loc[i, "e_a"] = (
                1.24
                * math.pow(abs(Ea / (df_out.loc[i, "T_a"] + 273.15)), 1 / 7)
                * (1 + 0.22 * math.pow(c, 2))
            )

            # Short Wave Radiation SW
            df_out.loc[i, "SW"] = (1 - a_w) * (
                df_out.loc[i, "Rad"] + df_out.loc[i, "DRad"]
            )

            # Long Wave Radiation LW
            if "oli000z0" not in list(df_out.columns):
                df_out.loc[i, "LW"] = df_out.loc[i, "e_a"] * bc * math.pow(
                    df_out.loc[i, "T_a"] + 273.15, 4
                ) - we * bc * math.pow(0 + 273.15, 4)
            else:
                df_out.loc[i, "LW"] = df_out.loc[i, "oli000z0"] - we * bc * math.pow(
                    0 + 273.15, 4
                )

            # Sensible Heat Qs
            df_out.loc[i, "Qs"] = (
                cw
                * rho_a
                * df_out.loc[i, "p_a"]
                / p0
                * math.pow(k, 2)
                * df_out.loc[i, "v_a"]
                * (df_out.loc[i, "T_a"])
                / (np.log(z / z0mi) * np.log(z / z0hi))
            )

            # Total Energy W/m2
            df_out.loc[i, "TotalE"] = (
                df_out.loc[i, "SW"] + df_out.loc[i, "LW"] + df_out.loc[i, "Qs"]
            )

        df_out.Discharge[df_out.TotalE < 0] = fountain["discharge"]  # litres per minute

    if option == "temperature":
        """ Use Temperature """
        mask = df_out["T_a"] < fountain["t_c"]
        mask_index = df_out[mask].index
        df_out.loc[mask_index, "Fountain"] = 1
        mask = df_out["When"] >= dates["fountain_off_date"]
        mask_index = df_out[mask].index
        df_out.loc[mask_index, "Fountain"] = 0

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
        "Fountain",
    ]
    df_out = df_out[cols]
    df_out = df_out.round(5)

if option == "temperature":
    filename = (
        folders["input_folder"] + site + "_" + option + "_" + str(fountain["t_c"])
    )
else:
    filename = folders["input_folder"] + site + "_" + option

df_out.to_csv(filename + "_input.csv")


# Plots
pp = PdfPages(filename + "_all_data" + ".pdf")

x = df_out["When"]
y1 = df_out["T_a"]
y2 = df_out["Prec"]

fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.plot(x, y1, "k-", linewidth=0.5)
ax1.set_ylabel("Temperature (C)")
ax1.set_xlabel("Days")

ax2 = ax1.twinx()
ax2.plot(x, y2, "b-", linewidth=0.5)
ax2.set_ylabel("Prec (m)", color="b")
for tl in ax2.get_yticklabels():
    tl.set_color("b")

# format the ticks
ax1.xaxis.set_major_locator(mdates.WeekdayLocator())
ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
ax1.xaxis.set_minor_locator(mdates.DayLocator())
ax1.grid()

# rotates and right aligns the x labels, and moves the bottom of the axes up to make room for them
fig.autofmt_xdate()
pp.savefig(bbox_inches="tight")
plt.clf()

y1 = df_out["v_a"]

fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.plot(x, y1, "k-", linewidth=0.5)
ax1.set_ylabel("Wind Speed ($m/s$)")
ax1.set_xlabel("Days")

# format the ticks
ax1.xaxis.set_major_locator(mdates.WeekdayLocator())
ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
ax1.xaxis.set_minor_locator(mdates.DayLocator())
ax1.grid()

fig.autofmt_xdate()
pp.savefig(bbox_inches="tight")
plt.clf()

y1 = df_out["p_a"]
fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.plot(x, y1, "k-", linewidth=0.5)
ax1.set_ylabel("Pressure (hPa)")
ax1.set_xlabel("Days")

# format the ticks
ax1.xaxis.set_major_locator(mdates.WeekdayLocator())
ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
ax1.xaxis.set_minor_locator(mdates.DayLocator())
ax1.grid()

fig.autofmt_xdate()
pp.savefig(bbox_inches="tight")
plt.clf()

y1 = df_out["RH"]

fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.plot(x, y1, "k-", linewidth=0.5)
ax1.set_ylabel("Relative Humidity (%)")
ax1.set_xlabel("Days")

# format the ticks
ax1.xaxis.set_major_locator(mdates.WeekdayLocator())
ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
ax1.xaxis.set_minor_locator(mdates.DayLocator())
ax1.grid()

fig.autofmt_xdate()
pp.savefig(bbox_inches="tight")
plt.clf()

if site != "schwarzsee":
    y1 = df_out["vp_a"]

    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.plot(x, y1, "k-", linewidth=0.5)
    ax1.set_ylabel("Vapour Pressure air (hPa)")
    ax1.set_xlabel("Days")

    # format the ticks
    ax1.xaxis.set_major_locator(mdates.WeekdayLocator())
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax1.xaxis.set_minor_locator(mdates.DayLocator())
    ax1.grid()

    # rotates and right aligns the x labels, and moves the bottom of the axes up to make room for them
    fig.autofmt_xdate()
    pp.savefig(bbox_inches="tight")
    plt.clf()

y1 = df_out["Rad"]
y2 = df_out["DRad"]

fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.plot(x, y1, "k-", linewidth=0.5)
ax1.set_ylabel("SWR ($W/m^{2}$)")
ax1.set_xlabel("Days")

ax2 = ax1.twinx()
ax2.plot(x, y2, "b-", linewidth=0.5)
ax2.set_ylabel("DR", color="b")
for tl in ax2.get_yticklabels():
    tl.set_color("b")

# format the ticks
ax1.xaxis.set_major_locator(mdates.WeekdayLocator())
ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
ax1.xaxis.set_minor_locator(mdates.DayLocator())
ax1.grid()

# rotates and right aligns the x labels, and moves the bottom of the axes up to make room for them
fig.autofmt_xdate()
pp.savefig(bbox_inches="tight")
plt.clf()


fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.plot(x, y1, "k-", linewidth=0.5)
ax1.set_ylabel("Fountain on/off")
ax1.set_xlabel("Days")

# format the ticks
ax1.xaxis.set_major_locator(mdates.WeekdayLocator())
ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
ax1.xaxis.set_minor_locator(mdates.DayLocator())
ax1.grid()

fig.autofmt_xdate()
pp.savefig(bbox_inches="tight")
plt.clf()

pp.close()


# Plots
pp = PdfPages(filename + "_data" + ".pdf")

fig, (ax1, ax2, ax3, ax4, ax5, ax6) = plt.subplots(
    nrows=6, ncols=1, sharex="col", sharey="row", figsize=(15, 10)
)

# fig.suptitle("Field Data", fontsize=14)
x = df_out.When

if option == "schwarzsee":
    y1 = df_out.Discharge * 5
else:
    y1 = df_out.Fountain
ax1.plot(x, y1, "k-", linewidth=0.5)
ax1.set_ylabel("Discharge ($l$)")
ax1.grid()

y2 = df_out.T_a
ax2.plot(x, y2, "k-", linewidth=0.5)
ax2.set_ylabel("T ($C$)")
ax2.grid()
ax2.xaxis.set_major_locator(mdates.WeekdayLocator())
ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
ax2.xaxis.set_minor_locator(mdates.DayLocator())


y3 = df_out.Rad
ax3.plot(x, y3, "k-", linewidth=0.5)
ax3.set_ylabel("SWR ($W/m^{2}$)")
ax3.set_ylim([0, 600])
ax3.grid()

ax3t = ax3.twinx()
ax3t.plot(x, df_out.DRad, "b-", linewidth=0.5)
ax3t.set_ylabel("Diffused ($W/m^{2}$)", color="b")
ax3t.set_ylim([0, 600])
for tl in ax3t.get_yticklabels():
    tl.set_color("b")

y4 = df_out.Prec * 1000
ax4.plot(x, y4, "k-", linewidth=0.5)
ax4.set_ylabel("Ppt ($mm$)")
ax4.grid()

y5 = df_out.p_a
ax5.plot(x, y5, "k-", linewidth=0.5)
ax5.set_ylabel("Pressure ($hPa$)")
ax5.grid()

y6 = df_out.v_a
ax6.plot(x, y6, "k-", linewidth=0.5)
ax6.set_ylabel("Wind ($m/s^{1}$)")
ax6.grid()

# rotates and right aligns the x labels, and moves the bottom of the axes up to make room for them
fig.autofmt_xdate()
pp.savefig(bbox_inches="tight")

plt.savefig(
    os.path.join(folders["input_folder"], site + "_data.jpg"),
    bbox_inches="tight",
    dpi=300,
)

plt.clf()

fig = plt.figure()
ax1 = fig.add_subplot(111)

y1 = df_out.T_a
ax1.plot(x, y1, "k-", linewidth=0.5)
ax1.set_ylabel("T ($C$)")
ax1.grid()

# format the ticks
ax1.xaxis.set_major_locator(mdates.WeekdayLocator())
ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
ax1.xaxis.set_minor_locator(mdates.DayLocator())
ax1.grid()
fig.autofmt_xdate()
pp.savefig(bbox_inches="tight")
plt.clf()

fig = plt.figure()
ax1 = fig.add_subplot(111)

ax1.plot(x, y2, "k-", linewidth=0.5)
ax1.set_ylabel("Fountain on/off ")
ax1.grid()

# format the ticks
ax1.xaxis.set_major_locator(mdates.WeekdayLocator())
ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
ax1.xaxis.set_minor_locator(mdates.DayLocator())
ax1.grid()
fig.autofmt_xdate()
pp.savefig(bbox_inches="tight")
plt.clf()

fig = plt.figure()
ax1 = fig.add_subplot(111)

y3 = df_out.Rad
ax1.plot(x, y3, "k-", linewidth=0.5)
ax1.set_ylabel("SWR ($W/m^{2}$)")
ax1.set_ylim([0, 600])
ax1.grid()

# format the ticks
ax1.xaxis.set_major_locator(mdates.WeekdayLocator())
ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
ax1.xaxis.set_minor_locator(mdates.DayLocator())
ax1.grid()
fig.autofmt_xdate()
pp.savefig(bbox_inches="tight")
plt.clf()

fig = plt.figure()
ax1 = fig.add_subplot(111)

y4 = df_out.Prec * 1000
ax1.plot(x, y4, "k-", linewidth=0.5)
ax1.set_ylabel("Ppt ($mm$)")
ax1.grid()

# format the ticks
ax1.xaxis.set_major_locator(mdates.WeekdayLocator())
ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
ax1.xaxis.set_minor_locator(mdates.DayLocator())
ax1.grid()
fig.autofmt_xdate()
pp.savefig(bbox_inches="tight")
plt.clf()

fig = plt.figure()
ax1 = fig.add_subplot(111)

y5 = df_out.p_a
ax1.plot(x, y5, "k-", linewidth=0.5)
ax1.set_ylabel("Pressure ($hPa$)")
ax1.grid()

# format the ticks
ax1.xaxis.set_major_locator(mdates.WeekdayLocator())
ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
ax1.xaxis.set_minor_locator(mdates.DayLocator())
ax1.grid()
fig.autofmt_xdate()
pp.savefig(bbox_inches="tight")
plt.clf()

fig = plt.figure()
ax1 = fig.add_subplot(111)

y6 = df_out.v_a
ax1.plot(x, y6, "k-", linewidth=0.5)
ax1.set_ylabel("Wind ($m/s^{1}$)")
ax1.grid()

# format the ticks
ax1.xaxis.set_major_locator(mdates.WeekdayLocator())
ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
ax1.xaxis.set_minor_locator(mdates.DayLocator())
ax1.grid()
fig.autofmt_xdate()
pp.savefig(bbox_inches="tight")
plt.clf()

pp.close()
