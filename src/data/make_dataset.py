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
from src.data.config import site, dates, folders, fountain, surface

start = time.time()

def discharge_rate(df, fountain):

    if option == 'schwarzsee':
        df["Fountain"] = 0  # Fountain run time

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


        for i in range(0, df_nights.shape[0]):
            df_nights.loc[i, "Start"] = df_nights.loc[i, "Start"] - pd.Timedelta(days=1)
            df.loc[
                (df["When"] >= df_nights.loc[i, "Start"])
                & (df["When"] <= df_nights.loc[i, "End"]),
                "Fountain",
            ] = 1

    if option == 'temperature':
        mask = df["T_a"] < fountain["crit_temp"]
        mask_index = df[mask].index
        df.loc[mask_index, "Fountain"] = 1
        mask = df["When"] >= dates["fountain_off_date"]
        mask_index = df[mask].index
        df.loc[mask_index, "Fountain"] = 0

    if option == "energy":

        """Constants"""
        Ls = 2848 * 1000  # J/kg Sublimation
        Le = 2514 * 1000  # J/kg Evaporation
        Lf = 334 * 1000  # J/kg Fusion
        cw = 4.186 * 1000  # J/kg Specific heat water
        ci = 2.108 * 1000  # J/kgC Specific heat ice
        rho_w = 1000  # Density of water
        rho_i = 916  # Density of Ice rho_i
        rho_a = 1.29  # kg/m3 air density at mean sea level
        k = 0.4  # Van Karman constant
        bc = 5.670367 * math.pow(10, -8)  # Stefan Boltzman constant
        g = 9.8  # gravity

        """Miscellaneous"""
        time_steps = 5 * 60  # s Model time steps
        p0 = 1013  # Standard air pressure hPa

        """ Estimating Albedo """
        df["a"] = albedo(df, surface)

        df["T_s"] = 0

        for i in range(1, df.shape[0]):

            """ Energy Balance starts """

            # Vapor Pressure empirical relations
            if "vp_a" not in list(df.columns):
                df.loc[i, "vp_a"] = (
                        6.11
                        * math.pow(
                    10, 7.5 * df.loc[i - 1, "T_a"] / (df.loc[i - 1, "T_a"] + 237.3)
                )
                        * df.loc[i, "RH"]
                        / 100
                )

            df.loc[i, "vp_ice"] = 6.112 * np.exp(
                22.46 * (df.loc[i - 1, "T_s"]) / ((df.loc[i - 1, "T_s"]) + 243.12)
            )

            # Sublimation only
            df.loc[i, "Ql"] = (
                    0.623
                    * Ls
                    * rho_a
                    / p0
                    * math.pow(k, 2)
                    * df.loc[i, "v_a"]
                    * (df.loc[i, "vp_a"] - df.loc[i, "vp_ice"])
                    / (
                            np.log(surface["h_aws"] / surface["z0mi"])
                            * np.log(surface["h_aws"] / surface["z0hi"])
                    )
            )

            # Short Wave Radiation SW
            df.loc[i, "SW"] = (1 - df.loc[i, "a"]) * (
                    df.loc[i, "SW_direct"] + df.loc[i, "DRad"]
            )

            # Cloudiness from diffuse fraction
            if df.loc[i, "SW_direct"] + df.loc[i, "DRad"] > 1:
                df.loc[i, "cld"] = df.loc[i, "DRad"] / (
                        df.loc[i, "SW_direct"] + df.loc[i, "DRad"]
                )
            else:
                df.loc[i, "cld"] = 0

            # atmospheric emissivity
            df.loc[i, "e_a"] = (
                                           1.24
                                           * math.pow(abs(df.loc[i, "vp_a"] / (df.loc[i, "T_a"] + 273.15)),
                                                      1 / 7)
                                   ) * (1 + 0.22 * math.pow(df.loc[i, "cld"], 2))

            # Long Wave Radiation LW
            if "oli000z0" not in list(df.columns):

                df.loc[i, "LW"] = df.loc[i, "e_a"] * bc * math.pow(
                    df.loc[i, "T_a"] + 273.15, 4
                ) - surface["ie"] * bc * math.pow(df.loc[i - 1, "T_s"] + 273.15, 4)
            else:
                df.loc[i, "LW"] = df.loc[i, "oli000z0"] - surface["ie"] * bc * math.pow(
                    df.loc[i - 1, "T_s"] + 273.15,
                    4)

            # Sensible Heat Qs
            df.loc[i, "Qs"] = (
                    ci
                    * rho_a
                    * df.loc[i, "p_a"]
                    / p0
                    * math.pow(k, 2)
                    * df.loc[i, "v_a"]
                    * (df.loc[i, "T_a"] - df.loc[i - 1, "T_s"])
                    / (
                            np.log(surface["h_aws"] / surface["z0mi"])
                            * np.log(surface["h_aws"] / surface["z0hi"])
                    )
            )

            # Total Energy W/m2
            df.loc[i, "TotalE"] = df.loc[i, "SW"] + df.loc[i, "LW"] + df.loc[i, "Qs"] + df.loc[i, "Ql"]

        x = df["When"]
        mask = df["TotalE"] < 0
        mask_index = df[mask].index
        df.loc[mask_index, "Fountain"] = 1
        mask = df["When"] >= dates["fountain_off_date"]
        mask_index = df[mask].index
        df.loc[mask_index, "Fountain"] = 0

    if option == 'schwarzsee':
        df.Discharge = df.Discharge * df.Fountain
    else:
        df.Discharge = fountain["discharge"] * df.Fountain

    return df["Fountain"], df["Discharge"]


if __name__ == '__main__':

    if site == "schwarzsee":

        # read files
        df_in = pd.read_csv(
            folders["data_file"],
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

        # Drop
        df_in = df_in.drop(["Pluviometer"], axis=1)

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

        # Add Radiation data
        df_in2 = pd.read_csv(
            os.path.join(folders["dirname"], "data/raw/plaffeien_rad.txt"),
            sep="\\s+",
            skiprows=2,
        )
        df_in2["When"] = pd.to_datetime(df_in2["time"], format="%Y%m%d%H%M")
        df_in2["ods000z0"] = pd.to_numeric(df_in2["ods000z0"], errors="coerce")
        df_in2["gre000z0"] = pd.to_numeric(df_in2["gre000z0"], errors="coerce")
        df_in2["SW_direct"] = df_in2["gre000z0"] - df_in2["ods000z0"]
        df_in2["DRad"] = df_in2["ods000z0"]

        # Add Precipitation data
        df_in2["Prec"] = pd.to_numeric(df_in2["rre150z0"], errors="coerce")
        df_in2["Prec"] = df_in2["Prec"] / 2  # 5 minute sum
        df_in2 = df_in2.set_index("When").resample("5T").ffill().reset_index()

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
        df["SW_direct"] = df_in2["SW_direct"]
        df["DRad"] = df_in2["DRad"]
        df["Prec"] = df_in2["Prec"] / 1000

        df["cld"] = 0
        df["SEA"] = 0
        df["e_a"] = 0

        # CSV output
        df.rename(
            columns={
                "Wind Speed": "v_a",
                "Temperature": "T_a",
                "Humidity": "RH",
                "Pressure": "p_a",
                "SW_direct": 'SW_direct',
                "DRad": 'SW_diffuse',
            },
            inplace=True,
        )

        # v_a mean
        v_a = df["v_a"].replace(0, np.NaN).mean()  # m/s Average Humidity
        df["v_a"] = df["v_a"].replace(0, v_a)

        for i in tqdm(range(1, df.shape[0])):
            if np.isnan(df.loc[i, "SW_direct"]):
                df.loc[i, "SW_direct"] = df.loc[i - 1, "SW_direct"]
            if np.isnan(df.loc[i, "SW_diffuse"]):
                df.loc[i, "SW_diffuse"] = df.loc[i - 1, "SW_diffuse"]
            if np.isnan(df.loc[i, "Prec"]):
                df.loc[i, "Prec"] = df.loc[i - 1, "Prec"]

        """Discharge Rate"""
        df["Fountain"], df["Discharge"] = discharge_rate(df,fountain)

        df["Discharge"] = df["Discharge"] * df["Fountain"]


        df_out = df[
            ["When", "T_a", "RH", "v_a", "Discharge", "SW_direct", "SW_diffuse", "Prec", "p_a"]
        ]

        df_out = df_out.round(5)

        df_out.to_csv(folders["input_folder"] + "raw_input.csv")

        fig, ax = plt.subplots()
        ax.plot(df.When, df.Discharge)
        plt.show()

    if site == "guttannen":

        # read files
        df_in = pd.read_csv(
            folders["data_file"],
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

        # Datetime
        df_in["When"] = pd.to_datetime(df_in["Date"] + " " + df_in["Time"])
        df_in["When"] = pd.to_datetime(df_in["When"], format="%Y.%m.%d %H:%M:%S")

        # Drop
        df_in = df_in.drop(["Pluviometer"], axis=1)
        df_in = df_in.drop(["Wind Direction"], axis=1)

        mask = (df_in["When"] >= dates["start_date"]) & (df_in["When"] <= dates["end_date"])
        df_in = df_in.loc[mask]
        df_in = df_in.reset_index()

        # Error Correction

        # v_a mean
        v_a = df_in["Wind Speed"].replace(5643.2, np.NaN).mean()  # m/s Average Humidity
        df_in["Wind Speed"] = df_in["Wind Speed"].replace(5643.2, v_a)

        """
                Parameter
                ---------
                          Unit                                 Description
                oli000z0
                prestas0  hPa                                  Pressure at station level (QFE); current value
                gre000z0  W/m²                                 Global radiation; ten minutes mean
                pva200s0  hPa                                  Vapour pressure 2 m above ground; current value
                rre150z0  mm                                   Precipitation; ten minutes total
                ure200s0  %                                    Relative air humidity 2 m above ground;
                fkl010z0  m/s                                  Wind speed scalar; ten minutes mean
                tre200s0  °C                                   Air temperature 2 m above ground; current
        """

        # Add Radiation data
        df_in2 = pd.read_csv(os.path.join(folders["dirname"], "data/raw/guttannen_2020_add.txt"), encoding="latin-1", skiprows=2, sep=";")
        df_in2["When"] = pd.to_datetime(df_in2["time"], format="%Y%m%d%H%M")  # Datetime

        # Convert to int
        df_in2["oli000z0"] = pd.to_numeric(
            df_in2["oli000z0"], errors="coerce"
        )  # Add Longwave Radiation data

        df_in2["gre000z0"] = pd.to_numeric(
            df_in2["gre000z0"], errors="coerce"
        )  # Add Radiation data

        # Add rest of data
        df_in3 = pd.read_csv(os.path.join(folders["dirname"], "data/raw/guttannen_2020_add_2.txt"), encoding="latin-1",
                             skiprows=2, sep=";")
        df_in3["When"] = pd.to_datetime(df_in3["time"], format="%Y%m%d%H%M")  # Datetime

        df_in3["Prec"] = pd.to_numeric(
            df_in3["rre150z0"], errors="coerce"
        )  # Add Precipitation data

        df_in3["vp_a"] = pd.to_numeric(
            df_in3["pva200s0"], errors="coerce"
        )  # Vapour pressure over air

        df_in3["Prec"] = df_in3["Prec"] / 1000

        df_in3["Temperature"] = pd.to_numeric(
            df_in3["tre200s0"], errors="coerce"
        )  # Air temperature

        df_in3["Wind Speed"] = pd.to_numeric(
            df_in3["fkl010z0"], errors="coerce"
        )  # Wind speed

        df_in3["Wind Speed"] = df_in3["Wind Speed"]/3.6

        df_in3["Humidity"] = pd.to_numeric(
            df_in3["ure200s0"], errors="coerce"
        )

        df_in2 = df_in2.set_index("When").resample("5T").ffill().reset_index()
        df_in3 = df_in3.set_index("When").resample("5T").ffill().reset_index()

        mask = (df_in["When"] >= dates["start_date"]) & (df_in["When"] <= dates["error_date"])
        df_in = df_in.loc[mask]
        df_in = df_in.reset_index()

        mask = (df_in2["When"] >= dates["start_date"]) & (
                df_in2["When"] <= dates["error_date"]
        )
        df_2 = df_in2.loc[mask]
        df_2 = df_2.reset_index()

        mask = (df_in3["When"] >= dates["start_date"]) & (
                df_in3["When"] <= dates["error_date"]
        )
        df_3 = df_in3.loc[mask]
        df_3 = df_3.reset_index()

        mask = (df_in3["When"] >= dates["error_date"]) & (
                df_in3["When"] <= dates["end_date"]
        )
        df_4 = df_in3.loc[mask]
        df_4 = df_4.reset_index()

        mask = (df_in2["When"] >= dates["error_date"]) & (
                df_in2["When"] <= dates["end_date"]
        )
        df_5 = df_in2.loc[mask]
        df_5 = df_5.reset_index()
        df_4["Pressure"] = df_5["prestas0"]
        df_4["gre000z0"] = df_5["gre000z0"]
        df_4["oli000z0"] = df_5["oli000z0"]


        df_4["Discharge"] = 0
        mask = df_4["Temperature"] < fountain["crit_temp"]
        mask_index = df_4[mask].index
        df_4.loc[mask_index, "Discharge"] = 15
        mask = df_4["When"] >= dates["fountain_off_date"]
        mask_index = df_4[mask].index
        df_4.loc[mask_index, "Discharge"] = 0

        # print(df_3.tail())
        # print(df_4.head())

        days = pd.date_range(start=dates["start_date"], end=dates["error_date"], freq="5T")
        days = pd.DataFrame({"When": days})

        df = pd.merge(
            days,
            df_in[
                [
                    "When",
                    "Discharge",
                    "Temperature",
                    "Wind Speed",
                    "Humidity",
                    "Pressure",
                ]
            ],
            on="When",
        )

        # Fill with other station-
        df["gre000z0"] = df_2["gre000z0"]
        df["oli000z0"] = df_2["oli000z0"]
        df["Prec"] = df_3["Prec"]
        df["vp_a"] = df_3["vp_a"]

        mask = (df["When"] >= dates["start_date"]) & (df["When"] <= dates["error_date"])
        df = df.loc[mask]
        df = df.reset_index()

        # Add rest of data
        df_4 = df_4.reindex(df_4.index.drop(0)).reset_index(drop=True)
        df = df.append(df_4, ignore_index=True)



        # Error Correction
        df["gre000z0"] = df["gre000z0"].replace(np.NaN, 0)
        df["Prec"] = df["Prec"].replace(np.NaN, 0)
        df["vp_a"] = df["vp_a"].replace(np.NaN, 0)

        cld = 0.5
        df["Rad"] = df_in2["gre000z0"] - df_in2["gre000z0"] * cld
        df["DRad"] = df_in2["gre000z0"] * cld
        df["cld"] = cld
        # df["SEA"] = 0
        # df["e_a"] = 0

        # CSV output
        df.rename(
            columns={
                "Wind Speed": "v_a",
                "Temperature": "T_a",
                "Humidity": "RH",
                "Pressure": "p_a",
                "Rad": 'SW_direct',
                "DRad": 'SW_diffuse',
                "oli000z0": 'LW_in',
            },
            inplace=True,
        )
        # for i in tqdm(range(1, df.shape[0])):

        #     """Solar Elevation Angle"""
        #     df.loc[i, "SEA"] = getSEA(
        #         df.loc[i, "When"],
        #         fountain["latitude"],
        #         fountain["longitude"],
        #         fountain["utc_offset"],
        #     )

        #     """ Vapour Pressure"""
        #     if "vpa" not in list(df.columns):
        #         df.loc[i, "vp_a"] = (6.11 * math.pow(10, 7.5 * df.loc[i - 1, "T_a"] / (df.loc[i - 1, "T_a"] + 237.3)) * df.loc[i, "RH"] / 100)
        #     else:
        #         df.loc[i, "vp_a"] = df.loc[i, "vpa"]

        #     # atmospheric emissivity
        #     df.loc[i, "e_a"] = ( 1.24 * math.pow(abs(df.loc[i, "vp_a"] / (df.loc[i, "T_a"] + 273.15)), 1 / 7)
        #                        ) * (1 + 0.22 * math.pow(df.loc[i, "cld"], 2))

        df_out = df[
            ["When", "T_a", "RH", "v_a", "Discharge", "SW_direct", "SW_diffuse", "LW_in", "cld", "Prec", "p_a", "vp_a"]
        ]

        df_out = df_out.round(5)

        print(df_out.tail())

        filename = folders["input_folder"] + site

        df_out.to_csv(filename + "_raw_input.csv")

        # fig, ax = plt.subplots()
        # ax.plot(df.When, df.v_a)
        # plt.show()
