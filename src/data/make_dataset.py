import sys

sys.path.append("/home/surya/Programs/Github/air_model")
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
from src.data.config import SITE, FOLDERS, FOUNTAIN, OPTION
from scipy import stats
from sklearn.linear_model import LinearRegression

start = time.time()


def discharge_rate(df, FOUNTAIN):

    if OPTION == "schwarzsee":
        df["Fountain"] = 0  # Fountain run time

        df_nights = pd.read_csv(
            os.path.join(FOLDERS["raw_folder"], "schwarzsee_fountain_time.txt"),
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

        for i in range(0, df_nights.shape[0]):
            df_nights.loc[i, "Start"] = df_nights.loc[i, "Start"] - pd.Timedelta(days=1)
            df.loc[
                (df["When"] >= df_nights.loc[i, "Start"])
                & (df["When"] <= df_nights.loc[i, "End"]),
                "Fountain",
            ] = 1

    if OPTION == "temperature":
        mask = df["T_a"] < FOUNTAIN["crit_temp"]
        mask_index = df[mask].index
        df.loc[mask_index, "Fountain"] = 1
        mask = df["When"] >= FOUNTAIN["fountain_off_date"]
        mask_index = df[mask].index
        df.loc[mask_index, "Fountain"] = 0

    if OPTION == "schwarzsee":
        df.Discharge = df.Discharge * df.Fountain
    else:
        df.Discharge = FOUNTAIN["discharge"] * df.Fountain

    return df["Fountain"], df["Discharge"]


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


if __name__ == "__main__":

    if SITE["name"] == "schwarzsee":

        # read files
        df_in = pd.read_csv(
            FOLDERS["raw_folder"] + SITE["name"] + "_aws.txt",
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

        # Correct datetime errors
        for i in tqdm(range(1, df_in.shape[0])):
            if str(df_in.loc[i, "When"].year) != "2019":
                df_in.loc[i, "When"] = df_in.loc[i - 1, "When"] + pd.Timedelta(
                    minutes=5
                )

        df_in = df_in.set_index("When").resample("5T").last().reset_index()

        mask = (df_in["When"] >= SITE["start_date"]) & (
            df_in["When"] <= SITE["end_date"]
        )
        df_in = df_in.loc[mask]
        df_in = df_in.reset_index()

        days = pd.date_range(start=SITE["start_date"], end=SITE["end_date"], freq="5T")
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

        # ERA5 begins
        df_in3 = pd.read_csv(
            FOLDERS["raw_folder"] + "ERA5_" + SITE["name"] + ".csv",
            sep=",",
            header=0,
            parse_dates=["When"],
        )

        df_out1 = df_in3
        time_steps = 60 * 60
        df_out1["ssrd"] /= time_steps
        df_out1["strd"] /= time_steps
        df_out1["fdir"] /= time_steps
        df_out1["v_a"] = np.sqrt(df_out1["u10"] ** 2 + df_out1["v10"] ** 2)
        # Derive RH
        df_out1["t2m"] -= 273.15
        df_out1["d2m"] -= 273.15
        df_out1["t2m_RH"] = df_out1["t2m"]
        df_out1["d2m_RH"] = df_out1["d2m"]
        df_out1 = df_out1.apply(lambda x: e_sat(x) if x.name == "t2m_RH" else x)
        df_out1 = df_out1.apply(lambda x: e_sat(x) if x.name == "d2m_RH" else x)
        df_out1["RH"] = 100 * df_out1["d2m_RH"] / df_out1["t2m_RH"]
        df_out1["sp"] = df_out1["sp"] / 100
        df_out1["tp"] = df_out1["tp"] * 1000 / 3600  # mm/s
        df_out1["SW_diffuse"] = df_out1["ssrd"] - df_out1["fdir"]
        df_out1 = df_out1.set_index("When")

        # CSV output
        df_out1.rename(
            columns={
                "t2m": "T_a",
                "sp": "p_a",
                "tcc": "cld",
                "tp": "Prec",
                "fdir": "SW_direct",
                "strd": "LW_in",
            },
            inplace=True,
        )

        df_in3 = df_out1[
            [
                "T_a",
                "RH",
                "Prec",
                "v_a",
                "SW_direct",
                "SW_diffuse",
                "LW_in",
                "cld",
                "p_a",
            ]
        ]

        df_in3 = df_in3.round(5)

        upsampled = df_in3.resample("5T")
        interpolated = upsampled.interpolate(method="linear")
        interpolated = interpolated.reset_index()

        interpolated["Discharge"] = 0
        mask = (interpolated["T_a"] < FOUNTAIN["crit_temp"]) & (
            interpolated["SW_direct"] < 100
        )
        mask_index = interpolated[mask].index
        interpolated.loc[mask_index, "Discharge"] = 2 * 60
        mask = interpolated["When"] >= FOUNTAIN["fountain_off_date"]
        mask_index = interpolated[mask].index
        interpolated.loc[mask_index, "Discharge"] = 0
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
                "cld",
                "p_a",
                "Prec",
            ]
        ]

        df_in3 = df_in3.reset_index()
        mask = (df_in3["When"] >= SITE["start_date"]) & (
            df_in3["When"] <= SITE["end_date"]
        )
        df_in3 = df_in3.loc[mask]
        df_in3 = df_in3.reset_index()

        df_in3.to_csv(FOLDERS["input_folder"] + "raw_input_ERA5.csv")

        df_ERA5 = interpolated[
            [
                "When",
                "T_a",
                "v_a",
                "RH",
                "SW_direct",
                "SW_diffuse",
                "LW_in",
                "cld",
                "p_a",
                "Prec",
                "Discharge",
            ]
        ]
        df_ERA5.loc[:, "Discharge"] = 0

        # Fill from ERA5
        df_ERA5 = df_ERA5.set_index("When")
        df = df.set_index("When")
        df["missing"] = 0
        df.loc[df["T_a"].isnull(), "missing"] = 1
        df["v_a"] = df["v_a"].replace(0, np.NaN)
        # Correct ERA5
        # v_shear = 0.143
        # df_ERA5["v_a"] = df_ERA5["v_a"] * 0.2 ** v_shear
        # print("RMSE temp", ((df.T_a - df_ERA5.T_a) ** 2).mean() ** 0.5)
        # print("RMSE wind", ((df.v_a - df_ERA5.v_a) ** 2).mean() ** 0.5)
        # print("RMSE pressure", ((df.p_a - df_ERA5.p_a) ** 2).mean() ** 0.5)
        # print("RMSE humidity", ((df.RH - df_ERA5.RH) ** 2).mean() ** 0.5)
        # print("Sign", np.sign(((df.v_a - df_ERA5.v_a)).mean()))

        df_ERA5 = df_ERA5.reset_index()
        mask = (df_ERA5["When"] >= SITE["start_date"]) & (
            df_ERA5["When"] <= SITE["end_date"]
        )

        for column in ["T_a", "RH", "v_a", "p_a"]:
            Y = df[column].values.reshape(-1, 1)
            X = df_ERA5[mask][column].values.reshape(-1, 1)
            slope, intercept = linreg(X, Y)
            print(column, slope, intercept)
            df_ERA5[column] = slope * df_ERA5[column] + intercept

        # Y = df["p_a"].values.reshape(-1, 1)  # values converts it into a numpy array
        # X = df_ERA5[mask]["p_a"].values.reshape(-1, 1)
        # slope, intercept = linreg(X, Y)
        # print(slope, intercept)
        # df_ERA5["p_a"] = slope * df_ERA5["p_a"] - intercept
        # Y_pred = slope * Y + intercept
        # fig = plt.figure()
        # ax1 = fig.add_subplot(111)
        # ax1.scatter(X, Y, s=2)
        # ax1.scatter(X, Y_pred, s=2, color="red")
        # ax1.set_xlabel("Plf ppt")
        # ax1.set_ylabel("ERA5 ppt")
        # ax1.grid()
        # lims = [
        #     np.min([ax1.get_xlim(), ax1.get_ylim()]),  # min of both axes
        #     np.max([ax1.get_xlim(), ax1.get_ylim()]),  # max of both axes
        # ]
        # # now plot both limits against eachother
        # ax1.plot(lims, lims, "--k", alpha=0.25, zorder=0)
        # ax1.set_aspect("equal")
        # ax1.set_xlim(lims)
        # ax1.set_ylim(lims)
        # plt.show()
        df_ERA5 = df_ERA5.set_index("When")

        # df_ERA5["p_a"] += (
        #     np.sign((df.p_a - df_ERA5.p_a).mean())
        #     * ((df.p_a - df_ERA5.p_a) ** 2).mean() ** 0.5
        # )
        # df_ERA5["T_a"] += (
        #     np.sign(((df.T_a - df_ERA5.T_a)).mean())
        #     * ((df.T_a - df_ERA5.T_a) ** 2).mean() ** 0.5
        # )
        # df_ERA5["v_a"] += (
        #     np.sign(((df.v_a - df_ERA5.v_a)).mean())
        #     * ((df.v_a - df_ERA5.v_a) ** 2).mean() ** 0.5
        # )
        # df_ERA5["RH"] += (
        #     np.sign(((df.RH - df_ERA5.RH)).mean())
        #     * ((df.RH - df_ERA5.RH) ** 2).mean() ** 0.5
        # )

        df.loc[df["T_a"].isnull(), ["T_a", "RH", "v_a", "p_a", "Discharge"]] = df_ERA5[
            ["T_a", "RH", "v_a", "p_a", "Discharge"]
        ]

        df.loc[df["v_a"].isnull(), "missing"] = 2
        df.loc[df["v_a"].isnull(), "v_a"] = df_ERA5["v_a"]

        df[["SW_direct", "SW_diffuse", "cld"]] = df_ERA5[
            ["SW_direct", "SW_diffuse", "cld"]
        ]

        # RMSE

        # print("RMSE Temp", ((df.T_a - df_ERA5.T_a) ** 2).mean() ** 0.5)
        # print("RMSE wind", ((df.v_a - df_ERA5.v_a) ** 2).mean() ** 0.5)
        # print("RMSE pressure", ((df.p_a - df_ERA5.p_a) ** 2).mean() ** 0.5)

        # Fill from Plaffeien
        df_in2 = pd.read_csv(
            os.path.join(FOLDERS["raw_folder"], "plaffeien_aws.txt"),
            sep=";",
            skiprows=2,
        )
        df_in2["When"] = pd.to_datetime(df_in2["time"], format="%Y%m%d%H%M")

        df_in2["Prec"] = pd.to_numeric(df_in2["rre150z0"], errors="coerce")
        df_in2["p_a"] = pd.to_numeric(df_in2["prestas0"], errors="coerce")
        df_in2["RH"] = pd.to_numeric(df_in2["ure200s0"], errors="coerce")
        df_in2["v_a"] = pd.to_numeric(df_in2["fkl010z0"], errors="coerce")
        df_in2["T_a"] = pd.to_numeric(df_in2["tre200s0"], errors="coerce")

        df_in2["Prec"] = df_in2["Prec"] / (10 * 60)  # ppt rate mm/s
        df_in2 = (
            df_in2.set_index("When")
            .resample("5T")
            .interpolate(method="linear")
            .reset_index()
        )

        mask = (df_in2["When"] >= SITE["start_date"]) & (
            df_in2["When"] <= SITE["end_date"]
        )
        df_in2 = df_in2.loc[mask]
        df_in2 = df_in2.set_index("When")
        df["Prec"] = df_in2["Prec"]
        df = df.reset_index()
        df_in2 = df_in2.reset_index()

        # df.loc[df["T_a"].isnull(), ["T_a", "RH", "p_a", "Discharge"]] = df_in2[
        #     ["T_a", "RH", "p_a", "Discharge"]
        # ]
        # df["v_a"] = df["v_a"].replace(0, np.NaN)
        # df.loc[df["v_a"].isnull(), "missing"] = 2
        # df.loc[df["v_a"].isnull(), "v_a"] = df_in2["v_a"]

        slope, intercept, r_value1, p_value, std_err = stats.linregress(
            df.T_a.values, df_in3.T_a.values
        )
        slope, intercept, r_value2, p_value, std_err = stats.linregress(
            df.v_a.values, df_in3.v_a.values
        )
        slope, intercept, r_value3, p_value, std_err = stats.linregress(
            df.RH.values, df_in3.RH.values
        )
        slope, intercept, r_value4, p_value, std_err = stats.linregress(
            df.p_a.values, df_in3.p_a.values
        )

        print("R2 temp", r_value1 ** 2)
        print("R2 wind", r_value2 ** 2)
        print("R2 RH", r_value3 ** 2)
        print("R2 pressure", r_value4 ** 2)

        """Discharge Rate"""
        df["Fountain"], df["Discharge"] = discharge_rate(df, FOUNTAIN)

        df["Discharge"] = df["Discharge"] * df["Fountain"]

        df_out = df[
            [
                "When",
                "T_a",
                "RH",
                "v_a",
                "Discharge",
                "SW_direct",
                "SW_diffuse",
                "Prec",
                "p_a",
                "cld",
                "missing",
            ]
        ]

        if df_out.isnull().values.any():
            print("Warning: Null values present")
            print(
                df[
                    [
                        "When",
                        "T_a",
                        "RH",
                        "v_a",
                        "Discharge",
                        "SW_direct",
                        "SW_diffuse",
                        "Prec",
                        "p_a",
                        "cld",
                        "missing",
                    ]
                ]
                .isnull()
                .sum()
            )

        df_out = df_out.round(5)
        df_out.loc[df_out["v_a"] < 0, "v_a"] = 0

        df_out.to_csv(FOLDERS["input_folder"] + "raw_input.csv")

        # Extend data
        df_ERA5["Prec"] = 0
        df_ERA5["missing"] = 1
        df_ERA5 = df_ERA5.reset_index()
        mask = (df_ERA5["When"] >= df_out["When"].iloc[-1]) & (
            df_ERA5["When"] <= datetime(2019, 5, 30)
        )
        df_ERA5 = df_ERA5.loc[mask]

        mask = (df_in2["When"] >= SITE["start_date"]) & (
            df_in2["When"] <= df_ERA5["When"].iloc[-1]
        )
        df_in2 = df_in2.loc[mask]
        df_in2 = df_in2.set_index("When")

        df_out = df_out.set_index("When")
        df_ERA5 = df_ERA5.set_index("When")

        df_ERA5 = df_ERA5.drop(["LW_in"], axis=1)
        df_ERA5["Prec"] = df_in2["Prec"]
        concat = pd.concat([df_out, df_ERA5])
        concat.loc[concat["Prec"].isnull(), "Prec"] = 0
        concat.loc[concat["v_a"] < 0, "v_a"] = 0
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
                        "Discharge",
                        "SW_direct",
                        "SW_diffuse",
                        "Prec",
                        "p_a",
                        "cld",
                        "missing",
                    ]
                ]
                .isnull()
                .sum()
            )

        concat.to_csv(FOLDERS["input_folder"] + "raw_input_extended.csv")
        concat.to_hdf(
            FOLDERS["input_folder"] + "raw_input_extended.h5", key="df", mode="w"
        )

        pp = PdfPages(FOLDERS["input_folder"] + "compare" + ".pdf")
        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        ax1.scatter(df_out.p_a, df_in3.p_a, s=2)
        ax1.set_xlabel("AWS p")
        ax1.set_ylabel("ERA5 p")
        ax1.grid()
        lims = [
            np.min([ax1.get_xlim(), ax1.get_ylim()]),  # min of both axes
            np.max([ax1.get_xlim(), ax1.get_ylim()]),  # max of both axes
        ]
        # now plot both limits against eachother
        ax1.plot(lims, lims, "--k", alpha=0.25, zorder=0)
        ax1.set_aspect("equal")
        ax1.set_xlim(lims)
        ax1.set_ylim(lims)
        pp.savefig(bbox_inches="tight")
        plt.clf()

        ax1 = fig.add_subplot(111)
        ax1.scatter(df_out.v_a, df_in3.v_a, s=2)
        ax1.set_xlabel("AWS v")
        ax1.set_ylabel("ERA5 v")
        ax1.grid()
        lims = [
            np.min([ax1.get_xlim(), ax1.get_ylim()]),  # min of both axes
            np.max([ax1.get_xlim(), ax1.get_ylim()]),  # max of both axes
        ]
        # now plot both limits against eachother
        ax1.plot(lims, lims, "--k", alpha=0.25, zorder=0)
        ax1.set_aspect("equal")
        ax1.set_xlim(lims)
        ax1.set_ylim(lims)
        pp.savefig(bbox_inches="tight")
        plt.clf()

        ax1 = fig.add_subplot(111)
        ax1.scatter(df_out.T_a, df_in3.T_a, s=2)
        ax1.set_xlabel("AWS T")
        ax1.set_ylabel("ERA5 T")
        ax1.grid()
        lims = [
            np.min([ax1.get_xlim(), ax1.get_ylim()]),  # min of both axes
            np.max([ax1.get_xlim(), ax1.get_ylim()]),  # max of both axes
        ]
        # now plot both limits against eachother
        ax1.plot(lims, lims, "--k", alpha=0.25, zorder=0)
        ax1.set_aspect("equal")
        ax1.set_xlim(lims)
        ax1.set_ylim(lims)
        pp.savefig(bbox_inches="tight")
        plt.clf()

        ax1 = fig.add_subplot(111)
        ax1.scatter(df_in2.Prec, df_in3.Prec, s=2)
        ax1.set_xlabel("Plf ppt")
        ax1.set_ylabel("ERA5 ppt")
        ax1.grid()
        lims = [
            np.min([ax1.get_xlim(), ax1.get_ylim()]),  # min of both axes
            np.max([ax1.get_xlim(), ax1.get_ylim()]),  # max of both axes
        ]
        # now plot both limits against eachother
        ax1.plot(lims, lims, "--k", alpha=0.25, zorder=0)
        ax1.set_aspect("equal")
        ax1.set_xlim(lims)
        ax1.set_ylim(lims)
        pp.savefig(bbox_inches="tight")
        plt.clf()
        pp.close()
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
