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
from scipy import stats
from sklearn.linear_model import LinearRegression

dirname = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.append(dirname)
# from src.data.config import SITE, FOLDERS, FOUNTAIN
from src.data.settings import config

import logging
import coloredlogs

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
coloredlogs.install(
    fmt="%(name)s %(levelname)s %(message)s",
    logger=logger,
)

start = time.time()


def field(site="schwarzsee"):
    if site == "guttannen":
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

    if site == "schwarzsee":
        df_in = pd.read_csv(
            raw_folder + SITE["name"] + "_aws.txt",
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


def era5(df, site="schwarzsee"):
    df_in3 = pd.read_csv(
        # FOLDERS["raw_folder"] + "ERA5_" + SITE["name"] + ".csv",
        # "/home/suryab/work/ERA5/results/schwarzsee_2019.csv",
        "/home/suryab/work/ERA5/outputs/" + site + "_2021.csv",
        sep=",",
        header=0,
        parse_dates=["When"],
    )
    df_in2 = pd.read_csv(
        "/home/suryab/work/ERA5/outputs/" + site + "_2020.csv",
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

    # interpolated["Discharge"] = 0
    # interpolated["Discharge"] = discharge_rate(interpolated, FOUNTAIN)

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
            # "Discharge",
        ]
    ]
    # df_ERA5.loc[:, "Discharge"] = 0

    logger.debug(df_ERA5.head())
    logger.debug(df_ERA5.tail())
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


def meteoswiss(site="schwarzsee"):

    if site == "schwarzsee":
        site = "plaffeien"
    df = pd.read_csv(
        os.path.join(raw_folder, site + "_meteoswiss.txt"),
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
    return df


if __name__ == "__main__":

    SITE, FOUNTAIN = config("Guttannen")

    raw_folder = os.path.join(dirname, "data/" + SITE["name"] + "/raw/")
    input_folder = os.path.join(dirname, "data/" + SITE["name"] + "/interim/")

    # df = field(site=SITE["name"])
    df = meteoswiss(SITE["name"])

    df_ERA5, df_in3 = era5(df, SITE["name"])

    df_ERA5 = df_ERA5.set_index("When")
    df = df.set_index("When")
    df["missing"] = 0
    df["missing_type"] = ""
    # df.loc[df["T_a"].isnull(), "missing"] = 1
    # df["v_a"] = df["v_a"].replace(0, np.NaN)

    df_ERA5 = df_ERA5.reset_index()
    mask = (df_ERA5["When"] >= SITE["start_date"]) & (
        df_ERA5["When"] <= SITE["end_date"]
    )

    # Fit ERA5 to field data
    # for column in ["T_a", "RH", "v_a", "p_a"]:
    #     Y = df[column].values.reshape(-1, 1)
    #     X = df_ERA5[mask][column].values.reshape(-1, 1)
    #     slope, intercept = linreg(X, Y)
    #     df_ERA5[column] = slope * df_ERA5[column] + intercept

    df_ERA5 = df_ERA5.set_index("When")

    # Fill from ERA5
    logger.debug(df.loc[df["T_a"].isnull()])
    for col in ["T_a", "RH", "v_a"]:
        df.loc[df[col].isnull(), "missing"] = 1
        df.loc[df[col].isnull(), "missing_type"] = col
        # logger.warning("%s converted to %0.1f" %(col, convertToNumber(col)))
        df.loc[df[col].isnull(), col] = df_ERA5[col]

    for col in ["p_a", "SW_direct", "SW_diffuse", "LW_in"]:
        logger.info("%s from ERA5" % col)
        df[col] = df_ERA5[col]

    # df_in2 = meteoswiss(SITE["name"])

    # df_in2 = df_in2.set_index("When")

    # df["Prec"] = df_in2["Prec"]
    # df_in2 = df_in2.reset_index()
    df = df.reset_index()

    # Compare ERA5 and field data
    # for column in ["T_a", "RH", "v_a", "p_a"]:

    #     slope, intercept, r_value, p_value, std_err = stats.linregress(
    #         df[column].values, df_in3[column].values
    #     )
    #     print("ERA5", column, r_value)
    # slope, intercept, r_value, p_value, std_err = stats.linregress(
    #     df[column].values, df_in2[column].values
    # )
    # print("Plf", column, r_value)

    df_out = df[
        [
            "When",
            "T_a",
            "RH",
            "v_a",
            # "Discharge",
            "SW_direct",
            "SW_diffuse",
            "Prec",
            "vp_a",
            "p_a",
            "missing",
            "LW_in",
        ]
    ]

    # df_out.loc[df_out["Prec"].isnull(), "Prec"] = 0
    if df_out.isnull().values.any():
        print("Warning: Null values present")
        print(
            df_out[
                [
                    "When",
                    "T_a",
                    "RH",
                    "v_a",
                    # "Discharge",
                    "SW_direct",
                    "SW_diffuse",
                    "Prec",
                    "vp_a",
                    "p_a",
                    "missing",
                    "LW_in",
                ]
            ]
            .isnull()
            .sum()
        )

    df_out.loc[:, "Prec"] = df_out.Prec.interpolate()
    df_out.loc[:, "vp_a"] = df_out.Prec.interpolate()

    df_out = df_out.round(3)
    # df_out.loc[df_out["v_a"] < 0, "v_a"] = 0
    logger.error(df_out[df_out.index.duplicated()])
    logger.info(df_out.tail())
    df_out.to_csv(input_folder + SITE["name"] + "_input_model.csv")

    # Extend data
    # df_ERA5["Prec"] = 0
    # df_ERA5["missing"] = 1
    # df_ERA5 = df_ERA5.reset_index()
    # mask = (df_ERA5["When"] > df_out["When"].iloc[-1]) & (
    #     df_ERA5["When"] <= datetime(2019, 5, 30)
    # )
    # df_ERA5 = df_ERA5.loc[mask]

    # mask = (df_in2["When"] >= SITE["start_date"]) & (
    #     df_in2["When"] <= SITE["end_date"]
    # )
    # df_in2 = df_in2.loc[mask]
    # df_in2 = df_in2.set_index("When")

    # df_out = df_out.set_index("When")
    # df_ERA5 = df_ERA5.set_index("When")

    # df_ERA5["Prec"] = df_in2["Prec"]
    # concat = pd.concat([df_out, df_ERA5])
    # concat.loc[concat["Prec"].isnull(), "Prec"] = 0
    # concat.loc[concat["v_a"] < 0, "v_a"] = 0
    # logger.error(concat[concat.index.duplicated()])
    # logger.info(concat.tail())

    # concat = concat.reset_index()

    # if concat.isnull().values.any():
    #     print("Warning: Null values present")
    #     print(
    #         concat[
    #             [
    #                 "When",
    #                 "T_a",
    #                 "RH",
    #                 "v_a",
    #                 # "Discharge",
    #                 "SW_direct",
    #                 "SW_diffuse",
    #                 "Prec",
    #                 "p_a",
    #                 "missing",
    #             ]
    #         ]
    #         .isnull()
    #         .sum()
    #     )

    # concat.to_csv(input_folder + SITE["name"] + "_input_model.csv")
    # concat.to_hdf(
    #     input_folder + SITE["name"] + "_input_model.h5",
    #     key="df",
    #     mode="w",
    # )

    # fig = plt.figure()
    # ax1 = fig.add_subplot(111)
    # ax1.scatter(df_in2.SW_g, df_in3.SW_g, s=2)
    # ax1.set_ylabel("ERA5 Global Solar radiation [$W\\,m^{-2}$]")
    # ax1.set_xlabel("Plaffeien Global Solar radiation [$W\\,m^{-2}$]")
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
    # # pp.savefig(bbox_inches="tight")
    # plt.savefig(FOLDERS["input_folder"] + "compare.jpg", dpi=300, bbox_inches="tight")
    # plt.clf()

    # ax1 = fig.add_subplot(111)
    # ax1.scatter(df_out.v_a, df_in3.v_a, s=2)
    # ax1.set_xlabel("AWS v")
    # ax1.set_ylabel("ERA5 v")
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
    # pp.savefig(bbox_inches="tight")
    # plt.clf()

    # ax1 = fig.add_subplot(111)
    # ax1.scatter(df_out.T_a, df_in3.T_a, s=2)
    # ax1.set_xlabel("AWS T")
    # ax1.set_ylabel("ERA5 T")
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
    # pp.savefig(bbox_inches="tight")
    # plt.clf()

    # ax1 = fig.add_subplot(111)
    # ax1.scatter(df_in2.Prec, df_in3.Prec, s=2)
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
    # pp.savefig(bbox_inches="tight")

    # plt.clf()
    # pp.close()
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
