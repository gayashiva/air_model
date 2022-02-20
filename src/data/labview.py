import sys, json
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
import glob
dirname = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.append(dirname)
from src.utils.settings import config
import re


def aws(location="guttannen22"):

    with open("data/common/constants.json") as f:
        CONSTANTS = json.load(f)

    SITE, FOLDER = config(location)
    # Get header colun list
    df = pd.read_csv(
        FOLDER["raw"] + location + "_aws.csv",
        # sep=";",
        sep="\t",
        header=0,
        # parse_dates=['date_time']
    )
    print(df.columns)
    logger.info(df['date_time'].head())
    logger.info(df['date_time'].tail())
    df["When"] = pd.to_datetime(df["date_time"], format="%d/%m/%y %H:%M")
    df.rename(
        columns={
            # "date_time": "When",
            " WndScMae": "v_a",
            " Tair_act_AVG": "T_a",
            " Rhum_act_AVG": "RH",
            "SWdown_cor": "SW_out",
            # " SWdown_AVG": "SW_out",
            "SWup_cor": "SW_in",
            # " SWup_AVG": "SW_in",
            # " LWdown_AVG": "LW_out",
            "LWdown_cor": "LW_out",
            # " LWup_AVG": "LW_in",
            "LWup_cor": "LW_in",
            " HS_act": "HS",
            " Baro_act_AVG": "p_a",
        },
        inplace=True,
    )
    df = df[["When", "v_a", "T_a", "RH", "SW_out", "SW_in", "LW_out", "LW_in", "HS", "p_a"]]
    # df = df[["When", "v_a", "T_a", "RH",  "HS", "p_a"]]

    df["a"] = df["SW_out"]/df["SW_in"]
    df.loc[df.a > 1, "a"] = np.NaN 
    df.loc[:, "a"] = df["a"].interpolate()
    df= df.set_index("When").sort_index()
    df= df.resample(pd.offsets.Minute(n=15)).mean().reset_index()

    # df["Prec"]= df.HS.diff()
    # df["Prec"] = df["Prec"] *10 / (15 * 60)  # ppt rate mm/s
    # df.loc[df.Prec < 0, "Prec"] = np.NaN 
    # df.loc[df.Prec *15*60 > 500, "Prec"] = np.NaN 
    # df.loc[df.Prec *15*60 < 4, "Prec"] = 0
    # logger.error(df.loc[df.Prec.isna(), "HS"].head())
    # df.loc[:, "Prec"] = df["Prec"].interpolate()

    # diffuse_fraction=0.25
    # df["SW_direct"] = (1-diffuse_fraction) * (df["SW_in"]) 
    # df["SW_diffuse"] = diffuse_fraction * (df["SW_in"]) 
    logger.warning(df.head())
    logger.warning(df.tail())

    fig, ax1 = plt.subplots()
    skyblue = "#9bc4f0"
    blue = "#0a4a97"
    x = df.time
    # y = df.Prec *15*60
    y = df.Discharge
    ax1.plot(
        x,
        y,
        linestyle="-",
        color=blue,
    )
    # ax1.set_ylabel("Temperature[$l\\, min^{-1}$]")
    ax1.set_ylabel("AWS ppt[$mm$]")
    ax1.xaxis.set_major_locator(mdates.WeekdayLocator())
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax1.xaxis.set_minor_locator(mdates.DayLocator())
    fig.autofmt_xdate()
    plt.savefig(FOLDER["input"]+ SITE["name"] + "test1.png")

    mask = (df["When"] >= SITE["start_date"]) & (
        df["When"] <= SITE["end_date"]
    )
    df= df.loc[mask]
    df= df.reset_index(drop=True)
    df = df.set_index("time")
    logger.info(pd.date_range(start = df.index[0], end = df.index[-1], freq='15T').difference(df.index))
    df = df.reset_index()
    return df

def labview(location):
    if location == "guttannen22":
        # path = "/home/suryab/ownCloud/Sites/Diavolezza/diavolezza_sdcard/"

        with open("data/common/constants.json") as f:
            CONSTANTS = json.load(f)
        SITE, FOLDER = config(location)

        path = FOLDER["raw"] + "auto/sdcard/"
        all_files = glob.glob(path + "*.txt")
        # print(all_files[0])

        li = []
        ctr = 0

        # # Get header colun list
        # df = pd.read_csv(
        #     all_files[0],
        #     sep=";",
        #     skiprows=3,
        #     usecols=["Q_Wasser ", "T_Luft", "r_Luft","w_Luft"],
        #     # header=0,
        #     encoding="latin-1",
        # )
        # df.rename(
        #     columns={
        #         "w_Luft": "v_a",
        #         "T_Luft": "T_a",
        #         "r_Luft": "RH",
        #         "Q_Wasser ": "Discharge",
        #     },
        #     inplace=True,
        # )
        # print(df.head())
        # sys.exit()
        # names = df.columns
        # names = names[:-3]

        for file in all_files:

            # var = re.split("[.|_| ]", file)
            # date = var[4:-1]
            # date = (' '.join(date))
            # date = re.sub(' +', ' ', date)
            # date= datetime.strptime(date, '%B %d %Y %I %M %S %p')
            # print(date)
            try:
                print(file)
                df = pd.read_csv(
                    file,
                    sep=";",
                    skiprows=3,
                    # header=0,
                    encoding="latin-1",
                    # usecols=names,
                    usecols=["Q_Wasser ", "T_Luft", "r_Luft","w_Luft"],
                )
                df = df[1:].reset_index(drop=True)
                df = df[["Q_Wasser ", "T_Luft", "r_Luft","w_Luft"]]
                for col in df.columns:
                    df[col] = df[col].astype(float)
                df = df.round(2)

                df.rename(
                    columns={
                        "w_Luft": "v_a",
                        "T_Luft": "T_a",
                        "r_Luft": "RH",
                        "Q_Wasser ": "Discharge",
                    },
                    inplace=True,
                )
                var = re.split("[.|_| ]", file)
                date = var[4:-1]
                date = (' '.join(date))
                date = re.sub(' +', ' ', date)
                date= datetime.strptime(date, '%B %d %Y %I %M %S %p')
                print(date)
                df["time"] = pd.to_datetime([date+timedelta(seconds=10 * h) for h in range(0,df.shape[0])])

                li.append(df)
            except:
                ctr += 1
                # a_file = open(file,encoding= 'latin-1')
                # line =a_file.readline()
                # line = line.split(";")[-1]
                # date = line.split("+")[0]
                # date = date[3:]
                # print(date)
                # date= datetime.strptime(date, ' %d %b %Y %I:%M:%S %p ')
                # print(date)
                the_type, the_value, the_traceback = sys.exc_info()
                print(the_type)
                pass

    df = pd.concat(li, axis=0, ignore_index=True)
    df = df.set_index("time").sort_index().reset_index()
    df= df.set_index("time").resample(pd.offsets.Minute(n=15)).mean().reset_index()


    # mask = (df["When"] >= SITE["start_date"]) & (
    #     df["When"] <= SITE["end_date"]
    # )
    # df= df.loc[mask]
    # df= df.reset_index(drop=True)

    print("Number of hours missing : %s" %ctr)

    # CSV output
    df.to_csv(FOLDER["raw"] + "labview.csv")
    print(df.tail(10))
    return df

if __name__ == "__main__":
    logger = logging.getLogger(__name__)
    coloredlogs.install(
        fmt="%(name)s %(levelname)s %(message)s",
        level=logging.INFO,
        logger=logger,
    )

    SITE, FOLDER = config("guttannen22", spray="auto")
    # df = aws()

    # sdcard = True
    sdcard = False
    
    if sdcard:
        df= labview("guttannen22")
    else:
        df= pd.read_csv(
                FOLDER["raw"] + "labview.csv",
                sep=",",
                header=0,
                parse_dates=["time"],
            )

    df = df.set_index("time")
    df = df[SITE["start_date"] : SITE["expiry_date"]]
    df = df[["Discharge"]]

    df= df.replace(np.NaN, 0)
    df = df.resample("H").mean()

    df.to_csv(FOLDER["input"] + "discharge_labview.csv")

    fig, ax = plt.subplots()
    # x = df.time
    y = df.Discharge
    ax.plot(y)
    # ax.set_ylim(0,10)

    # y1 = df["Tice_Avg(1)"]
    # y2 = df["Tice_Avg(3)"]
    # y3 = df["Tice_Avg(5)"]
    # ax.set_ylabel("Temp")
    # ax.plot(x,y1, label="1")
    # ax.plot(x,y2, label="2")
    # ax.plot(x,y3, label="3")
    # ax.legend()

    ax.xaxis.set_major_locator(mdates.WeekdayLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax.xaxis.set_minor_locator(mdates.DayLocator())
    fig.autofmt_xdate()
    plt.savefig(
        FOLDER['fig'] + "discharge.jpg",
        bbox_inches="tight",
    )
    plt.clf()


    # df_swiss = meteoswiss(SITE["name"])
    # df_ERA5, df_in3 = era5(SITE["name"])

    # df_ERA5 = df_ERA5.set_index("When")
    # df = df.set_index("When")
    # df_swiss = df_swiss.set_index("When")
    # df_field= df_field.set_index("When")

    # for col in ["Prec"]:
    #     logger.warning("%s from meteoswiss" % col)
    #     df[col] = df_swiss[col]

    # for col in ["Discharge"]:
    #     logger.warning("%s from field" % col)
    #     df[col] = df_field[col]


    # # Fit ERA5 to field data
    # if SITE["name"] in ["diavolezza21"]:
    #     fit_list = ["T_a", "RH", "v_a", "p_a"]

    # mask = df[fit_list].notna().any(axis=1).index

    # logger.warning(df.loc[mask][fit_list])

    # for column in fit_list:
    #     Y = df.loc[mask][column].values.reshape(-1, 1)
    #     X = df_ERA5.loc[mask][column].values.reshape(-1, 1)
    #     slope, intercept = linreg(X, Y)
    #     df_ERA5[column] = slope * df_ERA5[column] + intercept
    #     if column in ["v_a"]:
    #         # Correct negative wind
    #         df_ERA5.v_a.loc[df_ERA5.v_a<0] = 0

    # # Fill from ERA5
    # logger.warning("Temperature NaN percent: %0.2f" %(df["T_a"].isna().sum()/df.shape[0]*100))
    # logger.warning("wind NaN percent: %0.2f" %(df["v_a"].isna().sum()/df.shape[0]*100))

    # df['missing_type'] = ''

    # for col in ["T_a", "RH", "v_a", "Prec", "p_a", "SW_direct", "SW_diffuse", "LW_in"]:
    #     try:
    #         mask = df[col].isna()
    #         percent_nan = df[col].isna().sum()/df.shape[0] * 100
    #         logger.info(" %s has %s percent NaN values" %(col, percent_nan))
    #         if percent_nan > 1:
    #             logger.warning(" Null values filled with ERA5 in %s" %col)
    #             df.loc[df[col].isna(), "missing_type"] = df.loc[df[col].isna(), "missing_type"] + col
    #             df.loc[df[col].isna(), col] = df_ERA5[col]
    #         elif percent_nan > 0:
    #             logger.warning(" Null values interpolated in %s" %col)
    #             df.loc[:, col] = df[col].interpolate()
    #     except KeyError:
    #         logger.warning("%s from ERA5" % col)
    #         df[col] = df_ERA5[col]
    #         df["missing_type"] = df["missing_type"] + col
    # logger.info(df.missing_type.describe())
    # logger.info(df.missing_type.unique())

    # df = df.reset_index()

    # if SITE["name"] in ["diavolezza21"]:
    #     cols = [
    #         "When",
    #         "Discharge",
    #         "T_a",
    #         "RH",
    #         "v_a",
    #         "SW_direct",
    #         "SW_diffuse",
    #         "Prec",
    #         "p_a",
    #         "missing_type",
    #         "LW_in",
    #         # "a",
    #     ]

    # df_out = df[cols]

    # if df_out.isna().values.any():
    #     print(df_out[cols].isna().sum())
    #     for column in cols:
    #         if df_out[column].isna().sum() > 0 and column in ["a"]:
    #             albedo = df_out.a.replace(0, np.nan).mean()
    #             df_out.loc[df_out[column].isna(), column] = albedo
    #             logger.warning("Albedo Null values extrapolated in %s " %albedo)
    #         if df_out[column].isna().sum() > 0 and column in ["Discharge"]:
    #             discharge = df_out.Discharge.replace(0, np.nan).mean()

    #             # Logger switched off
    #             mask = df_out.When < datetime(2021,3,3) 
    #             mask &= df_out.When > datetime(2021,2,16) 
    #             mask &= df_out[column].isna()
    #             # logger.error(df_out.loc[mask].head())
    #             # logger.error(df_out.loc[mask].tail())
    #             df_out.loc[mask, column] = 0

    #             # Fill nan
    #             df_out.loc[df_out[column].isna(), column] = discharge

    #             # df_out = df_out.set_index("When")
    #             # df_out["Discharge"] = df_out.between_time('9:00', '18:00').Discharge.replace(np.nan, discharge)
    #             # df_out["Discharge"] = df_out.Discharge.replace(np.nan, 0)
    #             # df_out = df_out.reset_index()
    #             logger.warning(" Discharge Null values extrapolated in %s " %discharge)

    #         if df_out[column].isna().sum() > 0 and column not in ["missing_type", "Discharge"]:
    #             logger.warning(" Null values interpolated in %s" %column)
    #             df_out.loc[:, column] = df_out[column].interpolate()

    # df_out = df_out.round(3)
    # if len(df_out[df_out.index.duplicated()]):
    #     logger.error("Duplicate indexes")

    # logger.info(df_out.tail())
    # df_out.to_csv(FOLDER["input"]+ SITE["name"] + "_input_model.csv", index=False)

    # fig, ax1 = plt.subplots()
    # skyblue = "#9bc4f0"
    # blue = "#0a4a97"
    # x = df_out.When
    # y = df_out.Prec * 15 * 60
    # ax1.plot(
    #     x,
    #     y,
    #     linestyle="-",
    #     color=blue,
    # )
    # ax1.set_ylabel("Meteoswiss ppt[$mm$]")
    # ax1.xaxis.set_major_locator(mdates.WeekdayLocator())
    # ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    # ax1.xaxis.set_minor_locator(mdates.DayLocator())
    # fig.autofmt_xdate()
    # plt.savefig(FOLDER["input"]+ SITE["name"] + "test.png")


