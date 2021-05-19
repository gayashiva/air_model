 
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
import glob
dirname = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.append(dirname)
from src.utils.settings import config
from src.data.make_dataset import era5, linreg


def field(location="schwarzsee19"):
    if location == "diavolezza21":
        path = "/home/suryab/ownCloud/Sites/Diavolezza/diavolezza_sdcard/"
        all_files = glob.glob(path + "*.txt")

        SITE, FOLDER, df_h = config(location = "Diavolezza 2021")
        li = []
        ctr = 0

        # Get header colun list
        df = pd.read_csv(
            all_files[10],
            sep=";",
            skiprows=3,
            header=0,
            encoding="latin-1",
        )
        names = df.columns
        names = names[:-3]

        for file in all_files:

            try:
                df = pd.read_csv(
                    file,
                    sep=";",
                    skiprows=3,
                    header=0,
                    encoding="latin-1",
                    usecols=names,
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
                a_file = open(file,encoding= 'latin-1')
                line =a_file.readline()
                line = line.split(";")[-1]
                date = line.split("+")[0]
                date = date[3:]
                date= datetime.strptime(date, ' %d %b %Y %I:%M:%S %p ')
                df["When"] = pd.to_datetime([date+timedelta(seconds=10 * h) for h in range(0,df.shape[0])])

                li.append(df)
            except:
                ctr += 1
                a_file = open(file,encoding= 'latin-1')
                line =a_file.readline()
                line = line.split(";")[-1]
                date = line.split("+")[0]
                date = date[3:]
                date= datetime.strptime(date, ' %d %b %Y %I:%M:%S %p ')
                print(date)
                the_type, the_value, the_traceback = sys.exc_info()
                print(the_type)
                pass

    print("Number of hours missing : %s" %ctr)
    df_in = pd.concat(li, axis=0, ignore_index=True)
    df_in = df_in.set_index("When").sort_index().reset_index()

    # CSV output
    logger.info(df_in.head())
    logger.info(df_in.tail())
    df_in.to_csv(FOLDER["raw"] + SITE["name"] + "_field.csv")
    return df_in

if __name__ == "__main__":
    logger = logging.getLogger(__name__)
    coloredlogs.install(
        fmt="%(name)s %(levelname)s %(message)s",
        level=logging.INFO,
        logger=logger,
    )

    SITE, FOLDER, *args = config("Diavolezza 2021")
    # sdcard = True
    sdcard = False
    
    if sdcard:
        # raw_folder = os.path.join(dirname, "data/" + SITE["name"] + "/raw/")
        # input_folder = os.path.join(dirname, "data/" + SITE["name"] + "/interim/")
        df = field("diavolezza21")
    else:
        df = pd.read_csv(
                FOLDER["raw"]+ SITE["name"] + "_field.csv",
                sep=",",
                header=0,
                parse_dates=["When"],
            )

    df= df.set_index("When").resample(pd.offsets.Minute(n=30)).mean().reset_index()

#     days = pd.date_range(start=SITE["start_date"], end=SITE["end_date"], freq="30T")
#     days = pd.DataFrame({"When": days})
# 
#     df = pd.merge(
#         df_in[ df_in.columns
#         ],
#         days,
#         on="When",
#     )

    mask = (df["When"] >= SITE["start_date"]) & (
        df["When"] <= SITE["end_date"]
    )
    df= df.loc[mask]
    df= df.reset_index(drop=True)

    logger.info(df.head())
    logger.info(df.tail())
    logger.warning(df.loc[~df["Discharge"].isna(), "When"].head())

    df.loc[df.T_a < -30, "T_a"] = np.NaN

    df_ERA5, df_in3 = era5(df, SITE["name"])

    df_ERA5 = df_ERA5.set_index("When")
    df = df.set_index("When")

    # mask1 = (df_ERA5.index >= SITE["start_date"]) & (df_ERA5.index <= SITE["end_date"])

    # Fit ERA5 to field data
    if SITE["name"] in ["guttannen21", "guttannen20"]:
        fit_list = ["T_a", "RH", "v_a", "Prec"]

    if SITE["name"] in ["schwarzsee19"]:
        fit_list = ["T_a", "RH", "v_a", "p_a"]

    if SITE["name"] in ["diavolezza21"]:
        fit_list = ["T_a", "RH", "v_a"]

    mask = df[fit_list].notna().any(axis=1).index

    logger.warning(df_ERA5.loc[mask][fit_list])
    logger.warning(df.loc[mask][fit_list])

    for column in fit_list:
        Y = df.loc[mask][column].values.reshape(-1, 1)
        X = df_ERA5.loc[mask][column].values.reshape(-1, 1)
        slope, intercept = linreg(X, Y)
        df_ERA5[column] = slope * df_ERA5[column] + intercept
        if column in ["v_a"]:
            # Correct negative wind
            df_ERA5.v_a.loc[df_ERA5.v_a<0] = 0


    # Fill from ERA5
    logger.warning("Temperature NaN percent: %0.2f" %(df["T_a"].isna().sum()/df.shape[0]*100))
    logger.warning("wind NaN percent: %0.2f" %(df["v_a"].isna().sum()/df.shape[0]*100))

    df['missing_type'] = ''

    for col in ["T_a", "RH", "v_a", "Prec", "p_a", "SW_direct", "SW_diffuse", "LW_in"]:
        try:
            mask = df[col].isna()
            percent_nan = df[col].isna().sum()/df.shape[0] * 100
            logger.info(" %s has %s percent NaN values" %(col, percent_nan))
            if percent_nan > 1 or col in ["Prec"]:
                logger.warning(" Null values filled with ERA5 in %s" %col)
                df.loc[df[col].isna(), "missing_type"] = df.loc[df[col].isna(), "missing_type"] + col
                df.loc[df[col].isna(), col] = df_ERA5[col]
            else:
                logger.warning(" Null values interpolated in %s" %col)
                df.loc[:, col] = df[col].interpolate()
        except KeyError:
            logger.warning("%s from ERA5" % col)
            df[col] = df_ERA5[col]
            df["missing_type"] = df["missing_type"] + col
    logger.info(df.missing_type.describe())
    logger.info(df.missing_type.unique())


    # if SITE["name"] in ["schwarzsee19"]:
    #     for col in ["T_a", "RH", "v_a", "p_a"]:
    #         # df.loc[df[col].isna(), "missing"] = 1
    #         # df.loc[df[col].isna(), "missing_type"] = col
    #         df.loc[df[col].isna(), "missing_type"] = df.loc[df[col].isna(), "missing_type"] + col
    #         df.loc[df[col].isna(), col] = df_ERA5[col]

    #     for col in ["SW_direct", "SW_diffuse", "LW_in"]:
    #         logger.info("%s from ERA5" % col)
    #         df[col] = df_ERA5[col]

    df = df.reset_index()

    if SITE["name"] in ["diavolezza21"]:
        cols = [
            "When",
            "Discharge",
            "T_a",
            "RH",
            "v_a",
            "SW_direct",
            "SW_diffuse",
            "Prec",
            "p_a",
            "missing_type",
            "LW_in",
        ]

    if SITE["name"] in ["schwarzsee19"]:
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
            "missing_type",
            "LW_in",
        ]
    if SITE["name"] in ["guttannen20", "guttannen21"]:
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
            "missing_type",
            "LW_in",
        ]

    df_out = df[cols]

    if df_out.isna().values.any():
        print(df_out[cols].isna().sum())
        for column in cols:
            if df_out[column].isna().sum() > 0 and column in [ "Discharge"]:
                discharge = df_out.Discharge.replace(0, np.nan).mean()
                logger.warning(" Null values interpolated in %s set to 0" %discharge)
                df_out.loc[df_out[column].isna(), column] = discharge

            if df_out[column].isna().sum() > 0 and column not in ["missing_type", "Discharge"]:
                logger.warning(" Null values interpolated in %s" %column)
                df_out.loc[:, column] = df_out[column].interpolate()

    df_out = df_out.round(3)
    if len(df_out[df_out.index.duplicated()]):
        logger.error("Duplicate indexes")

    # df_out["Discharge"] += 1 #Discharge never zero

    logger.info(df_out.tail())
    df_out.to_csv(FOLDER["input"]+ SITE["name"] + "_input_model.csv")
    fig, ax1 = plt.subplots()
    skyblue = "#9bc4f0"
    blue = "#0a4a97"
    x = df_out.When
    y = df_out.T_a
    ax1.plot(
        x,
        y,
        linestyle="-",
        color=blue,
    )
    ax1.set_ylabel("Discharge [$l\\, min^{-1}$]")
    ax1.xaxis.set_major_locator(mdates.WeekdayLocator())
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax1.xaxis.set_minor_locator(mdates.DayLocator())
    fig.autofmt_xdate()
    plt.savefig(FOLDER["input"]+ SITE["name"] + "test.png")


