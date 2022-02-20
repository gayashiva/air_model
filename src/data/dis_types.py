"""Icestupa function that returns discharge rate as csv file
"""
import os, sys, time, json
import pandas as pd
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import math
from datetime import datetime
import numpy as np
from tqdm import tqdm
from functools import lru_cache
import logging
import coloredlogs
from lmfit.models import GaussianModel
import pytz

# Locals
dirname = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.append(dirname)
from src.utils.settings import config
from src.utils import setup_logger
import logging, coloredlogs

# Module logger
# logger = logging.getLogger("__main__")

def autoDis(a, b, c, d, amplitude, center, sigma, temp, time, rh, v):
    model = GaussianModel()
    params = {"amplitude": amplitude, "center": center, "sigma": sigma}
    return a * temp + b * rh + c * v + d + model.eval(x=time, **params)

def get_discharge(loc):  # Provides discharge info based on trigger setting

    with open("data/common/constants.json") as f:
        CONSTANTS = json.load(f)


    # locations = ["guttannen21", "guttannen20", "gangles21"]

    # for loc in locations:

    print(loc)
    SITE, FOLDER = config(loc, spray="man")

    time = pd.date_range(
        SITE["start_date"],
        SITE["expiry_date"],
        freq=(str(int(CONSTANTS["DT"] / 60)) + "T"),
    )
    sprays = ['man', 'auto', "man_field", "auto_field"]
     
    df = pd.DataFrame(index=time, columns=sprays)
    df = df.fillna(0)
    df = df.reset_index()
    df.rename(columns = {'index':'time'}, inplace = True)
    print(df.head())

    for spray in sprays:

        print(spray)
        if spray == "auto":
            SITE, FOLDER = config(loc, spray)

            with open(FOLDER["input"] + "auto/coeffs.json") as f:
                param_values = json.load(f)

            input_file = FOLDER["input"] + "aws.csv"
            df_aws = pd.read_csv(input_file, sep=",", header=0, parse_dates=["time"])

            for i in range(0,df_aws.shape[0]):
                df.loc[i, "auto"] = autoDis(**param_values, time=df_aws.time.dt.hour[i],
                                            temp=df_aws.temp[i],rh=df_aws.RH[i], v=df_aws.wind[i])
                df.loc[i, "auto"] *= math.pi * math.pow(SITE["R_F"],2)
                if df.auto[i] < SITE["dis_crit"]:
                    df.loc[i, "auto"] = 0
                if df.auto[i] >= SITE["dis_max"]:
                    df.loc[i, "auto"] = SITE["dis_max"]
            logger.warning(df.auto.describe())
            # SITE["D_F"] = df.auto[df.auto != 0].mean()

        if spray == "man":
            if loc  == "gangles21":
                SITE, FOLDER = config(loc, spray)
                df_f = pd.read_csv(
                    os.path.join("data/" + loc + "/raw/")
                    + loc
                    + "_fountain_runtime.csv",
                    sep=",",
                    index_col=False,
                )
                df_f = df_f.rename(columns={"When": "time"})
                df_f["time"] = pd.to_datetime(df_f["time"], format="%b-%d %H:%M")
                df_f["time"] += pd.DateOffset(years=121)
                df_f = (
                    df_f.set_index("time")
                    .resample(str(int(CONSTANTS["DT"] / 60)) + "T")
                    .ffill()
                    .reset_index()
                )

                mask = df_f["time"] >= SITE["start_date"]
                mask &= df_f["time"] <= SITE["expiry_date"]
                df_f = df_f.loc[mask]
                df_f = df_f.reset_index(drop=True)
                df_f = df_f.set_index("time")

                df = df.set_index("time")
                df.loc[df_f.index, spray] = SITE["D_F"] * df_f["fountain"]
                df = df.reset_index()


            if loc in ["guttannen21", "guttannen20", "guttannen22"]:
                SITE, FOLDER = config(loc, spray)
                df[spray] = SITE["D_F"]
                logger.info("Discharge constant")

        if loc == "guttannen22":
            if spray == "auto_field":
                SITE, FOLDER = config(loc, spray)
                df_f = pd.read_csv(
                    os.path.join("data/" + loc + "/interim/")
                    + "discharge_labview.csv",
                    sep=",",
                    parse_dates=["time"],
                )
                df_f = df_f.set_index("time")
                df = df.set_index("time")
                df[spray] = df_f["Discharge"]
                df = df.reset_index()
                df= df.replace(np.NaN, 0)
                # D_F = self.df.Discharge[self.df.Discharge != 0].mean()
                # logger.warning("Auto Discharge mean %.1f" % self.D_F)

            if spray == "man_field":
                SITE, FOLDER = config(loc, spray)
                df_f = pd.read_csv(
                    os.path.join("data/" + loc + "/interim/")
                    + "discharge_labview.csv",
                    sep=",",
                    parse_dates=["time"],
                )
                df_f = df_f.set_index("time")

                f_heights = [
                    {"time": SITE["start_date"], "h_f": 3},
                    {"time": datetime(2021, 12, 23, 16), "h_f": 4},
                    {"time": datetime(2022, 1, 3, 16), "h_f": 5},
                ]
                df_h = pd.DataFrame(f_heights)

                df[spray] = df_f.Discharge.max()
                dis_old= df_f.Discharge.max()
                for i in range(1,df_h.shape[0]):
                    dis_new = dis_old/2
                    df.loc[df.time > df_h.time[i], spray] *= dis_new/dis_old
                    dis_old = dis_new

                # D_F = self.df.Discharge[self.df.Discharge != 0].mean()
                # logger.warning("Manual Discharge mean %.1f" % self.D_F)
                # logger.warning("Manual Discharge used")

        if spray != "auto":
            mask = df["time"] > SITE["fountain_off_date"]
            mask_index = df[mask].index
            df.loc[mask_index, spray] = 0

    df.to_csv(FOLDER["input"]  + "discharge_types.csv", index=True)
    return df


if __name__ == "__main__":
    # Main logger
    logger = logging.getLogger(__name__)
    logger.setLevel("WARNING")
    # logger.setLevel("INFO")

    locations = ["gangles21", "guttannen21", "guttannen20", "guttannen22"]
    # locations = ["guttannen22"]

    for loc in locations:
        df = get_discharge(loc)
        SITE, FOLDER = config(loc, spray="man")
        fig, ax1 = plt.subplots()
        x = df.time
        y1 = df.auto_field
        y2 = df.man_field
        ax1.plot(
            x,
            y1,
            linestyle="-",
        )
        ax1.plot(
            x,
            y2,
            linestyle="--",
        )
        ax1.set_ylabel(loc + " discharge [$l/min$]")
        ax1.xaxis.set_major_locator(mdates.WeekdayLocator())
        ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
        ax1.xaxis.set_minor_locator(mdates.DayLocator())
        fig.autofmt_xdate()
        plt.savefig(FOLDER["input"] + "dis_types.png")


