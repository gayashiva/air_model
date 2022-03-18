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
import logging, coloredlogs

# Locals
dirname = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.append(dirname)
from src.utils.settings import config
from src.utils import setup_logger
# from src.automate.gen_auto_eqn import autoLinear
from src.models.methods.solar import get_solar
from src.models.icestupaClass import Icestupa
# from src.automate.autoDischarge import TempFreeze
# from lmfit.models import GaussianModel
from src.automate.autoDischarge import Scheduler

# Module logger
# logger = logging.getLogger("__main__")

def get_discharge(loc):  # Provides discharge info based on trigger setting

    with open("data/common/constants.json") as f:
        CONSTANTS = json.load(f)

    print(loc)

    # sprays = ['man', 'scheduled', "scheduled_field"]
    # sprays = ['scheduled', 'static', 'manual']
    # sprays = ["unscheduled_field","scheduled_field", "scheduled_icv", "scheduled_wue"]
    sprays = ["scheduled_icv", "scheduled_wue"]
    # sprays = ["unscheduled_field", "scheduled_icv", "scheduled_wue"]
    # sprays = ['manual']
     

    SITE, FOLDER = config(loc)

    times = pd.date_range(
        SITE["start_date"],
        SITE["expiry_date"],
        freq=(str(int(CONSTANTS["DT"] / 60)) + "T"),
    )
    df = pd.DataFrame(index=times, columns=sprays)
    df = df.fillna(0)
    df = df.reset_index()
    df.rename(columns = {'index':'time'}, inplace = True)

    # cld, df_solar = get_solar(
    #     coords=SITE["coords"],
    #     start=SITE["start_date"],
    #     end=SITE["expiry_date"],
    #     DT=CONSTANTS["DT"],
    #     alt=SITE["alt"],
    # )

    for spray in sprays:
        SITE, FOLDER = config(loc, spray)
        print(spray)

        if spray.split('_')[0] == "scheduled":
            obj = spray.split('_')[1]

            if obj == "field":
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
                # logger.warning("scheduled Discharge mean %.1f" % self.D_F)
                # D_F = self.df.Discharge[self.df.Discharge != 0].mean()
                # logger.warning("Manual Discharge mean %.1f" % self.D_F)
                # logger.warning("Manual Discharge used")

            if obj in ["wue", "icv"]:
                input_file = FOLDER["input"] + "aws.csv"
                df_aws = pd.read_csv(input_file, sep=",", header=0, parse_dates=["time"])

                for i in range(0,df_aws.shape[0]):
                    # df.loc[i, "scheduled_"+obj] = TempFreeze(data) + model.eval(x=df.time.dt.hour[i], **params)

                    df.loc[i, "scheduled_"+obj] = Scheduler(time=df_aws.time[i], temp=df_aws.temp[i], rh=df_aws.RH[i],
                                                            wind=df_aws.wind[i], obj=obj, r=SITE["R_F"],
                                                            alt=SITE["alt"], coords=SITE["coords"],
                                                            utc=SITE["utc"])
                    if df.loc[i, "scheduled_"+obj] > 0:
                        if obj == "icv":
                            if df.loc[i, "scheduled_"+obj] < SITE["dis_crit"]:
                                df.loc[i, "scheduled_"+obj] += SITE["dis_crit"]
                        if obj == "wue":
                            if df.loc[i, "scheduled_"+obj] < SITE["dis_crit"]:
                                df.loc[i, "scheduled_"+obj] = 0
                    else:
                        df.loc[i, "scheduled_"+obj] = 0

                    # if df.scheduled[i] >= SITE["dis_max"]:
                    #     df.loc[i, "scheduled"] = SITE["dis_max"]

                # df["static"] = df["scheduled"].max()
                
                # logger.warning("Static discharge is %s" %df.scheduled.max())
                # logger.warning("scheduled discharge varies from %s to %s" %(df.scheduled.min(), df.scheduled.max()))
                # SITE["D_F"] = df.scheduled[df.scheduled != 0].mean()

        if spray.split('_')[0] == "unscheduled":
            if spray.split('_')[1] == "field":
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

                    # df_h = pd.DataFrame(SITE["f_heights"])
                    # df[spray] = SITE["dis_max"]
                    # dis_old= SITE["dis_max"]
                    # for i in range(1,df_h.shape[0]):
                    #     h_change = round(df_h.h_f[i] - df_h.h_f[i-1],0)
                    #     print(h_change)
                    #     dis_new = dis_old/math.pow(2, h_change)
                    #     df.loc[df.time > df_h.time[i], spray] *= dis_new/dis_old
                    #     dis_old = dis_new

                    df = df.set_index("time")
                    df.loc[df_f.index, spray] = SITE["D_F"] * df_f["fountain"]
                    df = df.reset_index()

                if loc in ["guttannen22", "guttannen21", "guttannen20"]:
                    df_h = pd.DataFrame(SITE["f_heights"])

                    if loc in ["guttannen22"]:
                        df_f = pd.read_csv(
                            os.path.join("data/" + loc + "/interim/")
                            + "discharge_labview.csv",
                            sep=",",
                            parse_dates=["time"],
                        )
                        df_f = df_f.set_index("time")
                        SITE["dis_max"] = df_f["Discharge"].max()

                    df[spray] = SITE["dis_max"]
                    dis_old= SITE["dis_max"]
                    df.loc[df.time < df_h.time[0], spray] = 0
                    for i in range(1,df_h.shape[0]):
                        h_change = round(df_h.h_f[i] - df_h.h_f[i-1],0)
                        dis_new = dis_old/math.pow(2, h_change)
                        df.loc[df.time > df_h.time[i], spray] *= dis_new/dis_old
                        logger.warning("Discharge changed from %s to %s"%(dis_old,dis_new))
                        dis_old = dis_new
                    # SITE, FOLDER = config(loc, spray)
                    # df[spray] = SITE["D_F"]
                    # logger.info("Discharge constant")


        # if spray != "scheduledWUE":
        #     mask = df["time"] > SITE["fountain_off_date"]
        #     mask_index = df[mask].index
        #     df.loc[mask_index, spray] = 0

    df.to_csv(FOLDER["input"]  + "discharge_types.csv", index=True)
    return df


if __name__ == "__main__":
    # Main logger
    logger = logging.getLogger(__name__)
    logger.setLevel("WARNING")
    # logger.setLevel("INFO")


    with open("data/common/alt_coeffs.json") as f:
        param_values = json.load(f)

    # locations = ["gangles21", "guttannen21", "guttannen20", "guttannen22"]
    locations = ["guttannen22"]
    # locations = ["gangles21"]

    df = get_discharge(locations[0])
    print(df.head())

    for loc in locations:
        SITE, FOLDER = config(loc)

        df= df[df.time <= SITE["fountain_off_date"]]
       
        fig, ax1 = plt.subplots()
        x = df.time
        y1 = df.unscheduled_field
        y12 = df.scheduled_field
        y2 = df.scheduled_icv
        y3 = df.scheduled_wue
        ax1.plot(
            x,
            y1,
            linestyle="-",
            label = "Unscheduled field"
        )
        ax1.plot(
            x,
            y12,
            linestyle="-",
            label = "Scheduled field"
        )
        ax1.plot(
            x,
            y2,
            linestyle="--",
            label = "Scheduled ICV"
        )
        ax1.plot(
            x,
            y3,
            linestyle="-.",
            label = "Scheduled WUE"
        )
        ax1.set_ylabel("Discharge [$l/min$]")
        ax1.xaxis.set_major_locator(mdates.WeekdayLocator())
        ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
        ax1.xaxis.set_minor_locator(mdates.DayLocator())
        ax1.set_ylim([0,15])
        ax1.legend()
        fig.autofmt_xdate()
        plt.savefig(FOLDER["fig"] + "dis_types.png", dpi=300)

        # SITE, FOLDER = config(loc, spray="manual")
        # icestupa = Icestupa(loc, spray="manual")
        # icestupa.read_output()
        # dfi = icestupa.df
        # print(dfi.fountain_froze.describe()/60)
        # fig, ax1 = plt.subplots()
        # x = df.time
        # x1 = dfi.time
        # # y1 = df.manual
        # y1 = dfi.fountain_froze /60
        # y2 = df.scheduled
        # # y3 = df.scheduled_field
        # ax1.plot(
        #     x1,
        #     y1,
        #     linestyle="-",
        # )
        # ax1.plot(
        #     x,
        #     y2,
        #     linestyle="--",
        # )
        # # ax1.plot(
        # #     x,
        # #     y3,
        # #     linestyle="--",
        # # )
        # ax1.set_ylabel(loc + " discharge [$l/min$]")
        # ax1.xaxis.set_major_locator(mdates.WeekdayLocator())
        # ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
        # ax1.xaxis.set_minor_locator(mdates.DayLocator())
        # fig.autofmt_xdate()
        # plt.savefig(FOLDER["fig"] + "dis_types.png")


