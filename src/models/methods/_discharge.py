"""Icestupa class function that returns Discharge rate column(Discharge)
"""
import os, sys, time, json
import pandas as pd
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
from src.automate.projectile import get_projectile

# Module logger
logger = logging.getLogger("__main__")
logger.propagate = False

def autoDis(a, b, c, d, amplitude, center, sigma, temp, time, rh, v):
    model = GaussianModel()
    params = {"amplitude": amplitude, "center": center, "sigma": sigma}
    return a * temp + b * rh + c * v + d + model.eval(x=time, **params)

def get_discharge(self):  # Provides discharge info based on trigger setting

    if self.name == "schwarzsee19":
        self.df["Discharge"] = 0
        logger.debug("Initialised discharge as zero")

        df_f = pd.read_csv(
            os.path.join("data/" + self.name + "/interim/")
            + self.name
            + "_input_field.csv"
        )
        df_f = df_f.rename(columns={"When": "time"})
        df_f["time"] = pd.to_datetime(df_f["time"], format="%Y.%m.%d %H:%M:%S")
        df_f = df_f.set_index("time").resample(str(int(self.DT / 60)) + "T").mean()
        self.df = self.df.set_index("time")
        mask = df_f["Discharge"] != 0
        f_on = df_f[mask].index
        self.df.loc[f_on, "Discharge"] = df_f["Discharge"]
        self.df = self.df.reset_index()
        self.df["Discharge"] = self.df.Discharge.replace(np.nan, 0)
        self.discharge = self.df.Discharge.replace(0, np.nan).mean()
        logger.warning(
            f"Hours of spray : %.2f Mean Discharge:%.2f Max Discharge:%.2f"
            % (
                (self.df.Discharge.astype(bool).sum(axis=0) * self.DT / 3600),
                (self.df.Discharge.replace(0, np.nan).mean()),
                (self.df.Discharge.replace(0, np.nan).max()),
            )
        )
    if self.name in ["gangles21"]:
        self.df["Discharge"] = 0
        logger.debug("Initialised discharge as zero")
        if self.spray == "man":
            df_f = pd.read_csv(
                os.path.join("data/" + self.name + "/raw/")
                + self.name
                + "_fountain_runtime.csv",
                sep=",",
                index_col=False,
            )
            df_f = df_f.rename(columns={"When": "time"})
            df_f["time"] = pd.to_datetime(df_f["time"], format="%b-%d %H:%M")
            df_f["time"] += pd.DateOffset(years=121)
            df_f = (
                df_f.set_index("time")
                .resample(str(int(self.DT / 60)) + "T")
                .ffill()
                .reset_index()
            )

            mask = df_f["time"] >= self.start_date
            mask &= df_f["time"] <= self.expiry_date
            df_f = df_f.loc[mask]
            df_f = df_f.reset_index(drop=True)
            df_f = df_f.set_index("time")

            self.df = self.df.set_index("time")
            self.df.loc[df_f.index, "Discharge"] = self.D_F * df_f["fountain"]
            self.df = self.df.reset_index()

            f_heights = [
                {"time": self.start_date, "h_f": 5},
                {"time": datetime(2021, 1, 22, 16), "h_f": 9},
            ]

            df_h = pd.DataFrame(f_heights)

            # self.df["Discharge"] = self.df_f.Discharge.max()
            dis_old= self.D_F
            for i in range(1,df_h.shape[0]):
                dis_new= get_projectile(h_f=df_h.h_f[i], dia=0.006, dis=dis_old, theta_f=60)
                self.df.loc[self.df.time > df_h.time[i], "Discharge"] *= dis_new/dis_old
                logger.warning("Discharge changed from %.1f to %.1f" % (dis_old, dis_new))
                dis_old = dis_new
                # df_h[i]

            self.D_F = self.df.Discharge[self.df.Discharge != 0].mean()
            logger.warning("Manual Discharge mean %.1f" % self.D_F)

        if self.spray == "auto":
            with open(self.sim + "coeffs.json") as f:
                param_values = json.load(f)

            for i in range(0,self.df.shape[0]):
                self.df.loc[i, "Discharge"] = autoDis(**param_values, time=self.df.time.dt.hour[i], temp=self.df.temp[i],rh=self.df.RH[i], v=self.df.wind[i])
                # TODO correct with params
                if self.df.Discharge[i] < 2:
                    self.df.loc[i, "Discharge"] = 0
                # if self.df.Discharge[i] >= 13:
                #     self.df.loc[i, "Discharge"] = 13
            logger.warning(self.df.Discharge.describe())

    if self.name in ["guttannen21", "guttannen20"]:
        self.df["Discharge"] = self.D_F
        logger.info("Discharge constant")

        if self.spray == "auto":
            with open(self.sim + "coeffs.json") as f:
                param_values = json.load(f)

            for i in range(0,self.df.shape[0]):
                self.df.loc[i, "Discharge"] = autoDis(**param_values, time=self.df.time.dt.hour[i], temp=self.df.temp[i],rh=self.df.RH[i], v=self.df.wind[i])
                # TODO correct with params
                if self.df.Discharge[i] < 2:
                    self.df.loc[i, "Discharge"] = 0
                if self.df.Discharge[i] >= 13:
                    self.df.loc[i, "Discharge"] = 13
            logger.warning(self.df.Discharge.describe())

    if self.name == "guttannen22" and self.spray == "auto":
        df_f = pd.read_csv(
            os.path.join("data/" + self.name + "/interim/")
            + "discharge.csv",
            sep=",",
            parse_dates=["time"],
        )
        df_f = df_f.set_index("time")
        self.df = self.df.set_index("time")
        self.df["Discharge"] = df_f["Discharge"]
        self.df = self.df.reset_index()
        self.df= self.df.replace(np.NaN, 0)
        self.D_F = self.df.Discharge[self.df.Discharge != 0].mean()
        logger.warning("Auto Discharge mean %.1f" % self.D_F)

    if self.name == "guttannen22" and self.spray == "man":
        df_f = pd.read_csv(
            os.path.join("data/" + self.name + "/interim/")
            + "discharge.csv",
            sep=",",
            parse_dates=["time"],
        )
        df_f = df_f.set_index("time")

        f_heights = [
            {"time": self.start_date, "h_f": 3},
            {"time": datetime(2021, 12, 23, 16), "h_f": 4},
            {"time": datetime(2022, 1, 3, 16), "h_f": 5},
        ]
        df_h = pd.DataFrame(f_heights)

        self.df["Discharge"] = df_f.Discharge.max()
        dis_old= df_f.Discharge.max()
        for i in range(1,df_h.shape[0]):
            dis_new= get_projectile(h_f=df_h.h_f[i], dia=0.006, dis=dis_old, theta_f=60)
            self.df.loc[self.df.time > df_h.time[i], "Discharge"] *= dis_new/dis_old
            dis_old = dis_new
            # df_h[i]

        self.D_F = self.df.Discharge[self.df.Discharge != 0].mean()
        logger.warning("Manual Discharge mean %.1f" % self.D_F)
        # logger.warning("Manual Discharge used")

    if self.name in ["phortse20"]:
        self.df["Discharge"] = self.D_F

    if self.spray != "auto":
        mask = self.df["time"] > self.fountain_off_date
        mask_index = self.df[mask].index
        self.df.loc[mask_index, "Discharge"] = 0
