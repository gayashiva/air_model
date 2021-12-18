"""Icestupa class function that returns Discharge rate column(Discharge)
"""
import os, sys, time
import pandas as pd
import math
from datetime import datetime
import numpy as np
from tqdm import tqdm
from functools import lru_cache
import logging
import coloredlogs
import pytz

# Module logger
logger = logging.getLogger("__main__")
logger.propagate = False


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
        # df_f = df_f.tz_localize(pytz.country_timezones(self.country)[0])

        self.df = self.df.set_index("time")
        self.df.loc[df_f.index, "Discharge"] = self.D_F * df_f["fountain"]
        self.df = self.df.reset_index()

    if self.name in ["guttannen21", "guttannen20"]:
        self.df["Discharge"] = self.D_F
        logger.info("Discharge constant")

    if self.name in ["phortse20"]:
        self.df["Discharge"] = self.D_F

    mask = self.df["time"] > self.fountain_off_date
    mask_index = self.df[mask].index
    self.df.loc[mask_index, "Discharge"] = 0
