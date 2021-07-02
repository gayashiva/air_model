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

logger = logging.getLogger(__name__)


def get_discharge(self):  # Provides discharge info based on trigger setting

    if self.name == "schwarzsee19":
        self.df["Discharge"] = 0
        logger.debug("Initialised discharge as zero")

        df_f = pd.read_csv(
            os.path.join("data/" + self.name + "/interim/")
            + self.name
            + "_input_field.csv"
        )
        df_f["When"] = pd.to_datetime(df_f["When"], format="%Y.%m.%d %H:%M:%S")
        df_f = (
            df_f.set_index("When")
            .resample(str(int(self.DT / 60)) + "T")
            .mean()
        )
        self.df = self.df.set_index("When")
        mask = df_f["Discharge"] != 0
        f_on = df_f[mask].index
        self.df.loc[f_on, "Discharge"] = df_f["Discharge"]
        self.df = self.df.reset_index()
        self.df["Discharge"] = self.df.Discharge.replace(np.nan, 0)
        self.discharge = self.df.Discharge.replace(0, np.nan).mean()
        logger.warning(
            f"Hours of spray : %.2f Mean Discharge:%.2f Max Discharge:%.2f"
            % (
                (
                    self.df.Discharge.astype(bool).sum(axis=0)
                    * self.DT
                    / 3600
                ),
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
        df_f["When"] = pd.to_datetime(df_f["When"], format="%b-%d %H:%M")
        df_f["When"] += pd.DateOffset(years=121)
        df_f = (
            df_f.set_index("When")
            .resample(str(int(self.DT / 60)) + "T")
            .ffill().reset_index()
        )

        mask = df_f["When"] >= self.start_date
        mask &= df_f["When"] <= self.end_date
        df_f = df_f.loc[mask]
        df_f = df_f.reset_index(drop=True)
        df_f = df_f.set_index("When")

        self.df = self.df.set_index("When")
        self.df.loc[df_f.index, "Discharge"] = self.D_MEAN * df_f["fountain"]
        self.df = self.df.reset_index()
    if self.name in ['guttannen21', 'guttannen20']:
        self.df["Discharge"] = self.D_MEAN

    mask = self.df["When"] > self.fountain_off_date
    mask_index = self.df[mask].index
    self.df.loc[mask_index, "Discharge"] = 0
