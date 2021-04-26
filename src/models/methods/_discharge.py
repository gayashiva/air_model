"""Icestupa class function that returns Discharge rate column(Discharge)
"""
import os, sys, time
import pandas as pd
import math
import numpy as np
from tqdm import tqdm
from functools import lru_cache
import logging
import coloredlogs

logger = logging.getLogger(__name__)


def get_discharge(self):  # Provides discharge info based on trigger setting

    self.df["Discharge"] = 0

    if self.trigger == "Temperature":
        # self.df["Prec"] = 0
        mask = (self.df["T_a"] < self.crit_temp) & (self.df["SW_direct"] < 100)
        mask_index = self.df[mask].index
        self.df.loc[mask_index, "Discharge"] = 1 * self.discharge

        logger.debug(
            f"Hours of spray : %.2f"
            % (self.df.Discharge.astype(bool).sum(axis=0) * self.TIME_STEP / 3600)
        )

    if self.trigger == "Weather":

        logger.info("Spray radius %s used to calculate energy discharge"%self.r_spray)
        col = [
            "T_s",  # Surface Temperature
            "T_bulk",  # Bulk Temperature
            "f_cone",
            "TotalE",
            "SW",
            "LW",
            "Qs",
            "Ql",
            "Qf",
            "Qg",
            "ppt",
            "dpt",
            "cdt",
        ]

        for column in col:
            self.df[column] = 0

        self.df.Discharge = 0
        logger.debug("Calculating discharge from energy trigger ...")
        for row in tqdm(self.df[1:-1].itertuples(), total=self.df.shape[0]):
            self.get_energy(row, mode="trigger")

        # mask = self.df["TotalE"] < 0
        # mask_index = self.df[mask].index
        # self.df.loc[mask_index, "Discharge"] = 1 * self.discharge
        # spray = -(
        #     self.df.loc[i, "TotalE"] * self.TIME_STEP * self.df.loc[i, "SA"]
        # ) / (self.L_F)
        # self.df.loc[mask_index, "Discharge"] = spray

        col = [
            "T_s",  # Surface Temperature
            "T_bulk",  # Bulk Temperature
            "f_cone",
            "TotalE",
            "SW",
            "LW",
            "Qs",
            "Ql",
            "Qf",
            "Qg",
            "ppt",
            "dpt",
            "cdt",
        ]
        self.df.drop(columns=col)

        logger.debug(
            f"Hours of spray : %.2f"
            % (self.df.Discharge.astype(bool).sum(axis=0) * self.TIME_STEP / 3600)
        )

    if self.trigger == "None":
        self.df["Discharge"] = self.discharge

    if self.trigger == "Manual":
        if self.name in ["guttannen21", "guttannen20"]:
            df_f = pd.read_csv(
                os.path.join("data/" + self.name + "/raw/")
                + self.name
                + "_fountain_runtime.csv"
            )
            df_f["Label"] = df_f["Label"].str.split("_").str[-1]
            df_f["Label"] = df_f["Label"].str[:-3]
            df_f["When"] = pd.to_datetime(df_f["Label"], format="%b-%d %H")
            for index, row in df_f.iterrows():
                if row.When.month in [11, 12]:
                    df_f.loc[index, "When"] += pd.DateOffset(years=120)
                else:
                    df_f.loc[index, "When"] += pd.DateOffset(years=121)
            df_f = df_f.set_index("When").sort_index().reset_index()
            df_f.loc[df_f.index % 2 == 0, "fountain"] = 1
            df_f.loc[df_f.index % 2 != 0, "fountain"] = 0

            if self.name in ["guttannen20"]:
                df_f["When"] = df_f["When"] - pd.DateOffset(years=1)
                mask = df_f["When"] >= self.start_date
                df_f = df_f.loc[mask]
                df_f = df_f.reset_index(drop=True)
                df_f.loc[df_f.index % 2 == 0, "fountain"] = 0
                df_f.loc[df_f.index % 2 != 0, "fountain"] = 1
            df_f = df_f[["When", "fountain"]]
            df_f = (
                df_f.set_index("When")
                .resample(str(int(self.TIME_STEP / 60)) + "T")
                .ffill()
            )
            logger.debug(df_f.head())
            logger.debug(df_f.tail())
            self.df = self.df.set_index("When")
            self.df.loc[df_f.index, "Discharge"] = self.discharge * df_f["fountain"]
            self.df.loc[
                self.df[self.df.Discharge == 0].index, "Discharge"
            ] = 5  # Fountain was always on
            # ] = 0  # Fountain was always on
            self.df = self.df.reset_index()
            logger.info(
                f"Hours of spray : %.2f\n Mean Discharge:%.2f"
                % (
                    (
                        self.df.Discharge.astype(bool).sum(axis=0)
                        * self.TIME_STEP
                        / 3600
                    ),
                    (self.df.Discharge.replace(0, np.nan).mean()),
                )
            )
        if self.name == "schwarzsee19":

            df_f = pd.read_csv(
                os.path.join("data/" + self.name + "/interim/")
                + self.name
                + "_input_field.csv"
            )
            df_f["When"] = pd.to_datetime(df_f["When"], format="%Y.%m.%d %H:%M:%S")
            df_f = (
                df_f.set_index("When")
                .resample(str(int(self.TIME_STEP / 60)) + "T")
                .mean()
            )
            self.df = self.df.set_index("When")
            mask = df_f["Discharge"] != 0
            f_on = df_f[mask].index
            self.df.loc[f_on, "Discharge"] = df_f["Discharge"]
            self.df = self.df.reset_index()
            self.df["Discharge"] = self.df.Discharge.replace(np.nan, 0)
            self.discharge = self.df.Discharge.replace(0, np.nan).mean()
            logger.info(
                f"Hours of spray : %.2f\n Mean Discharge:%.2f"
                % (
                    (
                        self.df.Discharge.astype(bool).sum(axis=0)
                        * self.TIME_STEP
                        / 3600
                    ),
                    (self.df.Discharge.replace(0, np.nan).mean()),
                )
            )

    mask = self.df["When"] > self.fountain_off_date
    mask_index = self.df[mask].index
    self.df.loc[mask_index, "Discharge"] = 0
