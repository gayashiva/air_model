"""Function that returns discharge data
"""

# External modules
import sys, os
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
import logging
import coloredlogs

# Locals
dirname = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.append(dirname)
from src.utils.settings import config

def get_discharge(location="schwarzsee19"):

    SITE, FOLDER= config(location)
    df = pd.date_range(start ='1-1-2019', 
         end ='1-1-2020', freq ='15Min')
    df = df.reset_index()
    print(df.head())
    print(df.tail())
    if location in ["gangles21"]:
        self.df["Discharge"] = 0
        logger.debug("Initialised discharge as zero")
        df_f = pd.read_csv(
            os.path.join("data/" + location + "/raw/")
            + location
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
        self.df.loc[df_f.index, "Discharge"] = self.discharge * df_f["fountain"]
        self.df = self.df.reset_index()

    if location in ["guttannen21", "guttannen20"]:
        self.df["Discharge"] = 0
        logger.debug("Initialised discharge as zero")
        df_f = pd.read_csv(
            os.path.join("data/" + location + "/raw/")
            + location
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
        if (df_f.index[-1] % 2) != 0:
            df_f.loc[df_f.index % 2 == 0, "fountain"] = 1
            df_f.loc[df_f.index % 2 != 0, "fountain"] = 0

        df_f = df_f[["When", "fountain"]]
        df_f = (
            df_f.set_index("When")
            .resample(str(int(self.DT / 60)) + "T")
            .ffill().reset_index()
        )


        if location in ["guttannen20"]:
            df_f["When"] = df_f["When"] - pd.DateOffset(years=1)

            mask = df_f["When"] >= self.start_date
            mask &= df_f["When"] <= self.end_date
            df_f = df_f.loc[mask]
            df_f = df_f.reset_index(drop=True)

            self.df = self.df.set_index("When")

            # Use field discharge
            df_field = pd.read_csv(
                os.path.join("data/" + location + "/interim/")
                + location
                + "_input_field.csv"
            )
            df_field["When"] = pd.to_datetime(df_field["When"])

            df_field= df_field.set_index('When').resample(str(int(self.DT/60))+'T').mean().reset_index()

            mask = df_field["When"] >= self.start_date
            mask &= df_field["When"] <= self.end_date
            df_field = df_field.loc[mask]
            df_field = df_field.reset_index(drop=True)

            df_field = df_field.set_index("When")
            mask = df_field.Discharge != np.NaN
            mask &= df_field.Discharge != 0
            df_field = df_field.loc[mask]
            logger.warning("Discharge min %s, max %s" %(self.min_discharge,df_field.Discharge.max()))
            logger.info("Field discharge ends at %s" %df_field.index[-1])
            self.df.loc[df_field.index, "Discharge"] = df_field["Discharge"]

            df_f = df_f.set_index("When")
            self.df["Discharge_fill"] = 0
            self.df.loc[df_f.index, "Discharge_fill"] = df_f["fountain"] * df_field.Discharge.max()
            self.df.loc[
                self.df[self.df.Discharge_fill == 0].index, "Discharge_fill"
            ] = self.min_discharge  # Fountain was always on
            self.df['Discharge'] = self.df.apply(
                lambda row: row['Discharge_fill'] if np.isnan(row['Discharge']) else row['Discharge'],
                axis=1
            )
            self.df = self.df.drop(['Discharge_fill'], axis = 1)

        if location in ["guttannen21"]:
            mask = df_f["When"] >= self.start_date
            mask &= df_f["When"] <= self.end_date
            df_f = df_f.loc[mask]
            df_f = df_f.reset_index(drop=True)

            self.df = self.df.set_index("When")

            df_f = df_f.set_index("When")
            self.df.loc[df_f.index, "Discharge"] = self.discharge * df_f["fountain"]
            self.df.loc[
                self.df[self.df.Discharge== 0].index.intersection(self.df[self.df.index <= datetime(2020,12,26)].index), "Discharge"
            ] = 0  # Wood leak
            self.df.loc[
                self.df[self.df.Discharge== 0].index.intersection(self.df[self.df.index >=
                datetime(2020,12,26)].index), "Discharge"
            ] = self.min_discharge  # Fountain was always on

        logger.debug(self.df.Discharge.head())
        self.df = self.df.reset_index()
        logger.info(
            f"Hours of spray : %.2f Mean Discharge:%.2f"
            % (
                (
                    self.df.Discharge.astype(bool).sum(axis=0)
                    * self.DT
                    / 3600
                ),
                (self.df.Discharge.replace(0, np.nan).mean()),
            )
        )
    if location == "schwarzsee19":
        self.df["Discharge"] = 0
        logger.debug("Initialised discharge as zero")

        df_f = pd.read_csv(
            os.path.join("data/" + location + "/interim/")
            + location
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
    if location == "ravat20":
        self.df["Discharge"] = self.discharge
    mask = self.df["When"] > self.fountain_off_date
    mask_index = self.df[mask].index
    self.df.loc[mask_index, "Discharge"] = 0
