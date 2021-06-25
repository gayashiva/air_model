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
    index = pd.date_range(start ='1-1-2019', 
         end ='1-1-2022', freq ='H', name= "When")
    df = pd.DataFrame(columns=['Discharge'],index=index)
    df = df.reset_index()

    if location == "schwarzsee19":
        df["Discharge"] = 0
        # logger.debug("Initialised discharge as zero")

        df_f = pd.read_csv(
            os.path.join("data/" + location + "/interim/")
            + location
            + "_input_field.csv"
        )
        df_f["When"] = pd.to_datetime(df_f["When"], format="%Y.%m.%d %H:%M:%S")
        df_f = (
            df_f.set_index("When")
            # .resample(str(int(self.DT / 60)) + "T")
            .resample("H")
            .mean()
        )
        df = df.set_index("When")
        mask = df_f["Discharge"] != 0
        f_on = df_f[mask].index
        df.loc[f_on, "Discharge"] = df_f["Discharge"]
        df = df.reset_index()
        df["Discharge"] = df.Discharge.replace(np.nan, 0)
        # logger.warning(
        #     f"Hours of spray : %.2f Mean Discharge:%.2f Max Discharge:%.2f"
        #     % (
        #         (
        #             df.Discharge.astype(bool).sum(axis=0)
        #             * self.DT
        #             / 3600
        #         ),
        #         (df.Discharge.replace(0, np.nan).mean()),
        #         (df.Discharge.replace(0, np.nan).max()),
        #     )
        # )

    if location in ["gangles21"]:
        df["Discharge"] = 0
        # logger.debug("Initialised discharge as zero")
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
            # .resample(str(int(self.DT / 60)) + "T")
            .resample("H")
            .ffill()
        )

        df_f = df_f[SITE["start_date"]:SITE["end_date"]]

        df = df.set_index("When")
        df.loc[df_f.index, "Discharge"] = SITE["discharge"] * df_f["fountain"]
        df = df.reset_index()

    if location in ["guttannen21", "guttannen20"]:
        df["Discharge"] = 0
        # logger.debug("Initialised discharge as zero")
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
            # .resample(str(int(self.DT / 60)) + "T")
            .resample("H")
            .ffill().reset_index()
        )


        if location in ["guttannen20"]:
            df_f["When"] = df_f["When"] - pd.DateOffset(years=1)

            df_f = df_f.set_index("When")
            df_f = df_f[SITE["start_date"]:SITE["end_date"]]
            df_f = df_f.reset_index()

            df = df.set_index("When")

            # Use field discharge
            df_field = pd.read_csv(
                os.path.join("data/" + location + "/interim/")
                + location
                + "_input_field.csv"
            )
            df_field["When"] = pd.to_datetime(df_field["When"])

            # df_field= df_field.set_index('When').resample(str(int(self.DT/60))+'T').mean().reset_index()
            df_field= df_field.set_index('When').resample('H').mean().reset_index()

            df_field = df_field.set_index("When")
            df_field = df_field[SITE["start_date"]:SITE["end_date"]]

            mask = df_field.Discharge != np.NaN
            mask &= df_field.Discharge != 0
            df_field = df_field.loc[mask]
            df.loc[df_field.index, "Discharge"] = df_field["Discharge"]

            df_f = df_f.set_index("When")
            df["Discharge_fill"] = 0
            df.loc[df_f.index, "Discharge_fill"] = df_f["fountain"] * df_field.Discharge.max()
            # df.loc[
            #     df[df.Discharge_fill == 0].index, "Discharge_fill"
            # ] = self.min_discharge  # Fountain was always on
            df['Discharge'] = df.apply(
                lambda row: row['Discharge_fill'] if np.isnan(row['Discharge']) else row['Discharge'],
                axis=1
            )
            df = df.drop(['Discharge_fill'], axis = 1)

        if location in ["guttannen21"]:
            print(df_f.head())
            print(df_f.tail())
            # mask = df_f["When"] >= self.start_date
            # mask &= df_f["When"] <= self.end_date
            # df_f = df_f.loc[mask]
            # df_f = df_f.reset_index(drop=True)

            df_f = df_f.set_index("When")
            df_f = df_f[SITE["start_date"]:SITE["end_date"]]
            df = df.set_index("When")

            df.loc[df_f.index, "Discharge"] = SITE["discharge"] * df_f["fountain"]
            # df.loc[
            #     df[df.Discharge== 0].index.intersection(df[df.index <= datetime(2020,12,26)].index), "Discharge"
            # ] = 0  # Wood leak
            # df.loc[
            #     df[df.Discharge== 0].index.intersection(df[df.index >=
            #     datetime(2020,12,26)].index), "Discharge"
            # ] = SITE["min_discharge"]  # Fountain was always on

        # logger.debug(df.Discharge.head())
        df = df.reset_index()
        # logger.info(
        #     f"Hours of spray : %.2f Mean Discharge:%.2f"
        #     % (
        #         (
        #             df.Discharge.astype(bool).sum(axis=0)
        #             * self.DT
        #             / 3600
        #         ),
        #         (df.Discharge.replace(0, np.nan).mean()),
        #     )
        # )
    if location == "ravat20":
        df["Discharge"] = SITE["discharge"]

    start_date = (df.loc[df.Discharge!=0, "When"].iloc[0])
    df = df.set_index("When")
    df = df[start_date:SITE["end_date"]]
    df = df.reset_index()
    mask = df["When"] > SITE["fountain_off_date"]
    mask_index = df[mask].index
    df.loc[mask_index, "Discharge"] = 0
    return df
