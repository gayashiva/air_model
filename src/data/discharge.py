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

    SITE, FOLDER = config(location)
    index = pd.date_range(start="1-1-2019", end="1-1-2022", freq="H", name="When")
    df = pd.DataFrame(columns=["Discharge"], index=index)
    df = df.reset_index()

    if location == "schwarzsee19":
        df["Discharge"] = 0

        df_f = pd.read_csv(
            os.path.join("data/" + location + "/interim/")
            + location
            + "_input_field.csv"
        )
        df_f["When"] = pd.to_datetime(df_f["When"], format="%Y.%m.%d %H:%M:%S")
        df_f = df_f.set_index("When").resample("H").mean()
        df = df.set_index("When")
        mask = df_f["Discharge"] != 0
        f_on = df_f[mask].index
        df.loc[f_on, "Discharge"] = df_f["Discharge"]
        df = df.reset_index()
        df["Discharge"] = df.Discharge.replace(np.nan, 0)

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
            .resample("H").ffill()
        )

        df_f = df_f[SITE["start_date"] : SITE["end_date"]]

        df = df.set_index("When")
        df.loc[df_f.index, "Discharge"] = SITE["D_F"] * df_f["fountain"]
        df = df.reset_index()

    if location in ["guttannen21", "guttannen20"]:
        df = df.set_index("When")
        df = df[SITE["start_date"] : SITE["end_date"]]
        df = df.reset_index()
        df["Discharge"] = SITE["mean_discharge"]

    mask = df["When"] > SITE["fountain_off_date"]
    mask_index = df[mask].index
    df.loc[mask_index, "Discharge"] = 0
    return df
