"""Function that returns data from meteoswiss AWS
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


def get_meteoswiss(location="schwarzsee19"):

    SITE, FOLDER = config(location)
    if location == "schwarzsee19":
        location = "plaffeien19"

    location = location[:-2]

    df = pd.read_csv(
        os.path.join(FOLDER["raw"], location + "_meteoswiss.txt"),
        # sep="\s+",
        sep=";",
        skiprows=2,
    )
    for col in df.columns:
        if meteoswiss_parameter(col):
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df = df.rename(columns={col: meteoswiss_parameter(col)["name"]})
            # logger.info("%s from meteoswiss" % meteoswiss_parameter(col)["name"])
        else:
            df = df.drop(columns=col)
    df["time"] = pd.to_datetime(df["time"], format="%Y%m%d%H%M")

    # df["ppt"] = df["ppt"] / (10 * 60)  # ppt rate mm/s
    df_out = df.set_index("time").resample("H").mean().reset_index()
    df_out["ppt"] = df.set_index("time").resample("H").sum().reset_index()["ppt"]
    # mask = (df["time"] >= SITE["start_date"]) & (df["time"] <= SITE["end_date"])
    # df = df.loc[mask]
    return df_out


def meteoswiss_parameter(parameter):
    d = {
        "time": {
            "name": "time",
            "units": "(  )",
        },
        "rre150z0": {
            "name": "ppt",
            "units": "($mm$)",
        },
        "dkl010z0": {
            "name": "Wind direction",
            "units": "($\\degree$)",
        },
        "fkl010z0": {
            "name": "wind",
            "units": "($ms^{-1}$)",
        },
        "ure200s0": {
            "name": "RH",
            "units": "($%$)",
        },
        "prestas0": {
            "name": "PRESS",
            "units": "($hPa$)",
        },
        "pva200s0": {
            "name": "vp_a",
            "units": "($hPa$)",
        },
        "tde200s0": {
            "name": "T_ad",
            "units": "($\\degree C$)",
        },
        "tre200s0": {
            "name": "temp",
            "units": "($\\degree C$)",
        },
        "gre000z0": {
            "name": "SW_global",
            "units": "($W\\,m^{-2}$)",
        },
        "oli000z0": {
            "name": "LW_in",
            "units": "($W\\,m^{-2}$)",
        },
    }

    value = d.get(parameter)

    return value
