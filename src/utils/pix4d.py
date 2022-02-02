"""Convert raw pix4d data to model input format
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
import glob
from pathlib import Path
import logging
import coloredlogs
# Locals
dirname = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.append(dirname)
from src.utils.settings import config

if __name__ == "__main__":
    # Main logger
    logger = logging.getLogger(__name__)
    logger.setLevel("INFO")

    location="guttannen22"
    CONSTANTS, SITE, FOLDER = config(location)

    dfr = pd.read_csv(
        FOLDER["raw"] + "drone_rad.csv",
        sep="\t",
    )
    dfr = dfr.iloc[::2]

    dfr["time"] = pd.to_datetime(dfr["Name"], format="%d-%m-%y")

    dfr["rad"] = round(dfr["Terrain 3D Length  (m)"].astype(float)/(2*math.pi),2)
    dfr = dfr.set_index("time")
    dfr = dfr[["rad"]]

    print(dfr)

    dfv = pd.read_csv(
        FOLDER["raw"] + "drone_vol.csv",
        sep="\t",
    )
    dfv = dfv.iloc[::2]
    dfv["time"] = pd.to_datetime(dfv["Name"], format="%d-%m-%y")
    dfv["DroneV"] = round(dfv["Cut Volume  (m3)"].astype(float),2)
    dfv["DroneVError"] = dfv["DroneV"] * 0.2
    dfv = dfv.set_index("time")
    dfv = dfv[["DroneV", "DroneVError"]]
    print(dfv)
    df = pd.concat([dfr, dfv], axis=1)
    print(df)
    df.to_csv(FOLDER["input"] + "drone.csv")
