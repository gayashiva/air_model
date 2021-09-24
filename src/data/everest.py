"""Compile raw data from everest stations
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
from scipy import stats
from sklearn.linear_model import LinearRegression

# Locals
dirname = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.append(dirname)
from src.utils.settings import config
from src.data.field import get_field
from src.data.era5 import get_era5
from src.data.meteoswiss import get_meteoswiss


if __name__ == "__main__":

    logger = logging.getLogger(__name__)
    coloredlogs.install(
        fmt="%(name)s %(levelname)s %(message)s",
        level=logging.INFO,
        logger=logger,
    )

    location = "phortse20"

    SITE, FOLDER = config(location)

    col_list = [
        "TIMESTAMP",
        "T_HMP",
        "RH",
        "PRECIP",
        "WS_AVG",
        "SW_IN_AVG",
        "SW_OUT_AVG",
        "LW_IN_AVG",
        "PRESS",
        "SR50",
    ]

    df_in = pd.read_csv(
        FOLDER["raw"] + "Phortse_20201231.csv",
        sep=",",
        # skiprows=[0, 2, 3, 4],
        parse_dates=["TIMESTAMP"],
    )
    df_in = df_in[col_list]

    df_in.rename(
        columns={
            "TIMESTAMP": "When",
            "T_HMP": "T_a",
            "WS_AVG": "v_a",
            "PRESS": "p_a",
            "PRECIP": "Prec",
            "SW_IN_AVG": "SW_global",
            "LW_IN_AVG": "LW_in",
        },
        inplace=True,
    )

    print(df_in.head())
