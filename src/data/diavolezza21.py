 
import sys
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
import os
import logging
import coloredlogs

if __name__ == "__main__":
    logger = logging.getLogger(__name__)
    coloredlogs.install(
        fmt="%(name)s %(levelname)s %(message)s",
        level=logging.INFO,
        logger=logger,
    )
    file = "/home/suryab/ownCloud/Sites/Diavolezza/Daten Steuerung Icestupa/ICE_STUPA_LoggerFriday_ February 05_ 2021_01_00_01 PM.txt"

    df = pd.read_csv(
        file,
        sep=";",
        skiprows=3,
        header=0,
        encoding="latin-1",
    )
    df = df[1:].reset_index(drop=True)
    df = df[["Q_Wasser ", "T_Luft", "r_Luft","w_Luft"]]
    for col in df.columns:
        df[col] = df[col].astype(float)
    df = df.round(2)

    df.rename(
        columns={
            "w_Luft": "v_a",
            "T_Luft": "T_a",
            "r_Luft": "RH",
            "Q_Wasser ": "Discharge",
        },
        inplace=True,
    )
    a_file = open(file,encoding= 'latin-1')
    line =a_file.readline()
    line = line.split(";")[-1]
    date = line.split("+")[0]
    date = date[3:]
    print(date)
    date= datetime.strptime(date, ' %d %b %Y %I:%M:%S %p ')
    df["When"] = pd.to_datetime([date+timedelta(seconds=10 * h) for h in range(0,df.shape[0])])
    print(date)
    logger.info(df.head())
    logger.info(df.tail())

