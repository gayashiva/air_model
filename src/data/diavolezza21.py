 
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
import glob
dirname = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.append(dirname)
from src.utils.settings import config

if __name__ == "__main__":
    logger = logging.getLogger(__name__)
    coloredlogs.install(
        fmt="%(name)s %(levelname)s %(message)s",
        level=logging.INFO,
        logger=logger,
    )
    path = "/home/suryab/ownCloud/Sites/Diavolezza/Daten Steuerung Icestupa/"
    all_files = glob.glob(path + "*.txt")
    # path = r'C:\DRO\DCL_rawdata_files' # use your path
    # file = "/home/suryab/ownCloud/Sites/Diavolezza/Daten Steuerung Icestupa/ICE_STUPA_LoggerFriday_ February 05_ 2021_01_00_01 PM.txt"
    # file1 = "/home/suryab/ownCloud/Sites/Diavolezza/Daten Steuerung Icestupa/ICE_STUPA_LoggerFriday_ February 05_ 2021_02_00_03 AM.txt"
    # all_files = [file, file1]

    SITE, FOLDER, df_h = config(location = "Diavolezza 2021")
    li = []
    ctr = 0

    df = pd.read_csv(
        all_files[0],
        sep=";",
        skiprows=3,
        header=0,
        encoding="latin-1",
    )
    names = df.columns
    names = names[:-3]

    for file in all_files:

        try:
            df = pd.read_csv(
                file,
                sep=";",
                skiprows=3,
                header=0,
                encoding="latin-1",
                usecols=names,
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
            date= datetime.strptime(date, ' %d %b %Y %I:%M:%S %p ')
            df["When"] = pd.to_datetime([date+timedelta(seconds=10 * h) for h in range(0,df.shape[0])])

            li.append(df)
        except:
            ctr += 1
            a_file = open(file,encoding= 'latin-1')
            line =a_file.readline()
            line = line.split(";")[-1]
            date = line.split("+")[0]
            date = date[3:]
            date= datetime.strptime(date, ' %d %b %Y %I:%M:%S %p ')
            print(date)
            the_type, the_value, the_traceback = sys.exc_info()
            print(the_type)
            print(the_value)
            print(the_traceback)
            pass

    print("Number of hours missing : %s" %ctr)
    frame = pd.concat(li, axis=0, ignore_index=True)
    frame = frame.set_index("When").sort_index()
    logger.info(frame.head())
    logger.info(frame.tail())
    frame.to_csv(FOLDER["raw"] + SITE["name"] + ".csv")



