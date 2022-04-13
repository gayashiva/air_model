import sys, json
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
import re

def labview(location):
    if location == "guttannen22":

        with open("data/common/constants.json") as f:
            CONSTANTS = json.load(f)
        SITE, FOLDER = config(location)

        path = FOLDER["raw"] + "scheduled/sdcard/"
        all_files = glob.glob(path + "*.txt")

        li = []
        ctr = 0

        for file in all_files:

            # var = re.split("[.|_| ]", file)
            # date = var[4:-1]
            # date = (' '.join(date))
            # date = re.sub(' +', ' ', date)
            # date= datetime.strptime(date, '%B %d %Y %I %M %S %p')
            # print(date)
            try:
                print(file)

                df = pd.read_csv(
                    file,
                    sep=";",
                    skiprows=3,
                    encoding="latin-1",
                    usecols=["Q_Wasser ", "T_Luft", "r_Luft","w_Luft", "T_Wasser "],
                )
                df = df[1:].reset_index(drop=True)
                df = df[["Q_Wasser ", "T_Luft", "r_Luft","w_Luft", "T_Wasser "]]
                for col in df.columns:
                    df[col] = df[col].astype(float)
                df = df.round(2)

                df.rename(
                    columns={
                        "w_Luft": "v_a",
                        "T_Luft": "T_a",
                        "r_Luft": "RH",
                        "Q_Wasser ": "Discharge",
                        "T_Wasser ": "T_F",
                    },
                    inplace=True,
                )
                var = re.split("[.|_| ]", file)
                date = var[4:-1]
                date = (' '.join(date))
                date = re.sub(' +', ' ', date)
                date= datetime.strptime(date, '%B %d %Y %I %M %S %p')
                print(date)
                df["time"] = pd.to_datetime([date+timedelta(seconds=10 * h) for h in range(0,df.shape[0])])

                li.append(df)
            except:
                ctr += 1
                # a_file = open(file,encoding= 'latin-1')
                # line =a_file.readline()
                # line = line.split(";")[-1]
                # date = line.split("+")[0]
                # date = date[3:]
                # print(date)
                # date= datetime.strptime(date, ' %d %b %Y %I:%M:%S %p ')
                # print(date)
                the_type, the_value, the_traceback = sys.exc_info()
                print(the_type, the_value, the_traceback)
                pass

    df = pd.concat(li, axis=0, ignore_index=True)
    df = df.set_index("time").sort_index().reset_index()
    df= df.set_index("time").resample(pd.offsets.Minute(n=15)).mean().reset_index()

    print("Number of hours missing : %s" %ctr)

    # CSV output
    df.to_csv(FOLDER["raw"] + "labview.csv")
    print(df.tail(10))
    return df

if __name__ == "__main__":
    logger = logging.getLogger(__name__)
    coloredlogs.install(
        fmt="%(name)s %(levelname)s %(message)s",
        level=logging.INFO,
        logger=logger,
    )

    SITE, FOLDER = config("guttannen22")

    sdcard = True
    # sdcard = False
    
    if sdcard:
        df= labview("guttannen22")
    else:
        df= pd.read_csv(
                FOLDER["raw"] + "labview.csv",
                sep=",",
                header=0,
                parse_dates=["time"],
            )

    df = df.set_index("time")
    df = df[SITE["start_date"] : SITE["expiry_date"]]
    df = df[["Discharge", "T_F"]]

    df= df.replace(np.NaN, 0)
    df = df.resample("H").mean()

    df.to_csv(FOLDER["input"] + "discharge_labview.csv")

    fig, ax = plt.subplots()
    # x = df.time
    y = df.T_F
    ax.plot(y)
    ax.xaxis.set_major_locator(mdates.WeekdayLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax.xaxis.set_minor_locator(mdates.DayLocator())
    fig.autofmt_xdate()
    plt.savefig(
        FOLDER['fig'] + "water_temp.jpg",
        bbox_inches="tight",
    )
    plt.clf()

