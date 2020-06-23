
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.backends.backend_pdf import PdfPages
import math
import time
from tqdm import tqdm
import os
import glob
from src.data.config import site, dates, option, folders, fountain, surface
import fnmatch

dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

start = time.time()

if __name__ == '__main__':

    path = os.path.join(dir, "data/raw/FluxNotes/")
    all_files = glob.glob(
        os.path.join(path, "TOA5__Flux*.dat"))  
    pattern = "TOA5__FluxB*.dat"
    li = []
    li_B = []
    for filename in all_files:

        if 'B' in filename:
            df_inB = pd.read_csv(filename, header=1)
            df_inB = df_inB[2:].reset_index(drop=True)
            li_B.append(df_inB)
        else:
            df_in = pd.read_csv(filename, header=1)
            df_in = df_in[2:].reset_index(drop=True)
            li.append(df_in)

    df_A = pd.concat(li, axis=0, ignore_index=True)
    df_B = pd.concat(li_B, axis=0, ignore_index=True)

    for col in df_B.columns:
        if 'B' not in col:
            if col != 'TIMESTAMP':
                df_B = df_B.drop([col], axis=1)


    df_A["TIMESTAMP"] = pd.to_datetime(df_A["TIMESTAMP"], format='%Y-%m-%d %H:%M:%S')
    df_B["TIMESTAMP"] = pd.to_datetime(df_B["TIMESTAMP"], format='%Y-%m-%d %H:%M:%S')

    df = pd.merge(df_A, df_B, how='inner', left_index=True, on='TIMESTAMP')

    df["MO_LENGTH"] = pd.to_numeric(df["MO_LENGTH"], errors="coerce")

    mask = (df["TIMESTAMP"] >= dates["start_date"]) & (df["TIMESTAMP"] <= dates["end_date"])
    df = df.loc[mask]
    df = df.reset_index()

    df = df.fillna(method='ffill')

    print(df.describe())

    pp = PdfPages(folders["input_folder"] + site + "_notes" + ".pdf")

    fig = plt.figure()

    x = df.TIMESTAMP
    ax1 = fig.add_subplot(111)
    y7 = df.MO_LENGTH
    ax1.plot(x, y7, "k-", linewidth=0.5)
    ax1.set_ylabel("Ri")
    ax1.grid()


    # format the ticks
    ax1.xaxis.set_major_locator(mdates.WeekdayLocator())
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax1.xaxis.set_minor_locator(mdates.DayLocator())

    fig.autofmt_xdate()
    pp.savefig(bbox_inches="tight")
    plt.clf()
    pp.close()
