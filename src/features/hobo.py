import sys,os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import logging
import coloredlogs

# Locals
dirname = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.append(dirname)
from src.models.icestupaClass import Icestupa
from src.utils.settings import config

if __name__ == "__main__":

    # Main logger
    logger = logging.getLogger(__name__)
    coloredlogs.install(
        fmt="%(funcName)s %(levelname)s %(message)s",
        level=logging.WARNING,
        # level=logging.INFO,
        logger=logger,
    )

    location = "gangles21"

    SITE, FOLDER = config(location)

    col_list = [
        "Date",
        "Temperature, *C",
        "RH, %",
    ]

    df_in = pd.read_csv(
        FOLDER["raw"] + "Gangles_hobo.csv",
        sep=",",
    )
    df_in = df_in[col_list]
    df_in.rename(
        columns={
            "Date": "When",
            "Temperature, *C": "T_a",
            "RH, %": "RH",
        },
        inplace=True,
    )
    df_in["When"] = pd.to_datetime(df_in["When"], format="%y-%m-%d %H:%M:%S")
    df_in = df_in.set_index("When").resample("H").mean().reset_index()
    print(df_in.head())
    df_in.to_csv(FOLDER["input"] + SITE["name"] + "_input_hobo.csv")
    df = df_in.set_index("When")
    df = df[SITE['start_date']:SITE["end_date"]]
    print(df.tail())

    df2 = pd.read_csv(FOLDER['input'] + SITE['name']+ "_input_model.csv", sep=",", header=0, parse_dates=["When"])

    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.scatter(df.T_a, df2.T_a, s=2)
    ax1.set_xlabel("HOBO Temp")
    ax1.set_ylabel("AWS Temp")
    ax1.grid()
    lims = [
    np.min([ax1.get_xlim(), ax1.get_ylim()]),  # min of both axes
    np.max([ax1.get_xlim(), ax1.get_ylim()]),  # max of both axes
    ]
    ax1.plot(lims, lims, '--k', alpha=0.25, zorder=0)
    ax1.set_aspect('equal')
    ax1.set_xlim(lims)
    ax1.set_ylim(lims)
    plt.savefig(FOLDER["input"]+ "compare_hobo.jpg",bbox_inches="tight")
    plt.clf()
