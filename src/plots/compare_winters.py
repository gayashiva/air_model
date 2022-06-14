"""Plot comparison of CH22 and CH21 winter"""
import sys
import os
import numpy as np
import pandas as pd
import math
import matplotlib.colors
import statistics as st
from datetime import datetime, timedelta
import seaborn as sns
from matplotlib.offsetbox import AnchoredText
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import logging, coloredlogs
import multiprocessing
from time import sleep
import os, sys, time
from ast import literal_eval
import matplotlib.patches as mpatches

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
)

from src.utils.settings import config
from src.models.icestupaClass import Icestupa


if __name__ == "__main__":
    # Main logger
    logger = logging.getLogger(__name__)
    logger.setLevel("ERROR")

    # locations = ['guttannen20', 'guttannen21', 'guttannen22']
    locations = ['guttannen21', 'guttannen22']
    spray = 'unscheduled_field'

    print("Comparing weather of different locations")
    for location in locations:
        SITE, FOLDER = config(location, spray)
        icestupa = Icestupa(location, spray)
        print(location, spray)
        print()
        icestupa.read_output()
        cols = [
            "temp",
            "RH",
            "ppt",
            "wind",
            "press",
            "SW_global",
        ]
        separate_periods_index = icestupa.df.loc[icestupa.df.Discharge > 0].index[-1]
        df_jan = icestupa.df.loc[icestupa.df.time.dt.month == 1]
        df_e = icestupa.df[cols].describe().T[["mean", "std"]]
        # print(df_e)
        iceV_diff = df_jan.iceV[df_jan.index[-1]] - df_jan.iceV[df_jan.index[0]]
        print(df_jan[cols].describe().T[["mean", "std"]])
        print(f'\n\tVolume diff {iceV_diff}')
        print(f'\n\tSpray radius {icestupa.R_F}\n')
