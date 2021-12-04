
import numpy as np
from datetime import datetime, timedelta
import pandas as pd
import math
import logging, coloredlogs
from codetiming import Timer
import sys, os
import matplotlib.pyplot as plt

# Locals
dirname = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.append(dirname)
from src.utils.settings import config
from src.models.icestupaClass import Icestupa
from src.models.methods.calibration import get_calibration

# Module logger
logger = logging.getLogger("__main__")

if __name__ == "__main__":
    # Main logger
    logger = logging.getLogger(__name__)
    logger.setLevel("ERROR")

    locations = ["guttannen21", "gangles21"]
    # locations = ["guttannen21"]
    for loc in locations:
        CONSTANTS, SITE, FOLDER = config(loc)
        icestupa= Icestupa(loc)
        icestupa.read_output()

        if icestupa.name in ["guttannen21", "guttannen20"]:
            df_c, df_cam = get_calibration(site=icestupa.name, input=icestupa.raw)
        else:
            df_c = get_calibration(site=icestupa.name, input=icestupa.raw)
        
        print(icestupa.df.event[(icestupa.df.event == 1) & (icestupa.df.time < df_c.time[1])].count())
        df_c["diff"] = 0
        df_c["spray_hours"] = 0
        df_c["melt_hours"] = 0
        fig, ax = plt.subplots()
        # x = df_c.diff
        # y1 = df_c.rad
        for i in range(1, df_c.shape[0]):
            df_c.loc[i, "total_hours"] = -((df_c.time[0]- df_c.time[i]).total_seconds() / (60*60)) 
            # df_c.loc[i, "spray_energy"] = icestupa.df.Qfreeze[(icestupa.df.event == 1) & (icestupa.df.time<df_c.time[i])].sum()
            # df_c.loc[i, "melt_energy"] = icestupa.df.Qmelt[(icestupa.df.event == 0) & (icestupa.df.time <df_c.time[i])].sum()
            df_c.loc[i, "spray_hours"] = icestupa.df.event[(icestupa.df.event == 1) & (icestupa.df.time <df_c.time[i])].count() 
            df_c.loc[i, "melt_hours"] = icestupa.df.event[(icestupa.df.event == 0) & (icestupa.df.time < df_c.time[i])].count()
            df_c.loc[i, "spray_hours"] -=df_c.loc[i-1, "spray_hours"] 
            df_c.loc[i, "melt_hours"] -=df_c.loc[i-1, "melt_hours"] 
            # df_c.loc[i, "spray_mag"] = df_c.loc[i, "spray_energy"]/(df_c.loc[i,"spray_hours"] * icestupa.L_F)
            # df_c.loc[i, "melt_mag"] = df_c.loc[i, "melt_energy"]/(df_c.loc[i,"melt_hours"]* icestupa.L_F )
            # df_c.loc[i, "diff"] = (df_c.loc[i,"spray_mag"] - df_c.loc[i,"melt_mag"])
            df_c.loc[i, "diff"] = (df_c.loc[i,"spray_hours"] - df_c.loc[i,"melt_hours"])
            y = (df_c.loc[i,"rad"] - df_c.loc[i-1,"rad"])/df_c.loc[i,"diff"] * 1000
            # y = (df_c.loc[i,"rad"] - df_c.loc[0,"rad"])/df_c.loc[i,"spray_hours"] * 1000
            # y = (df_c.loc[i,"rad"] - df_c.loc[i-1,"rad"])/df_c.loc[i,"melt_hours"] * 1000
            x = df_c.index[i]
            ax.scatter(x, y)

        ax.set_ylabel("Radius growth rate[$mm$]")
        ax.set_xlabel("Drone flight")
        ax.grid()
        plt.savefig(FOLDER["fig"] + "r_correlate.jpg")
        print(df_c)
