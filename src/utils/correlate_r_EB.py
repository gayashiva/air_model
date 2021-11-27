
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

    # locations = ["guttannen21", "gangles21"]
    locations = ["guttannen21"]
    for loc in locations:
        CONSTANTS, SITE, FOLDER = config(loc)
        icestupa_sim = Icestupa(loc)
        icestupa_sim.read_output()

        if icestupa_sim.name in ["guttannen21", "guttannen20"]:
            df_c, df_cam = get_calibration(site=icestupa_sim.name, input=icestupa_sim.raw)
        else:
            df_c = get_calibration(site=icestupa_sim.name, input=icestupa_sim.raw)
        
        print(df_c)
        df_c["diff"] = 0
        fig, ax = plt.subplots()
        # x = df_c.diff
        # y1 = df_c.rad
        for i in range(1, df_c.shape[0]):
            df_c.loc[i, "diff"] = -((df_c.time[0]- df_c.time[i]).total_seconds() / (60*60)) 
            y = df_c.loc[i,"rad"]/df_c.loc[i,"diff"]
            x = df_c.index[i]
            ax.scatter(x, y)

        ax.grid()
        plt.savefig(FOLDER["sim"] + "r_correlate.jpg")
