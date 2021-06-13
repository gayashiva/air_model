"""Icestupa class function that returns additional field data_h to initialise ice radius and height and include drone validation
"""

import pandas as pd
import math
import numpy as np
from functools import lru_cache
from datetime import datetime
import logging
import coloredlogs
# from redis_cache import cache_it

logger = logging.getLogger(__name__)


# @cache_it(limit=1000, expire=None)
def get_calibration(site, input):
    # Add Validation data_h to input
    if site in ["guttannen21", "guttannen20", "gangles21", "diavolezza21"]:
        df_c = pd.read_csv(
            input + site + "_drone.csv",
            sep=",",
            header=0,
            parse_dates=["When"],
        )
        df_c = df_c.set_index("When")

        if site in ["guttannen21", "guttannen20"]:
            df_c = df_c.reset_index()
            df_c = df_c.set_index("When").sort_index().reset_index()
            df_cam = pd.read_csv(
                input + site + "_cam_temp.csv",
                sep=",",
                header=0,
                parse_dates=["When"],
            )

            # Correct thermal cam temp.
            if site == "guttannen20":
                # df_c = df_c.groupby(level=0).sum() # Correct duplicate datetime
                df_c = df_c.reset_index()
                mask = (df_cam["When"] >= datetime(2020,1,2)) & (df_cam["When"] <= datetime(2020,2,16))#No ice
                df_cam = df_cam.loc[mask]
                df_cam = df_cam.reset_index(drop=True)

            # Correct thermal cam temp.
            if site == "guttannen21":
                mask = (df_cam["When"] >= datetime(2020,12,5)) & (df_cam["When"] <= datetime(2021,3,25))#No ice
                mask2 = (df_cam["When"] >= datetime(2020,12,19,12)) & (df_cam["When"] <= datetime(2020,12,26,17))#No ice
                df_cam = df_cam.loc[mask & ~mask2]
                df_cam = df_cam.reset_index(drop=True)

            mask = (df_cam["std"] >= 1)# Cloudy and sunny times
            mask &= (df_cam["mean"] >= -7) #Bluish images
            df_cam = df_cam.loc[mask]
            df_cam = df_cam.reset_index(drop=True)
            df_cam.loc[(df_cam["cam_temp"] > 0), "cam_temp"] = np.NaN # Sunlight causes this
            df_cam = df_cam[["When", "cam_temp"]]
            df_cam = df_cam.set_index("When")

            return df_c, df_cam

        if site in ["gangles21", "diavolezza21"]:
            df_c = df_c.reset_index()
            return df_c

    if site in ["schwarzsee19"]:
        dataV = [
            {"When": datetime(2019, 2, 14, 16), "DroneV": 0.856575, "DroneVError": 0.5},
            {"When": datetime(2019, 3, 10, 18), "DroneV": 0.1295, "DroneVError": 0.1},
        ]
        df_c = pd.DataFrame(dataV)
        return df_c

    if site in ["schwarzsee19", "ravat20"]:
        dataV = [
            {"When": datetime(2020, 1, 1,17), "DroneV": 0.1, "DroneVError": 0.1},
        ]
        df_c = pd.DataFrame(dataV)
        return df_c
