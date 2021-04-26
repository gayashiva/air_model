"""Icestupa class function that returns additional field data to initialise ice radius and height and include drone validation
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
    # Add Validation data to input
    if site in ["guttannen21", "guttannen20", "gangles21"]:
        df_c = pd.read_csv(
            input + site + "_drone.csv",
            sep=",",
            header=0,
            parse_dates=["When"],
        )
        df_c = df_c.set_index("When")

        if site in ["guttannen21"]:
            data = [
                {"When": datetime(2020, 12, 30, 16), "h_s": 1},
                {"When": datetime(2021, 1, 11, 16), "h_s": 1},
                {"When": datetime(2021, 1, 7, 16), "h_s": 1},
            ]
        if site in ["guttannen20"]:
            data = [
                {"When": datetime(2020, 1, 24, 12), "h_s": 1},
                {"When": datetime(2020, 2, 5, 19), "h_s": -1},
            ]

        if site in ["guttannen21", "guttannen20"]:
            df_h = pd.DataFrame(data)
            df_c = df_c.reset_index()
            df_c = pd.concat([df_c, df_h], ignore_index=True, sort=False)
            df_c = df_c.set_index("When").sort_index().reset_index()
            df_cam = pd.read_csv(
                input + site + "_cam_temp.csv",
                sep=",",
                header=0,
                parse_dates=["When"],
            )

        # Correct thermal cam temp.
        if site == "guttannen21":
            mask = df_cam["When"] >= datetime(2020,12,5) #No ice
            df_cam = df_cam.loc[mask]
            df_cam = df_cam.reset_index(drop=True)
            mask = (df_cam["cam_temp_full"] >= -7.5) & (df_cam["cam_temp_full"] <= 5) # Cloudy and sunny times
            df_cam = df_cam.loc[mask]
            df_cam = df_cam.reset_index(drop=True)
            correct = df_cam.loc[df_cam.When == datetime(2021, 2, 11,11),  "cam_temp"].values + 0.9
            print("correcting temperature by %0.2f" %correct)
            df_cam.cam_temp -= correct
            df_cam = df_cam.set_index("When")
            return df_c, df_cam

        if site in ["gangles21"]:
            # df_c["h_s"] = 0
            df_c["h_s"] = np.NaN
            df_c = df_c.reset_index()
            return df_c

    if site in ["schwarzsee19"]:
        data = [
            {"When": datetime(2019, 2, 14, 16), "DroneV": 0.856575},
            {"When": datetime(2019, 3, 10, 18), "DroneV": 0.1295},
        ]
        df_c = pd.DataFrame(data)
        df_c["h_s"] = np.NaN
        logger.info(df_c.head(10))
        return df_c
