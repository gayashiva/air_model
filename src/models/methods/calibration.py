"""Icestupa class function that returns additional field data_h to initialise ice radius and height and include drone validation
"""

import pandas as pd
import math
import numpy as np
from functools import lru_cache
from datetime import datetime
import logging
import coloredlogs

logger = logging.getLogger("__main__")

def get_calibration(site, input):
    df_c = pd.read_csv(
        input + "drone.csv",
        sep=",",
        header=0,
        parse_dates=["time"],
    )
    df_c = df_c.reset_index()

    if site in ["guttannen21", "guttannen20"]:
        df_c = df_c.set_index("time").sort_index().reset_index()
        df_cam = pd.read_csv(
            input  + "cam_temp.csv",
            sep=",",
            header=0,
            parse_dates=["time"],
        )

        # Correct thermal cam temp.
        if site == "guttannen20":
            df_c = df_c.reset_index()
            mask = (df_cam["time"] >= datetime(2020, 1, 2)) & (
                df_cam["time"] <= datetime(2020, 2, 16)
            )  # No ice
            df_cam = df_cam.loc[mask]
            df_cam = df_cam.reset_index(drop=True)

        # Correct thermal cam temp.
        if site == "guttannen21":
            mask = (df_cam["time"] >= datetime(2020, 12, 5)) & (
                df_cam["time"] <= datetime(2021, 3, 25)
            )  # No ice
            mask2 = (df_cam["time"] >= datetime(2020, 12, 19, 12)) & (
                df_cam["time"] <= datetime(2020, 12, 26, 17)
            )  # No ice
            df_cam = df_cam.loc[mask & ~mask2]
            df_cam = df_cam.reset_index(drop=True)

        mask = df_cam["std"] >= 1  # Cloudy and sunny times
        mask &= df_cam["mean"] >= -7  # Bluish images
        df_cam = df_cam.loc[mask]
        df_cam = df_cam.reset_index(drop=True)
        df_cam.loc[
            (df_cam["cam_temp"] > 0), "cam_temp"
        ] = np.NaN  # Sunlight causes this
        df_cam = df_cam[["time", "cam_temp"]]
        df_cam = df_cam.dropna()
        df_cam = df_cam.set_index("time")

        return df_c, df_cam
    else:
        return df_c

