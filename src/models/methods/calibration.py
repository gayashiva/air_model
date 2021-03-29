"""Icestupa class function that returns additional field data to initialise ice radius and height and include drone validation
"""

import pandas as pd
import math
import numpy as np
from functools import lru_cache
from datetime import datetime
import logging
import coloredlogs

logger = logging.getLogger(__name__)


def get_calibration(site, input):
    # Add Validation data to input
    if site in ["guttannen"]:
        df_c = pd.read_csv(
            input + site + "_drone.csv",
            sep=",",
            header=0,
            parse_dates=["When"],
        )
        df_c = df_c.set_index("When")
        # self.df = self.df.set_index('When')
        # self.df['DroneV'] = df_c['DroneV']

        data = [
            {"When": datetime(2020, 12, 30, 16), "h_s": 1},
            {"When": datetime(2021, 1, 11, 16), "h_s": 1},
            {"When": datetime(2021, 1, 7, 16), "h_s": 1},
        ]
        df_h = pd.DataFrame(data)
        # df_c.loc[datetime(2020, 12, 30, 14), 'h_s'] = 1
        # df_c.loc[datetime(2021, 1, 7, 14), 'h_s'] = 1
        # df_c.loc[datetime(2021, 1, 11, 14), 'h_s'] = 1
        df_c = df_c.reset_index()
        df_c = pd.concat([df_c, df_h], ignore_index=True, sort=False)
        df_c = df_c.set_index("When").sort_index().reset_index()

    if site in ["schwarzsee"]:
        data = [
            {"When": datetime(2019, 2, 14, 16), "DroneV": 0.856575},
            {"When": datetime(2019, 3, 10, 18), "DroneV": 0.1295},
        ]
        df_c = pd.DataFrame(data)
    logger.info(df_c.head(10))
    return df_c
