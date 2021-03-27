import pandas as pd
import math
import numpy as np
from functools import lru_cache

import logging
import coloredlogs

logger = logging.getLogger(__name__)


@lru_cache
def get_albedo(
    self, row, s=0, f=0, site="schwarzsee"
):  # Albedo Scheme described in Section 3.2.1

    i = row.Index

    """Albedo"""
    if site in ["schwarzsee"]:
        # Precipitation event
        if (row.Discharge == 0) & (row.Prec > 0):
            if row.T_a < self.T_RAIN:  # Snow event
                s = 0
                f = 0

        if row.Discharge > 0:  # Spray event
            f = 1
            s = 0

        if f == 0:  # last snowed
            self.df.loc[i, "a"] = self.A_I + (self.A_S - self.A_I) * math.exp(
                -s / self.T_DECAY
            )
            s = s + 1
        else:  # last sprayed

            self.df.loc[i, "a"] = self.A_I

    if site in ["guttannen"]:

        # Precipitation event
        if row.Prec > 0:
            if row.T_a < self.T_RAIN:  # Snow event
                s = 0
        self.df.loc[i, "a"] = self.A_I + (self.A_S - self.A_I) * math.exp(
            -s / self.T_DECAY
        )
        s = s + 1

    return s, f
