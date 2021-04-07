"""Icestupa class function that returns albedo column(a)
"""
import pandas as pd
import math
import numpy as np
from functools import lru_cache
import logging
import coloredlogs

logger = logging.getLogger(__name__)


def get_albedo(
    self, i, s=0, f=0, site="schwarzsee"
):  # Albedo Scheme described in Section 3.2.1

    """Albedo"""
    if site in ["schwarzsee19"]:
        if self.df.T_a[i] < self.T_RAIN and self.df.Prec[i] > 0:  # Snow event
            s = 0
            f = 0

        if self.df.Discharge[i] > 0:  # Spray event
            f = 1
            s = 0

        if f == 0:  # last snowed
            self.df.loc[i, "a"] = self.A_I + (self.A_S - self.A_I) * math.exp(
                -s / self.T_DECAY
            )
            s = s + 1
        else:  # last sprayed
            self.df.loc[i, "a"] = self.A_I

    if site in ["guttannen20", "guttannen21", "gangles21"]:

        # Precipitation event
        if self.df.Prec[i] > 0:
            if self.df.T_a[i] < self.T_RAIN:  # Snow event
                f = 0
                s = 0
        if f == 0:  # last snowed
            self.df.loc[i, "a"] = self.A_I + (self.A_S - self.A_I) * math.exp(
                -s / self.T_DECAY
            )
            s = s + 1
        else:  # last sprayed
            self.df.loc[i, "a"] = self.A_I

    return s, f
