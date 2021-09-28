"""Icestupa class function that returns albedo
"""
import pandas as pd
import math
import numpy as np
from functools import lru_cache
import logging
import coloredlogs

logger = logging.getLogger(__name__)


def get_albedo(self, i, s=0, f=0):  # Albedo Scheme described in

    # Discharge event
    if self.df.isel(time=i).Discharge > 0:
        f = 1
    else:
        # Snow event
        if self.df.isel(time=i).T_A < self.T_PPT and self.df.isel(time=i).PRECIP > 0:
            f = 0
            s = 0

    if f == 0:  # last snowed
        self.df.isel(time=i)["a"] = self.A_I + (self.A_S - self.A_I) * math.exp(
            -s / self.A_DECAY
        )
        s = s + 1
    else:  # last sprayed
        self.df.isel(time=i)["a"] = self.A_I

    return s, f
