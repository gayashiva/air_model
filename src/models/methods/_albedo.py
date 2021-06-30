"""Icestupa class function that returns albedo
"""
import pandas as pd
import math
import numpy as np
from functools import lru_cache
import logging
import coloredlogs

logger = logging.getLogger(__name__)

def get_albedo(
    self, i, s=0, f=0
):  # Albedo Scheme described in

    # Precipitation event
    if self.df.T_a[i] < self.T_PPT and self.df.Prec[i] > self.H_PPT:  # Snow event
        f = 0
        s = 0

    # Discharge event
    if self.df.Discharge[i] > 0:
        f = 1

    if f == 0:  # last snowed
        self.df.loc[i, "a"] = self.A_I + (self.A_S - self.A_I) * math.exp(
            -s / self.A_DECAY
        )
        s = s + 1
    else:  # last sprayed
        self.df.loc[i, "a"] = self.A_I

    return s, f
