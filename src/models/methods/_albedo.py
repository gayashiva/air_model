"""Icestupa class function that returns albedo
"""
import pandas as pd
import math
import numpy as np
from functools import lru_cache
import logging
import coloredlogs
# from redis_cache import cache_it

logger = logging.getLogger(__name__)

# @cache_it(limit=1000, expire=None)
def get_albedo(
    self, i, s=0, f=0, site="schwarzsee"
):  # Albedo Scheme described in Section 3.2.1

    # Precipitation event
    if self.df.T_a[i] < self.T_RAIN and self.df.Prec[i] > 0:  # Snow event
        f = 0
        s = 0
    if f == 0:  # last snowed
        self.df.loc[i, "a"] = self.A_I + (self.A_S - self.A_I) * math.exp(
            -s / self.A_DECAY
        )
        s = s + 1
    else:  # last sprayed
        self.df.loc[i, "a"] = self.A_I

    return s, f
