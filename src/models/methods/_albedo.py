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
    if self.df.Discharge[i] > 0:
        f = 1

    # Rain event
    if self.df.temp[i] > self.T_PPT and self.df.ppt[i] > 0:
        f = 1

    # Snow event
    if self.df.temp[i] < self.T_PPT and self.df.ppt[i] > 0:
        f = 0
        s = 0

    scaling_factor = self.A_S/self.A_I
    self.df.loc[i, "alb"] = self.A_I + (self.A_S - self.A_I) * math.exp(
        -s / self.A_DECAY
    )

    if f == 0:  # last snowed
        s = s + 1

    if f == 1:  # last snowed
        s = s + scaling_factor * 1

    # if f == 0:  # last snowed
    #     self.df.loc[i, "alb"] = self.A_I + (self.A_S - self.A_I) * math.exp(
    #         -s / self.A_DECAY
    #     )
    #     s = s + 1
    # else:  # last sprayed
    #     self.df.loc[i, "alb"] = self.A_I

    return s, f
