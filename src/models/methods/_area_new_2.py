"""Icestupa class function that calculates surface area, ice radius and height
"""

import pandas as pd
import math
import numpy as np
from functools import lru_cache
import logging
import cmath  

logger = logging.getLogger("__main__")

def get_area(self, i):

    if not np.isnan(self.df.loc[i-1, "Qfreeze"]):
        EB = self.df.loc[i-1, "Qfreeze"]
    elif not np.isnan(self.df.loc[i-1, "Qmelt"]):
        EB = self.df.loc[i-1, "Qmelt"]
    else:
        EB=0

    # s = 2
    # s = 5
    dz = -1 * EB * self.DT/(self.RHO_I * self.L_F)
    self.df.loc[i, "dr"] = dz / self.df.loc[i - 1, "s_cone"]

    self.df.loc[i, "r_cone"] = self.df.loc[i-1, "r_cone"] + self.df.loc[i, "dr"]

    self.df.loc[i, "h_cone"] = (
        3 * self.df.loc[i, "iceV"] / (math.pi * self.df.loc[i, "r_cone"] ** 2)
    )

    self.df.loc[i, "s_cone"] = (
        self.df.loc[i - 1, "h_cone"] / self.df.loc[i - 1, "r_cone"]
    )

    self.df.loc[i, "A_cone"] = (
        math.pi
        * self.df.loc[i, "r_cone"]
        * math.pow(
            (
                math.pow(self.df.loc[i, "r_cone"], 2)
                + math.pow((self.df.loc[i, "h_cone"]), 2)
            ),
            1 / 2,
        )
    )

    self.df.loc[i, "f_cone"] = (
        0.5
        * self.df.loc[i, "h_cone"]
        * self.df.loc[i, "r_cone"]
        * math.cos(self.df.loc[i, "sea"])
        + math.pi
        * math.pow(self.df.loc[i, "r_cone"], 2)
        * 0.5
        * math.sin(self.df.loc[i, "sea"])
    ) / self.df.loc[i, "A_cone"]
