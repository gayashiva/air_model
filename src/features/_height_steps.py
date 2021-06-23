"""Icestupa class function that adjusts discharge when fountain height increases
"""

import pandas as pd
import math
import numpy as np
from functools import lru_cache


def get_height_steps(self, i):  # Updates discharge based on new fountain height
    h_steps = 1
    self.df.loc[i:, "Discharge"] /= self.discharge
    if self.name != "guttannen":
        if self.discharge != 0:
            Area = math.pi * math.pow(self.dia_f, 2) / 4
            if self.discharge < 6:
                discharge = 0
                self.v = 0
            else:
                self.v = np.sqrt(self.v ** 2 - 2 * self.G * h_steps)
                discharge = self.v * (60 * 1000 * Area)
            logger.warning(
                "Discharge changed from %.2f to %.2f" % (self.discharge, discharge)
            )
            self.discharge = discharge

    self.df.loc[i:, "Discharge"] *= self.discharge
