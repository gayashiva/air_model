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

    # s = self.df.loc[i-1, "s_cone"]
    s = 0.5
    a = math.pi/3* ( 2 * s * self.df.loc[i - 1, "r_cone"] + self.df.loc[i - 1, "h_cone"])
    b = math.pi/3* ( s * self.df.loc[i - 1, "r_cone"] ** 2 + 2 * self.df.loc[i - 1, "r_cone"] * self.df.loc[i - 1, "h_cone"])
    c = - (self.df.loc[i, "iceV"] - self.df.loc[i-1, "iceV"])

# calculate the discriminant  
    d = (b**2) - (4*a*c)  
      
# find two solutions  
    sol1 = (-b-math.sqrt(d))/(2*a)  
    sol2 = (-b+math.sqrt(d))/(2*a)  
    # print('The solution are {0} and {1}'.format(sol1,sol2))

    if abs(sol1) < 2:
        self.df.loc[i, "dr"] = sol1
    else:
        self.df.loc[i, "dr"] = sol2

    self.df.loc[i, "r_cone"] = self.df.loc[i-1, "r_cone"] + self.df.loc[i, "dr"]
    self.df.loc[i, "h_cone"] = self.df.loc[i-1, "h_cone"] + s * self.df.loc[i, "dr"]

    self.df.loc[i, "s_cone"] = (
        self.df.loc[i - 1, "h_cone"] / self.df.loc[i - 1, "r_cone"]
    )

    # Area of Conical Ice Surface
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
