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
        rho = self.RHO_I
    elif not np.isnan(self.df.loc[i-1, "Qmelt"]):
        EB = self.df.loc[i-1, "Qmelt"]
        rho = self.RHO_I
    else:
        EB=0

    # self.df.loc[i, "dy"] = math.sqrt(abs(EB)/(2 * math.pi * self.L_F * rho / self.DT * self.df.loc[i - 1, "r_ice"]))
    dV = math.pi * (self.df.loc[i, "dy"]**2 + 2 * self.df.loc[i - 1, "r_ice"] * self.df.loc[i, "dy"]) * self.df.loc[i, "dy"]

    if self.df.loc[i - 1, "Discharge"] > 0:
        # s = 4.2 * self.df.loc[i-1, "s_cone"] # fountain constant
        s = 1/self.df.loc[i-1, "s_cone"] # fountain constant
    else:
        s = self.df.loc[i-1, "s_cone"]

    # a = math.pi* self.df.loc[i - 1, "h_ice"]
    # b = math.pi * self.df.loc[i - 1, "r_ice"] * self.df.loc[i - 1, "h_ice"]
    # c = - (self.df.loc[i, "iceV"] - self.df.loc[i-1, "iceV"])
    a = math.pi/3* ( 2 * s * self.df.loc[i - 1, "r_ice"] + self.df.loc[i - 1, "h_ice"])
    b = math.pi/3* ( s * self.df.loc[i - 1, "r_ice"] ** 2 + 2 * self.df.loc[i - 1, "r_ice"] * self.df.loc[i - 1, "h_ice"])
    c = - (self.df.loc[i, "iceV"] - self.df.loc[i-1, "iceV"])

    # if -c > self.df.loc[i - 1, "fountain_runoff"]:
    #     c = self.df.loc[i - 1, "fountain_runoff"]
    #     logger.warning("Full Discharge used")
      
# calculate the discriminant  
    d = (b**2) - (4*a*c)  
      
# find two solutions  
    sol1 = (-b-math.sqrt(d))/(2*a)  
    sol2 = (-b+math.sqrt(d))/(2*a)  
    # print('The solution are {0} and {1}'.format(sol1,sol2))

    if abs(sol1) < 2:
        self.df.loc[i, "dy"] = sol1
    else:
        self.df.loc[i, "dy"] = sol2

    # if EB > 0:
    #     dV *= -1
    #     self.df.loc[i, "dy"] *=-1

    # if dV*self.RHO_I > self.df.loc[i - 1, "fountain_runoff"]:
    #     dV = self.df.loc[i - 1, "fountain_runoff"]
    #     logger.warning("Full Discharge used")

    self.df.loc[i - 1, "fountain_froze"] += dV* self.RHO_I
    self.df.loc[i - 1, "fountain_runoff"] -= dV* self.RHO_I

    self.df.loc[i, "r_ice"] = self.df.loc[i-1, "r_ice"] + self.df.loc[i, "dy"]
    self.df.loc[i, "h_ice"] = self.df.loc[i-1, "h_ice"] + s * self.df.loc[i, "dy"]

    # self.df.loc[i, "h_ice"] = (
    #     3 * (self.df.loc[i, "iceV"]+dV) / (math.pi * self.df.loc[i, "r_ice"]**2)
    # )

    # logger.warning(self.df.loc[i, "time"], self.df.loc[i, "dy"], self.df.loc[i, "r_ice"], self.df.loc[i, "iceV"])
    # print(self.df.loc[i, "time"], self.df.loc[i, "dy"], self.df.loc[i, "h_ice"], self.df.loc[i, "r_ice"], self.df.loc[i, "iceV"])

    self.df.loc[i, "s_cone"] = (
        self.df.loc[i - 1, "h_ice"] / self.df.loc[i - 1, "r_ice"]
    )

    # Area of Conical Ice Surface
    self.df.loc[i, "SA"] = (
        math.pi
        * self.SA_corr
        * self.df.loc[i, "r_ice"]
        * math.pow(
            (
                math.pow(self.df.loc[i, "r_ice"], 2)
                + math.pow((self.df.loc[i, "h_ice"]), 2)
            ),
            1 / 2,
        )
    )

    self.df.loc[i, "f_cone"] = (
        0.5
        * self.df.loc[i, "h_ice"]
        * self.df.loc[i, "r_ice"]
        * math.cos(self.df.loc[i, "sea"])
        + math.pi
        * math.pow(self.df.loc[i, "r_ice"], 2)
        * 0.5
        * math.sin(self.df.loc[i, "sea"])
    ) / self.df.loc[i, "SA"]
