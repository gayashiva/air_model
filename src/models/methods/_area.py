"""Icestupa class function that calculates surface area, ice radius and height
"""
import pandas as pd
import math
import numpy as np
from functools import lru_cache
import logging

logger = logging.getLogger("__main__")

def get_area(self, i):

    if (self.df.t_cone[i]> 0) & (
        self.df.loc[i - 1, "r_ice"] >= self.R_F
    ):  # Growth rate positive and radius goes beyond spray radius
        self.df.loc[i, "r_ice"] = self.df.loc[i - 1, "r_ice"]

        self.df.loc[i, "h_ice"] = (
            3 * self.df.loc[i, "iceV"] / (math.pi * self.df.loc[i, "r_ice"] ** 2)
        )

        self.df.loc[i, "s_cone"] = (
            self.df.loc[i - 1, "h_ice"] / self.df.loc[i - 1, "r_ice"]
        )

    else:
        # Maintain constant Height to radius ratio
        self.df.loc[i, "s_cone"] = self.df.loc[i - 1, "s_cone"]

        # Ice Radius
        self.df.loc[i, "r_ice"] = math.pow(
            3 * self.df.loc[i, "iceV"] / (math.pi * self.df.loc[i, "s_cone"]), 1 / 3
        )

        # Ice Height
        self.df.loc[i, "h_ice"] = self.df.loc[i, "s_cone"] * self.df.loc[i, "r_ice"]

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
