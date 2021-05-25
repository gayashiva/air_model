"""Icestupa class function that returns energy flux
"""

# External modules
import pandas as pd
import math
import numpy as np
from functools import lru_cache
import logging
import sys

# Module logger
logger = logging.getLogger(__name__)


def get_energy(self, row):
    i = row.Index

    self.df.loc[i, "vp_ice"] = (
        (
            1.0016
            + 3.15 * math.pow(10, -6) * self.df.loc[i, "p_a"]
            - 0.074 * math.pow(self.df.loc[i, "p_a"], -1)
        )
        * 6.112
        * np.exp(
            22.46 * (self.df.loc[i, "T_s"]) / ((self.df.loc[i, "T_s"]) + 272.62)
        )
    )
    self.df.loc[i, "Ql"] = (
        0.623
        * self.L_S
        * self.RHO_A
        / self.P0
        * math.pow(self.VAN_KARMAN, 2)
        * self.df.loc[i, "v_a"]
        * (row.vp_a - self.df.loc[i, "vp_ice"])
        / ((np.log(self.h_aws / self.Z)) ** 2)
    )
    if np.isnan(self.df.loc[i, "Ql"]):
        logger.error(f"When {self.df.When[i]},v_a {self.df.v_a[i]}, vp_ice {self.df.vp_ice[i]}")
        sys.exit("Ql nan")

    # Sensible Heat Qs
    self.df.loc[i, "Qs"] = (
        self.C_A
        * self.RHO_A
        * row.p_a
        / self.P0
        * math.pow(self.VAN_KARMAN, 2)
        * self.df.loc[i, "v_a"]
        * (self.df.loc[i, "T_a"] - self.df.loc[i, "T_s"])
        / ((np.log(self.h_aws / self.Z)) ** 2)
    )

    # Short Wave Radiation SW
    self.df.loc[i, "SW"] = (1 - row.a) * (
        row.SW_direct * self.df.loc[i, "f_cone"] + row.SW_diffuse
    )

    # Long Wave Radiation LW
    try:
        self.df.loc[i, "LW"] = row.LW_in - self.IE * self.STEFAN_BOLTZMAN * math.pow(
            self.df.loc[i, "T_s"] + 273.15, 4
        )
    except OverflowError:
        logger.error(
            f"When {self.df.When[i]},LW {self.df.LW[i]}, LW_in {self.df.LW_in[i]}, T_s {self.df.T_s[i]}"
        )
        sys.exit("LW nan")

    # if np.isnan(self.df.loc[i, "LW"]):
    #     logger.error(
    #         f"When {self.df.When[i]},LW {self.df.LW[i]}, LW_in {self.df.LW_in[i]}, T_s {self.df.T_s[i]}"
    #     )
    #     sys.exit("LW nan")

    if self.df.loc[i,'fountain_runoff']> 0:  # Can only find Qf if water discharge quantity known
        self.df.loc[i, "Qf"] = (
            (self.df.loc[i - 1, "solid"])
            # self.liquid
            * self.C_W
            * self.T_W
            / (self.TIME_STEP * self.df.loc[i, "SA"])
        )

        # if self.df.loc[i, "T_s"] < -self.delta_T_limit * self.TIME_STEP/60 : 
        # # Temperature change cannot by more than 1 C per minute
        #     self.df.loc[i, "Qf"] += (
        #         # (self.df.loc[i, "T_s"] - self.df.loc[i, "T_bulk"])
        #         (- self.delta_T_limit * self.TIME_STEP/60)
        #         * self.RHO_I
        #         * self.DX
        #         * self.C_I
        #         / self.TIME_STEP
        #     )
        #     logger.warning("Prevented temperature change from %s on %s" % (self.df.loc[i, "T_s"],self.df.loc[i, "When"]))

        # TODO add to paper
        self.df.loc[i, "Qf"] += (
            (self.df.loc[i, "T_s"])
            * self.RHO_I
            * self.DX
            * self.C_I
            / self.TIME_STEP
        )


    # TODO add to paper
    self.df.loc[i, "Qg"] = (
        self.K_I
        * (self.df.loc[i, "T_bulk"] - self.df.loc[i, "T_s"])
        # / (self.df.loc[i, "h_ice"] / 3)
        # / (self.df.loc[i, "r_ice"] / 2)
        / (self.df.loc[i, "r_ice"] / 3)
    )

    # Bulk Temperature
    self.df.loc[i + 1, "T_bulk"] = self.df.loc[i, "T_bulk"] - self.df.loc[i, "Qg"] * self.TIME_STEP * self.df.loc[i, "SA"] / (self.df.loc[i, "ice"] * self.C_I)

    # Total Energy W/m2
    self.df.loc[i, "Qsurf"] = (
        self.df.loc[i, "SW"]
        + self.df.loc[i, "LW"]
        + self.df.loc[i, "Qs"]
        + self.df.loc[i, "Qf"]
        + self.df.loc[i, "Qg"]
        + self.df.loc[i, "Ql"]
    )
