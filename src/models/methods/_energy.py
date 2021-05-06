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


def get_energy(self, row, mode="normal"):
    i = row.Index

    if mode == "trigger":  # Used while deriving discharge rate
        self.df.loc[i, "T_s"] = 0
        self.df.loc[i, "Qf"] = 0
        self.df.loc[i, "Qg"] = 0
        self.df.loc[i, "f_cone"] = 1

    if mode != "trigger":
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
            / ((np.log(self.h_aws / self.Z_I)) ** 2)
        )
        if np.isnan(self.df.loc[i, "Ql"]):
            logger.error(f"When {self.df.When[i]},p_a {self.df.p_a[i]}")
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
        / ((np.log(self.h_aws / self.Z_I)) ** 2)
    )

    # Short Wave Radiation SW
    if mode != "trigger":
        self.df.loc[i, "SW"] = (1 - row.a) * (
            row.SW_direct * self.df.loc[i, "f_cone"] + row.SW_diffuse
        )
    else:
        self.df.loc[i, "SW"] = (1 - self.A_I) * (row.SW_direct + row.SW_diffuse)

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

    if np.isnan(self.df.loc[i, "LW"]):
        logger.error(
            f"When {self.df.When[i]},LW {self.df.LW[i]}, LW_in {self.df.LW_in[i]}, T_s {self.df.T_s[i]}"
        )
        sys.exit("LW nan")

    if mode != "trigger":
        if self.df.loc[i,'fountain_in']> 0:  # Can only find Qf if water discharge quantity known
            self.df.loc[i, "Qf"] = (
                (self.df.loc[i - 1, "solid"])
                # self.liquid
                * self.C_W
                * self.T_w
                / (self.TIME_STEP * self.df.loc[i, "SA"])
            )

            # TODO add to paper
            self.df.loc[i, "Qf"] += (
                # (self.df.loc[i, "T_s"] - self.df.loc[i, "T_bulk"])
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
        self.df.loc[i + 1, "T_bulk"] = self.df.loc[i, "T_bulk"] - self.df.loc[
            i, "Qg"
        ] * self.TIME_STEP * self.df.loc[i, "SA"] / (self.df.loc[i, "ice"] * self.C_I)

    # Total Energy W/m2
    self.df.loc[i, "Qsurf"] = (
        self.df.loc[i, "SW"]
        + self.df.loc[i, "LW"]
        + self.df.loc[i, "Qs"]
        + self.df.loc[i, "Qf"]
        + self.df.loc[i, "Qg"]
        + self.df.loc[i, "Ql"]
    )

    if mode == "trigger":  # Used while deriving discharge rate
        area = math.pi * self.r_spray * self.r_spray
        if self.df.loc[i, "Qsurf"] < 0:
            self.df.loc[i, "Discharge"] = -(
                self.df.loc[i, "Qsurf"] * self.TIME_STEP * area
            ) / (self.L_F)
            self.df.loc[i, "Discharge"] = self.df.loc[i, "Discharge"] * 60 / self.TIME_STEP
