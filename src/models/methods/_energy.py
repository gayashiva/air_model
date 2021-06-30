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

def get_energy(self, i):

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
        * (self.df.loc[i, "vp_a"] - self.df.loc[i, "vp_ice"])
        / ((np.log(self.h_aws / self.Z)) ** 2)
    )

    # Sensible Heat Qs
    self.df.loc[i, "Qs"] = (
        self.C_A
        * self.RHO_A
        * self.df.loc[i, "p_a"]
        / self.P0
        * math.pow(self.VAN_KARMAN, 2)
        * self.df.loc[i, "v_a"]
        * (self.df.loc[i, "T_a"] - self.df.loc[i, "T_s"])
        / ((np.log(self.h_aws / self.Z)) ** 2)
    )

    # Short Wave Radiation SW
    self.df.loc[i, "SW"] = (1 - self.df.loc[i, "a"]) * (
        self.df.loc[i, "SW_direct"] * self.df.loc[i, "f_cone"] + self.df.loc[i, "SW_diffuse"]
    )

    # Long Wave Radiation LW
    self.df.loc[i, "LW"] = self.df.loc[i, "LW_in"] - self.IE * self.STEFAN_BOLTZMAN * math.pow(
        self.df.loc[i, "T_s"] + 273.15, 4
    )

    if self.df.loc[i,'Discharge']> 0:
        self.df.loc[i, "Qf"] = (
            (self.df.loc[i, "Discharge"] * self.DT / 60)
            * self.C_W
            * self.T_W
            / (self.DT * self.df.loc[i, "SA"])
        )

        self.df.loc[i, "Qf"] += (
            (self.df.loc[i, "T_s"])
            * self.RHO_I
            * self.DX
            * self.C_I
            / self.DT
        )

    self.df.loc[i, "Qg"] = (
        self.K_I
        * (self.df.loc[i, "T_bulk"] - self.df.loc[i, "T_s"])
        / (self.df.loc[i, "r_ice"] + self.df.loc[i, "h_ice"] / 2)
    )

    # Bulk Temperature
    self.df.loc[i + 1, "T_bulk"] = self.df.loc[i, "T_bulk"] - self.df.loc[i, "Qg"] * self.DT * self.df.loc[i, "SA"] / (self.df.loc[i, "ice"] * self.C_I)

    # Total Energy W/m2
    self.df.loc[i, "Qsurf"] = (
        self.df.loc[i, "SW"]
        + self.df.loc[i, "LW"]
        + self.df.loc[i, "Qs"]
        + self.df.loc[i, "Qf"]
        + self.df.loc[i, "Qg"]
        + self.df.loc[i, "Ql"]
    )

def test_get_energy(self, i):
    self.get_energy(i)

    if np.isnan(self.df.loc[i, "LW"]):
        logger.error(
            f"When {self.df.When[i]},LW {self.df.LW[i]}, LW_in {self.df.LW_in[i]}, T_s {self.df.T_s[i]}"
        )
        sys.exit("LW nan")

    if np.isnan(self.df.loc[i, "Ql"]):
        logger.error(f"When {self.df.When[i]},v_a {self.df.v_a[i]}, vp_ice {self.df.vp_ice[i]}")
        sys.exit("Ql nan")

    if np.isnan(self.df.loc[i, "s_cone"]):
        logger.error(
            f"When {self.df.When[i]}, r_ice{self.df.r_ice[i]}, SA {self.df.SA[i]}, h_ice{self.df.h_ice[i]}"
        )
        sys.exit("SW nan")

    if np.isnan(self.df.loc[i, "SW"]):
        logger.error(
            f"When {self.df.When[i]}, s_cone {self.df.f_cone[i]}, albedo {self.df.a[i]}, direct {self.df.SW_direct[i]},diffuse {self.df.SW_diffuse[i]}"
        )
        sys.exit("SW nan")

    if np.isnan(self.df.loc[i, "Qsurf"]):
        logger.error(
            f"When {self.df.When[i]}, SW {self.df.SW[i]}, LW {self.df.LW[i]}, Qs {self.df.Qs[i]}, Qf {self.df.Qf[i]}, Qg {self.df.Qg[i]}"
        )
        sys.exit("Energy nan")

    if math.fabs(self.df.loc[i, "Qsurf"]) > 1000:
        logger.warning(
            "Energy above 1000 %s,Fountain water %s,Sensible %s, SW %s, LW %s, Qg %s"
            % (
                self.df.loc[i, "When"],
                self.df.loc[i, "Qf"],
                self.df.loc[i, "Qs"],
                self.df.loc[i, "SW"],
                self.df.loc[i, "LW"],
                self.df.loc[i, "Qg"],
            )
        )
