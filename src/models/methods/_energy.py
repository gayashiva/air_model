"""Icestupa class function that returns energy flux
"""

# External modules
import pandas as pd
import math
import numpy as np
import logging
import sys

from src.utils import setup_logger

# Module logger
logger = logging.getLogger("__main__")


def get_energy(self, i):

    self.df.loc[i, "vp_ice"] = np.exp(
        43.494 - 6545.8 / (self.df.loc[i, "T_s"] + 278)
    ) / ((self.df.loc[i, "T_s"] + 868) ** 2 * 100)

    self.df.loc[i, "Ql"] = (
        0.623
        * self.L_S
        * self.RHO_A
        / self.P0
        * math.pow(self.VAN_KARMAN, 2)
        * self.df.loc[i, "wind"]
        * (self.df.loc[i, "vp_a"] - self.df.loc[i, "vp_ice"])
        / ((np.log(self.H_AWS / self.Z)) ** 2)
        * (1 + 0.5 * self.df.loc[i, "s_cone"])
    )

    # Sensible Heat Qs
    # if "Qs_meas" in list(self.df.columns):
    #     self.df.loc[i, "Qs"] = self.df.loc[i, "Qs_meas"]
    #     logger.warning("Using measured sensible heat")
    # else:
    self.df.loc[i, "Qs"] = (
        self.C_A
        * self.RHO_A
        * self.df.loc[i, "press"]
        / self.P0
        * math.pow(self.VAN_KARMAN, 2)
        * self.df.loc[i, "wind"]
        * (self.df.loc[i, "temp"] - self.df.loc[i, "T_s"])
        / ((np.log(self.H_AWS / self.Z)) ** 2)
        * (1 + 0.5 * self.df.loc[i, "s_cone"])
    )

    # Short Wave Radiation SW
    self.df.loc[i, "SW"] = (1 - self.df.loc[i, "alb"]) * (
        self.df.loc[i, "SW_direct"] * self.df.loc[i, "f_cone"]
        + self.df.loc[i, "SW_diffuse"]
    )

    # Long Wave Radiation LW
    self.df.loc[i, "LW"] = self.df.loc[i, "LW_in"] - self.IE * self.sigma * math.pow(
        self.df.loc[i, "T_s"] + 273.15, 4
    )

    # if self.df.loc[i, "Discharge"] > 0:
    self.df.loc[i, "Qf"] = (
        (self.df.loc[i, "Discharge"] * self.DT / 60)
        * self.C_W
        * self.T_F
        / (self.DT * self.df.loc[i, "A_cone"])
    )

    # self.df.loc[i, "Qf"] += (
    #     (self.df.loc[i, "T_s"]) * self.RHO_I * self.DX * self.C_I / self.DT
    # )

    self.df.loc[i, "Qg"] = (
        self.K_I
        * (self.df.loc[i, "T_bulk"] - self.df.loc[i, "T_s"])
        / (self.df.loc[i, "r_cone"] + self.df.loc[i, "h_cone"] / 2)
    )

    # Bulk Temperature
    self.df.loc[i + 1, "T_bulk"] = self.df.loc[i, "T_bulk"] - self.df.loc[
        i, "Qg"
    ] * self.DT * self.df.loc[i, "A_cone"] / (self.df.loc[i, "ice"] * self.C_I)

    # Total Energy W/m2
    self.df.loc[i, "Qtotal"] = (
        self.df.loc[i, "SW"]
        + self.df.loc[i, "LW"]
        + self.df.loc[i, "Qs"]
        + self.df.loc[i, "Ql"]
        + self.df.loc[i, "Qf"]
        + self.df.loc[i, "Qg"]
    )


def test_get_energy(self, i):

    self.get_energy(i)

    if np.isnan(self.df.loc[i, "LW"]):
        logger.error(
            f"time {self.df.time[i]},LW {self.df.LW[i]}, LW_in {self.df.LW_in[i]}, T_s {self.df.T_s[i]}"
        )
        sys.exit("LW nan")

    if np.isnan(self.df.loc[i, "Ql"]):
        logger.error(
            f"time {self.df.time[i]},wind {self.df.wind[i]}, vp_ice {self.df.vp_ice[i]}"
        )
        sys.exit("Ql nan")

    if np.isnan(self.df.loc[i, "s_cone"]):
        logger.error(
            f"time {self.df.time[i]}, r_cone{self.df.r_cone[i]}, A_cone {self.df.A_cone[i]}, h_cone{self.df.h_cone[i]}"
        )
        sys.exit("scone nan")

    if np.isnan(self.df.loc[i, "SW"]):
        logger.error(
            f"time {self.df.time[i]}, s_cone {self.df.f_cone[i]}, albedo {self.df.alb[i]}, direct {self.df.SW_direct[i]},diffuse {self.df.SW_diffuse[i]}"
        )
        sys.exit("SW nan")

    if np.isnan(self.df.loc[i, "Qtotal"]):
        logger.error(
            f"time {self.df.time[i]}, SW {self.df.SW[i]}, LW {self.df.LW[i]}, Qs {self.df.Qs[i]}, Qf {self.df.Qf[i]}, Qg {self.df.Qg[i]}, Dis {self.df.Discharge[i]}"
        )
        sys.exit("Energy nan")

    if math.fabs(self.df.loc[i, "Qtotal"]) > 1000:
        logger.warning(
            "Energy above 1000 %s,Fountain water %s,Sensible %s, SW %s, LW %s, Qg %s"
            % (
                self.df.loc[i, "time"],
                self.df.loc[i, "Qf"],
                self.df.loc[i, "Qs"],
                self.df.loc[i, "SW"],
                self.df.loc[i, "LW"],
                self.df.loc[i, "Qg"],
            )
        )
