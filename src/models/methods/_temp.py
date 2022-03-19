"""Icestupa class function that calculates surface temperature from energy flux
"""

# External modules
import math
import numpy as np
import logging
import sys

# Module logger
logger = logging.getLogger("__main__")

def get_temp(self, i):

    freezing_energy = self.df.loc[i, "Qtotal"] - self.df.loc[i, "Ql"]
    self.df.loc[i, "Qt"] = self.df.loc[i, "Ql"]

    if (
        self.df.loc[i, "Discharge"] > 0
        and freezing_energy < 0
        and self.df.loc[i, "Qtotal"] < 0
    ):
        self.df.loc[i, "event"] = 1

    else:
        self.df.loc[i, "event"] = 0

    if self.df.loc[i, "event"] == 1:
        self.df.loc[i, "Qfreeze"] = freezing_energy
        self.df.loc[i, "Qmelt"] = np.nan

        # Force surface temperature zero
        self.df.loc[i, "Qfreeze"] += (
            (self.df.loc[i, "T_s"]) * (self.df.loc[i, "rho_air"]) * self.DX * self.C_I / self.DT
        )
        self.df.loc[i, "Qt"] -= (
            (self.df.loc[i, "T_s"]) * self.df.loc[i,"rho_air"] * self.DX * self.C_I / self.DT
        )
    else:
        self.df.loc[i, "Qt"] += freezing_energy
        self.df.loc[i, "Qfreeze"] = np.nan

    self.df.loc[i, "delta_T_s"] = (
        self.df.loc[i, "Qt"] * self.DT / (self.df.loc[i,"rho_air"] * self.DX * self.C_I)
    )

    """Ice temperature above zero"""
    if (self.df.loc[i, "T_s"] + self.df.loc[i, "delta_T_s"]) > 0:
        self.df.loc[i, "Qmelt"] += (
            (self.df.loc[i, "T_s"] + self.df.loc[i, "delta_T_s"])
            * self.df.loc[i,"rho_air"]
            * self.DX
            * self.C_I
            / self.DT
        )

        self.df.loc[i, "Qt"] -= (
            (self.df.loc[i, "T_s"] + self.df.loc[i, "delta_T_s"])
            * self.df.loc[i,"rho_air"]
            * self.DX
            * self.C_I
            / self.DT
        )
        self.df.loc[i, "delta_T_s"] = -self.df.loc[i, "T_s"]

    if self.df.loc[i, "event"]:
        self.df.loc[i, "fountain_froze"] += (
            -self.df.loc[i, "Qfreeze"] * self.DT * self.df.loc[i, "A_cone"]
        ) / (self.L_F)

        self.df.loc[i, "wasted"] = (
            self.df.Discharge.loc[i] * self.DT / 60 - self.df.loc[i, "fountain_froze"]
        )

        if self.df.loc[i, "wasted"] < 0:
            logger.warning("Water not enough. Mean discharge exceeded")
            self.df.loc[i, "Qfreeze"] -= (
                self.df.loc[i, "wasted"]
                * (self.L_F)
                / (self.DT * self.df.loc[i, "A_cone"])
            )
            self.df.loc[i, "Qt"] += (
                self.df.loc[i, "wasted"] * self.DT * self.df.loc[i, "A_cone"]
            ) / (self.L_F)
            self.df.loc[i, "wasted"] = 0
            self.df.loc[i, "fountain_froze"] = (
                -self.df.loc[i, "Qfreeze"] * self.DT * self.df.loc[i, "A_cone"]
            ) / (self.L_F)
    else:
        self.df.loc[i, "wasted"] = self.df.Discharge.loc[i] * self.DT / 60
        self.df.loc[i, "fountain_froze"] = 0

    if np.isnan(self.df.loc[i, "Qmelt"]):
        self.df.loc[i, "melted"] = 0
    else:
        self.df.loc[i, "melted"] = (
            self.df.loc[i, "Qmelt"] * self.DT * self.df.loc[i, "A_cone"] / (self.L_F)
        )

def test_get_temp(self, i):
    self.get_temp(i)

    if not np.isnan(self.df.loc[i, "Qmelt"] * self.df.loc[i, "Qfreeze"]):
        sys.exit("Qmelt nonzero in freezing event")

    if np.isnan(self.df.loc[i, "delta_T_s"]):
        logger.error(
            f"time {self.df.time[i]},LW {self.df.LW[i]}, LW_in {self.df.LW_in[i]}, T_s {self.df.T_s[i - 1]}"
        )
        sys.exit("Ice Temperature nan")

    if self.df.loc[i, "wasted"] < 0:
        logger.error(
            f"time {self.df.time[i]},wasted {self.df.LW[i]}, LW_in {self.df.LW_in[i]}, T_s {self.df.T_s[i - 1]}"
        )
        logger.error("All discharge froze!")

    if np.isnan(self.df.loc[i, "wasted"]):
        logger.error(
            f"time {self.df.time[i]},wasted {self.df.LW[i]}, LW_in {self.df.LW_in[i]}, T_s {self.df.T_s[i - 1]}"
        )
        sys.exit("fountain runoff nan")

    if (
        self.df.loc[i, "wasted"] - self.df.loc[i, "Discharge"] * self.DT / 60
        > 2
    ):

        logger.error(
            f"Discharge exceeded time {self.df.time[i]}, Fountain in {self.df.wasted[i]}, Discharge in {self.df.Discharge[i]* self.DT / 60}"
        )

    if math.fabs(self.df.loc[i, "delta_T_s"]) > 40:
        logger.error(
            "Temperature change above 40C %s,Surface temp %i,Freezing energy %s, Temp flux %s, Density %i"
            % (
                self.df.loc[i, "time"],
                self.df.loc[i, "T_s"],
                self.df.loc[i, "Qfreeze"],
                self.df.loc[i, "Qt"],
                self.df.loc[i, "rho_air"],
            )
        )
        # sys.exit("temperature change")
