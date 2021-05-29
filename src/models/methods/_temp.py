"""Icestupa class function that returns energy flux
"""

# External modules
import math
import numpy as np
import logging
import sys

# Module logger
logger = logging.getLogger(__name__)

def get_temp(self, i):
    # Latent Heat
    if self.df.loc[i, "Ql"] < 0:
        # Sublimation
        L = self.L_S
        self.df.loc[i, "gas"] -= (
            self.df.loc[i, "Ql"]
            * self.TIME_STEP
            * self.df.loc[i, "SA"]
            / L
        )

        # Removing gas quantity generated from ice
        self.df.loc[i, "solid"] += (
            self.df.loc[i, "Ql"]
            * self.TIME_STEP
            * self.df.loc[i, "SA"]
            / L
        )

    else:
        # Deposition
        L = self.L_S
        self.df.loc[i, "dpt"] += (
            self.df.loc[i, "Ql"]
            * self.TIME_STEP
            * self.df.loc[i, "SA"]
            / self.L_S
        )

    freezing_energy = (self.df.loc[i, "Qsurf"] - self.df.loc[i, "Ql"])

    # TODO Add to paper
    process = 0

    if self.df.loc[i, "fountain_runoff"] > 0 and freezing_energy < 0 and self.df.loc[i, "Qsurf"] < 0:
        process = 1

    if process == 0:
        self.df.loc[i, "Qt"] = self.df.loc[i, "Qsurf"]
    else:
        self.df.loc[i, "Qt"] = self.df.loc[i, "Ql"]
        self.df.loc[i, "Qmelt"] = freezing_energy
        self.df.loc[i,"fountain_runoff"] += (
            freezing_energy* self.TIME_STEP * self.df.loc[i, "SA"]
        ) / (self.L_F)

    self.df.loc[i, "delta_T_s"] = (
        self.df.loc[i, "Qt"]
        * self.TIME_STEP
        / (self.RHO_I * self.DX * self.C_I)
    )

    """Ice temperature above zero"""
    if (self.df.loc[i, "T_s"] + self.df.loc[i, "delta_T_s"]) > 0:
        self.df.loc[i, "Qmelt"] += (
            (self.df.loc[i, "T_s"] + self.df.loc[i, "delta_T_s"])
            * self.RHO_I
            * self.DX
            * self.C_I
            / self.TIME_STEP
        )

        self.df.loc[i, "Qt"] -= (
            (self.df.loc[i, "T_s"] + self.df.loc[i, "delta_T_s"])
            * self.RHO_I
            * self.DX
            * self.C_I
            / self.TIME_STEP
        )

        self.df.loc[i, "delta_T_s"] = -self.df.loc[i, "T_s"]

    # TODO Remove
    if self.df.loc[i, "T_s"] < -100:
        self.df.loc[i, "T_s"] = -100
        logger.error(
            f"Surface temp rest to 100 When {self.df.When[i]},LW {self.df.LW[i]}, LW_in {self.df.LW_in[i]}, T_s {self.df.T_s[i - 1]}"
        )

def test_get_temp(self, i):
    self.get_temp(i)

    if self.df.loc[i, "delta_T_s"] > 1 * self.TIME_STEP/60:
        logger.warning("Too much fountain energy %s causes temperature change of %0.1f on %s" %(self.df.loc[i, "Qf"],self.df.loc[i, "delta_T_s"],self.df.loc[i, "When"]))
        if math.fabs(self.df.delta_T_s[i]) > 50:
            logger.error(
                "%s,Surface Temperature %s,Mass %s"
                % (
                    self.df.loc[i, "When"],
                    self.df.loc[i, "T_s"],
                    self.df.loc[i, "ice"],
                )
            )

    if np.isnan(self.df.loc[i, "delta_T_s"]):
        logger.error(
            f"When {self.df.When[i]},LW {self.df.LW[i]}, LW_in {self.df.LW_in[i]}, T_s {self.df.T_s[i - 1]}"
        )
        sys.exit("Ice Temperature nan")

    if self.df.loc[i,"fountain_runoff"] < 0 :
        logger.error(
            f"When {self.df.When[i]},fountain_runoff {self.df.LW[i]}, LW_in {self.df.LW_in[i]}, T_s {self.df.T_s[i - 1]}"
        )
        sys.exit("All discharge froze!")
        
    if np.isnan(self.df.loc[i, "fountain_runoff"]):
        logger.error(
            f"When {self.df.When[i]},fountain_runoff {self.df.LW[i]}, LW_in {self.df.LW_in[i]}, T_s {self.df.T_s[i - 1]}"
        )
        sys.exit("fountain runoff nan")

    if self.df.loc[i,'fountain_runoff'] - self.df.loc[i, 'Discharge'] * self.TIME_STEP / 60 > 2:

        logger.error(
            f"Discharge exceeded When {self.df.When[i]}, Fountain in {self.df.fountain_runoff[i]}, Discharge in {self.df.Discharge[i]* self.TIME_STEP / 60}"
        )

    if math.fabs(self.df.loc[i, "delta_T_s"]) > 20:
        logger.warning(
            "Temperature change above 20C %s,Surface temp %i,Bulk temp %i"
            % (
                self.df.loc[i, "When"],
                self.df.loc[i, "T_s"],
                self.df.loc[i, "T_bulk"],
            )
        )

    if math.fabs(self.df.loc[i,"freezing_discharge_fraction"]) > 1.01: 
        logger.error(
            "%s,temp flux %.1f,melt flux %.1f,total %.1f, Ql %.1f,freezing_discharge_fraction %.2f"
            % (
                self.df.loc[i, "When"],
                self.df.loc[i, "Qt"],
                self.df.loc[i, "Qmelt"],
                self.df.loc[i, "Qsurf"], 
                self.df.loc[i, "Ql"], 
                self.df.loc[i, "freezing_discharge_fraction"],
            )
        )
