import numpy as np
from datetime import datetime, timedelta
import pandas as pd
import math
import logging, coloredlogs
from codetiming import Timer
import os

# from src.models.methods.calibration import get_calibration

# Module logger
logger = logging.getLogger("__main__")


def self_attributes(self):
    logger.info("Initialising Icestupa attributes")

    if self.name in ["guttannen22"]:
        df_c = pd.read_csv(
            self.input+ self.spray + "/drone.csv",
            sep=",",
            header=0,
            parse_dates=["time"],
        )
        df_c = df_c.reset_index()
        df_c.to_hdf(
            self.input + self.spray+ "/input.h5",
            key="df_c",
            mode="w",
        )
    else:
        df_c = pd.read_csv(
            self.input + "drone.csv",
            sep=",",
            header=0,
            parse_dates=["time"],
        )
        df_c = df_c.reset_index()

        df_c.to_hdf(
            self.input + "input.h5",
            key="df_c",
            mode="w",
        )

    if hasattr(self, "R_F"):
        logger.error("Arbitrary spray radius of %s" % self.R_F)
    else:
        # TODO remove first index?
        self.R_F = df_c.loc[
            (df_c.time < self.fountain_off_date) & (df_c.index != 0), "rad"
            # (df_c.time < self.fountain_off_date), "rad"
        ].mean()
        logger.warning("Measured spray radius from drone %0.1f" % self.R_F)

    if self.name in ["guttannen22"]:
        self.V_dome = math.pi * math.pow(self.R_F,2) * self.h_dome
    else:
        self.V_dome = df_c.loc[0, "DroneV"]

    # Get initial height
    self.h_i = self.DX + 3 * self.V_dome / (math.pi * self.R_F ** 2)

    logger.warning("Dome Volume %0.1f" % self.V_dome)
