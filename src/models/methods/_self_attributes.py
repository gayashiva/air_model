import numpy as np
from datetime import datetime, timedelta
import pandas as pd
import math
import logging, coloredlogs
from codetiming import Timer
import os

from src.models.methods.calibration import get_calibration

# Module logger
logger = logging.getLogger("__main__")


def self_attributes(self):
    logger.info("Initialising Icestupa attributes")

    if hasattr(self, "R_F"):
        logger.error("Arbitrary spray radius of %s" % self.R_F)
        self.V_dome = 0
    else:
        if self.name in ["guttannen21", "guttannen20"]:
            df_c, df_cam = get_calibration(site=self.name, input=self.input)
            self.V_dome = df_c.loc[0, "DroneV"]
        elif self.name in ["guttannen22"]:
            df_c = get_calibration(site=self.name , input=self.input+ self.spray + "/")
            self.V_dome = 0
        else:
            df_c = get_calibration(site=self.name, input=self.input)
            self.V_dome = df_c.loc[0, "DroneV"]

        df_c.to_hdf(
            self.input + "input.h5",
            key="df_c",
            mode="w",
        )

        self.R_F = df_c.loc[
            (df_c.time < self.fountain_off_date) & (df_c.index != 0), "rad"
        ].mean()
        # self.R_F = df_c.loc[ 0, "rad"]
        #     (df_c.time < self.fountain_off_date) & (df_c.index != 0), "rad"
        # ].mean()
        logger.warning("Measured spray radius from drone %0.1f" % self.R_F)

        # Get initial height
        self.h_i = self.DX + 3 * self.V_dome / (math.pi * self.R_F ** 2)

    # if self.name in ["guttannen21", "guttannen20", "gangles21"]:
    #     self.V_dome = df_c.loc[0, "DroneV"]
    #     # Get initial height
    #     self.h_i = self.DX + 3 * self.V_dome / (math.pi * self.R_F ** 2)
    # else:
    #     self.V_dome = 0

    logger.warning("Dome Volume %0.1f" % self.V_dome)

    if self.name in ["guttannen21", "guttannen20"]:
        df_cam.to_hdf(
            self.input + "input.h5",
            key="df_cam",
            mode="a",
        )
