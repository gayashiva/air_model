import numpy as np
from datetime import datetime, timedelta
import pandas as pd
import math
import logging
from codetiming import Timer

from src.models.methods.calibration import get_calibration
from src.models.methods.droplet import get_droplet_projectile
logger = logging.getLogger(__name__)

def self_attributes(self, save=False):
    logger.info("Initialising Icestupa attributes")

    if self.name in ["guttannen21", "guttannen20"]:
        df_c, df_cam = get_calibration(site=self.name, input=self.raw)
    else:
        df_c = get_calibration(site=self.name, input=self.raw)

    # if self.name == "schwarzsee19":
    #     self.r_spray = get_droplet_projectile(
    #         dia=self.dia_f, h=self.h_f, d=self.discharge
    #     )
    #     logger.warning("Measured spray radius from fountain parameters %0.1f"%self.r_spray)
    # else:
    # Get spray radius
    if hasattr(self, "r_spray"):
        logger.error("Arbitrary spray radius of %s" %self.r_spray)
    else:
        self.r_spray= df_c.loc[(df_c.When < self.fountain_off_date) & (df_c.index!=0), "rad"].mean()
        logger.warning("Measured spray radius from drone %0.1f"%self.r_spray)

    if self.name == "schwarzsee19":
        self.V_dome = 0
    else:
        self.V_dome = df_c.loc[0, "DroneV"]

    logger.warning("Dome Volume %0.1f"%self.V_dome)

    # Get initial height
    self.h_i = self.DX + 3 * self.V_dome / (math.pi * self.r_spray ** 2)


    if save:
        df_c.to_hdf(
            self.input + "model_input_" + self.trigger + ".h5",
            key="df_c",
            mode="w",
        )

        if self.name in ["guttannen21", "guttannen20"]:
            df_cam.to_hdf(
                self.input + "model_input_" + self.trigger + ".h5",
                key="df_cam",
                mode="a",
            )
            df_cam.to_csv(self.input + "measured_temp.csv")
