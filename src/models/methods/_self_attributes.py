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

    if self.name == "schwarzsee19":
        self.r_spray = get_droplet_projectile(
            dia=self.dia_f, h=self.df.loc[0,"h_f"], d=self.discharge
        )
        self.dome_vol=0
        logger.warning("Measured spray radius from fountain parameters %0.1f"%self.r_spray)
    else:
        if hasattr(self, "perimeter"):
            self.r_spray = self.perimeter/(math.pi *2)
            logger.warning("Measured spray radius from perimeter %0.1f"%self.r_spray)
        else:
            self.r_spray= df_c.loc[df_c.When < self.fountain_off_date, "dia"].mean() / 2
            logger.warning("Measured spray radius from drone %0.1f"%self.r_spray)
        # Get initial height
        if hasattr(self, "dome_rad"):
            self.dome_vol = 2/3 * math.pi * self.dome_rad ** 3 # Volume of dome
            self.h_i = 3 * self.dome_vol/ (math.pi * self.r_spray ** 2)
            logger.warning("Initial height estimated from dome %0.1f"%self.h_i)
        else:
            self.h_i = 3 * df_c.loc[0, "DroneV"] / (math.pi * self.r_spray ** 2)
            self.dome_vol = df_c.loc[0, "DroneV"]
            logger.warning("Initial height estimated from drone %0.1f"%self.h_i)

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
