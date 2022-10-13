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

    if self.name in ["sibinacocha21", "sibinacocha22"]:
        logger.error("Arbitrary spray radius of %s" % self.R_F)
        logger.error("Arbitrary dome volume of %s" % self.V_dome)
    else:
        if self.name in ["guttannen22"]:
            df_c = pd.read_csv(
                self.input + self.spray.split('_')[0] + "/drone.csv",
                sep=",",
                header=0,
                parse_dates=["time"],
            )
            df_c = df_c.reset_index()
            df_c.to_hdf(
                self.input_sim + "/input.h5",
                key="df_c",
                mode="w",
            )
        elif self.name in ["guttannen21", "gangles21", "guttannen20"]:
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
            # # TODO remove first index?
            # self.R_F = df_c.loc[
            #     # (df_c.time < self.fountain_off_date) & (df_c.index != 0), "rad"
            #     (df_c.time < self.fountain_off_date), "rad"
            # ].mean()
            # rad_flights = df_c.loc[
            #     # (df_c.time < self.fountain_off_date) & (df_c.index != 0), "rad"
            #     (df_c.time < self.fountain_off_date), "rad"
            # ].values
            # logger.warning("Measured spray radius from all ice radius drone %0.1f using %i measurements" % (self.R_F, len(rad_flights)))

            df_c['cond'] = (df_c['rad'] - df_c.shift(1)['rad']>0) | (df_c['DroneV'] - df_c.shift(1)['DroneV']>0)
            rad_flights = df_c.loc[df_c['cond'].values > 0 , "rad"].values
            self.R_F = np.mean(rad_flights)
            logger.warning("Measured spray radius from increasing ice radius drone %0.1f using %i measurements" % (self.R_F, len(rad_flights)))

        if self.name in ["guttannen22"]:
            if self.spray.split('_')[0] == 'scheduled':
                self.V_dome = math.pi * math.pow(self.R_F,2) * self.h_dome
            if self.spray.split('_')[0] == 'unscheduled':
                self.V_dome = df_c.loc[df_c.shape[0]-1, "DroneV"]
        else:
            self.V_dome = df_c.loc[0, "DroneV"]

    # Get initial height
    self.h_i = self.DX + 3 * self.V_dome / (math.pi * self.R_F ** 2)

    logger.warning("Dome Volume %0.1f" % self.V_dome)
