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

    if self.spray == "ERA5_":
        self.R_F = 10
        self.V_dome = 0
        logger.error("Arbitrary spray radius of %s" % self.R_F)
        logger.error("Arbitrary dome volume of %s" % self.V_dome)

    # Get initial height
    self.h_i = self.DX + 3 * self.V_dome / (math.pi * self.R_F ** 2)
