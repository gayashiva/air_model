import pandas as pd
import math
import numpy as np
from functools import lru_cache

import logging
import coloredlogs

# Required for colored logging statements
print(__name__)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
# coloredlogs.install(
#     # fmt="%(name)s %(levelname)s %(message)s",
#     fmt="%(levelname)s %(message)s",
#     logger=logger,
# )


@lru_cache
def get_droplet_projectile(
    self, dia, h, d=0, x=0
):  # returns discharge or spray radius using projectile motion
    Area = math.pi * math.pow(dia, 2) / 4
    data_xy = []
    theta_f = math.radians(45)
    if x == 0:
        v = d / (60 * 1000 * Area)
        t = 0.0
        while True:
            # now calculate the height y
            y = h + (t * v * math.sin(theta_f)) - (self.G * t * t) / 2
            # projectile has hit ground level
            if y < 0:
                break
            # calculate the distance x
            x = v * math.cos(theta_f) * t
            # append the (x, y) tuple to the list
            data_xy.append((x, y))
            t += 0.01
        logger.info("Spray radius at height %s is %s" % (h, x))
        return x
    else:
        v = math.sqrt(
            self.G ** 2
            * x ** 2
            / (
                math.cos(theta_f) ** 2 * 2 * self.G * h
                + math.sin(2 * theta_f) * self.G * x
            )
        )
        d = v * (60 * 1000 * Area)
        logger.info("Discharge calculated is %s" % (d))
        return d
