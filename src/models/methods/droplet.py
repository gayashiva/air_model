"""Function that returns spray radius or discharge quantity by simulating projectile motion
"""

import pandas as pd
import math
import numpy as np
import logging
import coloredlogs


def get_droplet_projectile(
    dia=0, h_f=3, dis=0, x=0
):  # returns discharge or spray radius using projectile motion

    if dia == 0:
        Area = dis / (v * 60 * 1000)
        dia = math.sqrt(Area * 4 / math.pi)
        logger.warning("Aperture dia is %s" % (dia))
        return dia

    Area = math.pi * math.pow(dia, 2) / 4
    theta_f = math.radians(45)
    G = 9.8
    if x == 0:
        data_xy = []
        v = dis / (60 * 1000 * Area)
        t = 0.0
        while True:
            # now calculate the height y
            y = h_f + (t * v * math.sin(theta_f)) - (G * t * t) / 2
            # projectile has hit ground level
            if y < 0:
                break
            # calculate the distance x
            x = v * math.cos(theta_f) * t
            # append the (x, y) tuple to the list
            data_xy.append((x, y))
            t += 0.01
        logger.warning("Spray radius is %s" % (x))
        return x
    else:
        v = math.sqrt(
            G ** 2
            * x ** 2
            / (math.cos(theta_f) ** 2 * 2 * G * h_f + math.sin(2 * theta_f) * G * x)
        )
        dis = v * (60 * 1000 * Area)
        logger.warning("Discharge calculated is %s" % (dis))
        return dis


if __name__ == "__main__":
    # Main logger
    logger = logging.getLogger(__name__)
    logger.setLevel("INFO")
    get_droplet_projectile(h_f=3, dia=0.005, x=9)
