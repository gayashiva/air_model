"""Function that returns returns new discharge after height increase by dh
"""
import pandas as pd
import math
import numpy as np
import logging
import coloredlogs


def get_dis_new(
    dia=0.016, dh=1, dis=0
): 
    Area = math.pi * math.pow(dia, 2) / 4
    G = 9.8
    v= dis/ (60 * 1000 * Area)
    if v**2 - 2 * G * dh < 0 :
        dis_new = 0
    else:
        v_new = math.sqrt( v**2 - 2 * G * dh )
        dis_new = v_new * (60 * 1000 * Area)
    # logger.warning("Discharge calculated is %s" % (dis))
    return dis_new

# def get_projectile(
#     dia=0, h_f=3, dis=0, r=0, theta_f = 45
# ):  # returns discharge or spray radius using projectile motion

#     Area = math.pi * math.pow(dia, 2) / 4
#     theta_f = math.radians(theta_f)
#     G = 9.8
#     if r == 0:
#         data_ry = []
#         v = dis / (60 * 1000 * Area)
#         t = 0.0
#         while True:
#             # now calculate the height y
#             y = h_f + (t * v * math.sin(theta_f)) - (G * t * t) / 2
#             # projectile has hit ground level
#             if y < 0:
#                 break
#             r = v * math.cos(theta_f) * t
#             data_ry.append((r, y))
#             t += 0.01
#         # logger.warning("Spray radius is %s" % (r))
#         return r
#     else:
#         v = math.sqrt(
#             G ** 2
#             * r ** 2
#             / (math.cos(theta_f) ** 2 * 2 * G * h_f + math.sin(2 * theta_f) * G * r)
#         )
#         dis = v * (60 * 1000 * Area)
#         # logger.warning("Discharge calculated is %s" % (dis))
#         return dis


if __name__ == "__main__":
    # Main logger
    logger = logging.getLogger(__name__)
    logger.setLevel("ERROR")

    # print(get_dis_new(dh=4, dis_old=60))
    print(get_dis_new(dh=1, dis=13))
    # print(get_projectile(h_f=3, dia=0.006, r=3, theta_f=60))
    # get_droplet_projectile(h_f=3, dia=0.005, dis=8)
