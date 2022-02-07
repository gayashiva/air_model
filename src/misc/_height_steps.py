"""Icestupa class function that adjusts discharge when fountain height increases
"""
import os,sys
import pandas as pd
import math
import numpy as np
from functools import lru_cache

# Locals
dirname = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.append(dirname)
from src.models.methods.solar import get_solar
from src.utils.settings import config
from src.automate.projectile import get_projectile

# def get_height_steps(self, i):  # Updates discharge based on new fountain height
def get_height_steps(dis_old, dia_f):  # Updates discharge based on new fountain height
    h_steps = 1
    G = 9.8
    Area = math.pi * math.pow(dia_f, 2) / 4
    # self.df.loc[i:, "Discharge"] /= self.discharge
    # if self.name != "guttannen":
    if dis_old != 0:
        v_old = dis_old / (60 * 1000 * Area)
        if v_old ** 2 - 2 * G * h_steps < 0:
            dis_new = 0
            v_new = 0
        else:
            v_new = np.sqrt(v_old ** 2 - 2 * G * h_steps)
            dis_new = v_new * (60 * 1000 * Area)
        # logger.warning(
        #     "Discharge changed from %.2f to %.2f" % (dis_old, dis_new)
        # )
        return dis_new

    # self.df.loc[i:, "Discharge"] *= self.discharge

if __name__ == "__main__":
    dis= 11
    print(get_projectile(h_f=3, dia=0.006, dis=dis, theta_f=60))
    dis= get_height_steps(dis, dia_f=0.006)
    print(get_projectile(h_f=4, dia=0.006, dis=dis, theta_f=60))
    dis= get_height_steps(dis, dia_f=0.006)
    print(get_projectile(h_f=5, dia=0.006, dis=dis, theta_f=60))
    # f_heights = [
    #     {"time": SITE["start_date"], "h_f": 2.68},
    #     {"time": datetime(2020, 12, 30, 16), "h_f": 3.75},
    #     {"time": datetime(2021, 1, 7, 16), "h_f": 4.68},
    #     {"time": datetime(2021, 1, 11, 16), "h_f": 5.68},
    # ]
    # df_h = pd.DataFrame(f_heights)
