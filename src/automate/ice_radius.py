import sys
import os
import json
import numpy as np
import pandas as pd
import math
import matplotlib.colors
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import logging, coloredlogs
from lmfit.models import GaussianModel
from tqdm import tqdm

dirname = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.append(dirname)
from src.utils.settings import config
from src.models.icestupaClass import Icestupa
from src.models.methods.metadata import get_parameter_metadata

def autoDis(a, b, c, d, amplitude, center, sigma, temp, time=10, rh=50, v=2):
    model = GaussianModel()
    params = {"amplitude": amplitude, "center": center, "sigma": sigma}
    return a * temp + b * rh + c * v + d + model.eval(x=time, **params)

def datetime_to_int(dt):
    return int(dt.strftime("%H"))

if __name__ == "__main__":
    # Main logger
    logger = logging.getLogger(__name__)
    logger.setLevel("ERROR")

    # locations = ["guttannen21", "gangles21"]
    locations = ["guttannen21"]
    for loc in locations:
        CONSTANTS, SITE, FOLDER = config(loc)
        df = pd.read_csv(FOLDER["sim"] + "auto_dis.csv")

        q = 0.5
        dis_max = 8
        dis_left = dis_max
        r_real = list(range(1,15))

        dis_freeze = []
        dis_freeze_total = []
        for r in r_real:
            if dis_left> 0:
                dis_freeze.append( df.dis_freeze.quantile(q) * math.pi * math.sqrt(2) * (r ** 2 - (r-1) ** 2) )
                dis_left = dis_max - sum(dis_freeze)
            else:
                dis_freeze.append(0)

            dis_freeze_total.append(sum(dis_freeze)) 

                
        r_max = 1
        cross_sec_area =math.pi * math.sqrt(2) * (r_max ** 2 - (r_max-1) ** 2)
        # growth_rate = dis_freeze[r_max-1]/(1000 * cross_sec_area) * number_of_hours * 100 * 60
        number_of_hours =df.dis_freeze.count() * (1-q)
        growth_rate = round(df.dis_freeze.quantile(q) * 60 /1000,4)
        print(f"{round(growth_rate* number_of_hours/cross_sec_area*100,2)} cm thickness grown in {(1-q)*100} percent of the fountain runtime")
        print(f"at the rate of {growth_rate*100} cm/h")

        
        plt.figure()
        plt.scatter(r_real, dis_freeze)
        plt.legend()
        plt.grid()
        plt.savefig(FOLDER["sim"] + "ice_rad.jpg")
