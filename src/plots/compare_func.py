import sys
import os
import json
import numpy as np
import pandas as pd
import math
import matplotlib.colors
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import logging, coloredlogs
from lmfit.models import GaussianModel
from tqdm import tqdm

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
)

from src.utils.settings import config
from src.models.icestupaClass import Icestupa
from src.models.methods.metadata import get_parameter_metadata

def datetime_to_int(dt):
    return int(dt.strftime("%H"))

def autoDis(a, b, c, d, amplitude, center, sigma, temp, time=10, rh=50, v=2):
    model = GaussianModel()
    params = {"amplitude": amplitude, "center": center, "sigma": sigma}
    return a * temp + b * rh + c * v + d + model.eval(x=time, **params)

if __name__ == "__main__":
    # Main logger
    logger = logging.getLogger(__name__)
    logger.setLevel("ERROR")

    # locations = ["guttannen21", "gangles21"]
    locations = ["guttannen21"]
    for loc in locations:
        CONSTANTS, SITE, FOLDER = config(loc)
        icestupa_sim = Icestupa(loc)
        icestupa_sim.read_output()

        with open(FOLDER["raw"] + "automate_info.json") as f:
            params = json.load(f)

        with open(FOLDER["sim"] + "coeffs.json") as f:
            param_values = json.load(f)
        print(param_values)

        df = icestupa_sim.df

        df = df[df.time <= icestupa_sim.fountain_off_date]

        t = tqdm(
            df.itertuples(),
            total=icestupa_sim.total_hours,
        )

        t.set_description("Simulating %s Icestupa" % icestupa_sim.name)

        df["dis_fountain"] = 0
        df["dis_iceV"] = 0
        df.loc[0, "dis_iceV"] = icestupa_sim.V_dome
        for row in t:
            i = row.Index
            if i !=0:
                hour = datetime_to_int(row.time)
                df.loc[i, "dis_fountain"] = autoDis(
                    **param_values, time=hour, temp=row.temp, rh=row.RH, v=row.wind
                )
                df.loc[i, "dis_freeze"] = df.loc[i, "dis_fountain"]/params['scaling_factor']
                 
                if df.loc[i, "dis_fountain"] <= params['dis_min']:
                    df.loc[i, "dis_fountain"] = 0
                    df.loc[i, "dis_freeze"] = np.nan

                df.loc[i, "dis_iceV"] = (
                    df.loc[i - 1, "dis_iceV"]
                    + df.loc[i, "dis_fountain"] * 60 / (icestupa_sim.RHO_I * params['scaling_factor'])
                )
                    
        print(df.dis_freeze.describe())
        frozen_vol = df.dis_freeze.sum() * 60/1000
        total_vol = df.dis_fountain.sum() * 60/1000
        wasted_percent = (total_vol - frozen_vol) / total_vol * 100
        print(wasted_percent)

        df.to_csv(FOLDER["sim"] + "auto_dis.csv")
        fig, ax = plt.subplots()
        x = df.time
        y1 = df.iceV
        y2 = df.dis_iceV
        ax.plot(x, y1, label="Manual Icestupa")
        ax.plot(x, y2, label="Automated Icestupa" )
        # df.dis_freeze.plot.kde()
        # plt.plot(x, y3)
        ax.grid()
        ax.legend()
        ax.set_ylabel("Ice Volume [$m^{3}$]")
# use formatters to specify major and minor ticks
        ax.xaxis.set_major_locator(mdates.WeekdayLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
        ax.xaxis.set_minor_locator(mdates.DayLocator())
        fig.autofmt_xdate()
        plt.savefig(FOLDER["fig"] + "auto_manual_dis.jpg")

