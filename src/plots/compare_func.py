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

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
)

from src.utils.settings import config
from src.models.icestupaClass import Icestupa
from src.models.methods.metadata import get_parameter_metadata


def datetime_to_int(dt):
    return int(dt.strftime("%H"))


def autoDis(a1, a2, a3, b, amplitude, center, sigma, temp, time=10, rh=50, v=2):
    model = GaussianModel()
    params = {"amplitude": amplitude, "center": center, "sigma": sigma}
    return a1 * temp + a2 * rh + a3 * v + b + model.eval(x=time, **params)


if __name__ == "__main__":
    # Main logger
    logger = logging.getLogger(__name__)
    logger.setLevel("WARNING")

    locations = ["guttannen21", "gangles21"]
    sources = ["model", "func"]
    for i, loc in enumerate(locations):
        CONSTANTS, SITE, FOLDER = config(loc)
        icestupa_sim = Icestupa(loc)
        icestupa_sim.read_output()
        icestupa_sim.self_attributes()
        with open(FOLDER["input"] + "coeff.json") as f:
            param_values = json.load(f)
        print(param_values)

        df = icestupa_sim.df
        t = tqdm(
            df.itertuples(),
            total=icestupa_sim.total_hours,
        )

        t.set_description("Simulating %s Icestupa" % icestupa_sim.name)

        df["dis_freeze"] = 0
        df["dis_iceV"] = 0
        for row in t:
            i = row.Index
            hour = datetime_to_int(row.time)
            df.loc[i, "dis_freeze"] = autoDis(
                **param_values, time=hour, temp=row.temp, rh=row.RH, v=row.wind
            )
            if i != 0:
                df.loc[i, "dis_iceV"] = (
                    df.loc[i - 1, "dis_iceV"]
                    + df.loc[i, "dis_freeze"] * 60 / icestupa_sim.RHO_I
                )
            else:
                df.loc[i, "dis_iceV"] = icestupa_sim.V_dome

        print(df.dis_iceV.describe())

        plt.figure()
        x = df.time
        y1 = df.iceV / df.SA
        y2 = df.dis_iceV
        y3 = df.SA
        plt.plot(x, y1)
        # plt.plot(x, y2)
        # plt.plot(x, y3)
        plt.legend()
        plt.grid()
        plt.savefig(FOLDER["sim"] + "dis.jpg")
