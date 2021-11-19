import numpy as np
from scipy.stats import norm
import scipy
import matplotlib.pyplot as plt
from pvlib import location, atmosphere
import pandas as pd
import xarray as xr
import math
from lmfit.models import LinearModel, GaussianModel
import json

import os, sys
dirname = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.append(dirname)
from src.utils.settings import config

def datetime_to_int(dt):
    return int(dt.strftime("%H"))

def SunMelt(site='guttannen21'):

    CONSTANTS, SITE, FOLDER = config(site)
    # with open("data/" + site + "/info.json") as f:
    with open(FOLDER["raw"] + "info.json") as f:
        params = json.load(f)

    times = pd.date_range(
        params["solar_day"],
        freq="H",
        periods=1 * 24,
    )

    times -= pd.Timedelta(hours=params["utc"])
    loc = location.Location(
        params["lat"],
        params["long"],
        altitude=params["alt"],
    )

    solar_position = loc.get_solarposition(times=times, method="ephemeris")
    clearsky = loc.get_clearsky(times=times)

    df = pd.DataFrame(
        {
            "ghi": clearsky["ghi"],
            "sea": np.radians(solar_position["elevation"]),
        }
    )
    df.index += pd.Timedelta(hours=params["utc"])
    df.loc[df["sea"] < 0, "sea"] = 0
    df = df.reset_index()
    df["hour"] = df["index"].apply(lambda x: datetime_to_int(x))
    df["f_cone"] = 0

    SA = math.pi * math.pow(params["r"],2) 

    for i in range(0, df.shape[0]):
        df.loc[i, "f_cone"] = (
            math.pi * math.pow(params["r"], 2) * 0.5 * math.sin(df.loc[i, "sea"])
            + 0.5 * math.pow(params["r"], 2) * math.cos(df.loc[i, "sea"])
        ) / SA

        df.loc[i, "SW_direct"] = (
            (1 - params["cld"])
            * df.loc[i, "f_cone"]
            * df.loc[i, "ghi"]
        )
        df.loc[i, "SW_diffuse"] = (
            params["cld"]  * df.loc[i, "ghi"]
        )
    df["dis"] = -1 * (1 - params["alb"]) * (df["SW_direct"] + df["SW_diffuse"]) * SA / CONSTANTS["L_F"] * 1000 / 60

    model = GaussianModel()
    gauss_params = model.guess(df.dis, df.hour)
    result = model.fit(df.dis, gauss_params, x=df.hour)
    return result

if __name__ == "__main__":

    sites = ["gangles21", "guttannen21"]
    for site in sites:
        result = Daymelt(site)

        # x = df.hour
        x = list(range(0,24))
        param_values = dict(result.best_values)

        plt.figure()
        plt.plot(x, result.best_fit, "-")
        plt.ylabel("Daymelt [l min-1]")
        plt.xlabel("Time of day [hour]")
        plt.legend()
        plt.grid()
        plt.savefig("data/" + site + "/figs/daymelt.jpg")

        print(param_values)

        with open("data/" + site + "/daymelt.json", "w") as f:
            json.dump(param_values, f)
