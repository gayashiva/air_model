"""Functions to produce automation coefficients
"""
import math
import numpy as np
import pandas as pd
from pvlib import location, atmosphere
from datetime import datetime
from projectile import get_projectile
import json
import logging
import coloredlogs
from lmfit.models import GaussianModel
import os, sys

dirname = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.append(dirname)
from src.utils.settings import config

def TempFreeze(aws, loc="guttannen21"):

    constants, SITE, FOLDER = config(loc)

    with open(FOLDER["raw"] + "automate_info.json") as f:
        params = json.load(f)

    # AWS
    temp = aws[0]
    rh = aws[1]
    wind = aws[2]

    # Derived
    press = atmosphere.alt2pres(params["alt"]) / 100

    # Check eqn
    vp_a = (
        6.107
        * math.pow(
            10,
            7.5 * temp / (temp + 237.3),
        )
        * rh
        / 100
    )

    # Check eqn
    vp_ice = np.exp(43.494 - 6545.8 / (params["temp_i"] + 278)) / ((params["temp_i"] + 868) ** 2 * 100)

    e_a = (1.24 * math.pow(abs(vp_a / (temp + 273.15)), 1 / 7)) * (
        1 + 0.22 * math.pow(params["cld"], 2)
    )

    # Note assumptions
    LW = e_a * constants["sigma"] * math.pow(
        temp + 273.15, 4
    ) - constants["IE"] * constants["sigma"] * math.pow(273.15 + params["temp_i"], 4)

    # Check eqn
    Qs = (
        constants["C_A"]
        * constants["RHO_A"]
        * press
        / constants["P0"]
        * math.pow(constants["VAN_KARMAN"], 2)
        * wind
        * (temp - params["temp_i"])
        / ((np.log(constants["H_AWS"] / constants["Z"])) ** 2)
    )

    # Check eqn
    Ql = (
        0.623
        * constants["L_S"]
        * constants["RHO_A"]
        / constants["P0"]
        * math.pow(constants["VAN_KARMAN"], 2)
        * wind
        * (vp_a - vp_ice)
        / ((np.log(constants["H_AWS"] / constants["Z"])) ** 2)
    )

    # Check eqn
    Qf = (
        constants["RHO_I"]
        * constants["DX"]
        * constants["C_I"]
        / constants["DT"]
        * params["temp_i"]
    )

    freezing_energy = Ql + Qs + LW + Qf
    dis = -1 * freezing_energy / constants["L_F"] * 1000 / 60

    SA = math.pi * math.pow(params['r'],2) * math.pow(2,0.5) # Assuming h=r cone
    dis *= SA

    # if scaling_factor:
    #     dis *= scaling_factor

    # if r_virtual:
    #     VA = math.pi * math.pow(r_virtual,2) * math.pow(2,0.5) # Assuming h=r cone
    #     dis *= VA
    # else:
    #     SA = math.pi * math.pow(params['r'],2) * math.pow(2,0.5) # Assuming h=r cone
    #     dis *= SA

    return dis

def datetime_to_int(dt):
    return int(dt.strftime("%H"))

def SunMelt(loc='guttannen21'):

    constants, SITE, FOLDER = config(loc)

    with open(FOLDER["raw"] + "automate_info.json") as f:
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

    SA = math.pi * math.pow(params["r"],2) * math.pow(2,0.5) # Assuming h=r cone

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
    df["dis"] = -1 * (1 - params["alb"]) * (df["SW_direct"] + df["SW_diffuse"]) * SA / constants["L_F"] * 1000 / 60

    model = GaussianModel()
    gauss_params = model.guess(df.dis, df.hour)
    result = model.fit(df.dis, gauss_params, x=df.hour)
    return result

if __name__ == "__main__":

    locations = ["gangles21", "guttannen21"]
    for loc in locations:
        result = SunMelt(loc)

        x = list(range(0,24))
        param_values = dict(result.best_values)

        # plt.figure()
        # plt.plot(x, result.best_fit, "-")
        # plt.ylabel("Daymelt [l min-1]")
        # plt.xlabel("Time of day [hour]")
        # plt.legend()
        # plt.grid()
        # plt.savefig("data/" + site + "/figs/daymelt.jpg")

        print(param_values)
