"""Functions to produce automation coefficients
"""
import math
import numpy as np
import pandas as pd
from pvlib import location, atmosphere, irradiance
from datetime import datetime
import json
import logging
import coloredlogs
from lmfit.models import GaussianModel
import os, sys

dirname = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.append(dirname)
from src.utils.settings import config
from src.models.methods.solar import get_offset
# from src.automate.projectile import get_projectile

def TempFreeze(aws, cld, loc="guttannen22"):

    CONSTANTS, SITE, FOLDER = config(loc)

    with open("data/common/info.json") as f:
        params = json.load(f)

    # AWS
    temp = aws[0]
    rh = aws[1]
    wind = aws[2]


    vp_a = (
        6.107
        * math.pow(
            10,
            7.5 * temp / (temp + 237.3),
        )
        * rh
        / 100
    )

    vp_ice = np.exp(43.494 - 6545.8 / (params["temp_i"] + 278)) / ((params["temp_i"] + 868) ** 2 * 100)

    e_a = (1.24 * math.pow(abs(vp_a / (temp + 273.15)), 1 / 7)) * (
        1 + 0.22 * math.pow(cld, 2)
    )

    LW = e_a * CONSTANTS["sigma"] * math.pow(
        temp + 273.15, 4
    ) - CONSTANTS["IE"] * CONSTANTS["sigma"] * math.pow(273.15 + params["temp_i"], 4)

    # Derived
    press = atmosphere.alt2pres(SITE["alt"]) / 100

    Qs = (
        CONSTANTS["C_A"]
        * CONSTANTS["RHO_A"]
        * press
        / CONSTANTS["P0"]
        * math.pow(CONSTANTS["VAN_KARMAN"], 2)
        * wind
        * (temp - params["temp_i"])
        / ((np.log(CONSTANTS["H_AWS"] / CONSTANTS["Z"])) ** 2)
    )

    Ql = (
        0.623
        * CONSTANTS["L_S"]
        * CONSTANTS["RHO_A"]
        / CONSTANTS["P0"]
        * math.pow(CONSTANTS["VAN_KARMAN"], 2)
        * wind
        * (vp_a - vp_ice)
        / ((np.log(CONSTANTS["H_AWS"] / CONSTANTS["Z"])) ** 2)
    )


    EB = Ql + Qs + LW
    dis = -1 * EB / CONSTANTS["L_F"] * 1000 / 60

    # SA = math.pi * math.pow(params['spray_r'],2)
    # dis *= SA

    return dis

def SunMelt(loc):

    CONSTANTS, SITE, FOLDER = config(loc)

    with open("data/common/info.json") as f:
        params = json.load(f)

    times = pd.date_range(
        params["solar_day"],
        freq="H",
        periods=1 * 24,
    )

    # Derived
    utc = get_offset(*SITE["coords"], date=SITE["start_date"])

    times -= pd.Timedelta(hours=utc)
    loc = location.Location(
        *SITE["coords"],
        altitude=SITE["alt"],
    )

    solar_position = loc.get_solarposition(times=times, method="ephemeris")
    clearsky = loc.get_clearsky(times=times)
    clearness = irradiance.erbs(ghi = clearsky["ghi"], zenith = solar_position['apparent_zenith'],
                                      datetime_or_doy= times) 

    df = pd.DataFrame(
        {
            "ghi": clearsky["ghi"],
            "dhi": clearness["dhi"],
            "cld": 1 - clearness["kt"],
            "sea": np.radians(solar_position["elevation"]),
        }
    )
    bad_values = df["sea"]< 0 
    df["cld"]= np.where(bad_values, np.nan, df["cld"])
    cld = df["cld"].mean()
    print(df.describe())
                            
    df.index += pd.Timedelta(hours=utc)
    df.loc[df["sea"] < 0, "sea"] = 0
    df = df.reset_index()
    df["hour"] = df["index"].apply(lambda x: int(x.strftime("%H")))
    df["f_cone"] = 0

    # SA = math.pi * math.pow(params["spray_r"],2)

    for i in range(0, df.shape[0]):
        # df.loc[i, "f_cone"] = 0.3
        df.loc[i, "f_cone"] = (math.pi * math.sin(df.loc[i, "sea"]) + math.cos(df.loc[i, "sea"]))/(2*math.sqrt(2)*math.pi)

        df.loc[i, "SW_direct"] = (
            (1 - SITE["cld"])
            * df.loc[i, "f_cone"]
            * df.loc[i, "ghi"]
        )
        df.loc[i, "SW_diffuse"] = (
            SITE["cld"]  * df.loc[i, "ghi"]
        )
    df["dis"] = -1 * (1 - CONSTANTS["A_I"]) * (df["SW_direct"] + df["SW_diffuse"]) / CONSTANTS["L_F"] * 1000 / 60

    model = GaussianModel()
    gauss_params = model.guess(df.dis, df.hour)
    result = model.fit(df.dis, gauss_params, x=df.hour)
    return cld, result

if __name__ == "__main__":

    # params={
    #   "cld": 0.5,
    #   "temp_i": 0,
    #   "crit_dis": 2,
    #   "spray_r": 5,
    #   "solar_day": "2019-02-01",
    # }


    cld, result = SunMelt("gangles21")
    param_values = dict(result.best_values)

    aws = [-5,10,2]
    print(TempFreeze(aws, cld))
    print(param_values)
    # locations = ["gangles21", "guttannen21"]
    # for loc in locations:
    #     result = SunMelt(loc)

    #     x = list(range(0,24))
    #     param_values = dict(result.best_values)

        # plt.figure()
        # plt.plot(x, result.best_fit, "-")
        # plt.ylabel("Daymelt [l min-1]")
        # plt.xlabel("Time of day [hour]")
        # plt.legend()
        # plt.grid()
        # plt.savefig("data/" + site + "/figs/daymelt.jpg")

        # print(param_values)
