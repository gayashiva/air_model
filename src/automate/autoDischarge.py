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

def TempFreeze(temp,rh,wind,alt,cld):

    with open("data/common/auto.json") as f:
        params = json.load(f)

    with open("data/common/constants.json") as f:
        CONSTANTS = json.load(f)

    vp_a = np.exp(
        34.494 - 4924.99/ (temp + 237.1)
    ) / ((temp + 105) ** 1.57 * 100)
    vp_a *= rh/100

    vp_ice = np.exp(43.494 - 6545.8 / (params["temp_i"] + 278)) / ((params["temp_i"] + 868) ** 2 * 100)

    e_a = (1.24 * math.pow(abs(vp_a / (temp + 273.15)), 1 / 7)) * (
        1 + 0.22 * math.pow(cld, 2)
    )

    LW = e_a * CONSTANTS["sigma"] * math.pow(
        temp + 273.15, 4
    ) - CONSTANTS["IE"] * CONSTANTS["sigma"] * math.pow(273.15 + params["temp_i"], 4)

    # Derived
    alt *= 1000
    press = atmosphere.alt2pres(alt) / 100

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


    dis = -1 * (Ql / CONSTANTS["L_V"] + (Qs+LW) / CONSTANTS["L_F"]) * 1000 / 60

    # SA = math.pi * math.pow(params['spray_r'],2)
    # dis *= SA

    return dis

def SunMelt(time, coords, utc, alt):

    with open("data/common/auto.json") as f:
        params = json.load(f)

    with open("data/common/constants.json") as f:
        CONSTANTS = json.load(f)

    times = pd.date_range(
        time,
        freq="H",
        periods=1 * 24,
    )

    times -= pd.Timedelta(hours=utc)
    loc = location.Location(
        *coords,
        altitude=alt,
    )

    solar_position = loc.get_solarposition(times=times, method="ephemeris")
    clearsky = site_location.get_clearsky(times=times, model = 'simplified_solis')
    # Not using measured GHI due to shading effects
    clearness = irradiance.erbs(ghi = clearsky["ghi"], zenith = solar_position['zenith'],
                                      datetime_or_doy= times) 

    df = pd.DataFrame(
        {
            "SW_diffuse": clearness["dhi"],
            "SW_global": clearsky["ghi"],
            "sea": np.radians(solar_position["elevation"]),
            # "cld": 1 - clearness["kt"],
        }
    )
    bad_values = df["sea"]< 0 
    df["sea"]= np.where(bad_values, 0, df["sea"])
    # df["cld"]= np.where(bad_values, np.nan, df["cld"])
    # cld = df["cld"].mean()
                            
    df.index += pd.Timedelta(hours=utc)
    df = df.reset_index()
    df["hour"] = df["index"].apply(lambda x: int(x.strftime("%H")))
    df["f_cone"] = 0

    # SA = math.pi * math.pow(params["spray_r"],2)

    for i in range(0, df.shape[0]):
        df.loc[i, "f_cone"] = (math.pi * math.sin(df.loc[i, "sea"]) + math.cos(df.loc[i, "sea"]))/(2*math.sqrt(2)*math.pi)

    df["SW_direct"]= df["SW_global"] - df["SW_diffuse"]
    df["dis"] = -1 * (1 - CONSTANTS["A_I"]) * (df["SW_direct"] * df["f_cone"] + df["SW_diffuse"]) / CONSTANTS["L_F"] * 1000 / 60

    model = GaussianModel()
    gauss_params = model.guess(df.dis, df.hour)
    result = model.fit(df.dis, gauss_params, x=df.hour)
    return result

def dayMelt(times, coords, alt, utc, opt="auto"):
    with open("data/common/auto.json") as f:
        params = json.load(f)

    with open("data/common/constants.json") as f:
        CONSTANTS = json.load(f)

    times -= pd.Timedelta(hours=utc)

    loc = location.Location(*coords, altitude=alt)

    solar_position = loc.get_solarposition(times=times, method="ephemeris")
    clearsky = loc.get_clearsky(times=times, model = 'simplified_solis')
    clearness = irradiance.erbs(ghi = clearsky["ghi"], zenith = solar_position['zenith'],
                                      datetime_or_doy= times) 
    df = pd.DataFrame(
        {
            "SW_diffuse": clearness["dhi"],
            "SW_global": clearsky["ghi"],
            "sea": np.radians(solar_position["elevation"]),
            "cld": 1 - clearness["kt"],
        }
    )

    bad_values = df["sea"] < 0 
    df["sea"]= np.where(bad_values, 0, df["sea"])
    df["cld"]= np.where(bad_values, np.nan, df["cld"])
    cld = df["cld"].mean()

    times += pd.Timedelta(hours=utc)

    if sea < 0:
        SW_diffuse, SW_global = 0
        dis = 0
    else:
        f_cone = (math.pi * math.sin(sea) + math.cos(sea))/(2*math.sqrt(2)*math.pi)
        SW_direct= SW_global - SW_diffuse
        dis = -1 * (1 - CONSTANTS["A_I"]) * (SW_direct * f_cone + SW_diffuse) / CONSTANTS["L_F"] * 1000 / 60

    return dis


if __name__ == "__main__":

    # params={
    #   "cld": 0.5,
    #   "temp_i": 0,
    #   "crit_dis": 2,
    #   "spray_r": 5,
    #   "solar_day": "2019-02-01",
    # }

    loc="guttannen21"
    SITE, FOLDER = config(loc,spray="man")
    utc = get_offset(*SITE["coords"], date=SITE["start_date"])
    print(dayMelt(time=SITE["start_date"], coords = SITE["coords"], utc = utc, alt=SITE["alt"]))
    # result = SunMelt(coords = SITE["coords"], utc = utc, alt=SITE["alt"])
    # param_values = dict(result.best_values)


    aws1 = [-5,10,2,4000]
    aws2 = [-5,10,2,0]
    print(TempFreeze(*aws1))
    print(TempFreeze(*aws2))
    # print(param_values)
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
