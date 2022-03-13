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

def TempFreeze(data):

    with open("data/common/constants.json") as f:
        CONSTANTS = json.load(f)

    #Assumptions
    temp_i = 0

    if data["obj"] == "WUE":
        cld = 1
        mu_cone = 1
    elif data["obj"]== "ICV":
        cld = 0
        mu_cone = 1.5
    else:
        logger.error("Wrong Objective")
        sys.exit()

    vp_a = np.exp(
        34.494 - 4924.99/ (data["temp"] + 237.1)
    ) / ((data["temp"] + 105) ** 1.57 * 100)
    vp_a *= data["rh"]/100

    vp_ice = np.exp(43.494 - 6545.8 / (temp_i + 278)) / ((temp_i + 868) ** 2 * 100)

    e_a = (1.24 * math.pow(abs(vp_a / (data["temp"] + 273.15)), 1 / 7)) * (
        1 + 0.22 * math.pow(cld, 2)
    )

    LW = e_a * CONSTANTS["sigma"] * math.pow(
        data["temp"] + 273.15, 4
    ) - CONSTANTS["IE"] * CONSTANTS["sigma"] * math.pow(273.15 + temp_i, 4)

    # Derived
    press = atmosphere.alt2pres(data["alt"] * 1000) / 100

    Qs = (
        mu_cone
        * CONSTANTS["C_A"]
        * CONSTANTS["RHO_A"]
        * press
        / CONSTANTS["P0"]
        * math.pow(CONSTANTS["VAN_KARMAN"], 2)
        * data["wind"]
        * (data["temp"] - temp_i)
        / ((np.log(CONSTANTS["H_AWS"] / CONSTANTS["Z"])) ** 2)
    )

    Ql = (
        mu_cone
        * 0.623
        * CONSTANTS["L_S"]
        * CONSTANTS["RHO_A"]
        / CONSTANTS["P0"]
        * math.pow(CONSTANTS["VAN_KARMAN"], 2)
        * data["wind"]
        * (vp_a - vp_ice)
        / ((np.log(CONSTANTS["H_AWS"] / CONSTANTS["Z"])) ** 2)
    )

    j_cone = -1 * (Ql / CONSTANTS["L_V"] + (Qs+LW) / CONSTANTS["L_F"]) * 60

    return j_cone

def SunMelt(time, coords, utc, alt, obj="WUE"):

    with open("data/common/constants.json") as f:
        CONSTANTS = json.load(f)

    if obj== "WUE":
        cld = 0
        alpha = CONSTANTS["A_I"]
    elif obj== "ICV":
        cld = 1
        alpha = CONSTANTS["A_S"]
    else:
        logger.error("Wrong Objective")
        sys.exit()

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
    clearsky = loc.get_clearsky(times=times, model = 'simplified_solis')
    # Not using measured GHI due to shading effects
    # clearness = irradiance.erbs(ghi = clearsky["ghi"], zenith = solar_position['zenith'],
    #                                   datetime_or_doy= times) 

    df = pd.DataFrame(
        {
            "SW_global": clearsky["ghi"],
            "sea": np.radians(solar_position["elevation"]),
            # "SW_diffuse": clearness["dhi"],
            # "cld": 1 - clearness["kt"],
        }
    )
    bad_values = df["sea"]< 0 
    df["sea"]= np.where(bad_values, 0, df["sea"])
    df["SW_diffuse"]= cld * df["SW_global"]
    df["SW_direct"]= (1-cld) * df["SW_global"]
                            
    df.index += pd.Timedelta(hours=utc)
    df = df.reset_index()
    df["hour"] = df["index"].apply(lambda x: int(x.strftime("%H")))
    df["f_cone"] = 0

    for i in range(0, df.shape[0]):
        if obj == "WUE":
            df.loc[i, "f_cone"] = math.sin(df.loc[i, "sea"])/2
        elif obj == "ICV":
            df.loc[i, "f_cone"] = (math.cos(df.loc[i, "sea"]) + math.pi * math.sin(df.loc[i, "sea"]))/(2*math.sqrt(2)*math.pi)
        else:
            logger.error("Wrong Objective")
            sys.exit()

    df["SW_direct"]= df["SW_global"] - df["SW_diffuse"]
    df["j_cone"] = -1 * (1 - alpha) * (df["SW_direct"] * df["f_cone"] + df["SW_diffuse"]) / CONSTANTS["L_F"] * 60

    model = GaussianModel()
    gauss_params = model.guess(df.j_cone, df.hour)
    result = model.fit(df.j_cone, gauss_params, x=df.hour)
    return result

if __name__ == "__main__":

    loc="gangles21"
    SITE, FOLDER = config(loc,spray="manual")
    utc = get_offset(*SITE["coords"], date=SITE["start_date"])
    result = SunMelt(time=SITE["fountain_off_date"], coords = SITE["coords"], utc = utc, alt=SITE["alt"])
    param_values = dict(result.best_values)
    print(param_values)
    data = {'temp':-5, 'rh':50, 'wind':1, 'alt':1, 'obj':"ICV"}
    print(TempFreeze(data)*math.pi*7**2)

