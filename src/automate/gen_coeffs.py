"""Generate coefficients for empirical fountain scheduler"""
import pandas as pd
import xarray as xr
import numpy as np
import os, sys
import json
import math
from datetime import datetime, timedelta

import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from lmfit.models import GaussianModel
import logging
import coloredlogs

dirname = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.append(dirname)
from src.utils.settings import config
from src.models.icestupaClass import Icestupa
from src.models.methods.metadata import get_parameter_metadata
from src.automate.autoDischarge import TempFreeze, SunMelt
from src.models.methods.solar import get_offset
# from src.automate.projectile import get_projectile

def line(x, a, b, c, d, e):
    x1 = x[:, 0]
    x2 = x[:, 1]
    x3 = x[:, 2]
    x4 = x[:, 3]
    return a * x1 + b * x2 + c * x3 + d * x4 + e

def autoDis(a, b, c, d, e, amplitude, center, sigma, time, temp, rh, wind, alt):
    model = GaussianModel()
    params = {"amplitude": amplitude, "center": center, "sigma": sigma}
    return a * temp + b * rh + c * v + d * alt + e + model.eval(x=time, **params)


if __name__ == "__main__":
    # Main logger
    logger = logging.getLogger(__name__)
    logger.setLevel("INFO")

    # locations = ["gangles21", "guttannen21"]
    locations = ["guttannen21", "guttannen22", "guttannen20", "gangles21"]

    for loc in locations:
        with open("data/common/auto.json") as f:
            params = json.load(f)
        with open("data/common/constants.json") as f:
            CONSTANTS= json.load(f)

        SITE, FOLDER = config(loc, spray="man")

        # icestupa_sim = Icestupa(loc)
        # icestupa_sim.read_output()

        # df = icestupa_sim.df

        # temp_cutoff = 0.25
        # wind_cutoff = 0.5
        # crit_temp = df.loc[df.time < datetime(2021, 3, 1)].temp.quantile(temp_cutoff)
        # crit_rh = df.loc[df.time < datetime(2021, 3, 1)].RH.quantile(temp_cutoff)
        # crit_wind = df.loc[df.time < datetime(2021, 3, 1)].wind.quantile(wind_cutoff)
        # freeze_when = [crit_temp,crit_rh, crit_wind]


        # """Calculate Virtual radius"""
        # print(f"The temperature, humidity and wind were less/more than {freeze_when} for {temp_cutoff} of the months of Jan and Feb" )
        # dis = TempFreeze(freeze_when,loc)
        # scaling_factor = params['crit_dis']/dis

        # print(f"Radius for {loc} is {params['spray_r']}" )
        # print(f"So only {freezing_fraction*100}% froze from the discharge rate" )
        # print(f"Corresponding discharge for {loc} is {dis}" )
        # print(f"Recommended scaling factor is {scaling_factor}" )

        # with open(FOLDER["raw"] + "auto/auto_info.json", "w") as f:
        #     json.dump(params, f)

        """Calculate Solar gaussian coeffs"""
        utc = get_offset(*SITE["coords"], date=SITE["start_date"])
        cld, result = SunMelt(coords = SITE["coords"], utc = utc, alt = SITE["alt"])

        with open(FOLDER["input"] + "auto/sunmelt.json", "w") as f:
            json.dump(dict(result.best_values), f)

        compile = True
        if not os.path.exists(FOLDER["input"] + "auto/sims.nc") or compile:
            logger.warning("=> Computing temp coeffs for location {}".format(loc))
            """Compute Temp coeffs"""
            temp = list(range(-30, 20))
            rh = list(range(0, 100, 5))
            v = list(range(0, 20, 1))

            da = xr.DataArray(
                data=np.zeros(len(temp) * len(rh) * len(v)).reshape(
                    len(temp), len(rh), len(v)
                ),
                dims=["temp", "rh", "v"],
                coords=dict(
                    temp=temp,
                    rh=rh,
                    v=v,
                ),
                attrs=dict(
                    long_name="Freezing rate",
                    description="Max. freezing rate",
                    units="l min-1",
                ),
            )

            da.temp.attrs["units"] = "deg C"
            da.temp.attrs["description"] = "Air Temperature"
            da.temp.attrs["long_name"] = "Air Temperature"
            da.rh.attrs["units"] = "%"
            da.rh.attrs["long_name"] = "Relative Humidity"
            da.v.attrs["units"] = "m s-1"
            da.v.attrs["long_name"] = "Wind Speed"

            for temp in da.temp.values:
                for rh in da.rh.values:
                    for v in da.v.values:
                        aws = [temp, rh, v]
                        da.sel(temp=temp, rh=rh, v=v).data+= TempFreeze(aws, cld, SITE["alt"])
            da.to_netcdf(FOLDER["input"] + "auto/sims.nc")
        else:
            logger.info("=> Skipping temp coeffs for location {}".format(loc))
            da = xr.open_dataarray(FOLDER["input"] + "auto/sims.nc")


        x = []
        y = []
        for temp in da.temp.values:
            for rh in da.rh.values:
                for v in da.v.values:
                    aws = [temp, rh, v]
                    x.append(aws)
                    y.append(da.sel(temp=temp, rh=rh, v=v).data)

        popt, pcov = curve_fit(line, x, y)
        a, b, c, d = popt
        print("For %s, dis = %.5f * temp + %.5f * rh + %.5f * wind + %.5f" % (loc, a, b, c, d))

        """Combine all coeffs"""
        param_values = {}

        with open(FOLDER["input"] + "auto/sunmelt.json") as f:
            param_values = json.load(f)

        param_values["a"] = a
        param_values["b"] = b
        param_values["c"] = c
        param_values["d"] = d

        # TODO Scale all coeffs ?
        # param_values.update((x, y*scaling_factor) for x, y in param_values.items())

        with open(FOLDER["input"] + "auto/coeffs.json", "w") as f:
            json.dump(param_values, f)

        print(
            "Max freezing rate:",
            autoDis(**param_values, time=6, temp=-20, rh=0, v=10),
        )

        # print(param_values)
        print(
            "dis = %.5f * temp + %.5f * rh + %.5f * wind + %.5f + Gaussian(time; Amplitude = %.5f, center = %.5f, sigma = %.5f) "
            % (
                param_values["a"],
                param_values["b"],
                param_values["c"],
                param_values["d"],
                param_values["amplitude"],
                param_values["center"],
                param_values["sigma"],
            )
        )
