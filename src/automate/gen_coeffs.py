
import pandas as pd
import xarray as xr
import numpy as np
import os, sys
import json
import math
from datetime import datetime, timedelta

from autoDischarge import TempFreeze, SunMelt
from projectile import get_projectile
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

def line(x, a, b, c, d):
    x1 = x[:, 0]
    x2 = x[:, 1]
    x3 = x[:, 2]
    return a * x1 + b * x2 + c * x3 + d

def autoDis(a, b, c, d, amplitude, center, sigma, temp, time, rh, v):
    model = GaussianModel()
    params = {"amplitude": amplitude, "center": center, "sigma": sigma}
    return a * temp + b * rh + c * v + d + model.eval(x=time, **params)


if __name__ == "__main__":
    # Main logger
    logger = logging.getLogger(__name__)
    logger.setLevel("ERROR")

    # locations = ["gangles21", "guttannen21"]
    locations = ["guttannen21"]

    for loc in locations:
        constants, SITE, FOLDER = config(loc)
        icestupa_sim = Icestupa(loc)
        icestupa_sim.read_output()

        df = icestupa_sim.df

        temp_cutoff = 0.25
        wind_cutoff = 0.5
        crit_temp = df.loc[df.time < datetime(2021, 3, 1)].temp.quantile(temp_cutoff)
        crit_rh = df.loc[df.time < datetime(2021, 3, 1)].RH.quantile(temp_cutoff)
        crit_wind = df.loc[df.time < datetime(2021, 3, 1)].wind.quantile(wind_cutoff)
        freeze_when = [crit_temp,crit_rh, crit_wind]

        with open(FOLDER["raw"] + "info.json") as f:
            params = json.load(f)

        """Calculate Virtual radius"""
        print(f"The temperature, humidity and wind were less/more than {freeze_when} for {temp_cutoff} of the months of Jan and Feb" )
        growth_rate = TempFreeze(freeze_when,loc)

        dis_real = get_projectile(h_f=params["h_f"], dia=params["dia_f"], r=params["r_real"])

        VA = dis_real/growth_rate
        # params["r_virtual"] = round(math.sqrt(VA/(math.pi*math.sqrt(2))),2)
        r_virtual = round(math.sqrt(VA/(math.pi*math.sqrt(2))),2)
        freezing_fraction = round(params['r_real'] ** 2/r_virtual ** 2,2)

        print(f"Virtual radius for {loc} is {r_virtual} for recommended radius of {params['r_real']}" )
        print(f"So only {freezing_fraction*100}% froze from the discharge rate" )
        print(f"Recommended discharge for {loc} is {dis_real}" )

        with open(FOLDER["raw"] + "info.json", "w") as f:
            json.dump(params, f)

        """Calculate Solar gaussian coeffs"""
        result = SunMelt(loc)

        with open(FOLDER["input"] + "sunmelt.json", "w") as f:
            json.dump(dict(result.best_values), f)

        """Compute Temp coeffs"""
        temp = list(range(params["temp"][0], params["temp"][1] + 1))
        rh = list(range(params["rh"][0], params["rh"][1] + 1))
        v = list(range(params["wind"][0], params["wind"][1] + 1))

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

        x = []
        y = []
        for temp in da.temp.values:
            for rh in da.rh.values:
                for v in da.v.values:
                    aws = [temp, rh, v]
                    da.sel(temp=temp, rh=rh, v=v).data += TempFreeze(aws, loc, r_virtual)
                    x.append(aws)
                    y.append(da.sel(temp=temp, rh=rh, v=v).data)

        da.to_netcdf(FOLDER["sim"] + "auto_sims.nc")

        popt, pcov = curve_fit(line, x, y)
        a, b, c, d = popt
        print("For %s, dis = %.5f * temp + %.5f * rh + %.5f * wind + %.5f" % (loc, a, b, c, d))

        """Combine all coeffs"""
        param_values = {}

        with open(FOLDER["input"] + "sunmelt.json") as f:
            param_values = json.load(f)

        param_values["a"] = a
        param_values["b"] = b
        param_values["c"] = c
        param_values["d"] = d

        with open(FOLDER["sim"] + "coeffs.json", "w") as f:
            json.dump(param_values, f)

        print(
            "Max freezing rate:",
            autoDis(**param_values, time=6, temp=params["temp"][0], rh=params["rh"][0], v=params["wind"][1]+ 1),
        )

        print(
            "y = %.5f * temp + %.5f * rh + %.5f * wind + %.5f + Gaussian(time; Amplitude = %.5f, center = %.5f, sigma = %.5f) "
            % (
                a,
                b,
                c,
                d,
                param_values["amplitude"],
                param_values["center"],
                param_values["sigma"],
            )
        )
