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
    return a * temp + b * rh + c * wind + d * alt + e + model.eval(x=time, **params)


if __name__ == "__main__":
    # Main logger
    logger = logging.getLogger(__name__)
    logger.setLevel("INFO")


    opts = [opt for opt in sys.argv[1:] if opt.startswith("-")]

    if opts==[]:
        opts = ["-nc", "-solar", "-json", "-test"]

    if "-nc" in opts:
        logger.info("=> Calculation of temp coeffs")
        temp = list(range(-20, 5))
        rh = list(range(0, 100, 5))
        wind = list(range(0, 15, 1))
        alt = list(np.arange(0, 5.1, 0.1))
        cld = list(np.arange(0, 1.1, 0.1))
        spray_r = list(np.arange(5, 11, 1))

        da = xr.DataArray(
            data=np.zeros(len(temp) * len(rh) * len(wind)* len(alt) * len(cld) * len(spray_r)).reshape(
                len(temp), len(rh), len(wind), len(alt), len(cld), len(spray_r)
            ),
            dims=["temp", "rh", "wind", "alt", "cld", "spray_r"],
            coords=dict(
                temp=temp,
                rh=rh,
                wind=wind,
                alt=alt,
                cld=cld,
                spray_r=spray_r,
            ),
            attrs=dict(
                long_name="Freezing rate",
                description="Mean freezing rate",
                units="$l\\, min^{-1}$",
            ),
        )

        da.temp.attrs["units"] = "$\\degree C$"
        da.temp.attrs["description"] = "Air Temperature"
        da.temp.attrs["long_name"] = "Air Temperature"
        da.rh.attrs["units"] = "%"
        da.rh.attrs["long_name"] = "Relative Humidity"
        da.wind.attrs["units"] = "$m\\, s^{-1}$"
        da.wind.attrs["long_name"] = "Wind Speed"
        da.alt.attrs["units"] = "$km$"
        da.alt.attrs["long_name"] = "Altitude"
        da.cld.attrs["units"] = " "
        da.cld.attrs["long_name"] = "Cloudiness"
        da.spray_r.attrs["units"] = "$m$"
        da.spray_r.attrs["long_name"] = "Spray radius"

        for temp in da.temp.values: 
            for rh in da.rh.values:
                for wind in da.wind.values:
                    for alt in da.alt.values:
                        for cld in da.cld.values:
                            for spray_r in da.spray_r.values: 
                                da.sel(temp=temp, rh=rh, wind=wind, alt=alt, cld=cld, spray_r = spray_r).data +=TempFreeze(temp, rh, wind, alt, cld)
                                da.sel(temp=temp, rh=rh, wind=wind, alt=alt, cld=cld, spray_r = spray_r).data *= math.pi * spray_r * spray_r


        da.to_netcdf("data/common/alt_cld_sims.nc")

    locations = ["gangles21", "guttannen21"]
    # locations = ["guttannen21", "guttannen22", "guttannen20", "gangles21"]

    for loc in locations:
        with open("data/common/constants.json") as f:
            CONSTANTS= json.load(f)

        SITE, FOLDER = config(loc, spray="manual")

        if "-solar" in opts:
            logger.info("=> Calculation of solar coeffs")
            utc = get_offset(*SITE["coords"], date=SITE["start_date"])
            result = SunMelt(time = SITE["fountain_off_date"], coords = SITE["coords"], utc = utc, alt = SITE["alt"])

            with open(FOLDER["input"] + "dynamic/sun_coeffs.json", "w") as f:
                json.dump(dict(result.best_values), f, indent=4)

            params = dict(result.best_values)
            print(
                "dis = Gaussian(time; Amplitude = %.5f, center = %.5f, sigma = %.5f) for %s "
                % (
                    params["amplitude"],
                    params["center"],
                    params["sigma"],
                    loc,
                )
            )
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


    if "-json" in opts:
        logger.info("=> Performing regression analysis")
        da = xr.open_dataarray("data/common/temp_sims.nc")
        x = []
        y = []
        for temp in da.temp.values:
            for rh in da.rh.values:
                for wind in da.wind.values:
                    for alt in da.alt.values:
                        aws = [temp, rh, wind, alt]
                        x.append(aws)
                        y.append(da.sel(temp=temp, rh=rh, wind=wind, alt=alt, spray_r=7).data/(math.pi * 7 * 7))

        popt, pcov = curve_fit(line, x, y)
        a, b, c, d, e = popt
        print("dis = %.5f * temp + %.5f * rh + %.5f * wind + %.5f * alt + %.5f" % (a, b, c, d, e))

        """Combine all coeffs"""
        params= {}
        params["a"] = a
        params["b"] = b
        params["c"] = c
        params["d"] = d
        params["e"] = e

        with open("data/common/temp_coeffs.json", "w") as f:
            json.dump(params, f, indent=4)

        with open(FOLDER["input"] + "dynamic/sun_coeffs.json") as f:
            sun_params = json.load(f)

        params = dict(params, **sun_params)

        with open(FOLDER["input"] + "dynamic/coeffs.json", "w") as f:
            json.dump(params, f, indent=4)

    if "-test" in opts:
        # TODO Scale all coeffs ?
        # param_values.update((x, y*scaling_factor) for x, y in param_values.items())

        for loc in locations:

            SITE, FOLDER = config(loc, spray="manual")

            with open(FOLDER["input"] + "dynamic/coeffs.json", "w") as f:
                json.dump(params, f)
            print(params)

            max_freeze = autoDis(**params, time=6, temp=-20, rh=0, wind=10, alt=SITE["alt"]/1000) * math.pi * 7 **2
            print(
                "Max freezing rate: %0.1f for loc %s"%(max_freeze, loc)
            )

            # print(param_values)
            # print(
            #     "dis = %.5f * temp + %.5f * rh + %.5f * wind + %.5f + Gaussian(time; Amplitude = %.5f, center = %.5f, sigma = %.5f) "
            #     % (
            #         params["a"],
            #         params["b"],
            #         params["c"],
            #         params["d"],
            #         params["amplitude"],
            #         params["center"],
            #         params["sigma"],
            #     )
            # )

    if "-png" in opts:
        logger.info("=> Producing figs")


