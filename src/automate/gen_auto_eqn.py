""" Generate auto equation"""
import sys, json
import os
import xarray as xr
import seaborn as sns
import dask.array as da
import numpy as np
import pandas as pd
import math
import matplotlib.colors
from datetime import datetime, timedelta
from matplotlib.offsetbox import AnchoredText
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import logging, coloredlogs
from scipy.optimize import curve_fit

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
)
from src.utils.settings import config
from src.models.icestupaClass import Icestupa
from src.automate.autoDischarge import TempFreeze, SunMelt
from src.automate.gen_coeffs import line

def autoLinear(a, b, c, d, e, f, temp, rh, wind, alt, cld):
    return a * temp + b * rh + c * wind + d * alt + e * cld + f

if __name__ == "__main__":

    # Main logger
    logger = logging.getLogger(__name__)
    logger.setLevel("INFO")

    opts = [opt for opt in sys.argv[1:] if opt.startswith("-")]

    if opts==[]:
        opts = ["-png"]

    if "-nc" in opts:
        logger.info("=> Calculation of coeffs")

        temp = list(range(-20, 5))
        rh = list(range(0, 100, 10))
        wind = list(range(0, 15, 1))
        alt = list(np.arange(0, 5.1, 0.5))
        cld = list(np.arange(0, 1.1, 0.5))
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

        da.to_netcdf("data/common/alt_sims.nc")

    if "-json" in opts:
        logger.info("=> Performing regression analysis")
        da = xr.open_dataarray("data/common/alt_sims.nc")
        x = []
        y = []
        for temp in da.temp.values:
            for rh in da.rh.values:
                for wind in da.wind.values:
                    for alt in da.alt.values:
                        for cld in da.cld.values:
                            aws = [temp, rh, v, alt, cld]
                            x.append(aws)
                            y.append(da.sel(temp=temp, rh=rh, v=v, alt=alt, cld=cld, spray_r=7).data/(math.pi * 7 * 7))

        popt, pcov = curve_fit(line, x, y)
        a, b, c, d, e, f = popt
        print("dis = %.5f * temp + %.5f * rh + %.5f * wind + %.5f * alt + %.5f * cld + %.5f" % (a, b, c, d, e, f))

        """Combine all coeffs"""
        param_values = {}
        param_values["a"] = a
        param_values["b"] = b
        param_values["c"] = c
        param_values["d"] = d
        param_values["e"] = e
        param_values["f"] = f

        with open("data/common/alt_coeffs.json", "w") as f:
            json.dump(param_values, f)

    if "-png" in opts:
        logger.info("=> Producing figs")

        with open("data/common/alt_coeffs.json") as f:
            param_values = json.load(f)
        print(
            "dis = %.5f * temp + %.5f * rh + %.5f * wind + %.5f * alt + %.5f * cld + %.5f "
            % (
                param_values['a'],
                param_values["b"],
                param_values["c"],
                param_values["d"],
                param_values["e"],
                param_values["f"],
            )
        )

        da = xr.open_dataarray("data/common/alt_sims.nc")

        df_l = pd.DataFrame(dict(x=[4,1], y=[0, 2], text=['Ladakh', 'Swiss']))
        a = pd.concat({'x': df_l.x, 'y': df_l.y, 'text': df_l.text}, axis=1)

        fig, ax = plt.subplots(1, 1)
        # ax = df_l.set_index('x')['y'].plot(style='.', color='k', ms=10)
        # for i, point in a.iterrows():
        #     print(i,point)
        #     ax.text(point['x']+0.125, point['y'], str(point['text']))
        da.sel(rh = 30, v=2, cld =1, spray_r=7 ).plot()
        # ax.set_ylim([0,3])
        plt.savefig("data/figs/paper3/alt_temp.png", bbox_inches="tight", dpi=300)
