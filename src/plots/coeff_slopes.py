""" Plot comparing auto and manual discharge at guttannen"""
import sys, json
import os
import xarray as xr
import seaborn as sns
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

def autoDis(a, b, c, d, e, f, temp, time, rh, v, alt, cld):
    return a * temp + b * rh + c * v + d * alt + e * cld + f

if __name__ == "__main__":

    # Main logger
    logger = logging.getLogger(__name__)
    logger.setLevel("INFO")

    opts = [opt for opt in sys.argv[1:] if opt.startswith("-")]

    if opts==[]:
        opts = ["-nc"]

    if "-nc" in opts:
        logger.info("=> Calculation of coeffs")

        temp = list(range(-10, 10))
        rh = list(range(0, 100, 10))
        v = list(range(0, 15, 1))
        alt = list(np.arange(0, 5.1, 0.5))
        cld = list(np.arange(0, 1.1, 0.5))

        da = xr.DataArray(
            data=np.zeros(len(temp) * len(rh) * len(v)* len(alt)* len(cld)).reshape(
                len(temp), len(rh), len(v), len(alt), len(cld)
            ),
            dims=["temp", "rh", "v", "alt", "cld"],
            coords=dict(
                temp=temp,
                rh=rh,
                v=v,
                alt=alt,
                cld=cld,
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
        da.alt.attrs["units"] = "km"
        da.alt.attrs["long_name"] = "Altitude"
        da.alt.attrs["units"] = " "
        da.alt.attrs["long_name"] = "Cloudiness"

        data = []

        for temp in da.temp.values:
            for rh in da.rh.values:
                for v in da.v.values:
                    for alt in da.alt.values:
                        for cld in da.cld.values:
                            da.sel(temp=temp, rh=rh, v=v, alt=alt, cld=cld).data += TempFreeze(temp, rh, v, alt, cld)

        da.to_netcdf("data/common/alt_sims.nc")

    if "-json" in opts:
        logger.info("=> Performing regression analysis")
        da = xr.open_dataarray("data/common/alt_sims.nc")
        x = []
        y = []
        for temp in da.temp.values:
            for rh in da.rh.values:
                for v in da.v.values:
                    for alt in da.alt.values:
                        for cld in da.cld.values:
                            aws = [temp, rh, v, alt, cld]
                            x.append(aws)
                            y.append(da.sel(temp=temp, rh=rh, v=v, alt=alt, cld=cld).data)

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

        da = xr.open_dataarray("data/common/alt_sims.nc")
        df_l = pd.DataFrame(dict(x=[4,1], y=[0, 2], text=['Ladakh', 'Swiss']))
        a = pd.concat({'x': df_l.x, 'y': df_l.y, 'text': df_l.text}, axis=1)
        print(da)

        fig, ax = plt.subplots(1, 1)
        ax = df_l.set_index('x')['y'].plot(style='.', color='k', ms=10)
        for i, point in a.iterrows():
            print(i,point)
            ax.text(point['x']+0.125, point['y'], str(point['text']))
        da.sel(rh=50, v=2,cld=0.5).plot()
        plt.savefig("data/figs/paper3/alt_temp.png", bbox_inches="tight", dpi=300)

        # ax.legend(title = "Altitude")
        # ax.set_ylabel("Night freezing with 5m spray radius [$l/min$]")
        # ax.set_xlabel("Air Temperature [$C$]")
    #     # x_vals = list(range(-10, 10))
    #     # df = df.round(4)
    #     # print(df.tail())
    #     # fig, ax = plt.subplots(1, 1)
    #     # for i in range(0, df.shape[0], 4):
    #     #     y_vals = []
    #     #     intercept = df.constant[i]
    #     #     slope = df.temp[i]
    #     #     for x in x_vals:
    #     #         y_vals.append((intercept + slope * x) * math.pi * 25)
    #     #     ax.plot(x_vals, y_vals, '--', label = str(df.alt[i]))
    #     # # ax.scatter(df.alt, df.dis, s=100, c=df.cld, cmap='Blues')
    #     # ax.legend(title = "Altitude")
    #     # ax.set_ylabel("Night freezing with 5m spray radius [$l/min$]")
    #     # ax.set_xlabel("Air Temperature [$C$]")
    #     # plt.savefig("data/figs/paper3/coeff_slopes.png", bbox_inches="tight", dpi=300)

