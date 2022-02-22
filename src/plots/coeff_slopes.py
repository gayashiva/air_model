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

def abline(slope, intercept):
    """Plot a line from slope and intercept"""
    axes = plt.gca()
    x_vals = np.array(axes.get_xlim())
    y_vals = intercept + slope * x_vals
    plt.plot(x_vals, y_vals, '--')

if __name__ == "__main__":

    # Main logger
    logger = logging.getLogger(__name__)
    logger.setLevel("INFO")

    opts = [opt for opt in sys.argv[1:] if opt.startswith("-")]
    # opts = ["-c"]

    if "-c" in opts:

        # temp = list(range(-20, 20,10))
        # rh = list(range(0, 100, 50))
        # v = list(range(0, 20, 10))
        # alt = list(range(0, 5000, 1000))

        temp = list(range(-20, 20))
        rh = list(range(0, 100, 5))
        v = list(range(0, 15, 1))
        alt = list(arange(0, 5, 0.25))
        spray_r = 7

        da = xr.DataArray(
            data=np.zeros(len(temp) * len(rh) * len(v)* len(alt)).reshape(
                len(temp), len(rh), len(v), len(alt)
            ),
            dims=["temp", "rh", "v", "alt"],
            coords=dict(
                temp=temp,
                rh=rh,
                v=v,
                alt=alt,
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

        data = []

        for temp in da.temp.values:
            for rh in da.rh.values:
                for v in da.v.values:
                    for alt in da.alt.values:
                        alt *= 1000
                        da.sel(temp=temp, rh=rh, v=v, alt = alt).data += TempFreeze(temp, rh, v, alt)

        da.to_netcdf("data/common/alt_sims.nc")

    else:
        logger.info("=> Skipping calculation of coeffs")
        da = xr.open_dataarray("data/common/alt_sims.nc")
        x = []
        y = []
        for temp in da.temp.values:
            for rh in da.rh.values:
                for v in da.v.values:
                    for alt in da.alt.values:
                        aws = [temp, rh, v, alt]
                        x.append(aws)
                        y.append(da.sel(temp=temp, rh=rh, v=v,alt=alt).data)

        popt, pcov = curve_fit(line, x, y)
        a, b, c, d, e = popt
        print("dis = %.5f * temp + %.5f * rh + %.5f * wind + %.5f * alt + %.5f" % (a, b, c, d, e))

        """Combine all coeffs"""
        param_values = {}

        with open("data/common/alt_coeffs.nc", "w") as f:
            json.dump(param_values, f)

        print(
            "Max freezing rate:",
            autoDis(**param_values, time=6, temp=-20, rh=0, v=10),
        )


    # else:
    #     df = pd.read_csv("data/common/alt_cld_dependence.csv")
    #     da = xr.open_dataarray("data/common/alt_sims.nc")

    #     df_l = pd.DataFrame(dict(x=[4000,1000], y=[0, 2], text=['Ladakh', 'Swiss']))


    #     a = pd.concat({'x': df_l.x, 'y': df_l.y, 'text': df_l.text}, axis=1)

    #     fig, ax = plt.subplots(1, 1)
    #     ax = df_l.set_index('x')['y'].plot(style='.', color='k', ms=10)
    #     for i, point in a.iterrows():
    #         print(i,point)
    #         ax.text(point['x']+0.125, point['y'], str(point['text']))
    #     da.sel(rh=50, v=2).plot()
    #     # ax.legend(title = "Altitude")
    #     # ax.set_ylabel("Night freezing with 5m spray radius [$l/min$]")
    #     # ax.set_xlabel("Air Temperature [$C$]")
    #     plt.savefig("data/figs/paper3/alt_temp.png", bbox_inches="tight", dpi=300)
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

