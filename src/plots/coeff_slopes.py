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
# from src.models.methods.metadata import get_parameter_metadata

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

    compile = False

    if compile:

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

        alt_list = list(range(0,5000,1000))
        cld_list = list(np.arange(0,1,0.25))
        data = [] 

        for alt in alt_list:
            for cld in cld_list:
                logger.warning("=> Computing coeffs for altitude {} and cloudiness {}".format(alt, cld))

                for temp in da.temp.values:
                    for rh in da.rh.values:
                        for v in da.v.values:
                            aws = [temp, rh, v]
                            da.sel(temp=temp, rh=rh, v=v).data+= TempFreeze(aws, cld=0, alt=alt)
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
                data.append([alt,cld,a,b,c,d])
                # print(param_list)
                print("For %s, dis = %.5f * temp + %.5f * rh + %.5f * wind + %.5f" % (alt, a, b, c, d))

        df = pd.DataFrame(data, columns=['alt', 'cld', 'temp', 'rh', 'wind', 'constant'])
        print(df.head())
        df.to_csv("data/common/alt_cld_dependence.csv")

        temp = list(range(-30, 20))
        rh = list(range(0, 100, 5))
        v = list(range(0, 20, 1))
        alt = list(range(0, 5000, 1000))

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

        data = [] 

        for temp in da.temp.values:
            for rh in da.rh.values:
                for v in da.v.values:
                    for i in range(0, df.shape[0], 4):
                        aws = [temp, rh, v]
                        da.sel(temp=temp, rh=rh, v=v, alt = df.alt[i]).data= df.temp[i] * temp + df.rh[i] * rh +df.wind[i] * v + df.constant[i]
                        da.sel(temp=temp, rh=rh, v=v, alt = df.alt[i]).data *= math.pi * 5 * 5

        da.to_netcdf("data/common/alt_sims.nc")
    else:
        df = pd.read_csv("data/common/alt_cld_dependence.csv")
        da = xr.open_dataarray("data/common/alt_sims.nc")

        # x_vals = list(range(-10, 10))
        # df = df.round(4)
        # print(df.tail())
        # fig, ax = plt.subplots(1, 1)
        # for i in range(0, df.shape[0], 4):
        #     y_vals = []
        #     intercept = df.constant[i]
        #     slope = df.temp[i]
        #     for x in x_vals:
        #         y_vals.append((intercept + slope * x) * math.pi * 25)
        #     ax.plot(x_vals, y_vals, '--', label = str(df.alt[i]))
        # # ax.scatter(df.alt, df.dis, s=100, c=df.cld, cmap='Blues')
        # ax.legend(title = "Altitude")
        # ax.set_ylabel("Night freezing with 5m spray radius [$l/min$]")
        # ax.set_xlabel("Air Temperature [$C$]")
        # plt.savefig("data/figs/paper3/coeff_slopes.png", bbox_inches="tight", dpi=300)



#     with open("data/common/constants.json") as f:
#         CONSTANTS = json.load(f)
#     SITE, FOLDER = config(location)
#     with open(FOLDER["input"] + "auto/coeffs.json", "w") as f:
#         json.dump(param_values, f)

# countries = ['Ireland', 'Norway',
#              'Switzerland', 'Cayman Islands',
#              'China, Macao Special Administrative Region']
# fig, ax = plt.subplots(1, figsize=(10,10))
# for i in countries:
#     temp = df[df['Country or Area'] == i]
#     plt.plot(temp.Year, temp.Value)
#     
#     ax.set_ylabel("Discharge [$l/min$]")


#     plt.savefig("data/figs/paper3/coeff_slopes.png", bbox_inches="tight", dpi=300)
