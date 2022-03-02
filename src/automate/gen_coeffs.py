"""Generate coefficients for empirical fountain scheduler"""
import pandas as pd
import multiprocessing
import xarray as xr
import numpy as np
import os, sys
import json
import math
from datetime import datetime, timedelta
import seaborn as sns
import matplotlib.dates as mdates
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

def line(x, a, b, c, d):
    x1 = x[:, 0]
    x2 = x[:, 1]
    x3 = x[:, 2]
    return a * x1 + b * x2 + c * x3 + d

def autoDis(a, b, c, d, amplitude, center, sigma, time, temp, rh, wind):
    model = GaussianModel()
    params = {"amplitude": amplitude, "center": center, "sigma": sigma}
    return a * temp + b * rh + c * wind + d + model.eval(x=time, **params)

if __name__ == "__main__":
    # Main logger
    logger = logging.getLogger(__name__)
    logger.setLevel("INFO")

    opts = [opt for opt in sys.argv[1:] if opt.startswith("-")]

    if opts==[]:
        # opts = ["-nc", "-solar", "-json", "-test"]
        # opts = ["-png"]
        opts = ["-json", "-png"]

    locations = ["gangles21", "guttannen21"]
    # locations = ["guttannen21", "guttannen22", "guttannen20", "gangles21"]

    for loc in locations:
        print(loc)

        with open("data/common/constants.json") as f:
            CONSTANTS= json.load(f)

        SITE, FOLDER = config(loc, spray="manual")

        with open(FOLDER["output"] + "manual/results.json", "r") as read_file:
            results = json.load(read_file)

        if "-json" in opts:

            logger.info("=> Calculation of solar coeffs")
            utc = get_offset(*SITE["coords"], date=SITE["start_date"])
            result = SunMelt(time = SITE["fountain_off_date"], coords = SITE["coords"], utc = utc, alt = SITE["alt"])

            sun_params = dict(result.best_values)
            print(
                "dis = Gaussian(time; Amplitude = %.5f, center = %.5f, sigma = %.5f) for %s "
                % (
                    sun_params["amplitude"],
                    sun_params["center"],
                    sun_params["sigma"],
                    loc,
                )
            )

            # with open(FOLDER["input"] + "dynamic/sun_coeffs.json", "w") as f:
            #     json.dump(dict(result.best_values), f, indent=4)

            logger.info("=> Performing regression analysis")
            da = xr.open_dataarray("data/common/alt_cld_sims.nc")
            x = []
            y = []
            for temp in da.temp.values:
                for rh in da.rh.values:
                    for wind in da.wind.values:
                        aws = [temp, rh, wind]
                        x.append(aws)
                        y.append(da.sel(
                                     temp=temp, 
                                     rh=rh, 
                                     wind=wind,
                                     alt=round(SITE["alt"]/1000,0),
                                     cld=round(SITE["cld"],0),
                                     spray_r=round(results["R_F"],0)).data)

            popt, pcov = curve_fit(line, x, y)
            a, b, c, d = popt
            print("dis = %.5f * temp + %.5f * rh + %.5f * wind + %.5f" % (a, b, c, d))

            """Combine all coeffs"""
            params= {}
            params["a"] = a
            params["b"] = b
            params["c"] = c
            params["d"] = d

            params = dict(params, **sun_params)

            with open(FOLDER["input"] + "dynamic/coeffs.json", "w") as f:
                json.dump(params, f, indent=4)

        if "-test" in opts:

            # TODO Scale all coeffs ?
            # param_values.update((x, y*scaling_factor) for x, y in param_values.items())

            for loc in locations:

                SITE, FOLDER = config(loc, spray="manual")

                with open(FOLDER["input"] + "dynamic/coeffs.json") as f:
                    params = json.load(f)

                max_freeze = autoDis(**params, time=6, temp=-20, rh=50, wind=0)
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

            spray = "static"
            mypal = sns.color_palette("Set1", 2)
            for loc in locations:

                SITE, FOLDER = config(loc, spray)

                with open(FOLDER["input"] + "dynamic/coeffs.json") as f:
                    params = json.load(f)

                icestupa = Icestupa(loc, spray)
                icestupa.read_output()
                df = icestupa.df
                df = df[df.time <= icestupa.fountain_off_date]

                df["Discharge_sim"] = 0 
                
                for i in range(0, df.shape[0]):
                    df.loc[i, "Discharge_sim"] = autoDis(**params, time=df.time[i].hour, temp=df.temp[i],
                                                         rh=df.RH[i], wind = df.wind[i])
                    if df.Discharge_sim[i] <0:
                        df.loc[i, "Discharge_sim"] = 0
                        

                fig, ax1 = plt.subplots()

                ax1.scatter(df.fountain_froze/60, df.Discharge_sim, s=2)
                ax1.set_xlabel("Freezing rate")
                ax1.set_ylabel("Scheduled discharge")
                ax1.grid()

                lims = [
                np.min([ax1.get_xlim(), ax1.get_ylim()]),  # min of both axes
                np.max([ax1.get_xlim(), ax1.get_ylim()]),  # max of both axes
                ]

                ax1.plot(lims, lims, '--k', alpha=0.25, zorder=0)
                ax1.set_aspect('equal')
                ax1.set_xlim(lims)
                ax1.set_ylim(lims)

                plt.savefig(FOLDER["fig"] + "scheduled_discharge_corr.png", bbox_inches="tight", dpi=300)
                plt.clf()

                fig, ax1 = plt.subplots()
                x = df.time
                y1 = df.Discharge_sim
                y2 = df.fountain_froze /60
                ax1.plot(
                    x,
                    y1,
                    color=mypal[0],
                )
                ax1.set_ylim(0, 30)

                ax2 = ax1.twinx()
                ax2.set_ylabel('Freezing rate [$l/min$]', color = mypal[1])
                ax2.plot(x, y2, color = mypal[1], alpha=0.5)
                ax2.tick_params(axis ='y', labelcolor = mypal[1])
                ax2.set_ylim(0, 30)

                # ax2.spines["top"].set_visible(False)
                ax1.spines["top"].set_visible(False)
                ax1.set_ylabel("Discharge rate [$l/min$]")
                ax1.xaxis.set_major_locator(mdates.WeekdayLocator())
                ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
                fig.autofmt_xdate()
                plt.savefig(FOLDER["fig"] + "scheduled_discharge.png", bbox_inches="tight", dpi=300)


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

