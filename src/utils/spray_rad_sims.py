import sys
import os
import numpy as np
import pandas as pd

pd.options.mode.chained_assignment = None  # default='warn'
import xarray as xr
import math
import matplotlib.colors
import statistics as st
from datetime import datetime, timedelta
from matplotlib.offsetbox import AnchoredText
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import logging, coloredlogs
import multiprocessing
from time import sleep
import os, sys, time

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
)

from src.utils.settings import config
from src.models.icestupaClass import Icestupa
from src.models.methods.metadata import get_parameter_metadata

# define worker function
def calculate(process_name, tasks, results):
    print("[%s] evaluation routine starts" % process_name)

    while True:
        new_value = tasks.get()
        if new_value == "None":
            print("[%s] evaluation routine quits" % process_name)

            # Indicate finished
            results.put(-1)
            break
        else:
            icestupa = Icestupa()
            icestupa_sim.R_F = new_value
            icestupa.derive_parameters()
            icestupa.melt_freeze()
            # Compute result and mimic a long-running task
            compute = icestupa.df.iceV.max()

            # Output which process received the value
            print("[%s] received value: %s" % (process_name, new_value))
            print("[%s] calculated max ice volume: %.1f" % (process_name, compute))

            # Add result to the queue
            results.put(compute)

    return


if __name__ == "__main__":
    # Main logger
    logger = logging.getLogger(__name__)
    logger.setLevel("WARNING")
    loc = "guttannen21"
    rads = list(range(20, 51, 15))

    # compile = True
    compile = False

    if compile:
        time = pd.date_range("2020-11-01", freq="H", periods=365 * 24)
        ds = xr.DataArray(
            dims=["time", "rads"],
            coords={"time": time, "rads": rads},
            attrs=dict(description="coords with matrices"),
        )

        SITE, FOLDER = config(loc)
        for rad in rads:
            df = pd.DataFrame()
            icestupa_sim = Icestupa(loc)
            icestupa_sim.R_F = rad
            icestupa_sim.derive_parameters()
            icestupa_sim.melt_freeze()
            df = icestupa_sim.df[["time", "iceV"]]
            df = df.loc[df.iceV != 0].reset_index()
            ds.loc[dict(time=df.time.values[1:], rads=rad)] = (
                df.iceV.values[1:] - icestupa_sim.V_dome
            )

        ds.to_netcdf("data/slides/rads.nc")
    else:
        ds = xr.open_dataarray("data/slides/rads.nc")
        icestupa = Icestupa(loc)
        plt.figure()
        ax = plt.gca()
        for rad in rads:
            ds.sel(rads=rad).plot(label=rad)

            V = round(np.nanmax(ds.sel(rads=rad).data), 0)
            days = round(sum(~np.isnan(ds.sel(rads=rad).values)) / 24, 0)
            print(days, V)
        plt.legend()
        plt.grid()
        plt.savefig("data/slides/rads.jpg")
