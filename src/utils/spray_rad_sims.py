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
def calculate(process_name, tasks, loc, results, results_list):
    print("[%s] evaluation routine starts" % process_name)

    while True:
        new_value = tasks.get()
        if new_value == "None":
            print("[%s] evaluation routine quits" % process_name)
            results.put(-1)
            break
        else:
            icestupa = Icestupa(loc)
            icestupa.R_F = new_value
            icestupa.derive_parameters()
            icestupa.melt_freeze()
            df = icestupa.df[["time", "iceV"]]
            df = df.loc[df.iceV != 0].reset_index()
            days = int(df.index[-1] / 24)
            maxV = int(df.iceV.max())
            # Compute result and mimic a long-running task
            compute = maxV

            # Output which process received the value
            print("[%s] received value: %s" % (process_name, new_value))
            print("[%s] calculated max ice volume: %.1f" % (process_name, compute))

            # Add result to the queue
            results.put(compute)
            results_list.append([new_value, days, maxV])
    return


if __name__ == "__main__":
    # Main logger
    logger = logging.getLogger(__name__)
    logger.setLevel("WARNING")
    loc = "guttannen21"
    SITE, FOLDER = config(loc)
    rads = list(range(5, 101, 5))
    file_path = "rad_sims.csv"

    # Define IPC manager
    manager = multiprocessing.Manager()

    # Define a list (queue) for tasks and computation results
    tasks = manager.Queue()
    results = manager.Queue()
    results_list = manager.list()
    # Create process pool with four processes
    num_processes = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(processes=num_processes)
    processes = []
    num_finished_tasks = 0

    print()
    ctr = len(rads)
    runtime = 40
    days = ctr * runtime / (num_processes * 60 * 60 * 24)
    print("Total hours expected : %0.01f" % int(days * 24))
    print("Total days expected : %0.01f" % days)

    # Initiate the worker processes
    for i in range(num_processes):

        # Set process name
        process_name = "P%i" % i

        # Create the process, and connect it to the worker function
        new_process = multiprocessing.Process(
            target=calculate, args=(process_name, tasks, loc, results, results_list)
        )

        # Add new process to the list of processes
        processes.append(new_process)

        # Start the process
        new_process.start()

    # Fill task queue
    task_list = rads

    for single_task in task_list:
        tasks.put(single_task)

    # Wait while the workers process
    sleep(0.5)

    # Quit the worker processes by sending them -1
    for i in range(num_processes):
        tasks.put("None")

    # Read calculation results
    num_finished_processes = 0

    while True:
        # Read result
        new_result = results.get()
        num_finished_tasks += 1

        # Have a look at the results
        if new_result == -1:
            # Process has finished
            num_finished_processes += 1

            if num_finished_processes == num_processes:
                print(results_list)
                df = pd.DataFrame.from_records(
                    results_list, columns=["rad", "days", "maxV"]
                )
                df = df.set_index("rad").sort_index()
                print(df.head())
                df.to_csv(FOLDER["sim"] + file_path)
                break
        else:
            # Print percentage of completed tasks
            print()
            print(
                print(
                    "\tCompleted : %0.1f" % (num_finished_tasks / len(task_list) * 100)
                )
            )

    # compile = False

    # if compile:
    #     time = pd.date_range("2020-11-01", freq="H", periods=365 * 24)
    #     ds = xr.DataArray(
    #         dims=["time", "rads"],
    #         coords={"time": time, "rads": rads},
    #         attrs=dict(description="coords with matrices"),
    #     )

    #     SITE, FOLDER = config(loc)
    #     for rad in rads:
    #         df = pd.DataFrame()
    #         icestupa_sim = Icestupa(loc)
    #         icestupa_sim.R_F = rad
    #         icestupa_sim.derive_parameters()
    #         icestupa_sim.melt_freeze()
    #         df = icestupa_sim.df[["time", "iceV"]]
    #         df = df.loc[df.iceV != 0].reset_index()
    #         ds.loc[dict(time=df.time.values[1:], rads=rad)] = (
    #             df.iceV.values[1:] - icestupa_sim.V_dome
    #         )

    #     ds.to_netcdf("data/slides/rads.nc")
    # else:
    #     ds = xr.open_dataarray("data/slides/rads.nc")
    #     icestupa = Icestupa(loc)
    #     plt.figure()
    #     ax = plt.gca()
    #     for rad in rads:
    #         ds.sel(rads=rad).plot(label=rad)

    #         V = round(np.nanmax(ds.sel(rads=rad).data), 0)
    #         days = round(sum(~np.isnan(ds.sel(rads=rad).values)) / 24, 0)
    #         print(days, V)
    #     plt.legend()
    #     plt.grid()
    #     plt.savefig("data/slides/rads.jpg")
