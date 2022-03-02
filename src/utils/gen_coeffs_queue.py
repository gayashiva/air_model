import multiprocessing
from time import sleep
import os, sys, time
import logging
import coloredlogs
import xarray as xr
import numpy as np
import math

# Locals
dirname = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.append(dirname)

from src.models.icestupaClass import Icestupa
from src.utils.settings import config
from src.automate.autoDischarge import TempFreeze, SunMelt

# define worker function
def calculate(process_name, tasks, results, results_list, da):
    print("[%s] evaluation routine starts" % process_name)

    while True:
        new_value = tasks.get()
        if new_value == "None":
            print("[%s] evaluation routine quits" % process_name)

            # Indicate finished
            results.put(-1)
            break
        else:

            # Compute result and mimic a long-running task
            compute = TempFreeze(new_value)

            # Output which process received the value
            # print("[%s] received value: %s" % (process_name, new_value))
            # print("[%s] calculated thickness rate: %.1f" % (process_name, compute))

            # Add result to the queue
            results.put(compute)
            results_list.append([new_value, compute])

    return


if __name__ == "__main__":
    # Main logger
    logger = logging.getLogger(__name__)
    logger.setLevel("ERROR")

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

    # Define xarray
    temp = list(range(-20, 5))
    rh = list(range(0, 100, 10))
    wind = list(range(0, 15))
    alt = list(np.arange(0, 6.1, 0.5))
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

    # Initiate the worker processes
    for i in range(num_processes):

        # Set process name
        process_name = "P%i" % i

        # Create the process, and connect it to the worker function
        new_process = multiprocessing.Process(
            target=calculate, args=(process_name, tasks, results, results_list, da)
        )

        # Add new process to the list of processes
        processes.append(new_process)

        # Start the process
        new_process.start()


    # Fill task queue
    task_list = []

    for temp in da.temp.values: 
        for rh in da.rh.values:
            for wind in da.wind.values:
                for alt in da.alt.values:
                    for cld in da.cld.values:
                        task_list.append({'temp':temp, 'rh':rh, 'wind':wind, 'alt':alt, 'cld':cld})

    for single_task in task_list:
        tasks.put(single_task)

    # Wait while the workers process
    sleep(2)

    # Quit the worker processes by sending them -1
    for i in range(num_processes):
        tasks.put("None")

    # Read calculation results
    num_finished_processes = 0

    while True:
        # Read result
        new_result = results.get()

        # Have a look at the results
        if new_result == -1:
            # Process has finished
            num_finished_processes += 1

            if num_finished_processes == num_processes:
                for item in results_list:
                    input = item[0]
                    output = item[1]
                    for spray_r in da.spray_r.values:
                        input['spray_r'] = spray_r
                        da.sel(input).data += output
                        da.sel(input).data *= math.pi * spray_r * spray_r

                print(da.data.mean())
                da.to_netcdf("data/common/alt_cld_sims_test.nc")

                break
