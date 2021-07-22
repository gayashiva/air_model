"""Icestupa validation function
"""
from sklearn.model_selection import train_test_split, cross_val_score, ParameterGrid, GroupKFold
from sklearn.metrics import mean_squared_error

import multiprocessing
from time import sleep
import os, sys, time
import pandas as pd
import math
import sys
import os
import pickle
import logging
import coloredlogs
import numpy as np
from codetiming import Timer
from datetime import datetime
import inspect
import json

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
)
from src.utils.cv import CV_Icestupa, setup_params
from src.utils.settings import config
from src.models.icestupaClass import Icestupa

# define worker function
def calculate(process_name,location, tasks, X, y, results, results_list, kind):
    print('[%s] evaluation routine starts' % process_name)

    while True:
        new_value = tasks.get()
        if new_value == "None":
            print('[%s] evaluation routine quits' % process_name)

            # Indicate finished
            results.put(-1)
            break
        else:
            # Initialise icestupa object
            clf = CV_Icestupa(name = location, kind = kind)

            # Set new parameter
            clf.set_params(**new_value)

            # Fit new parameter
            clf.fit(X,y)
            # diff = clf.predict_survival()
            y_pred, x_pred = clf.predict_sa_v(X)
            rmse1 = mean_squared_error(y_pred,y, squared=False)
            rmse2 = mean_squared_error(x_pred,x, squared=False)
            rmse = np.sqrt(rmse1**2+rmse2**2)

            # Compute result and mimic a long-running task
            compute = rmse
            # compute = diff

            # Output which process received the value
            print('[%s] received value: %s' % (process_name, new_value))
            print('[%s] calculated rmse: %.1f' % (process_name, compute))

            # Add result to the queue
            results.put(compute)
            results_list.append([new_value, compute])

    return
if __name__ == "__main__":
    # Main logger
    logger = logging.getLogger(__name__)
    coloredlogs.install(
        fmt="%(funcName)s %(levelname)s %(message)s",
        level=logging.ERROR,
        logger=logger,
    )

    location = "gangles21"
    # location = "guttannen21"
    # location = "schwarzsee19"

    icestupa = Icestupa(location)
    SITE, FOLDER = config(location)

    icestupa.read_input()
    icestupa.self_attributes()


    # Loading measurements
    obs = list()
    kind = 'volume'
    # kind = 'temp'

    if kind == 'volume':
        df_c = pd.read_hdf(FOLDER["input"] + "model_input.h5", "df_c")

        # Remove dome volume
        df_c = df_c[1:]

        df_c["Where"] = location

        # print(df_c.loc[df_c.Area=='NaN', 'Area'])
        # df_c.loc[-1, 'Area']= math.pi * icestupa.r_F **2
        obs.extend(df_c.reset_index()[["Where", 'When', 'DroneV', 'Area']].values.tolist())

    else:
        if location != 'gangles21':
            df_cam = pd.read_hdf(FOLDER["input"] + "model_input.h5", "df_cam")
            df_cam = df_cam.reset_index()
            df_cam["Where"] = location
            obs.extend(df_cam.reset_index()[["Where", 'When', 'cam_temp']].values.tolist())
        else:
            logger.error("%s doesnt have cam temp" %location)

    X = [[a[0], a[1]] for a in obs]
    y = [a[2] for a in obs]
    x = [a[3] for a in obs]
    print(x)

    # if location == 'gangles21':
    #     params = ['IE', 'A_I', 'Z', 'T_F', 'DX']
    # else:
    #     params = ['IE', 'A_I', 'A_S','A_DECAY', 'T_PPT', 'Z', 'T_F', 'DX']
    # params = ['DX']
    # params = ['IE', 'A_I', 'Z', 'DX']
    params = ['DX', 'SA_corr', 'Z']
    tuned_params = setup_params(params)

    file_path = 'cv-'+kind+'-'
    file_path += '-'.join('{}'.format(key) for key, value in tuned_params.items())

    # Define IPC manager
    manager = multiprocessing.Manager()

    # Define a list (queue) for tasks and computation results
    tasks = manager.Queue()
    results = manager.Queue()
    results_list = manager.list()

    # Create process pool with four processes
    num_processes = multiprocessing.cpu_count()
    # num_processes = 2
    pool = multiprocessing.Pool(processes=num_processes)
    processes = []

    print()
    ctr = len(list(ParameterGrid([tuned_params]))) 
    runtime = 40
    days = (ctr*runtime/(num_processes*60*60*24))
    print("Total hours expected : %0.01f" % int(days*24))
    print("Total days expected : %0.01f" % days)
    num_finished_tasks = 0

    # Initiate the worker processes
    for i in range(num_processes):

        # Set process name
        process_name = 'P%i' % i

        # Create the process, and connect it to the worker function
        new_process = multiprocessing.Process(target=calculate, args=(process_name,location, tasks, X, y, results,
            results_list, kind))

        # Add new process to the list of processes
        processes.append(new_process)

        # Start the process
        new_process.start()

    # Fill task queue
    task_list = list(ParameterGrid(tuned_params))

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
                df = pd.DataFrame.from_records(results_list, columns=["params", "rmse"])
                df = df.set_index('rmse').sort_index().reset_index()
                print(df.head())
                df.to_csv(FOLDER['sim'] + file_path, index=False)
                # save_obj(FOLDER['sim'], file_path, results_list)
                break
        else:
            # Print percentage of completed tasks
            print()
            print(print("\tCompleted : %0.1f" % (num_finished_tasks/len(task_list) * 100)))
