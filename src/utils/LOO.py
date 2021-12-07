"""Icestupa leave one out cv
"""
import pickle
# pickle.HIGHEST_PROTOCOL = 4 # For python version 2.7
from sklearn.model_selection import train_test_split, cross_val_score, ParameterGrid, LeaveOneOut, GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.utils import parallel_backend
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
            clf = CV_Icestupa(name = location)

            # Set new parameter
            clf.set_params(**new_value)

            # # Fit new parameter
            # clf.fit(X,y)

            # Compute result and mimic a long-running task
            compute = -1 * cross_val_score(clf, X, y, cv=y.shape[0], verbose = 4, scoring='neg_root_mean_squared_error').mean()
            # print(compute)
            # compute = rmse

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
    logger.setLevel("ERROR")

    # location = "gangles21"
    location = "guttannen21"

    CONSTANTS, SITE, FOLDER = config(location)

    # Loading measurements
    obs = list()
    kind = 'volume'
    # kind = 'area'

    df_c = pd.read_hdf(FOLDER["input"] + "model_input.h5", "df_c")

    # Remove dome volume
    df_c = df_c[1:]
    print(df_c)

    df_c["Where"] = location

    obs.extend(df_c.reset_index()[["Where", 'time', 'DroneV', 'Area']].values.tolist())

    X = np.array([[a[0], a[1]] for a in obs])

    if kind == 'area':
        y = np.array([[a[3]] for a in obs])
    else:
        y = np.array([[a[2]] for a in obs])

    print(X.shape, y.shape)


    # params = [ 'Z', 'SA_corr', 'DX']
    # params = [ 'SA_corr', 'DX']
    params = ['DX']
    tuned_params = setup_params(params)

    file_path = 'loo-cv-'+kind+'-'
    file_path += '-'.join('{}'.format(key) for key, value in tuned_params.items())

    # Define IPC manager
    manager = multiprocessing.Manager()

    # Define a list (queue) for tasks and computation results
    tasks = manager.Queue()
    results = manager.Queue()
    results_list = manager.list()

    # Create process pool with four processes
    num_processes = int(multiprocessing.cpu_count()/2)
    pool = multiprocessing.Pool(processes=num_processes)
    processes = []

    print()
    ctr = len(list(ParameterGrid([tuned_params]))) 
    runtime = 60
    days = (ctr*runtime/(num_processes*60*60*24))
    print("Total hours expected : %0.01f" % int(days*24))
    print("Total days expected : %0.01f" % days)

    answer = input("Continue?")
    if not answer.lower() in ["n","no"]:
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
                    break
            else:
                # Print percentage of completed tasks
                print()
                print(print("\tCompleted : %0.1f" % (num_finished_tasks/len(task_list) * 100)))
