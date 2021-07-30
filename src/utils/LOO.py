"""Icestupa leave one out cv
"""
import pickle
pickle.HIGHEST_PROTOCOL = 4 # For python version 2.7
from sklearn.model_selection import train_test_split, cross_val_score, ParameterGrid, LeaveOneOut, GridSearchCV
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

if __name__ == "__main__":
    # Main logger
    logger = logging.getLogger(__name__)
    coloredlogs.install(
        fmt="%(funcName)s %(levelname)s %(message)s",
        level=logging.ERROR,
        logger=logger,
    )

    # location = "gangles21"
    location = "guttannen21"
    # location = "schwarzsee19"

    icestupa = Icestupa(location)
    SITE, FOLDER = config(location)

    icestupa.read_input()
    icestupa.self_attributes()

    # Loading measurements
    obs = list()
    kind = 'volume'
    # kind = 'area'
    # kind = 'temp'

    df_c = pd.read_hdf(FOLDER["input"] + "model_input.h5", "df_c")

    # Remove dome volume
    df_c = df_c[1:]
    print(df_c)

    df_c["Where"] = location

    obs.extend(df_c.reset_index()[["Where", 'When', 'DroneV', 'Area']].values.tolist())

    X = np.array([[a[0], a[2]] for a in obs])
    y = np.array([[a[2]] for a in obs])
    print(X.shape, y.shape)

    loo = LeaveOneOut()
    for train_index, test_index in loo.split(X):
        print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        # print(X_train, X_test, y_train, y_test)

    params = [ 'Z', 'SA_corr', 'DX']
    tuned_params = setup_params(params)

    file_path = 'cv-'+kind+'-'
    file_path += '-'.join('{}'.format(key) for key, value in tuned_params.items())

    clf = GridSearchCV(
        CV_Icestupa(name=location), tuned_params, n_jobs=-1, scoring='neg_root_mean_squared_error'
    )
    clf.fit(X_train,y_train)

    with open(FOLDER['sim'] + file_path + '.json', 'w') as fp:
        json.dump(clf.best_params_, fp, sort_keys=True, indent=4)
    best_parameters = clf.best_params_

    print("Best parameters set found on development set:")
    print()
    for param_name in sorted(tuned_params[0].keys()):
        print("\t%s: %r" % (param_name, best_parameters[param_name]))
    print()
    print("Grid scores on development set:")
    print()
    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']

    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r"
              % (mean, std * 2, params))

    save_obj(FOLDER['sim'], file_path, clf.cv_results_)
