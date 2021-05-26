from sklearn import datasets
from sklearn.model_selection import train_test_split, cross_val_score, ParameterGrid, GroupKFold
from sklearn.experimental import enable_halving_search_cv # noqa
from sklearn.model_selection import HalvingGridSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.base import BaseEstimator 
from sklearn.model_selection import ShuffleSplit, KFold

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
from src.utils.cv import CV_Icestupa, save_obj, load_obj, bounds
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

    # locations = ["Guttannen 2021"]
    locations = ["guttannen21", "schwarzsee19"]#"guttannen20"]

    icestupa = Icestupa("guttannen21")

    icestupa.read_input()
    icestupa.self_attributes()

    obs = list()
    for location in locations:
        # Loading measurements
        SITE, FOLDER, df_h = config(location)
        df_c = pd.read_hdf(FOLDER["input"] + "model_input_Manual.h5", "df_c")

        df_c["Where"] = location
        df_c["Group"] = locations.index(location)
        obs.extend(df_c.reset_index()[["Where", 'When', 'DroneV', 'Group']].values.tolist())

    X = [[a[0], a[1]] for a in obs]
    y = [a[2] for a in obs]
    groups =[a[3] for a in obs] 
    print(y,groups)
    print(X)

    # Set the parameters by cross-validation
    tuned_params = [{
        # 'r_spray': bounds(var=icestupa.r_spray, change=10, res = 0.5),
        'DX': np.arange(0.018, 0.022, 0.001).tolist(), 
        # 'IE': np.arange(0.949, 0.994 , 0.005).tolist(),
        # 'A_I': bounds(var=icestupa.A_I, res = 0.01),
        # 'A_S': bounds(var=icestupa.A_S, res = 0.01),
        # 'T_RAIN': np.arange(0, 2 , 0.5).tolist(),
        # 'T_W': np.arange(1, 5, 1).tolist(),
        # 'A_DECAY': np.arange(1, 23 , 2).tolist(),
        # 'Z': bounds(var=icestupa.Z, res = 0.005),
    }]

    file_path = 'cv-'
    file_path += '-'.join('{}'.format(key) for key, value in tuned_params[0].items())

    print()
    ctr = len(list(ParameterGrid(tuned_params))) 
    days = (ctr*70/(12*60*60*24))
    print("Total hours expected : %0.01f" % int(days*24))
    print("Total days expected : %0.01f" % days)
    # for dict in list(ParameterGrid(tuned_params)):
    #     clf = CV_Icestupa()
    #     clf.set_params(**dict)

    gkf = GroupKFold(n_splits=2)
    for train, test in gkf.split(X, y, groups=groups):
        print("%s %s" % (train, test))

    clf = GridSearchCV(
        CV_Icestupa(), tuned_params, n_jobs=12, cv=gkf, scoring='neg_root_mean_squared_error', error_score=0, verbose=10
    )
    print(groups)
    clf.fit(X,y,groups=groups)

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
