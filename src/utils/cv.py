from sklearn import datasets
from sklearn.model_selection import train_test_split, cross_val_score, ParameterGrid
from sklearn.experimental import enable_halving_search_cv # noqa
from sklearn.model_selection import HalvingGridSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.svm import SVC
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

from src.utils.settings import config
from src.models.icestupaClass import Icestupa
from src.models.methods.calibration import get_calibration
from src.models.methods.metadata import get_parameter_metadata
from src.models.methods.solar import get_solar
from src.models.methods.droplet import get_droplet_projectile

def save_obj(path, name, obj ):
    with open(path + name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(path, name ):
    with open(path + name + '.pkl', 'rb') as f:
        return pickle.load(f)

def bounds(var, res, change = 5):
    return np.arange(var * (100-change)/100, var * (100+change)/100 + res, res).tolist()

class CV_Icestupa(BaseEstimator,Icestupa):
    def __init__(self, name = "guttannen21", DX = 0.020, DT = 60*60, A_I = 0.15, A_S = 0.85, IE = 0.97, T_PPT = 1, T_W
        = 1, A_DECAY= 10, Z=0.001):
        super(Icestupa, self).__init__()

        print("Initializing classifier:\n")

        args, _, _, values = inspect.getargvalues(inspect.currentframe())
        values.pop("self")

        for arg, val in values.items():
            setattr(self, arg, val)
            # print("{} = {}".format(arg,val))

        SITE, FOLDER = config(location = self.name)
        initial_data = [SITE, FOLDER]
         # Initialise all variables of dictionary
        for dictionary in initial_data:
            for key in dictionary:
                setattr(self, key, dictionary[key])
        
        self.read_input()
        self.self_attributes()
           
    @Timer(text="Simulation executed in {:.2f} seconds")
    def fit(self, X,y,groups=None):

#         if self.r_spray in parameters.keys():
#             self.self_attributes()
# 
#         if self.D_MEAN in parameters.keys():
#             self.get_discharge()

        if self.A_DECAY !=10 or self.A_I != 0.15 or self.A_S != 0.85 or self.T_PPT!= 1: 
            """Albedo Decay parameters initialized"""
            self.A_DECAY = self.A_DECAY * 24 * 60 * 60 / self.DT
            s = 0
            f = 1
            for i, row in self.df.iterrows():
                s, f = self.get_albedo(i, s, f)
 
        self.melt_freeze()

        return self

    def predict(self, X, y=None, groups=None):
        y_pred = []
        ctr = 0
        for x in X:
            if (self.df[self.df.When == x[1]].shape[0]): 
                y_pred.append(self.df.loc[self.df.When == x[1], "iceV"].values[0])
            else:
                y_pred.append(0)
                # print(x,y[ctr])
                # y_pred.append(y[ctr])
                # y_pred.append(0)
            ctr +=1

        return y_pred


if __name__ == "__main__":
    # Main logger
    logger = logging.getLogger(__name__)
    coloredlogs.install(
        fmt="%(funcName)s %(levelname)s %(message)s",
        level=logging.ERROR,
        logger=logger,
    )

    locations = ["Guttannen 2021"]
    # locations = ["Schwarzsee 2019", "Guttannen 2021", "Guttannen 2020"]

    icestupa = Icestupa("guttannen21")
    icestupa.read_input()
    icestupa.self_attributes()

    obs = list()
    for location in locations:
        # Loading measurements
        SITE, FOLDER = config(location)
        df_c = pd.read_hdf(FOLDER["input"] + "model_input_Manual.h5", "df_c")

        if location in ["Guttannen 2021", "Guttannen 2020"]:
            df_c = df_c.iloc[1:]

        df_c["Where"] = location
        obs.extend(df_c.reset_index()[["Where", 'When', 'DroneV']].values.tolist())

    X = [[a[0], a[1]] for a in obs]
    y = [a[2] for a in obs]

    # Set the parameters by cross-validation
    tuned_params = [{
        'IE': np.arange(0.949, 0.994 , 0.005).tolist(),
        'A_I': bounds(var=icestupa.A_I, res = 0.01),
        # 'A_S': bounds(var=icestupa.A_S, res = 0.01),
        # 'A_DECAY': np.arange(1, 23 , 2).tolist(),
        # 'T_PPT': np.arange(0, 2 , 0.5).tolist(),
        # 'MU_CONE': np.arange(0, 1, 0.5).tolist(),
        # 'DX': bounds(var=icestupa.DX, res = 0.1),
        # 'r_spray': bounds(var=icestupa.r_spray, res = 0.25),
    }]
    # ctr = 1
    # for item in tuned_params[0]:
        # ctr *=len(tuned_params[0][item])
    # days = (ctr*70/(12*60*60*24))
    # print("Total hours expected : %0.01f" % int(days*24))
    # print("Total days expected : %0.01f" % days)

    file_path = 'cv-'
    file_path += '-'.join('{}'.format(key) for key, value in tuned_params[0].items())


    print()
    # custom_cv = custom_cv_1folds(X)
    # for train_index, test_index in kf.split(X):
    #     print("TRAIN:", train_index, "TEST:", test_index)

    # clf = HalvingGridSearchCV(
        # CV_Icestupa(), tuned_params, n_jobs=12, cv=2, scoring='neg_root_mean_squared_error', error_score=-100, verbose=2
        # CV_Icestupa(), tuned_params, n_jobs=12, cv=custom_cv, scoring='neg_root_mean_squared_error', error_score=-100, verbose=10
        # CV_Icestupa(), tuned_params, n_jobs=12, cv=[train_index, test_index], scoring='neg_root_mean_squared_error', error_score=-100, verbose=10
        # CV_Icestupa(), tuned_params, n_jobs=12, cv=kf, scoring='neg_root_mean_squared_error', error_score=-100, verbose=10
    # )
    ctr = len(list(ParameterGrid(tuned_params))) 
    days = (ctr*70/(12*60*60*24))
    print("Total hours expected : %0.01f" % int(days*24))
    print("Total days expected : %0.01f" % days)
    for dict in list(ParameterGrid(tuned_params)):
        clf = CV_Icestupa()
        clf.set_params(**dict)
        # print(clf.T_RAIN)
        # print(clf.A_S)

    # for key in tuned_params[0]:
    #     print(key)
    #     for value in tuned_params[0][key]:
    #         print(key,value)
    #         CV_Icestupa.set_params(key=value)
    # clf.fit(X,y)
    # clf.fit(X_train,y_train)
    # scores = cross_val_score(clf, X, y, cv=2)
    # print("%0.2f accuracy with a standard deviation of %0.2f" % (scores.mean(), scores.std()))

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
