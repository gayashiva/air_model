"""Icestupa cross validation class object definition
"""
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


# Module logger
logger = logging.getLogger("__main__")
logger.propagate = False

def setup_params(params):

    params_range = []
    for param in params:
        y_lim=get_parameter_metadata(param)['ylim']
        step=get_parameter_metadata(param)['step']
        # param_range = np.linspace(y_lim[0], y_lim[1], step)
        param_range = np.arange(y_lim[0], y_lim[1]+step/2, step)
        param_range = np.round(param_range, 4)
        params_range.append(param_range)
        print(param, param_range)

    tuned_params = {params[i]: params_range[i] for i in range(len(params))}
    return tuned_params

def save_obj(path, name, obj ):
    with open(path + name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(path, name ):
    with open(path + name + '.pkl', 'rb') as f:
        return pickle.load(f)

def bounds(var, res, change = 5):
    return np.arange(var * (100-change)/100, var * (100+change)/100 + res, res).tolist()


class CV_Icestupa(BaseEstimator,Icestupa):
    def __init__(self, name, DX = 0.020, Z=0.003):
        super(Icestupa, self).__init__()

        print("Initializing classifier:\n")

        args, _, _, values = inspect.getargvalues(inspect.currentframe())
        values.pop("self")

        for arg, val in values.items():
            setattr(self, arg, val)
            # print("{} = {}".format(arg,val))

        CONSTANTS, SITE, FOLDER = config(location = self.name)

        for key in ["DX", "Z"]:
            CONSTANTS.pop(key)

        initialize = [CONSTANTS, SITE, FOLDER]
        for dictionary in initialize:
            for key in dictionary:
                setattr(self, key, dictionary[key])
                logger.info(f"%s -> %s" % (key, str(dictionary[key])))

        diff = SITE["melt_out"] - SITE["start_date"]
        days, seconds = diff.days, diff.seconds
        self.total_hours = days * 24 + seconds // 3600

        self.read_input()
        self.self_attributes()
        self.diff = self.total_hours
           
    @Timer(text="Simulation executed in {:.2f} seconds")
    def fit(self, X,y):

        if self.A_DECAY !=17.5 or self.A_I != 0.25 or self.A_S != 0.85 or self.T_PPT!= 1: 
            """Albedo Decay parameters initialized"""
            self.A_DECAY = self.A_DECAY * 24 * 60 * 60 / self.DT
            s = 0
            f = 1
            for i, row in self.df.iterrows():
                s, f = self.get_albedo(i, s, f)
 
        self.melt_freeze()

        self.duration = self.df.index[-1] # total hours

        return self

    def predict(self, X, y=None, groups=None):
        y_pred = []
        ctr = 0
        for x in X:
            if (self.df[self.df.time == x[1]].shape[0]): 
                y_pred.append(self.df.loc[self.df.time == x[1], "iceV"].values[0])
            else:
                y_pred.append(self.V_dome)
                # y_pred.append((1 - (self.total_hours - self.duration)/self.total_hours) * self.V_dome)
            ctr +=1

        return y_pred

    def predict_survival(self):
        self.diff = abs(self.total_hours - self.duration)
        print("\n\tHour diff %0.1f\n" % (self.diff))
        print()
        print(self.df.iceV.index[-1])

        return self.diff

    def predict_sa_v(self, X):
        y_pred = []
        x_pred = []
        ctr = 0
        for x in X:
            if (self.df[self.df.time == x[1]].shape[0]): 
                y_pred.append(self.df.loc[self.df.time == x[1], "iceV"].values[0])
                x_pred.append(self.df.loc[self.df.time == x[1], "SA"].values[0])
            else:
                # y_pred.append((1 - (self.total_hours - self.duration)/self.total_hours) * self.V_dome)
                y_pred.append(self.V_dome)
                x_pred.append(math.pi * self.r_F**2)
            ctr +=1

        return y_pred, x_pred

    def predict_sa(self, X):
        x_pred = []
        ctr = 0
        for x in X:
            if (self.df[self.df.time == x[1]].shape[0]): 
                x_pred.append(self.df.loc[self.df.time == x[1], "SA"].values[0])
            else:
                x_pred.append(math.pi * self.r_F**2)
                # x_pred.append(0)
            ctr +=1

        return x_pred

