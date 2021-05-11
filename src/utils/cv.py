from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.experimental import enable_halving_search_cv # noqa
from sklearn.model_selection import HalvingGridSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.svm import SVC
from sklearn.base import BaseEstimator 

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

class CV_Icestupa(BaseEstimator,Icestupa):
    def __init__(self,location = "Guttannen 2021", DX = 0.010, TIME_STEP = 30*60, A_I = 0.35, A_S = 0.85, IE = 0.95, T_RAIN = 1 ):
        super(Icestupa, self).__init__()

        print("Initializing classifier:\n")

        args, _, _, values = inspect.getargvalues(inspect.currentframe())
        values.pop("self")

        for arg, val in values.items():
            setattr(self, arg, val)
            # print("{} = {}".format(arg,val)
           
    @Timer(text="Simulation executed in {:.2f} seconds")
    def fit(self, X,y=None):
        SITE, FOLDER, df_h = config(self.location)
        initial_data = [SITE, FOLDER]

        # Initialise all variables of dictionary
        for dictionary in initial_data:
            for key in dictionary:
                setattr(self, key, dictionary[key])
                # logger.info(f"%s -> %s" % (key, str(dictionary[key])))

        self.df = pd.read_hdf(self.input + "model_input_" + self.trigger + ".h5", "df")

        self.read_input()

        """Albedo Decay parameters initialized"""
        self.T_DECAY = self.T_DECAY * 24 * 60 * 60 / self.TIME_STEP
        s = 0
        if self.name in ["schwarzsee19", "guttannen20"]:
            f = 0  # Start with snow event
        else:
            f = 1
        for i, row in self.df.iterrows():
            s, f = self.get_albedo(i, s, f, site=self.name)
 
        self.melt_freeze()

        return self

    def predict(self, X, y=None):
        y_pred = []
        for x in X:
            if (self.df[self.df.When == x].shape[0]):
                y_pred.append(self.df.loc[self.df.When == x, "iceV"].values[0])
            else:
                y_pred = [0 for i in range(0,len(X))]
                break

        # y_pred = [(self.df.loc[self.df.When == x, "iceV"].values[0]) for x in X]
        # if (self.df[self.df.When == x].shape[0]):
        #     y_pred = [(self.df.loc[self.df.When == x, "iceV"].values[0]) for x in X]
        # else:
        #   y_pred = [np.NaN for i in range(0,len(X))]

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
    # locations = ["Guttannen 2021"]
    # locations = ["Schwarzsee 2019"]

    for location in locations:
        # Loading measurements
        SITE, FOLDER, df_h = config(location)
        df_c = pd.read_hdf(FOLDER["input"] + "model_input_Manual.h5", "df_c")
        df_c = df_c.iloc[1:]
        y = df_c.DroneV.to_list()
        X = df_c.When.to_list()

        # Split the dataset
        # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8)

#         # Set the parameters by cross-validation
#         tuned_params = [{
#             'location': [location],
#             'DX': np.arange(0.010, 0.011, 0.001).tolist(), 
#             'TIME_STEP': np.arange(30*60, 35*60, 30*60).tolist(),
#             'IE': np.arange(0.9, 0.999 , 0.03).tolist(),
#             # 'A_I': np.arange(0.3, 0.4 , 0.03).tolist(),
#             # 'A_S': np.arange(0.8, 0.9 , 0.02).tolist(),
#             # 'T_RAIN': np.arange(0, 2 , 1).tolist(),
#             # 'T_DECAY': np.arange(1, 22 , 1).tolist(),
#             # 'v_a_limit': np.arange(4, 10, 1).tolist(),
#             # 'dia_f': np.arange(0.003, 0.010 , 0.001).tolist(),
#             # 'Z_I': np.arange(0.0010, 0.0020, 0.0001).tolist(),
#         }]
# 
# 
#         print()
# 
#         # clf = GridSearchCV(
#         #     CV_Icestupa(), tuned_params, n_jobs=12 , cv=3, scoring='neg_root_mean_squared_error'
#         # )
#         clf = HalvingGridSearchCV(
#             CV_Icestupa(), tuned_params, n_jobs=4 , cv=3, scoring='neg_root_mean_squared_error'
#         )
#         clf.fit(X,y)
#         # clf.fit(X_train,y_train)
# 
#         print("Best parameters set found on development set:")
#         print()
#         print(clf.best_params_)
#         print()
#         print("Grid scores on development set:")
#         print()
#         means = clf.cv_results_['mean_test_score']
#         stds = clf.cv_results_['std_test_score']
#         for mean, std, params in zip(means, stds, clf.cv_results_['params']):
#             print("%0.3f (+/-%0.03f) for %r"
#                   % (mean, std * 2, params))
#         # print()
#         # print("Detailed classification report:")
#         # print()
#         # print("The model is trained on the full development set.")
#         # print("The scores are computed on the full evaluation set.")
#         # print()
#         # y_true, y_pred = y_test, clf.predict(X_test)
#         # print(classification_report(y_true, y_pred))
#         # print()
# 
#         save_obj(FOLDER['output'], "cv_try", clf.cv_results_)
#         # with open(FOLDER["output"] + "cv_results.pkl", 'wb') as f:
#         #     pickle.dump(clf.cv_results_, f, pickle.HIGHEST_PROTOCOL)
