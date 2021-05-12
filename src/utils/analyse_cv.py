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
import json

from sklearn.metrics import classification_report

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
)

from src.utils.settings import config

def save_obj(path, name, obj ):
    with open(path + name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(path, name ):
    with open(path + name + '.pkl', 'rb') as f:
        return pickle.load(f)

if __name__ == "__main__":
    # Main logger
    logger = logging.getLogger(__name__)
    coloredlogs.install(
        fmt="%(funcName)s %(levelname)s %(message)s",
        level=logging.ERROR,
        logger=logger,
    )

    # Set the parameters by cross-validation
    tuned_params = [{
        # 'location': locations,
        'DX': np.arange(0.010, 0.011, 0.001).tolist(), 
        'TIME_STEP': np.arange(30*60, 35*60, 30*60).tolist(),
        'IE': np.arange(0.9, 0.999 , 0.02).tolist(),
        'A_I': np.arange(0.3, 0.4 , 0.02).tolist(),
        'A_S': np.arange(0.8, 0.9 , 0.02).tolist(),
        'T_RAIN': np.arange(0, 2 , 1).tolist(),
        # 'T_DECAY': np.arange(1, 22 , 1).tolist(),
        # 'v_a_limit': np.arange(8, 12, 1).tolist(),
        # 'dia_f': np.arange(0.003, 0.011 , 0.001).tolist(),
        # 'Z_I': np.arange(0.0010, 0.0020, 0.0001).tolist(),
    }]
    
    file_path = 'cv-'
    file_path += '-'.join('{}'.format(key) for key, value in tuned_params[0].items())

    location = "Guttannen 2021"
    SITE, FOLDER, df_h = config(location)

    data = load_obj(FOLDER['output'], file_path)
    # with open(FOLDER['sim'] + file_path + '.json', 'r') as fp:
    #     best_parameters = json.load(fp)
    # best_parameters = data._best_params_

    # print("Best parameters set:")
    # for param_name in sorted(tuned_params[0].keys()):
    #     print("\t%s: %r" % (param_name, best_parameters[param_name]))
    print("Grid scores on development set:")
    print()
    means = data['mean_test_score']
    stds = data['std_test_score']
    for mean, std, params in zip(means, stds, data['params']):
        # print("%0.3f (+/-%0.03f) for %r"
        #       % (mean, std * 2, params))
        if mean == data['mean_test_score'].max():
            print("%0.3f (+/-%0.03f) for %r"
                  % (mean, std * 2, params))
            for param_name in sorted(params.keys()):
                print("\t%s: %r" % (param_name, params[param_name]))

    # print("Best parameters set:")
    # means = data['mean_test_score'].min()
    # print("\t%s: %r" % (param_name, best_parameters[param_name]))
