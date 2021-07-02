
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
from ast import literal_eval

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

    location = "guttannen21"
    icestupa = Icestupa(location)
    SITE, FOLDER = config(location)

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

    file_path = 'cv-'
    file_path += '-'.join('{}'.format(key) for key, value in tuned_params[0].items())

    df = pd.read_csv(FOLDER['sim'] + file_path)
    df = df.set_index('rmse').sort_index().reset_index()
    df['params'] = df['params'].apply(literal_eval)

    for i in range(0,5):
        print(df.rmse[i])
        for param_name in sorted(df.params[0].keys()):
            print("\t%s: %r" % (param_name, df.params[i][param_name]))

