
import os, sys, time
import pandas as pd
import pickle
import logging, coloredlogs
import numpy as np
from ast import literal_eval

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
)
from src.utils.cv import CV_Icestupa, save_obj, load_obj, bounds
from src.utils.settings import config
from src.models.icestupaClass import Icestupa
from src.models.methods.metadata import get_parameter_metadata

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
    icestupa.read_output()

    tuned_params = [{
        'IE': np.arange(0.95, 0.991, 0.01).tolist(),
        'A_I': np.arange(0.01, 0.35, 0.05).tolist(),
        'A_S': bounds(var=icestupa.A_S, res = 0.05),
        'A_DECAY': bounds(var=icestupa.A_DECAY, res = 0.5),
        'Z': np.arange(0.001, 0.003, 0.001).tolist(),
        'T_PPT': np.arange(0, 2 , 1).tolist(),
        # 'T_W': np.arange(0, 5 , 1).tolist(),
        # 'DX': bounds(var=icestupa.DX, res = 0.0005),
    }]

    file_path = 'cv-'
    file_path += '-'.join('{}'.format(key) for key, value in tuned_params[0].items())

    df = pd.read_csv(FOLDER['sim'] + file_path)
    df = df.set_index('rmse').sort_index().reset_index()
    df['params'] = df['params'].apply(literal_eval)

    best_params = dict(df.params[0])
    print(best_params)
    save_obj(FOLDER['sim'], 'best_params' , best_params)
