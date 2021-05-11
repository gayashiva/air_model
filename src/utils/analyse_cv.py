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

    locations = ["Guttannen 2021"]
    for location in locations:
        SITE, FOLDER, df_h = config(location)
        df_c = pd.read_hdf(FOLDER["input"] + "model_input_Manual.h5", "df_c")
        df_c = df_c.iloc[1:]
        clf= load_obj(FOLDER['output'], "cv_try")

        print()
        means = clf['mean_test_score']
        stds = clf['std_test_score']
        for mean, std, params in zip(means, stds, clf['params']):
            print("%0.3f (+/-%0.03f) for %r"
                  % (mean, std * 2, params))
        print()
        print("Detailed classification report:")
        print()
        print("The model is trained on the full development set.")
        print("The scores are computed on the full evaluation set.")
        print()
        y_true, y_pred = y_test, clf.predict(X_test)
        print(classification_report(y_true, y_pred))
        print()
