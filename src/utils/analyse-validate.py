
import os, sys, time
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
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
from src.utils.cv import param_ranges
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
    icestupa.read_output()

    tuned_params = param_ranges(icestupa)

    file_path = 'cv-'
    file_path += '-'.join('{}'.format(key) for key, value in tuned_params.items())

    df = pd.read_csv(FOLDER['sim'] + file_path)
    df = df.set_index('rmse').sort_index().reset_index()
    df['params'] = df['params'].apply(literal_eval)

    print(df.shape[0])
    print(df.head())
    for i in range(0,10):
        print(df.rmse[i])
        for param_name in sorted(df.params[0].keys()):
            print("\t%s: %r" % (param_name, df.params[i][param_name]))

    df = df[:101]
    df['rmse_percent'] = df['rmse']/icestupa.df.iceV.max() * 100
    df.plot(y='rmse_percent')
    plt.savefig(FOLDER["sim"]+ "rmse.jpg", bbox_inches="tight", dpi=300)
    plt.clf()

    df = pd.concat([df.drop(['params'], axis=1), df['params'].apply(pd.Series)], axis=1)

    tuned_params[param_name] =[round(num, 3) for num in tuned_params[param_name]]
    df = df.round(3)
    sns.set(style="darkgrid")
    fig, ax = plt.subplots(
        nrows=1, ncols=len(tuned_params), sharey="row", figsize=(20, 10)
    )
    for i,param_name in enumerate(tuned_params):
        print(param_name, i)
        # param_range = [tuned_params[param_name][0], tuned_params[param_name][-1]]
        ax[i] = sns.countplot( x=param_name, data=df, order = tuned_params[param_name], ax=ax[i])
        ax[i].set_xlabel(param_name)
        if param_name in ['DX', 'Z']:
            multiple = 1000
            labels = [item.get_text() for item in ax[i].get_xticklabels()]
            ax[i].set_xticklabels([str(float(label)* 1000) for label in labels])
        ax[i].set_ylim([0,100])
        # ax.set_ylabel("Count [$\%$]")

    plt.savefig(FOLDER["sim"]+"param_hist.jpg", bbox_inches="tight", dpi=300)
    plt.clf()
