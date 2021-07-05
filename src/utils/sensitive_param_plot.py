
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

    print("Total evaluations %s"%df.shape[0])

    df = df[:101]
    df = pd.concat([df.drop(['params'], axis=1), df['params'].apply(pd.Series)], axis=1)

    tot_num = (len(tuned_params[0]))
    # fig = plt.figure(figsize=(12, 14))
    ctr = 0
    fig, axs = plt.subplots(1,tot_num, figsize=(15, 6), facecolor='w', edgecolor='k')
    fig.subplots_adjust(hspace = 1, wspace=1)
    # axs = axs.ravel()
    for key, value in tuned_params[0].items():
        # axs[ctr] = fig.add_subplot(1, tot_num, ctr)
        # if key != 'A_DECAY':
        sns.boxplot( y=key,  data=df,  width=0.5, ax= axs[ctr])
        sns.despine(offset=10,  bottom=True, ax=axs[ctr])
        axs[ctr].set(xticks=[])
        axs[ctr].set_xlabel(get_parameter_metadata(key)['latex'])
        axs[ctr].set_ylabel("")
        axs[ctr].set_ylim(get_parameter_metadata(key)['ylim'])
        axs[ctr].yaxis.set_major_locator(plt.LinearLocator(numticks=2))
        ctr+=1
    # plt.gca().axes.xaxis.set_ticks([])
    plt.savefig(FOLDER["sim"]+"hist.jpg", bbox_inches="tight", dpi=300)
    plt.clf()
    # ax = sns.boxplot( y="A_I",  data=df,  width=0.5)
    # ax.set_xlabel(get_parameter_metadata('A_I')['latex'])
    # ax.set_ylabel("Sensitivity of RMSE [$\%$]")
    # ax.set_ylim(get_parameter_metadata('A_I')['ylim'])
    # plt.savefig(FOLDER["sim"]+"hist.jpg", bbox_inches="tight", dpi=300)
    # plt.clf()

    # df['rmse_percent'] = df['rmse']/icestupa.df.iceV.max() * 100
    # df.plot(y='rmse_percent')
    df.boxplot()
    plt.savefig(FOLDER["sim"]+ "rmse.jpg", bbox_inches="tight", dpi=300)
    plt.clf()

