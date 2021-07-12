
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
import matplotlib.ticker as mtick
import math

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
)
from src.utils.cv import param_ranges
from src.utils.settings import config
from src.models.icestupaClass import Icestupa
from src.models.methods.metadata import get_parameter_metadata

def truncate(number, digits) -> float:
    stepper = 10.0 ** digits
    return math.trunc(stepper * number) / stepper

if __name__ == "__main__":
    # Main logger
    logger = logging.getLogger(__name__)
    coloredlogs.install(
        fmt="%(funcName)s %(levelname)s %(message)s",
        level=logging.ERROR,
        logger=logger,
    )

    locations = ["gangles21", "guttannen21", "guttannen20"]
    # locations = ["guttannen21", "gangles21"]
    # location = "guttannen21"

    sns.set(style="darkgrid")
    # fig, ax = plt.subplots(
    #     nrows=1, ncols=len(tuned_params), sharey="row", figsize=(20, 10)
    # )
    fig = plt.figure(figsize=(18, 12))
    subfigs = fig.subfigures(len(locations), 1)

    for ctr, location in enumerate(locations):
        icestupa = Icestupa(location)
        SITE, FOLDER = config(location)
        icestupa.read_output()

        tuned_params = param_ranges(icestupa)

        file_path = 'cv-'
        file_path += '-'.join('{}'.format(key) for key, value in tuned_params.items())

        df = pd.read_csv(FOLDER['sim'] + file_path)
        df = df.set_index('rmse').sort_index().reset_index()
        df['params'] = df['params'].apply(literal_eval)

        num_selected = int(0.1 * df.shape[0])
        num_total = df.shape[0]
        print()
        print("\tSelected %s out of %s" % (num_selected, num_total))
        print("\tRMSE %s upto %s" % (df.rmse[0], df.rmse[num_selected]))
        df = df[:num_selected]

        df = pd.concat([df.drop(['params'], axis=1), df['params'].apply(pd.Series)], axis=1)
        df = df.round(4)

        df['A_I'] = df['A_I'].map(lambda x: (truncate(x,3)))
        print(df.head())
        custom_colors = sns.color_palette("Set1", len(tuned_params))

        ax = subfigs[ctr].subplots(1, len(tuned_params), sharey=True)
        for i,param_name in enumerate(tuned_params):
            tuned_params[param_name] =[round(num, 4) for num in tuned_params[param_name]]
            ax[i] = sns.countplot( x=param_name, color =custom_colors[i], data=df, order = tuned_params[param_name],
                ax=ax[i], label = param_name)
            print(param_name)
            v = get_parameter_metadata(param_name)
            if ctr == 2:
                ax[i].set_xlabel(v['latex'] + v['units'], fontsize=16)
            else:
                ax[i].set_xlabel('')

            if param_name in ['DX', 'Z']:
                labels = [item.get_text() for item in ax[i].get_xticklabels()]
                # ax[i].set_xticklabels([str(round(float(label)* 1000,1)) for label in labels])
                ax[i].set_xticklabels([str(num*1000) for num in tuned_params[param_name]])
                # ax[i].set_xlabel(param_name + ' [mm]')
            else:
                labels = [item.get_text() for item in ax[i].get_xticklabels()]
                ax[i].set_xticklabels([str(round(float(label),2)) for label in labels])
            ax[i].set_ylabel("")
            ax[i].set_ylim([0,num_selected*0.6])
            ax[i].yaxis.set_major_formatter(mtick.PercentFormatter(num_selected))
        subfigs[ctr].text(
            0.04,
            0.5,
            get_parameter_metadata(location)["shortname"],
            va="center",
            rotation="vertical",
            fontsize="x-large",
        )
        subfigs[ctr].subplots_adjust(hspace=0.05, wspace=0.025)

    plt.savefig(
        "data/paper/param_hist.jpg",
        dpi=300,
        bbox_inches="tight",
    )
