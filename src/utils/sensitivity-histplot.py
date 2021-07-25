
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
from matplotlib.lines import Line2D
import math

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
)
from src.utils.cv import setup_params
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

    locations = ["gangles21", "guttannen21"]
    # locations = ["guttannen21", "guttannen20"]
    # locations = ["guttannen21"]
    # location = "guttannen21"

    # params = ['IE', 'A_I', 'Z', 'SA_corr', 'DX']
    params = ['Z', 'SA_corr', 'DX']
    # params = ['DX', 'SA_corr']

    # Creating an empty Dataframe with column names only
    dfx = pd.DataFrame(columns=params)
    for ctr, location in enumerate(locations):
        icestupa = Icestupa(location)
        SITE, FOLDER = config(location)
        icestupa.read_output()

        tuned_params = setup_params(params)

        # kind = 'temp'
        # kind = 'volume'
        kind = 'area'
        file_path = 'cv-'+kind+'-'
        # file_path = 'cv-'
        file_path += '-'.join('{}'.format(key) for key, value in tuned_params.items())

        df = pd.read_csv(FOLDER['sim'] + file_path)
        df = df.set_index('rmse').sort_index().reset_index()
        df['params'] = df['params'].apply(literal_eval)

        num_selected = int(0.1 * df.shape[0])
        # num_selected = 100
        num_total = df.shape[0]
        print()
        print("\tSelected %s out of %s" % (num_selected, num_total))
        print("\tRMSE %s upto %s" % (df.rmse[0], df.rmse[num_selected]))
        df = df[:num_selected]

        df = pd.concat([df.drop(['params'], axis=1), df['params'].apply(pd.Series)], axis=1)
        df = df.round(4)
        #df = df.loc[df.DX==0.02]

        print()
        for col in params:
            print("\t%s from %s upto %s with percentage %s" % (col, df[col].min(), df[col].max(),
                df[col].value_counts(normalize=True)))

        df = df.loc[df.SA_corr >= 1.2]
        df['AIR'] = get_parameter_metadata(location)['shortname']
        dfx = dfx.append(df, ignore_index = True)


    dfx[['Z', 'DX']] *= 1000
    dfx['Z'] = pd.to_numeric(dfx['Z'], downcast='integer')
    dfx['DX'] = pd.to_numeric(dfx['DX'], downcast='integer')
    print(dfx.head())
    print(dfx.tail())

    sns.set(style="darkgrid")

    fig, ax = plt.subplots(
        nrows=1, ncols=len(params), sharey="row", figsize=(18, 4)
    )

    for i,param_name in enumerate(tuned_params):
        sns.countplot( x=param_name, hue ='AIR', palette="Set1", data=dfx,
            ax=ax[i])
        v = get_parameter_metadata(param_name)
        label = v['latex'] + v['units']
        ax[i].set_xlabel(label)
        # ax[i].set_xlim(v['ylim'])
        # ax[i].set_xticks(np.arange(v['ylim'][0], v['ylim'][1], v['step']))
        if i != 0:
            ax[i].set_ylabel('')
        if i != len(tuned_params) - 1:
            ax[i].get_legend().remove()
        else:
            ax[i].legend(loc='upper right', title = 'AIR')
        ax[i].set_ylabel("")
        ax[i].yaxis.set_major_formatter(mtick.PercentFormatter(num_selected))


    plt.savefig(
        "data/paper/param_hist.jpg",
        dpi=300,
        bbox_inches="tight",
    )
