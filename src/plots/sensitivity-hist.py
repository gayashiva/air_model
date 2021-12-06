
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
    # locations = ["guttannen21"]
    # location = "guttannen21"

    # params = ['IE', 'A_I', 'Z', 'SA_corr', 'DX']
    # params = ['DX', 'SA_corr']
    params = ['Z', 'SA_corr', 'DX']
    kind = ['volume', 'area']
    # kind = ['volume']

    sns.set(style="darkgrid")
    fig, ax = plt.subplots(
        nrows=len(kind), ncols=len(params), sharey="row", figsize=(18, 8)
    )

    for obj in kind:
        # Creating an empty Dataframe with column names only
        dfx = pd.DataFrame(columns=params)
        for ctr, location in enumerate(locations):
            icestupa = Icestupa(location)
            CONSTANTS, SITE, FOLDER = config(location)
            icestupa.read_output()

            file_path = 'loo-cv-'+obj+'-'
            # file_path = 'cv-'+obj+'-'
            file_path += '-'.join('{}'.format(key) for key in params)

            df = pd.read_csv(FOLDER['sim'] + file_path)
            df = df.set_index('rmse').sort_index().reset_index()
            df['params'] = df['params'].apply(literal_eval)

            num_selected = int(0.2 * df.shape[0])
            num_total = df.shape[0]
            print()
            print("\tObjective %s Site %s" % (obj, location))
            print("\tSelected %s out of %s" % (num_selected, num_total))
            print("\tRMSE %s upto %s" % (df.rmse[0], df.rmse[num_selected]))
            df = df[:num_selected]

            df = pd.concat([df.drop(['params'], axis=1), df['params'].apply(pd.Series)], axis=1)
            # df = df.round(4)
            #df = df.loc[df.DX==0.02]
            # df = df.loc[df.SA_corr >= 1.2]

            print()
            for col in params:
                print("\t%s from %s upto %s with percentage %s" % (col, df[col].min(), df[col].max(),
                    df[col].value_counts(normalize=True)))

            df['AIR'] = get_parameter_metadata(location)['shortname']
            df[['Z', 'DX']] *= 1000
            # df['Z'] = pd.to_numeric(df['Z'], downcast='integer')
            # df['DX'] = pd.to_numeric(df['DX'], downcast='integer')
            dfx = dfx.append(df, ignore_index = True)

        print(dfx.head())
        print(dfx.tail())

        for i,param_name in enumerate(params):
            if obj == 'volume':
                j=0
                sns.countplot( x=param_name, hue ='AIR', palette="Set1", data=dfx,
                    ax=ax[j,i])
                if i == 0:
                    ax[j,i].set_ylabel('Volume Objective')
                else:
                    ax[j,i].set_ylabel('')
                ax[j,i].set_xlabel('')
                if i != len(params) - 1:
                    ax[j,i].get_legend().remove()
            if obj == 'area':
                j=1
                sns.countplot( x=param_name, hue ='AIR', palette="Set1", data=dfx,
                    ax=ax[j,i])
                if i == 0:
                    ax[j,i].set_ylabel('Area Objective')
                else:
                    ax[j,i].set_ylabel('')
                v = get_parameter_metadata(param_name)
                label = v['latex'] + v['units']
                ax[j,i].set_xlabel(label)
                ax[j,i].get_legend().remove()
            ax[j,i].yaxis.set_major_formatter(mtick.PercentFormatter(num_selected))


    plt.savefig(
        "data/paper1/param_hist.jpg",
        dpi=300,
        bbox_inches="tight",
    )
