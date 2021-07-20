
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

    sns.set(style="darkgrid")
    # fig, ax = plt.subplots(
    #     nrows=1, ncols=len(tuned_params), sharey="row", figsize=(20, 10)
    # )
    fig = plt.figure(figsize=(18, 12))
    subfigs = fig.subfigures(len(locations), 1)
    custom_colors = sns.color_palette("Set1", len(locations))
    custom_lines = [Line2D([0], [0], color=custom_colors[0], lw=4),
                    Line2D([0], [0], color=custom_colors[1], lw=4)]#,
                    # Line2D([0], [0], color=custom_colors[2], lw=4)]

    for ctr, location in enumerate(locations):
        icestupa = Icestupa(location)
        SITE, FOLDER = config(location)
        icestupa.read_output()

        params = ['IE', 'A_I', 'Z', 'DX']
        # params = ['Z', 'DX']
        tuned_params = setup_params(params)

        kind = 'temp'
        # kind = 'volume'
        # file_path = 'cv-'+kind+'-'
        file_path = 'cv-'
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

        # df['A_I'] = df['A_I'].map(lambda x: (truncate(x,3)))
        df['Z'] = df['Z'].map(lambda x: (round(x,3)))
        # df = df.loc[df.Z == 0.002]
        # df = df.loc[df.DX == 0.05]
        print(df.head())
        print()
        for col in params:
            print("\t%s from %s upto %s with percentage %s" % (col, df[col].min(), df[col].max(),
                df[col].value_counts(normalize=True)))
            # print(df[col].value_counts(normalize=True).reset_index()[0])

            # g = df[col]
            # df_c = pd.concat([g.value_counts(), 
            #                 g.value_counts(normalize=True).mul(100)],axis=1, keys=('counts','percentage'))
            # df_c = df_c.reset_index()
            # print(df_c)
            # if df_c.loc[0,'percentage'] > 40:
            #     print(df_c.loc[0,'index'])
            #     df = df.loc[df[col]==df_c.loc[0,'index']]
            #     print()
            #     print(df.head())

        ax = subfigs[ctr].subplots(1, len(tuned_params), sharey=True)
        for i,param_name in enumerate(tuned_params):
            tuned_params[param_name] =[round(num, 4) for num in tuned_params[param_name]]
            ax[i] = sns.countplot( x=param_name, color =custom_colors[ctr], data=df, order = tuned_params[param_name],
                ax=ax[i], label = param_name)
            print(param_name)
            v = get_parameter_metadata(param_name)
            if ctr == len(locations) - 1:
                ax[i].set_xlabel(v['latex'] + v['units'], fontsize="x-large")
            else:
                ax[i].set_xlabel('')
            if ctr == 0 and i == len(tuned_params)-1:
                labels = [get_parameter_metadata(item)['shortname'] for item in locations]
                ax[i].legend(custom_lines, labels, fontsize = "x-large")

            if param_name in ['DX', 'Z']:
                labels = [item.get_text() for item in ax[i].get_xticklabels()]
                # ax[i].set_xticklabels([str(round(float(label)* 1000,1)) for label in labels])
                ax[i].set_xticklabels([int(num*1000) if i%2==0 else None for i,num in enumerate(tuned_params[param_name])])
                # ax[i].set_xlabel(param_name + ' [mm]')
            else:
                labels = [item.get_text() for item in ax[i].get_xticklabels()]
                ax[i].set_xticklabels([round(float(label),3) if i%2==0 else None for i,label in enumerate(labels)])
            ax[i].set_ylabel("")
            ax[i].set_ylim([0,num_selected*0.6])
            ax[i].yaxis.set_major_formatter(mtick.PercentFormatter(num_selected))
        subfigs[ctr].subplots_adjust(hspace=0.05, wspace=0.025)

    plt.savefig(
        "data/paper/param_hist.jpg",
        dpi=300,
        bbox_inches="tight",
    )
