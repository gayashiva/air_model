"""Icestupa class function that generates tables for latex
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os, sys
import matplotlib.dates as mdates
import matplotlib as mpl
from matplotlib.colors import ListedColormap
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.offsetbox import AnchoredText
from mpl_toolkits.axisartist.axislines import Axes
from mpl_toolkits import axisartist
import matplotlib.ticker as ticker

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
)

from src.utils.settings import config
from src.models.methods.metadata import get_parameter_metadata
from src.models.icestupaClass import Icestupa

if __name__ == "__main__":
    locations = ["gangles21", "guttannen21", "guttannen20"]
    # locations = ['guttannen21',  'gangles21']
    # locations = ['guttannen21']

    for ctr, location in enumerate(locations):
        SITE, FOLDER = config(location)
        icestupa = Icestupa(location)
        icestupa.read_output()
        icestupa.df.loc[icestupa.df.Qfreeze == 0, 'Qfreeze'] = np.nan
        icestupa.df.loc[icestupa.df.Qmelt == 0, 'Qmelt'] = np.nan
        icestupa.df = icestupa.df.rename(
            {
                "SW": "$q_{SW}$",
                "LW": "$q_{LW}$",
                "Qs": "$q_S$",
                "Ql": "$q_L$",
                "Qf": "$q_{F}$",
                "Qg": "$q_{G}$",
                "Qsurf": "$q_{surf}$",
                "Qmelt": "$q_{melt}$",
                "Qfreeze": "$q_{freeze}$",
                "Qt": "$q_{T}$",
            },
            axis=1,
        )
        cols = ["$q_{SW}$", "$q_{LW}$","$q_S$","$q_L$","$q_{F}$","$q_{G}$", "$q_{freeze}$", "$q_{melt}$", "$q_{T}$"]
        df_e = icestupa.df[cols].describe().T[['mean', 'std']]
        # df_e = df_e.astype('int32')
        print(df_e)
        # df_e['table'] = '$' + df_e['mean'].astype(str) + ' \pm '+ df_e['std'].astype(str) + '$'
        # df_e = df_e['table']
        # print(df_e.to_latex())

        # print(df_e.describe().loc[["mean", "std"]].T)
        # print(df_e.describe().loc[["mean", "std"]].T)

        cols = [
            "T_a",
            "RH",
            "v_a",
            "SW_direct",
            "SW_diffuse",
            "Prec",
            "p_a",
        ]
        icestupa.df['Prec'] *= 1000
        df_i = icestupa.df[cols].describe().T[['mean', 'std']]
        # df_i = df_i.astype('int32')
        print(df_i)
        # df_i = df_i.astype('int32')
        # df_i['table'] = '$' + df_i['mean'].astype(str) + ' \pm '+ df_i['std'].astype(str) + '$'
        # df_i = df_i['table']
        # print(df_i.to_latex())

