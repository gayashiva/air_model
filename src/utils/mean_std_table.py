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
    locations = ["gangles21", "guttannen21", "guttannen20", "schwarzsee19"]
    # locations = ['guttannen21',  'gangles21']
    # locations = ['guttannen21']

    for ctr, location in enumerate(locations):
        SITE, FOLDER = config(location)
        icestupa = Icestupa(location)
        icestupa.read_input()
        pd.options.display.float_format = '{:,.1f}'.format

        cols = [
            "T_a",
            "RH",
            "v_a",
            "SW_direct",
            "SW_diffuse",
            "LW_in",
            "Prec",
            "p_a",
        ]
        icestupa.df['Prec'] *= 1000
        df_i = icestupa.df[cols].describe().T[['mean', 'std']]
        print(df_i)
        print()

        icestupa.read_output()

        # print(icestupa.df.loc[icestupa.df.Discharge> 0, 'Discharge'].count())
        # print(icestupa.df.loc[icestupa.df.fountain_froze >= SITE['D_F'] * 60, 'fountain_runoff'].count())

        icestupa.df.loc[icestupa.df.Qfreeze == 0, 'Qfreeze'] = np.nan
        icestupa.df.loc[icestupa.df.Qmelt == 0, 'Qmelt'] = np.nan
        icestupa.df.loc[icestupa.df.Discharge== 0, 'fountain_froze'] = np.nan
        icestupa.df['melted'] /= 60
        icestupa.df['fountain_froze'] /= 60
        # print(icestupa.df.fountain_froze.max())


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
        cols = ["$q_{SW}$", "$q_{LW}$","$q_S$","$q_L$","$q_{F}$","$q_{G}$","$q_{surf}$", "$q_{freeze}$", "$q_{melt}$",
            "$q_{T}$", "SA", "fountain_froze", "melted"]
        df_e = icestupa.df[cols].describe().T[['mean', 'std']]
        print(df_e)
        print()

        dfds = icestupa.df.set_index("When").resample("D").sum().reset_index()
        dfds['mb'] *=1000
        df_e = dfds['mb'].describe().T
        print(df_e)
        print()

        df = dfds['mb'].reset_index()

        # normalized_df=(df-df.mean())/df.std()
        # normalized_df=(df-df.min())/(df.max()-df.min()) * 100
# assign as variable because I'm going to use it more than once.
        # s = (df.index.to_series() / 5).astype(int)
        # s = (df.index-df.index.min())/(df.index.max()-df.index.min()) * 100
        # s = s.astype(int)
        # df.groupby(s).mean().set_index(s)
        df['index']=(df.index-df.index.min())/(df.index.max()-df.index.min()) * 100
        df['index']=df['index'].astype(int)
        # df.groupby(df['index']).mean().set_index(df['index'])
        print(df.tail())
        print(df.index[-1])
        print()
        if df.index[-1] > 100:
            df = (df.groupby(df['index']).mean())
        else:
            df = df.set_index('index')
            df = df.reindex(np.arange(df.index.min(),df.index.max()+1))
            df.loc[:, 'mb'] = df['mb'].interpolate(method = 'ffill')

        # df['mb'] *=1000
        df_e = df['mb'].describe().T
        print(df_e)
        print()


