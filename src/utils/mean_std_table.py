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

pd.options.display.float_format = "{:,.1f}".format

if __name__ == "__main__":
    locations = ["gangles21", "guttannen21", "guttannen20", "schwarzsee19"]
    # locations = ['guttannen21',  'gangles21']
    # locations = ['guttannen21']

    for ctr, location in enumerate(locations):
        SITE, FOLDER = config(location)
        icestupa = Icestupa(location)
        # icestupa.read_input()

        # cols = [
        #     "temp",
        #     "RH",
        #     "wind",
        #     "SW_direct",
        #     "SW_diffuse",
        #     "LW_in",
        #     "ppt",
        #     "press",
        # ]
        # icestupa.df["ppt"] *= 1000
        # df_i = icestupa.df[cols].describe().T[["mean", "std"]]
        # print(df_i)
        print()

        icestupa.read_output()

        # icestupa.df.loc[icestupa.df.event == 0, 'Qfreeze'] = np.nan
        # icestupa.df.loc[icestupa.df.event == 1, 'Qmelt'] = np.nan
        icestupa.df.loc[icestupa.df.Discharge == 0, "fountain_froze"] = np.nan
        icestupa.df["melted"] /= 60
        icestupa.df["fountain_froze"] /= 60
        # print(icestupa.df.fountain_froze.max())

        # icestupa.df = icestupa.df.rename(
        #     {
        #         "SW": "$q_{SW}$",
        #         "LW": "$q_{LW}$",
        #         "Qs": "$q_S$",
        #         "Ql": "$q_L$",
        #         "Qf": "$q_{F}$",
        #         "Qg": "$q_{G}$",
        #         "Qsurf": "$q_{surf}$",
        #         "Qmelt": "$q_{melt}$",
        #         "Qfreeze": "$q_{freeze}$",
        #         "Qt": "$q_{T}$",
        #     },
        #     axis=1,
        # )

        separate_periods_index = icestupa.df.loc[icestupa.df.Discharge > 0].index[-1]
        df_ac = icestupa.df[icestupa.df.index <= separate_periods_index]
        df_ab = icestupa.df[icestupa.df.index > separate_periods_index]

        # cols = ["$q_{SW}$", "$q_{LW}$","$q_S$","$q_L$","$q_{F}$","$q_{G}$","$q_{surf}$", "$q_{freeze}$", "$q_{melt}$",
        #     "$q_{T}$", "SA", "fountain_froze", "melted"]
        cols = [
            "SW",
            "LW",
            "Qs",
            "Ql",
            "Qf",
            "Qg",
            "Qsurf",
            "Qfreeze",
            "Qmelt",
            "Qt",
            "SA",
            "fountain_froze",
            "melted",
        ]
        df_e = icestupa.df[cols].describe().T[["mean", "std"]]
        print(df_e)
        print()

        print("Accumulation")
        df_e = df_ac[cols].describe().T[["mean", "std"]]
        print(df_e)
        print("Sublimation", df_ac.vapour.tail(1).values)
        print()
        print("Ablation")
        df_e = df_ab[cols].describe().T[["mean", "std"]]
        print(df_e)
        print()

        # pd.options.display.float_format = '{:,.3f}'.format
        dfds = icestupa.df.set_index("time").resample("D").sum().reset_index()
        dfds["t_cone"] *= 1000
        separate_periods_index = dfds.loc[dfds.Discharge > 0].index[-1]
        df_ac = dfds[dfds.index <= separate_periods_index]
        df_ab = dfds[dfds.index > separate_periods_index]
        df_e = dfds["t_cone"].describe().T[["mean", "std"]]
        print(df_e)
        print()

        print("Accumulation")
        print(df_ac.shape[0])
        df_e = df_ac["t_cone"].describe().T[["mean", "std"]]
        print(df_e)
        print()
        print("Ablation")
        print(df_ab.shape[0])
        df_e = df_ab["t_cone"].describe().T[["mean", "std"]]
        print(df_e)
        print()
        print("Absolute Energies")
        dfd = icestupa.df.set_index("time").resample("D").mean().reset_index()
        separate_periods_index = dfd.loc[dfd.Discharge > 0].index[-1]
        dfd_ac = dfd[dfd.index <= separate_periods_index]
        dfd_ab = dfd[dfd.index > separate_periods_index]
        # Total = dfd.Qsurf.abs().sum()
        Total1 = dfd.Qmelt.abs().sum() + dfd.Qfreeze.abs().sum() + dfd.Qt.abs().sum()
        print(
            "Percent of Qmelt: %.1f \n Qfreeze: %.1f \n Qt: %.1f"
            % (
                dfd.Qmelt.abs().sum() / Total1 * 100,
                dfd.Qfreeze.abs().sum() / Total1 * 100,
                dfd.Qt.abs().sum() / Total1 * 100,
            )
        )
        print("Accumulation Energies")
        Total2 = (
            dfd_ac.SW.abs().sum()
            + dfd_ac.LW.abs().sum()
            + dfd_ac.Qs.abs().sum()
            + dfd_ac.Ql.abs().sum()
            + dfd_ac.Qf.abs().sum()
            + dfd_ac.Qg.abs().sum()
        )
        print(
            "Percent of SW: %.1f \n LW: %.1f \n Qs: %.1f \n Ql: %.1f \n Qf: %.1f\n Qg: %.1f"
            % (
                dfd_ac.SW.abs().sum() / Total2 * 100,
                dfd_ac.LW.abs().sum() / Total2 * 100,
                dfd_ac.Qs.abs().sum() / Total2 * 100,
                dfd_ac.Ql.abs().sum() / Total2 * 100,
                dfd_ac.Qf.abs().sum() / Total2 * 100,
                dfd_ac.Qg.abs().sum() / Total2 * 100,
            )
        )
        print("Ablation Energies")
        Total2 = (
            dfd_ab.SW.abs().sum()
            + dfd_ab.LW.abs().sum()
            + dfd_ab.Qs.abs().sum()
            + dfd_ab.Ql.abs().sum()
            + dfd_ab.Qf.abs().sum()
            + dfd_ab.Qg.abs().sum()
        )
        print(
            "Percent of SW: %.1f \n LW: %.1f \n Qs: %.1f \n Ql: %.1f \n Qf: %.1f\n Qg: %.1f"
            % (
                dfd_ab.SW.abs().sum() / Total2 * 100,
                dfd_ab.LW.abs().sum() / Total2 * 100,
                dfd_ab.Qs.abs().sum() / Total2 * 100,
                dfd_ab.Ql.abs().sum() / Total2 * 100,
                dfd_ab.Qf.abs().sum() / Total2 * 100,
                dfd_ab.Qg.abs().sum() / Total2 * 100,
            )
        )
        print("Full Energies")
        Total2 = (
            dfd.SW.abs().sum()
            + dfd.LW.abs().sum()
            + dfd.Qs.abs().sum()
            + dfd.Ql.abs().sum()
            + dfd.Qf.abs().sum()
            + dfd.Qg.abs().sum()
        )
        print(
            "Percent of SW: %.1f \n LW: %.1f \n Qs: %.1f \n Ql: %.1f \n Qf: %.1f\n Qg: %.1f"
            % (
                dfd.SW.abs().sum() / Total2 * 100,
                dfd.LW.abs().sum() / Total2 * 100,
                dfd.Qs.abs().sum() / Total2 * 100,
                dfd.Ql.abs().sum() / Total2 * 100,
                dfd.Qf.abs().sum() / Total2 * 100,
                dfd.Qg.abs().sum() / Total2 * 100,
            )
        )
        # print(
        #     f"% of SW {dfd.SW.abs().sum()/Total*100}, LW {dfd.LW.abs().sum()/Total*100},
        #     Qs{dfd.Qs.abs().sum()/Total*100}, Ql {dfd.Ql.abs().sum()/Total*100}, Qf{dfd.Qf.abs().sum()/Total*100}, Qg {dfd.Qg.abs().sum()/Total*100}"
        # )
