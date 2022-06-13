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
import logging, coloredlogs

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
)

from src.utils.settings import config
# from data.common.metadata import get_parameter_metadata
from src.models.icestupaClass import Icestupa

pd.options.display.float_format = "{:,.1f}".format

if __name__ == "__main__":
    # Main logger
    logger = logging.getLogger(__name__)
    logger.setLevel("ERROR")


    # locations = ['guttannen20', 'guttannen21', 'guttannen22']
    locations = ['guttannen21', 'guttannen22']
    spray = 'unscheduled_field'

    print("Comparing weather of different locations")
    for location in locations:
        SITE, FOLDER = config(location, spray)
        icestupa = Icestupa(location, spray)
        print(location, spray)
        print()
        icestupa.read_output()
        cols = [
            "temp",
            "RH",
            "wind",
            "SW_direct",
            "SW_diffuse",
            "ppt",
            "press",
        ]
        separate_periods_index = icestupa.df.loc[icestupa.df.Discharge > 0].index[-1]
        df_jan = icestupa.df.loc[icestupa.df.time.dt.month == 1]
        df_ac = icestupa.df[icestupa.df.index <= separate_periods_index]
        df_ab = icestupa.df[icestupa.df.index > separate_periods_index]
        df_e = icestupa.df[cols].describe().T[["mean", "std"]]
        # print(df_e)
        iceV_diff = df_jan.iceV[df_jan.index[-1]] - df_jan.iceV[df_jan.index[0]]
        print(df_jan[cols].describe().T[["mean", "std"]])
        print(f'\n\tVolume diff {iceV_diff}')
        print(f'\n\tSpray radius {icestupa.R_F}\n')

    # locations = ['gangles21',  'guttannen21']
    # locations = ['guttannen22']
    locations = ['guttannen21', 'guttannen22']
    # sprays = ['unscheduled_field', 'scheduled_field']
    sprays = ['unscheduled_field']
    print("Comparing processes of same location")
    for location in locations:
        for spray in sprays:
            SITE, FOLDER = config(location, spray)
            icestupa = Icestupa(location, spray)
            print(location, spray)
            print()
            icestupa.read_output()

            # icestupa.df.loc[icestupa.df.event == 0, 'Qfreeze'] = np.nan
            # icestupa.df.loc[icestupa.df.event == 1, 'Qmelt'] = np.nan
            icestupa.df.loc[icestupa.df.Discharge == 0, "fountain_froze"] = np.nan
            icestupa.df["melted"] /= 60
            icestupa.df["fountain_froze"] /= 60
            # print(icestupa.df.fountain_froze.max())

            separate_periods_index = icestupa.df.loc[icestupa.df.Discharge > 0].index[-1]
            df_ac = icestupa.df[icestupa.df.index <= separate_periods_index]
            df_ab = icestupa.df[icestupa.df.index > separate_periods_index]

            cols = [
                "SW",
                "LW",
                "Qs",
                "Ql",
                "Qf",
                "Qg",
                "Qtotal",
                "Qfreeze",
                "Qmelt",
                "Qt",
                "A_cone",
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

            print("Accumulation")
            print("Total hours", df_ac.shape[0])
            print("Max freezing rate", df_ac.fountain_froze.quantile(0.9))
            print("Hours reaching max", df_ac.loc[df_ac.fountain_froze>=icestupa.D_F/2].shape[0])
            print("% with no freezing", df_ac.loc[df_ac.fountain_froze==0].shape[0]/ df_ac.shape[0])

            # pd.options.display.float_format = '{:,.3f}'.format
            dfds = icestupa.df.set_index("time").resample("D").sum().reset_index()
            dfds["j_cone"] *= 1000
            separate_periods_index = dfds.loc[dfds.Discharge > 0].index[-1]
            df_ac = dfds[dfds.index <= separate_periods_index]
            df_ab = dfds[dfds.index > separate_periods_index]
            df_e = dfds["j_cone"].describe().T[["mean", "std"]]
            print(df_e)
            print()

            print("Accumulation")
            df_e = df_ac["j_cone"].describe().T[["mean", "std"]]
            print(df_e)
            print()
            print("Ablation")
            print(df_ab.shape[0])
            df_e = df_ab["j_cone"].describe().T[["mean", "std"]]
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
                + dfd.Qr.abs().sum()
                + dfd.Qg.abs().sum()
            )
            print(
                "Percent of \n SW: %.1f \n LW: %.1f \n Qs: %.1f \n Ql: %.1f \n Qf: %.1f\n Qr: %.1f\n Qg: %.1f"
                % (
                    dfd.SW.abs().sum() / Total2 * 100,
                    dfd.LW.abs().sum() / Total2 * 100,
                    dfd.Qs.abs().sum() / Total2 * 100,
                    dfd.Ql.abs().sum() / Total2 * 100,
                    dfd.Qf.abs().sum() / Total2 * 100,
                    dfd.Qr.abs().sum() / Total2 * 100,
                    dfd.Qg.abs().sum() / Total2 * 100,
                )
            )
            # print(
            #     f"% of SW {dfd.SW.abs().sum()/Total*100}, LW {dfd.LW.abs().sum()/Total*100},
            #     Qs{dfd.Qs.abs().sum()/Total*100}, Ql {dfd.Ql.abs().sum()/Total*100}, Qf{dfd.Qf.abs().sum()/Total*100}, Qg {dfd.Qg.abs().sum()/Total*100}"
            # )
