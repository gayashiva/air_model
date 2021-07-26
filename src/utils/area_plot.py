"""Command line interface to create or display Icestupa class
"""
# External modules
import os, sys
import logging, coloredlogs
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime, timedelta
from matplotlib.offsetbox import AnchoredText
import matplotlib.dates as mdates
import seaborn as sns
from matplotlib.lines import Line2D
from operator import truediv
import numpy as np

# Locals
dirname = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.append(dirname)
from src.models.icestupaClass import Icestupa
from src.utils.settings import config
from src.utils import setup_logger
from src.models.methods.metadata import get_parameter_metadata

if __name__ == "__main__":

    # Main logger
    logger = logging.getLogger(__name__)
    coloredlogs.install(
        fmt="%(funcName)s %(levelname)s %(message)s",
        # level=logging.WARNING,
        level=logging.INFO,
        logger=logger,
    )

    # locations = ['gangles21', 'guttannen21', 'guttannen20']
    locations = ['gangles21', 'guttannen21']

    fig, ax = plt.subplots()
    custom_colors = sns.color_palette("Set1", len(locations))

    for i,location in enumerate(locations):
        SITE, FOLDER = config(location)

        icestupa = Icestupa(location)
        icestupa.self_attributes()
        icestupa.read_output()

        # icestupa.df = icestupa.df[1:-1]
        icestupa.df = icestupa.df[1:icestupa.last_hour-1]

        df_c = pd.read_csv(
            FOLDER["raw"] + location + "_drone.csv",
            sep=",",
            header=0,
            parse_dates=["When"],
        )

#         a_pred = []
#         a_true = df_c.Area.values
#         for date in df_c.When.values:
#             if icestupa.df[icestupa.df.When == date].shape[0]:
#                 a_pred.append(icestupa.df.loc[icestupa.df.When == date, "SA"].values[0])
#             else:
#                 a_pred.append(np.nan)
# 
#         res = list(map(truediv, a_true, a_pred))
# # printing original lists 
#         print ("The original list 1 is : " + str(a_true))
#         print ("The original list 2 is : " + str(a_pred))
# # printing result
#         print ("The division list is : " + str(res))

        if location == 'guttannen20':
            SITE["start_date"] +=pd.offsets.DateOffset(year=2023)
        if location == 'guttannen21':
            SITE["start_date"] +=pd.offsets.DateOffset(year=2022)
        if location == 'gangles21':
            SITE["start_date"] +=pd.offsets.DateOffset(year=2023)

        days = pd.date_range(
            start=SITE["start_date"],
            end=SITE["start_date"]+ timedelta(hours=icestupa.total_hours - 1),
            freq="1H",
        )
        days2 = pd.date_range(
            start=SITE["start_date"]+ timedelta(hours= 1),
            end=SITE["start_date"]+ timedelta(hours=icestupa.total_hours - 1),
            freq="1H",
        )

        df = icestupa.df[["When","SA", "ice", "fountain_froze", "melted"]]
        dfv = df_c[["When", "Area"]]
        if location == 'guttannen20':
            df['When'] = df['When'].mask(icestupa.df['When'].dt.year == 2019, 
                                         icestupa.df['When'] + pd.offsets.DateOffset(year=2022))
            df['When'] = df['When'].mask(icestupa.df['When'].dt.year == 2020, 
                                         icestupa.df['When'] + pd.offsets.DateOffset(year=2023))
            dfv['When'] = dfv['When'].mask(df_c['When'].dt.year == 2019, 
                                         df_c['When'] + pd.offsets.DateOffset(year=2022))
            dfv['When'] = dfv['When'].mask(df_c['When'].dt.year == 2020, 
                                         df_c['When'] + pd.offsets.DateOffset(year=2023))
        if location == 'guttannen21':
            df['When'] = df['When'].mask(icestupa.df['When'].dt.year == 2020, 
                                         icestupa.df['When'] + pd.offsets.DateOffset(year=2022))
            dfv['When'] = dfv['When'].mask(df_c['When'].dt.year == 2020, 
                                         df_c['When'] + pd.offsets.DateOffset(year=2022))
        df['When'] = df['When'].mask(icestupa.df['When'].dt.year == 2021, 
                                     icestupa.df['When'] + pd.offsets.DateOffset(year=2023))
        dfv['When'] = dfv['When'].mask(df_c['When'].dt.year == 2021, 
                                     df_c['When'] + pd.offsets.DateOffset(year=2023))
        df= df.reset_index()


        x = df.When[1:]
        y1 = df.SA[1:]
        x2 = dfv.When
        y2 = dfv.Area
        v = get_parameter_metadata(location)
        ax.plot(
            x,
            y1,
            linewidth=1,
            color=custom_colors[i],
            zorder=1,
            label = v['shortname']
        )
        ax.scatter(x2, y2, s=5, zorder=2, color=custom_colors[i])
        # Hide the right and top spines
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['left'].set_color('grey')
        ax.spines['bottom'].set_color('grey')
        # Only show ticks on the left and bottom spines
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
        ax.legend()
        fig.autofmt_xdate()

#         df = df.set_index("When").resample("D").mean().reset_index()
#         x = df.When[1:]
#         # y1 = df.fountain_froze[1:] *60/1000 - df.melted[1:] *60/1000
#         y1 = df.ice[1:].diff()/1000
# 
#         v = get_parameter_metadata(location)
#         ax.plot(
#             x,
#             y1,
#             linewidth=1,
#             color=custom_colors[i],
#             zorder=1,
#             label = v['shortname']
#         )
#         # Hide the right and top spines
#         ax.spines['right'].set_visible(False)
#         ax.spines['top'].set_visible(False)
#         ax.spines['left'].set_color('grey')
#         ax.spines['bottom'].set_color('grey')
#         # Only show ticks on the left and bottom spines
#         ax.xaxis.set_major_locator(mdates.MonthLocator())
#         ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
#         ax.legend()
#         fig.autofmt_xdate()

    plt.savefig(
        "data/paper/area.jpg",
        dpi=300,
        bbox_inches="tight",
    )
