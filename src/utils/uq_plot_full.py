import matplotlib.pyplot as plt
import sys
import os
import numpy as np
import pandas as pd
import math
from matplotlib.offsetbox import AnchoredText
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.colors
import uncertainpy as un
import statistics as st
from datetime import datetime, timedelta
import matplotlib.dates as mdates

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
)

from src.utils.settings import config
from src.models.icestupaClass import Icestupa
from src.models.methods.metadata import get_parameter_metadata

if __name__ == "__main__":

    locations = ['gangles21', 'guttannen21','guttannen20']

    blue = "#0a4a97"
    red = "#e23028"
    purple = "#9673b9"
    green = "#28a745"
    orange = "#ffc107"
    pink = "#ce507a"
    skyblue = "#9bc4f0"
    grey = '#ced4da'
    CB91_Blue = "#2CBDFE"
    CB91_Green = "#47DBCD"
    CB91_Pink = "#F3A0F2"
    CB91_Purple = "#9D2EC5"
    CB91_Violet = "#661D98"
    CB91_Amber = "#F5B14C"


    index = pd.date_range(start ='1-1-2022', 
         end ='1-1-2024', freq ='D', name= "When")
    df_out = pd.DataFrame(columns=locations,index=index)

    # fig, ax = plt.subplots(len(locations), 1, sharex='col', figsize=(12, 14))
    fig, ax = plt.subplots(len(locations), 1, sharex='col')
    # fig.subplots_adjust(hspace=0.1, wspace=0.1)

    for i,location in enumerate(locations):
        SITE, FOLDER = config(location)
        icestupa = Icestupa(location)
        icestupa.read_output()
        icestupa.self_attributes()

        # total_days = int(icestupa.df.index[-1] * icestupa.DT / (60 * 60 * 24))
        if location == "guttannen21":
            total_days = 180
        if location == "schwarzsee19":
            total_days = 60
        if location == "guttannen20":
            total_days = 110
        if location == "gangles21":
            total_days = 150

        variance = []
        mean = []
        evaluations = []

        data = un.Data()
        filename1 = FOLDER['sim']+ "efficiency.h5"
        data.load(filename1)

        survived_days = icestupa.df.index[-1] * icestupa.DT / (60 * 60 * 24)
        if location == 'schwarzsee19':
            SITE["start_date"] +=pd.offsets.DateOffset(year=2023)
        if location == 'guttannen20':
            SITE["start_date"] +=pd.offsets.DateOffset(year=2023)
        if location == 'guttannen21':
            SITE["start_date"] +=pd.offsets.DateOffset(year=2022)
        if location == 'gangles21':
            SITE["start_date"] +=pd.offsets.DateOffset(year=2023)

        days = pd.date_range(
            start=SITE["start_date"],
            end=SITE["start_date"]+ timedelta(hours=total_days * 24 - 1),
            freq="1H",
        )
        days2 = pd.date_range(
            start=SITE["start_date"]+ timedelta(hours= 1),
            end=SITE["start_date"]+ timedelta(hours=survived_days * 24 - 1),
            freq="1H",
        )

        data = data[location]
        data['percentile_5'] = data['percentile_5'][1:len(days2)+1]
        data['percentile_95'] = data['percentile_95'][1:len(days2)+1]
        data["When"] = days2

        df = icestupa.df[["When","iceV"]]
        df_c = pd.read_hdf(FOLDER["input"] + "model_input.h5", "df_c")
        if icestupa.name in ["guttannen21", "guttannen20", "gangles21"]:
            df_c = df_c[1:]
        df_c = df_c.set_index("When").resample("D").mean().reset_index()
        dfv = df_c[["When", "DroneV", "DroneVError"]]
        if location == 'schwarzsee19':
            df['When'] = df['When'].mask(icestupa.df['When'].dt.year == 2019, 
                                         icestupa.df['When'] + pd.offsets.DateOffset(year=2023))
            dfv['When'] = dfv['When'].mask(df_c['When'].dt.year == 2019, 
                                         df_c['When'] + pd.offsets.DateOffset(year=2023))
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
        # df_out[location] = icestupa.df["iceV"] 
        df= df.reset_index()

        x = df.When[1:]
        y1 = df.iceV[1:]
        x2 = dfv.When
        y2 = dfv.DroneV
        ax[i].plot(
            x,
            y1,
            "b-",
            label="Modelled Volume",
            linewidth=1,
            color=CB91_Blue,
            zorder=1,
        )
        ax[i].fill_between(
            data["When"],
            data.percentile_5,
            data.percentile_95,
            color="skyblue",
            alpha=0.3,
            label="90% prediction interval",
        )
        ax[i].scatter(x2, y2, color=CB91_Green, s=5, label="Measured Volume", zorder=2)
        # ax[i].fill_between(x, y1=icestupa.V_dome, y2=0, color=grey, label = "Dome Volume", zorder=0)
        ax[i].set_ylim(round(icestupa.V_dome,0), round(data.percentile_95.max(),0))
        v = get_parameter_metadata(location)
        at = AnchoredText( v['shortname'], prop=dict(size=10), frameon=True, loc="upper left")
        at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
        if i != 2:
            x_axis = ax[i].axes.get_xaxis()
            x_axis.set_visible(False)
            ax[i].spines['bottom'].set_visible(False)
        ax[i].add_artist(at)
        # Hide the right and top spines
        ax[i].spines['right'].set_visible(False)
        ax[i].spines['top'].set_visible(False)
        ax[i].spines['left'].set_color('grey')
        ax[i].spines['bottom'].set_color('grey')
        [t.set_color('grey') for t in ax[i].xaxis.get_ticklines()]
        [t.set_color('grey') for t in ax[i].yaxis.get_ticklines()]
        # ax[i].tick_params(axis='x', colors='grey')
        # ax[i].axes.get_yaxis().label.set_color('red')
        # Only show ticks on the left and bottom spines
        ax[i].yaxis.set_ticks_position('left')
        ax[i].xaxis.set_ticks_position('bottom')
        ax[i].yaxis.set_major_locator(plt.LinearLocator(numticks=2))
        ax[i].xaxis.set_major_locator(mdates.MonthLocator())
        ax[i].xaxis.set_major_formatter(mdates.DateFormatter("%b"))
        fig.autofmt_xdate()

    fig.text(0.04, 0.5, 'Ice Volume[$m^3$]', va='center', rotation='vertical')
    handles, labels = ax[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right')
    # fig.suptitle('Artificial Ice Reservoirs', fontsize=16)
    # plt.legend()
    plt.savefig("data/paper/icev_results.jpg", bbox_inches="tight", dpi=300)

