""" Plot comparing auto and manual discharge at guttannen"""

import sys, json
import os
import numpy as np
import pandas as pd
import math
import matplotlib.colors
from datetime import datetime, timedelta
from matplotlib.offsetbox import AnchoredText
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.lines import Line2D
import logging, coloredlogs

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
)
from src.utils.settings import config
from src.models.icestupaClass import Icestupa
from src.automate.projectile import get_projectile
# from src.models.methods.metadata import get_parameter_metadata

if __name__ == "__main__":
    # Main logger
    logger = logging.getLogger(__name__)
    logger.setLevel("ERROR")
    location = 'guttannen22'
    sprays = ['scheduled_field', 'unscheduled_field']
    # sprays = ['dynamic_field']

    mypal = sns.color_palette("Set1", 2)
    default = "#284D58"
    grey = "#ced4da"
    legend_elements = [Line2D([0], [0], color=mypal[0], lw=4, label='Scheduled'),
                        Line2D([0], [0], color=mypal[1], lw=4, label='Non-scheduled'),
                        Line2D([0], [0], color=default, lw=4, label='Measured Temperature'),
                       Line2D([0], [0], marker='.', color='w', label='Measured Volume',
                              markerfacecolor='k', markersize=15),
                        Line2D([0], [0], color=grey, lw=4, label='Dome Volume'),
                       ]

    fig, ax = plt.subplots(2, 1, gridspec_kw={'height_ratios': [1,1]}, sharex="col")
    for i, spray in enumerate(sprays):
        SITE, FOLDER = config(location, spray)
        icestupa = Icestupa(location, spray)
        icestupa.read_output()
        df=icestupa.df

        x = df.time[1:]
        if spray == "scheduled_field":
            spray = "Scheduled"
            y1 = df.temp[1:]
            ax[0].plot(
                x,
                y1,
                linewidth=0.8,
                color=default,
            )
            ax[0].axhline(y=0, color = 'k', linestyle = '--', alpha = 0.5, linewidth=0.9)
            ax[0].spines["right"].set_visible(False)
            ax[0].spines["top"].set_visible(False)
            ax[0].spines["left"].set_color("grey")
            ax[0].spines["bottom"].set_color("grey")
            ax[0].set_ylabel("Temperature [$\degree C$]")
            ax[0].set_ylim([-20,20])
        else:
            spray = "Non-scheduled"


        # y2 = df.ppt[1:]
        # ax[1].plot(
        #     x,
        #     y2,
        #     linewidth=0.8,
        #     color=default,
        # )
        # ax[1].spines["right"].set_visible(False)
        # ax[1].spines["top"].set_visible(False)
        # ax[1].spines["left"].set_color("grey")
        # ax[1].spines["bottom"].set_color("grey")
        # ax[1].set_ylabel("Precipitation [$mm$]", size=6)

        y2 = df.Discharge[1:]
        ax[1].plot(
            x,
            y2,
            label= spray,
            linewidth=1,
            color=mypal[i],
            # color=default,
        )
        ax[1].spines["right"].set_visible(False)
        ax[1].spines["top"].set_visible(False)
        ax[1].spines["left"].set_color("grey")
        ax[1].spines["bottom"].set_color("grey")
        ax[1].set_ylabel("Discharge [$l/min$]")

        ax[1].set_ylim([0,15])
        # ax[0].set_ylim([-13,10])

    ax[1].xaxis.set_major_locator(mdates.WeekdayLocator())
    ax[1].xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    fig.subplots_adjust(hspace=None, wspace=None)
    fig.autofmt_xdate()
    handles, labels = ax[1].get_legend_handles_labels()
    ax[1].legend(handles, labels, loc="upper right", prop={"size": 8}, title="Fountain spray")
    plt.savefig("data/figs/paper3/disvstemp.png", bbox_inches="tight", dpi=300)
    plt.close()

    sprays = ['scheduled_field', 'unscheduled_field']
    fig, ax = plt.subplots(2, 1, gridspec_kw={'height_ratios': [1,1]}, sharex="col")
    for i, spray in enumerate(sprays):
        SITE, FOLDER = config(location, spray)
        icestupa = Icestupa(location, spray)
        icestupa.read_output()
        df=icestupa.df
        dfd = df.set_index("time").resample("D").mean().reset_index()

        # dfd["time"] = dfd["time"].dt.strftime("%b %d")
        # z = dfd[
        #     [
        #         "alb",
        #     ]
        # ]

        if spray == "scheduled_field":
            spray = "Scheduled"
        else:
            spray = "Non-scheduled"

        x = dfd.time[1:]
        y1 = dfd.alb[1:]
        y2 = dfd.Qf[1:]

        # z.plot.bar(
        #     # stacked=True,
        #     # edgecolor="black",
        #     linewidth=0.5,
        #     alpha = (i+1)*0.5,
        #     color=mypal[i],
        #     ax=ax[0],
        # )
        ax[0].plot(
            x,
            y1,
            linewidth=0.8,
            color=mypal[i],
        )
        ax[0].spines["right"].set_visible(False)
        ax[0].spines["top"].set_visible(False)
        ax[0].spines["left"].set_color("grey")
        ax[0].spines["bottom"].set_color("grey")
        ax[0].set_ylabel("Albedo", size=8)
        ax[0].set_ylim([0,1])
        at = AnchoredText("(a)", prop=dict(size=10), frameon=True, loc="upper left")
        at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
        ax[0].add_artist(at)

        ax[1].plot(
            x,
            y2,
            label= spray,
            linewidth=0.8,
            color=mypal[i],
        )
        ax[1].spines["right"].set_visible(False)
        ax[1].spines["top"].set_visible(False)
        ax[1].spines["left"].set_color("grey")
        ax[1].spines["bottom"].set_color("grey")
        ax[1].set_ylabel("Fountain heat flux [$W\\,m^{-2}$]", size=8)
        # ax[1].set_ylim([0,100])
        at = AnchoredText("(b)", prop=dict(size=10), frameon=True, loc="upper left")
        at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
        ax[1].add_artist(at)

    ax[1].xaxis.set_major_locator(mdates.WeekdayLocator())
    ax[1].xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    fig.subplots_adjust(hspace=None, wspace=None)
    fig.autofmt_xdate()
    handles, labels = ax[1].get_legend_handles_labels()
    ax[1].legend(handles, labels, loc="upper right", prop={"size": 8}, title="Fountain spray")
    plt.savefig("data/figs/paper3/dis_processes.png", bbox_inches="tight", dpi=300)
    plt.close()

    fig, ax = plt.subplots()
    for i, spray in enumerate(sprays):
        SITE, FOLDER = config(location, spray)
        icestupa = Icestupa(location, spray)
        icestupa.read_output()
        df=icestupa.df

        df_c = pd.read_hdf(FOLDER["input_sim"]  + "/input.h5", "df_c")
        df_c = df_c[["time", "DroneV", "DroneVError"]]

        # tol = pd.Timedelta("15T")
        # df_c = df_c.set_index("time")
        # df = df.set_index("time")
        # df_c = pd.merge_asof(
        #     left=df,
        #     right=df_c,
        #     right_index=True,
        #     left_index=True,
        #     direction="nearest",
        #     tolerance=tol,
        # )
        # df_c = df_c[["DroneV", "DroneVError", "iceV"]]
        # df = df.reset_index()

        if spray == "scheduled_field":
            spray = "Scheduled"
        else:
            spray = "Non-scheduled"

        x = df.time[1:]
        y1 = df.iceV[1:]
        x2 = df_c.time
        y2 = df_c.DroneV
        yerr = df_c.DroneVError
        ax.set_ylabel("Ice Volume[$m^3$]")
        ax.plot(
            x,
            y1,
            label=spray,
            linewidth=1,
            color=mypal[i],
        )
        # ax.fill_between(x, y1=icestupa.V_dome, y2=0, color=grey, label="Dome Volume")
        ax.scatter(x2, y2, color=mypal[i], label="Measured Volume")
        ax.errorbar(x2, y2, yerr, color=mypal[i], linewidth=0, elinewidth=1)
        ax.set_ylim([0,70])
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.spines["left"].set_color("grey")
        ax.spines["bottom"].set_color("grey")
        ax.xaxis.set_major_locator(mdates.WeekdayLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
        # ax.xaxis.set_minor_locator(mdates.DayLocator())
        fig.autofmt_xdate()

    legend_elements = [Line2D([0], [0], color=mypal[0], lw=4, label='Automated'),
                        Line2D([0], [0], color=mypal[1], lw=4, label='Manual'),
                       Line2D([0], [0], marker='.', color='w', label='Measured',
                              markerfacecolor='k', markersize=15),
                       ]
    ax.legend(handles=legend_elements, prop={"size": 8}, title='AIR Volume')
    plt.savefig("data/figs/paper3/validation.png", bbox_inches="tight", dpi=300)
    plt.clf()

    # mypal[1] = default
    # sprays = ['scheduled_field', 'scheduled_icv']
    # # sprays = ['scheduled_icv']
    # fig, ax = plt.subplots()
    # ax.axhline(y=2, color='k', linestyle='--', alpha=0.5)
    # # location = 'gangles21'
    # for i, spray in enumerate(sprays):
    #     SITE, FOLDER = config(location, spray)
    #     icestupa = Icestupa(location, spray)
    #     icestupa.read_output()
    #     df=icestupa.df

    #     # df["Discharge"] = np.where(df.Discharge== 0, np.nan, df.Discharge)
    #     # print(f'Median discharge of {spray} is {df.Discharge.median()}')

    #     if spray == "scheduled_field":
    #         spray = "Automation system"
    #     else:
    #         spray = "Model simulated"

    #     x = df.time[1:]
    #     y1 = df.Discharge[1:]
    #     ax.set_ylabel("Scheduled discharge rate [$l/min$]")
    #     ax.plot(
    #         x,
    #         y1,
    #         label=spray,
    #         linewidth=1,
    #         color=mypal[i],
    #     )
    #     ax.set_ylim([0,15])
    #     ax.spines["right"].set_visible(False)
    #     ax.spines["top"].set_visible(False)
    #     ax.spines["left"].set_color("grey")
    #     ax.spines["bottom"].set_color("grey")
    #     ax.xaxis.set_major_locator(mdates.WeekdayLocator())
    #     ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    #     ax.xaxis.set_minor_locator(mdates.DayLocator())
    #     fig.autofmt_xdate()

    # legend_elements = [Line2D([0], [0], color=mypal[0], lw=2, label='Measured'),
    #                     Line2D([0], [0], color=mypal[1], lw=2, label='Modelled'),
    #                     Line2D([0], [0], color='k', lw=2, ls='--', alpha=0.5, label='Minimum'),
    #                    ]
    # ax.legend(handles=legend_elements, prop={"size": 8}, title='Type')
    # plt.savefig("data/figs/paper3/simvsreal.png", bbox_inches="tight", dpi=300)
    # plt.clf()


    icestupa = Icestupa('guttannen21')
    icestupa.read_output()
    df2=icestupa.df
    df2_winter= df2.loc[(df2.time.dt.month <4) | (df2.time.dt.month ==12)]
    # df2_winter['time'] = df2_winter.time.apply(lambda dt: dt.replace(year=2021))

    df2_c = pd.read_csv('/home/bsurya/work/air_model/data/guttannen21/interim/drone.csv')
    df2_c = df2_c[["time", "rad"]]
    df2_c['time'] = pd.to_datetime(df2_c['time'])
    df2_c['time'] = df2_c.time.apply(lambda dt: dt.replace(year=2021))
    df2_c.loc[df2_c.time.dt.month <4, 'time'] = df2_c[df2_c.time.dt.month <4].time.apply(lambda dt: dt.replace(year=2022))
    df2_c= df2_c.loc[(df2_c.time.dt.month <4) | (df2_c.time.dt.month ==12)]
    print(df2_c)

    location = 'guttannen22'
    # sprays = ['scheduled_field', 'unscheduled_field']
    dias = [10,7]
    sprays = ['scheduled_field']
    mypal = sns.color_palette("Set1", 2)
    legend_elements = [Line2D([0], [0], color=mypal[0], lw=4, label='CH22'),
                        Line2D([0], [0], color=mypal[1], lw=4, label='CH21'),
                       ]

    fig, ax = plt.subplots(1, 1)
    for i, spray in enumerate(sprays):
        SITE, FOLDER = config(location, spray)
        icestupa = Icestupa(location, spray)
        icestupa.read_output()
        df=icestupa.df

        for j in range(0,df.shape[0]):
            if df.Discharge[j] !=0:
                df.loc[j,'radf'], df.loc[j,'flight_time'] = get_projectile(h_f=4, dia=dias[i]/1000,
                        dis=df.Discharge[j], theta_f=45)
            else:
                # df.loc[j,'radf'] = np.nan
                df.loc[j,'radf'] = 0
                df.loc[j,'flight_time'] = 0

        df_c = pd.read_hdf(FOLDER["input_sim"]  + "/input.h5", "df_c")
        df_c = df_c[["time", "rad"]]
        df_c= df_c.loc[(df_c.time.dt.month <4) | (df_c.time > SITE['start_date'])]
        df_winter= df.loc[(df.time.dt.month <4) | (df.time > SITE['start_date'])]
        df_winter = df_winter.reset_index()
        df2_winter = df2_winter.reset_index()
        df_c= df_c.reset_index()
        df2_c= df2_c.reset_index()
        # print(df_winter.shape[0], df2_winter.shape[0])
        # print(df_winter.head(), df2_winter.head())

        x = df_winter.time[1:-1]
        y1 = df_winter.radf[1:-1] + df_winter.flight_time[1:-1] * df_winter.wind[1:-1]
        y3 = df_winter.radf[1:-1] + df_winter.flight_time[1:-1] * df2_winter.wind[1:-1]
        # y3 = df.h_cone[1:-1]
        x2 = df_c.time
        y2 = df_c.rad
        x4 = df2_c.time
        y4 = df2_c.rad
        ax.plot(
            x,
            y1,
            linewidth=0.8,
            color=mypal[i],
        )
        ax.plot(
            x,
            y3,
            linewidth=0.8,
            color=mypal[i+1],
        )
        ax.scatter(
            x2,
            y2,
            s=10,
            color=mypal[i],
        )
        ax.scatter(
            x4,
            y4,
            s=10,
            color=mypal[i + 1],
        )
        ax.set_ylabel("Spray Radius [$m$]")
        # ax.axhline(y=icestupa.R_F, linewidth=0.8, linestyle='--', color=mypal[i])
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.spines["left"].set_color("grey")
        ax.spines["bottom"].set_color("grey")
        ax.xaxis.set_major_locator(mdates.WeekdayLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax.set_ylim([0,14])
    fig.subplots_adjust(hspace=None, wspace=None)
    fig.autofmt_xdate()
    ax.legend(handles=legend_elements, prop={"size": 8}, title='Wind influence')
    plt.savefig("data/figs/paper3/radf.png", bbox_inches="tight", dpi=300)
    plt.close()
