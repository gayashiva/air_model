import logging
import os
import os.path
import time
from datetime import datetime
from logging import StreamHandler
from pandas.plotting import register_matplotlib_converters

register_matplotlib_converters()
import matplotlib
from matplotlib.offsetbox import AnchoredText
import math
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from src.data.config import site, dates, option, folders, fountain, surface
from src.models.air_forecast import icestupa, projectile_xy

plt.rcParams["figure.figsize"] = (10,7)

start = time.time()

# Create the Logger
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Create the Handler for logging data to a file
logger_handler = logging.FileHandler(
    os.path.join(os.path.join(folders["dirname"], "data/logs/"), site + "_site.log"),
    mode="w",
)
logger_handler.setLevel(logging.DEBUG)

# Create the Handler for logging data to console.
console_handler = StreamHandler()
console_handler.setLevel(logging.CRITICAL)

# Create a Formatter for formatting the log messages
logger_formatter = logging.Formatter("%(name)s - %(levelname)s - %(message)s")

# Add the Formatter to the Handler
logger_handler.setFormatter(logger_formatter)
console_handler.setFormatter(logger_formatter)

# Add the Handler to the Logger
logger.addHandler(logger_handler)
logger.addHandler(console_handler)

#  read files
if option == "temperature":
    filename0 = (
            folders["input_folder"] + site + "_" + option + "_" + str(fountain["crit_temp"])
    )
    filename1 = (
        folders["output_folder"] + site + "_" + option + "_" + str(fountain["crit_temp"])
    )
else:
    filename0 = folders["input_folder"] + site + "_" + option
    filename1 = folders["output_folder"] + site + "_" + option

filename1 = os.path.join(filename1 + "_model_results.csv")

same = True

if same:
    if os.path.isfile(filename1):
        print("Simulation Exists")
        df = pd.read_csv(filename1, sep=",")
        df["When"] = pd.to_datetime(df["When"], format="%Y.%m.%d %H:%M:%S")

    else:
        filename0 = os.path.join(filename0 + "_input.csv")
        df_in = pd.read_csv(filename0, sep=",")
        df_in["When"] = pd.to_datetime(df_in["When"], format="%Y.%m.%d %H:%M:%S")

        df = icestupa(df_in, fountain, surface)

        total = time.time() - start

        print("Total time : ", total / 60)

else:
    filename0 = os.path.join(filename0 + "_input.csv")
    df_in = pd.read_csv(filename0, sep=",")
    df_in["When"] = pd.to_datetime(df_in["When"], format="%Y.%m.%d %H:%M:%S")

    df = icestupa(df_in, fountain, surface)

    total = time.time() - start

    print("Total time : ", total / 60)

# # Output for manim
# filename2 = os.path.join(folders["output_folder"], site + "_model_gif.csv")
# cols = ["When", "h_ice", "h_f", "r_ice", "ice", "T_a", "Discharge"]
# df[cols].to_csv(filename2, sep=",")

# Output for energy balance
# filename3 = os.path.join(folders["output_folder"], site + "_model_energy.csv")
# cols = ["When", "SW", "LW", "Qs", "Ql", "SA", "iceV"]
# df[cols].to_csv(filename3, sep=",")

# Full Output
if option == "temperature":
    filename2 = (
        folders["output_folder"] + site + "_" + option + "_" + str(fountain["crit_temp"])
    )
else:
    filename2 = folders["output_folder"] + site + "_" + option
filename4 = os.path.join(filename2 + "_model_results.csv")
df.to_csv(filename4, sep=",")


# Plots
pp = PdfPages(filename2 + "_results.pdf")

x = df.When
y1 = df.iceV

fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.plot(x, y1, "k-")
ax1.set_ylabel("Ice Volume [$m^3$]")
ax1.set_xlabel("Days")

#  format the ticks
ax1.xaxis.set_major_locator(mdates.WeekdayLocator())
ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
ax1.xaxis.set_minor_locator(mdates.DayLocator())
ax1.grid()
fig.autofmt_xdate()
pp.savefig(bbox_inches="tight")
plt.clf()

y1 = df.meltwater
y2 = df.sprayed
fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.plot(x, y1, "k-")
ax1.set_ylabel("Water used [$l$]")
ax1.set_xlabel("Days")

ax2 = ax1.twinx()
ax2.plot(x, y2, "b-", linewidth=0.5)
ax2.set_ylabel("Water sprayed [$l$]", color="b")
for tl in ax2.get_yticklabels():
    tl.set_color("b")

#  format the ticks
ax1.xaxis.set_major_locator(mdates.WeekdayLocator())
ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
ax1.xaxis.set_minor_locator(mdates.DayLocator())
ax1.grid()
fig.autofmt_xdate()
pp.savefig(bbox_inches="tight")
plt.clf()

y1 = df.SA

fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.plot(x, y1, "k-")
ax1.set_ylabel("Surface Area [$m^2$]")
ax1.set_xlabel("Days")

#  format the ticks
ax1.xaxis.set_major_locator(mdates.WeekdayLocator())
ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
ax1.xaxis.set_minor_locator(mdates.DayLocator())
ax1.grid()
fig.autofmt_xdate()
pp.savefig(bbox_inches="tight")
plt.clf()

dfd = df.set_index("When").resample("D").mean().reset_index()
dfd['When'] = dfd['When'].dt.strftime("%b %d")
dfd["Discharge"] = dfd["Discharge"] == 0
dfd["Discharge"] = dfd["Discharge"].astype(int)
dfd["Discharge"] = dfd["Discharge"].astype(str)


dfd = dfd.rename({'SW': 'Shortwave', 'LW': 'Longwave', 'Qs': 'Sensible', 'Ql': 'Latent'}, axis=1)

dfd["label"] = ' '
labels = ["Jan 29", "Feb 05", "Feb 12", "Feb 19", "Feb 26", "Mar 05", "Mar 12", "Mar 19"]
for i in range(0,dfd.shape[0]):
    for item in labels:
        if dfd.When[i] == item:
            dfd.loc[i, 'label'] = dfd.When[i]

dfd = dfd.set_index("label")

fig= plt.figure()
y= dfd[['Shortwave','Longwave','Sensible','Latent' ]]

y.plot.bar( stacked=True, edgecolor = dfd['Discharge'], linewidth=0.5)
plt.xlabel('Days')
plt.ylabel('Energy [$W\,m^{-2}$]')
plt.legend(loc = 'upper left')
plt.ylim(-150, 150)
plt.xticks(rotation=45)

pp.savefig(bbox_inches="tight")
plt.clf()

y1 = df.h_ice
y2 = df.r_ice

fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.plot(x, y1, "k-")
ax1.set_ylabel("Ice Cone Height [$m$]")
ax1.set_xlabel("Days")

ax2 = ax1.twinx()
ax2.plot(x, y2, "b-", linewidth=0.5)
ax2.set_ylabel("Ice Radius", color="b")
for tl in ax2.get_yticklabels():
    tl.set_color("b")

#  format the ticks
ax1.xaxis.set_major_locator(mdates.WeekdayLocator())
ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
ax1.xaxis.set_minor_locator(mdates.DayLocator())
ax1.grid()
fig.autofmt_xdate()
pp.savefig(bbox_inches="tight")
plt.clf()

y1 = df.SRf

fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.plot(x, y1, "k-", linewidth=0.5)
ax1.set_ylabel("Solar Area fraction")
ax1.set_xlabel("Days")

#  format the ticks
ax1.xaxis.set_major_locator(mdates.WeekdayLocator())
ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
ax1.xaxis.set_minor_locator(mdates.DayLocator())
ax1.grid()
fig.autofmt_xdate()
pp.savefig(bbox_inches="tight")
plt.clf()

y1 = df.iceV
y2 = df.TotalE + df.Ql

fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.plot(x, y1, "k-")
ax1.set_ylabel("Ice Volume [$m^3$]")
ax1.set_xlabel("Days")

ax2 = ax1.twinx()
ax2.plot(x, y2, "b-", linewidth=0.5)
ax2.set_ylabel("Energy [$W\,m^{-2}$]", color="b")
for tl in ax2.get_yticklabels():
    tl.set_color("b")

#  format the ticks
ax1.xaxis.set_major_locator(mdates.WeekdayLocator())
ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
ax1.xaxis.set_minor_locator(mdates.DayLocator())
ax1.grid()
fig.autofmt_xdate()
pp.savefig(bbox_inches="tight")
plt.clf()

y1 = df.a

fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.plot(x, y1, "k-", linewidth=1)
ax1.set_ylabel("Albedo")
ax1.set_xlabel("Days")

#  format the ticks
ax1.xaxis.set_major_locator(mdates.WeekdayLocator())
ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
ax1.xaxis.set_minor_locator(mdates.DayLocator())
ax1.grid()
fig.autofmt_xdate()
pp.savefig(bbox_inches="tight")
plt.clf()

y1 = df["SA"] / df["iceV"]

fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.plot(x, y1, linewidth=0.5)
ax1.set_ylim(0, 50)
ax1.set_ylabel("SA/V ratio [$m^{-1}$]")
ax1.set_xlabel("Days")

#  format the ticks
ax1.xaxis.set_major_locator(mdates.WeekdayLocator())
ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
ax1.xaxis.set_minor_locator(mdates.DayLocator())
ax1.grid()
fig.autofmt_xdate()
pp.savefig(bbox_inches="tight")
plt.clf()

y1 = df["h_ice"] / df["r_ice"]

fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.plot(x, y1, linewidth=0.5)
ax1.set_ylim(0, 0.5)
ax1.set_ylabel("h/r ratio")
ax1.set_xlabel("Days")

#  format the ticks
ax1.xaxis.set_major_locator(mdates.WeekdayLocator())
ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
ax1.xaxis.set_minor_locator(mdates.DayLocator())
ax1.grid()
fig.autofmt_xdate()
pp.savefig(bbox_inches="tight")
plt.clf()

y1 = df.T_s

fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.plot(x, y1, "k-", linewidth=0.5)
ax1.set_ylabel("Surface Temperature (C)")
ax1.set_xlabel("Days")

#  format the ticks
ax1.xaxis.set_major_locator(mdates.WeekdayLocator())
ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
ax1.xaxis.set_minor_locator(mdates.DayLocator())
ax1.grid()
fig.autofmt_xdate()
pp.savefig(bbox_inches="tight")
plt.clf()

y1 = df.solid / 5

fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.plot(x, y1, "b-", linewidth=0.5)
ax1.set_ylabel("Ice Production rate [$l\,min^{-1}$]")
ax1.set_xlabel("Days")

#  format the ticks
ax1.xaxis.set_major_locator(mdates.WeekdayLocator())
ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
ax1.xaxis.set_minor_locator(mdates.DayLocator())
ax1.grid()
fig.autofmt_xdate()
pp.savefig(bbox_inches="tight")
plt.clf()

y1 = df.theta_s
fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.plot(x, y1, "k-", linewidth=0.5)
ax1.set_ylabel("Solar Elevation angle [$\degree$]")
ax1.set_xlabel("Days")

#  format the ticks
ax1.xaxis.set_major_locator(mdates.WeekdayLocator())
ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
ax1.xaxis.set_minor_locator(mdates.DayLocator())
ax1.grid()
fig.autofmt_xdate()
pp.savefig(bbox_inches="tight")
plt.clf()

y1 = df.cld
y2 = df.e_a
fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.plot(x, y1, "k-", linewidth=0.5)
ax1.set_ylabel("Cloudiness")
ax1.set_xlabel("Days")

ax2 = ax1.twinx()
ax2.plot(x, y2, "b-", linewidth=0.5)
ax2.set_ylabel("Atmospheric emissivity ", color="b")
for tl in ax2.get_yticklabels():
    tl.set_color("b")

#  format the ticks
ax1.xaxis.set_major_locator(mdates.WeekdayLocator())
ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
ax1.xaxis.set_minor_locator(mdates.DayLocator())
ax1.grid()
fig.autofmt_xdate()
pp.savefig(bbox_inches="tight")
plt.clf()

y1 = df.Discharge
y2 = df.ppt * 1000/5

fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.plot(x, y1, "k-")
ax1.set_ylabel("Discharge [$l\,min^{-1}$]")
ax1.set_xlabel("Days")

ax2 = ax1.twinx()
ax2.plot(x, y2, "b-", linewidth=0.5)
ax2.set_ylabel("Precipitation [$l\,min^{-1}$]", color="b")
for tl in ax2.get_yticklabels():
    tl.set_color("b")

#  format the ticks
ax1.xaxis.set_major_locator(mdates.WeekdayLocator())
ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
ax1.xaxis.set_minor_locator(mdates.DayLocator())
ax1.grid()
fig.autofmt_xdate()
pp.savefig(bbox_inches="tight")
plt.clf()
plt.close('all')

pp.close()

# Plots
if option == "schwarzsee":
    filename = os.path.join(filename2 + "_paper.pdf")
    pp = PdfPages(filename)

    x = df.When
    y1 = df.iceV

    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.plot(x, y1, "b-", lw=1)
    ax1.set_ylabel("Ice Volume [$m^3$]")
    ax1.set_xlabel("Days")

    # Include Validation line segment 1
    ax1.plot(
        [datetime(2019, 2, 14, 16), datetime(2019, 2, 14, 16)],
        [0.67115, 1.042],
        color="green",
        lw=1,
    )
    ax1.scatter(datetime(2019, 2, 14, 16), 0.856575, color="green", marker="o")

    # Include Validation line segment 2
    ax1.plot(
        [datetime(2019, 3, 10, 18), datetime(2019, 3, 10, 18)],
        [0.037, 0.222],
        color="green",
        lw=1,
    )
    ax1.scatter(datetime(2019, 3, 10, 18), 0.1295, color="green", marker="o")

    #  format the ticks
    ax1.xaxis.set_major_locator(mdates.WeekdayLocator())
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax1.xaxis.set_minor_locator(mdates.DayLocator())
    ax1.grid()
    fig.autofmt_xdate()
    pp.savefig(bbox_inches="tight")

    plt.savefig(
        os.path.join(folders["output_folder"], site + "_result.jpg"),
        bbox_inches="tight",
        dpi=300,
    )

    plt.clf()

    fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, ncols=1, sharex=True, figsize=(10, 5))

    # fig.suptitle("Mass and Energy balance", fontsize=14)

    x = df.When
    y1 = df.ice

    ax1.plot(x, y1, "b-")
    ax1.set_ylabel("Ice [kg]")
    ax1.grid()
    ax1.xaxis.set_major_locator(mdates.WeekdayLocator())
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax1.xaxis.set_minor_locator(mdates.DayLocator())

    y2 = df.SA
    ax2.plot(x, y2, "b-")
    ax2.set_ylabel("Surface Area [$m^2$]")
    ax2.grid()

    y3 = df.TotalE + df.Ql
    ax3.plot(x, y3, "b-")
    ax3.set_ylabel("E [$W\,m^{-2}$]")
    ax3.grid()

    # rotates and right aligns the x labels, and moves the bottom of the axes up to make room for them
    fig.autofmt_xdate()
    pp.savefig(bbox_inches="tight")
    plt.clf()
    plt.close('all')

    fig, (ax1, ax2, ax3) = plt.subplots(
        nrows=3, ncols=1, sharex=True, sharey=True, figsize=(10, 5)
    )

    # fig.suptitle("Mass balance", fontsize=14)

    x = df.When
    y1 = df.ice

    ax1.plot(x, y1, "b-")
    ax1.set_ylabel("Ice [kg]")
    ax1.grid()
    ax1.xaxis.set_major_locator(mdates.WeekdayLocator())
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax1.xaxis.set_minor_locator(mdates.DayLocator())

    y2 = df.meltwater
    ax2.plot(x, y2, "b-")
    ax2.set_ylabel("Meltwater [kg]")
    ax2.grid()

    y3 = df.vapour
    ax3.plot(x, y3, "b-")
    ax3.set_ylabel("Vapour [kg]")
    ax3.grid()
    # rotates and right aligns the x labels, and moves the bottom of the axes up to make room for them
    fig.autofmt_xdate()
    pp.savefig(bbox_inches="tight")
    plt.clf()

    y = dfd[['Shortwave', 'Longwave', 'Sensible', 'Latent']]
    y.plot.bar(stacked=True, edgecolor=dfd['Discharge'], linewidth=0.5)
    plt.xlabel('Days')
    plt.ylabel('Energy [$W\,m^{-2}$]')
    plt.legend(loc='upper left')
    plt.ylim(-150, 150)
    plt.xticks(rotation=45)
    pp.savefig(bbox_inches="tight")
    plt.clf()
    plt.close('all')

    fig, (ax1, ax2, ax3) = plt.subplots(
        nrows=3, ncols=1, sharex=True, sharey=True, figsize=(10, 5)
    )

    # fig.suptitle("Daily ice and melt", fontsize=14)

    x = df.When
    y1 = df.ice

    ax1.plot(x, y1, "b-")
    ax1.set_ylabel("Ice [kg]")
    ax1.grid()
    ax1.xaxis.set_major_locator(mdates.WeekdayLocator())
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax1.xaxis.set_minor_locator(mdates.DayLocator())

    y2 = df.meltwater
    ax2.plot(x, y2, "b-")
    ax2.set_ylabel("Meltwater [kg]")
    ax2.grid()

    y3 = df.vapour
    ax3.plot(x, y3, "b-")
    ax3.set_ylabel("Vapour [kg]")
    ax3.grid()
    # rotates and right aligns the x labels, and moves the bottom of the axes up to make room for them
    fig.autofmt_xdate()
    pp.savefig(bbox_inches="tight")
    plt.clf()

    # Day melt and Night freeze Plots

    # df = df.reset_index()
    for i in df.index:
        if df.loc[i, 'solid'] < 0:
            df.loc[i, 'solid'] = 0

    dfds = df.set_index("When").resample("D").sum().reset_index()
    dfd['meltwater'] = dfd['meltwater'] * -1 / 1000
    dfds['melted'] = dfds['melted'] * -1 / 1000
    dfds['solid'] = dfds['solid'] / 1000
    dfds["When"] = pd.to_datetime(dfds["When"], format="%Y.%m.%d %H:%M:%S")
    dfds['When'] = dfds['When'].dt.strftime("%b %d")

    dfds2 = dfds[['When','solid', 'melted']]
    dfds2 = dfds2.rename({'solid': 'Daily Ice frozen', 'melted': 'Daily Meltwater discharged'}, axis=1)

    dfds2["label"] = ' '
    labels = ["Jan 29", "Feb 05", "Feb 12", "Feb 19", "Feb 26", "Mar 05", "Mar 12", "Mar 19"]
    for i in range(0, dfds2.shape[0]):
        for item in labels:
            if dfds2.When[i] == item:
                dfds2.loc[i, 'label'] = dfds2.When[i]

    dfds2 = dfds2.set_index("label")


    fig, (ax1, ax2, ax3) = plt.subplots(
        nrows=3, ncols=1, sharex=True, figsize=(10, 5)
    )
    fig.subplots_adjust(hspace=0)

    y1 = dfds2[['Daily Ice frozen', 'Daily Meltwater discharged']]
    y2 = dfd[['Shortwave', 'Longwave', 'Sensible', 'Latent']]
    y3 = dfd['SA']


    y1.plot(kind='bar', stacked=True, edgecolor='black', linewidth=0.5, color=['#D9E9FA', '#0C70DE'], ax=ax1)
    ax1.set_ylabel('Volume [$m^{3}$]')
    ax1.legend(loc='upper right' , prop={'size': 6})

    ax1.grid( axis="y",color="black", alpha=.3, linewidth=.5, which="major")
    at = AnchoredText("(a)",
                      prop=dict(size=6), frameon=True,
                      loc='upper left',
                      )
    at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
    ax1.add_artist(at)


    y2.plot.bar(stacked=True, edgecolor=dfd['Discharge'], linewidth=0.5, ax=ax2)
    ax2.set_ylabel('Energy [$W\,m^{-2}$]')
    ax2.legend(loc='lower right', prop={'size': 6})
    ax2.set_ylim(-120, 120)
    ax2.grid(axis="y", color="black", alpha=.3, linewidth=.5, which="major")
    at = AnchoredText("(b)",
                      prop=dict(size=6), frameon=True,
                      loc='upper left',
                      )
    at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
    ax2.add_artist(at)

    y3.plot.bar(edgecolor='#0C70DE',  color=['#D9E9FA'], linewidth=0.5, ax=ax3)
    ax3.set_ylabel('Area [$m^2$]')
    ax3.grid(axis="y", color="black", alpha=.3, linewidth=.5, which="major")
    at = AnchoredText("(c)",
                      prop=dict(size=6), frameon=True,
                      loc='upper left',
                      )
    at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
    ax3.add_artist(at)

    plt.xlabel('Days')
    plt.xticks(rotation=45)
    fig.autofmt_xdate()
    pp.savefig(bbox_inches="tight")
    plt.clf()
    pp.close()





# # Plots
# if fountain["discharge"] == 11.5:
#     filename = os.path.join(filename2 + "_suggestion.pdf")
#     pp = PdfPages(filename)
#
#     x = df.When
#     y1 = df.iceV
#
#     fig = plt.figure()
#     ax1 = fig.add_subplot(111)
#     ax1.plot(x, y1, "b-", lw=1)
#     ax1.set_ylabel("Ice Volume [$m^3$]")
#     ax1.set_xlabel("Days")
#
#     # Include Validation line segment 1
#     ax1.plot(
#         [datetime(2019, 2, 14, 16), datetime(2019, 2, 14, 16)],
#         [0.67115, 1.042],
#         color="green",
#         lw=1,
#     )
#     ax1.scatter(datetime(2019, 2, 14, 16), 0.856575, color="green", marker="o")
#
#     # Include Validation line segment 2
#     ax1.plot(
#         [datetime(2019, 3, 10, 18), datetime(2019, 3, 10, 18)],
#         [0.037, 0.222],
#         color="green",
#         lw=1,
#     )
#     ax1.scatter(datetime(2019, 3, 10, 18), 0.1295, color="green", marker="o")
#
#     #  format the ticks
#     ax1.xaxis.set_major_locator(mdates.WeekdayLocator())
#     ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
#     ax1.xaxis.set_minor_locator(mdates.DayLocator())
#     ax1.grid()
#     fig.autofmt_xdate()
#     pp.savefig(bbox_inches="tight")
#
#     plt.savefig(
#         os.path.join(folders["output_folder"], site + "_result.jpg"),
#         bbox_inches="tight",
#         dpi=300,
#     )
#
#     plt.clf()
#
#     fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, ncols=1, sharex=True, figsize=(10, 5))
#
#     # fig.suptitle("Mass and Energy balance", fontsize=14)
#
#     x = df.When
#     y1 = df.ice
#
#     ax1.plot(x, y1, "b-")
#     ax1.set_ylabel("Ice [kg]")
#     ax1.grid()
#     ax1.xaxis.set_major_locator(mdates.WeekdayLocator())
#     ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
#     ax1.xaxis.set_minor_locator(mdates.DayLocator())
#
#     y2 = df.SA
#     ax2.plot(x, y2, "b-")
#     ax2.set_ylabel("Surface Area [$m^2$]")
#     ax2.grid()
#
#     y3 = df.TotalE + df.Ql
#     ax3.plot(x, y3, "b-")
#     ax3.set_ylabel("E [$W\,m^{-2}$]")
#     ax3.grid()
#
#     # rotates and right aligns the x labels, and moves the bottom of the axes up to make room for them
#     fig.autofmt_xdate()
#     pp.savefig(bbox_inches="tight")
#     plt.clf()
#
#     fig, (ax1, ax2, ax3) = plt.subplots(
#         nrows=3, ncols=1, sharex=True, sharey=True, figsize=(10, 5)
#     )
#
#     # fig.suptitle("Mass balance", fontsize=14)
#
#     x = df.When
#     y1 = df.ice
#
#     ax1.plot(x, y1, "b-")
#     ax1.set_ylabel("Ice [kg]")
#     ax1.grid()
#     ax1.xaxis.set_major_locator(mdates.WeekdayLocator())
#     ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
#     ax1.xaxis.set_minor_locator(mdates.DayLocator())
#
#     y2 = df.meltwater
#     ax2.plot(x, y2, "b-")
#     ax2.set_ylabel("Meltwater [kg]")
#     ax2.grid()
#
#     y3 = df.vapour
#     ax3.plot(x, y3, "b-")
#     ax3.set_ylabel("Vapour [kg]")
#     ax3.grid()
#     # rotates and right aligns the x labels, and moves the bottom of the axes up to make room for them
#     fig.autofmt_xdate()
#     pp.savefig(bbox_inches="tight")
#     plt.clf()
#
#     y = dfd[['Shortwave', 'Longwave', 'Sensible', 'Latent']]
#     y.plot.bar(stacked=True, edgecolor=dfd['Discharge'], linewidth=0.5)
#     plt.xlabel('Days')
#     plt.ylabel('Energy [$W\,m^{-2}$]')
#     plt.legend(loc='upper left')
#     plt.ylim(-150, 150)
#     plt.xticks(rotation=45)
#     pp.savefig(bbox_inches="tight")
#     plt.clf()
#
#     pp.close()
