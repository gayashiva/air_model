import logging
import os
import os.path
import time
from datetime import datetime
from logging import StreamHandler
from pandas.plotting import register_matplotlib_converters

register_matplotlib_converters()
import matplotlib
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from src.data.config import site, dates, option, folders, fountain, surface
from src.models.air_forecast import icestupa

# python -m src.features.build_features
plt.rcParams["figure.figsize"] = (10,7)
# matplotlib.rc('xtick', labelsize=5)

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
            folders["input_folder"] + site + "_" + option + "_" + str(fountain["t_c"])
    )
    filename1 = (
        folders["output_folder"] + site + "_" + option + "_" + str(fountain["t_c"])
    )
else:
    filename0 = folders["input_folder"] + site + "_" + option
    filename1 = folders["output_folder"] + site + "_" + option

filename1 = os.path.join(filename1 + "_model_results.csv")

# print(filename1)
# if os.path.isfile(filename1):
#     print("Simulation Exists")
#     df = pd.read_csv(filename1, sep=",")
#     df["When"] = pd.to_datetime(df["When"], format="%Y.%m.%d %H:%M:%S")
#
# else:
#     filename0 = os.path.join(filename0 + "_input.csv")
#     df_in = pd.read_csv(filename0, sep=",")
#     df_in["When"] = pd.to_datetime(df_in["When"], format="%Y.%m.%d %H:%M:%S")
#
#     df = icestupa(df_in, fountain, surface)
#
#     total = time.time() - start
#
#     print("Total time : ", total / 60)

filename0 = os.path.join(filename0 + "_input.csv")
df_in = pd.read_csv(filename0, sep=",")
df_in["When"] = pd.to_datetime(df_in["When"], format="%Y.%m.%d %H:%M:%S")

df = icestupa(df_in, fountain, surface)

total = time.time() - start

print("Total time : ", total / 60)

# Output for manim
filename2 = os.path.join(folders["output_folder"], site + "_model_gif.csv")
cols = ["When", "h_ice", "h_f", "r_ice", "ice", "T_a", "Discharge"]
df[cols].to_csv(filename2, sep=",")

# Output for energy balance
# filename3 = os.path.join(folders["output_folder"], site + "_model_energy.csv")
# cols = ["When", "SW", "LW", "Qs", "Ql", "SA", "iceV"]
# df[cols].to_csv(filename3, sep=",")

# Full Output
if option == "temperature":
    filename2 = (
        folders["output_folder"] + site + "_" + option + "_" + str(fountain["t_c"])
    )
else:
    filename2 = folders["output_folder"] + site + "_" + option
filename4 = os.path.join(filename2 + "_model_results.csv")
df.to_csv(filename4, sep=",")

# x = df.When
# y0 = df.T_a
# # y1 = df.Ea - df.Ew
# y2 = df.T_a - df.T_s
# y3 = df.Qs
# fig = plt.figure()
# ax1 = fig.add_subplot(311)
# ax1.plot(x, y0, "k-")
# ax1.axhline(y=0)
# ax1.set_ylabel("Air Temperature ($m^3$)")
# ax1.set_xlabel("Days")
# # ax1 = fig.add_subplot(412)
# # ax1.plot(x, y1, "k-")
# # ax1.set_ylabel("Ice Volume ($m^3$)")
# # ax1.set_xlabel("Days")
# ax1 = fig.add_subplot(312)
# ax1.plot(x, y2, "k-")
# ax1.axhline(y=0)
# ax1.set_ylabel("Air-ice temp. diff. ($m^3$)")
# ax1.set_xlabel("Days")
# ax1 = fig.add_subplot(313)
# ax1.plot(x, y3, "k-")
# ax1.axhline(y=0)
# ax1.set_ylabel("Sensible heat ($m^3$)")
# ax1.set_xlabel("Days")
# plt.show()



# Plots
pp = PdfPages(filename2 + "_results.pdf")

x = df.When
y1 = df.iceV

fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.plot(x, y1, "k-")
ax1.set_ylabel("Ice Volume ($m^3$)")
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
ax1.set_ylabel("Water used ($l$)")
ax1.set_xlabel("Days")

ax2 = ax1.twinx()
ax2.plot(x, y2, "b-", linewidth=0.5)
ax2.set_ylabel("Water sprayed ($l$)", color="b")
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
ax1.set_ylabel("Surface Area ($m^2$)")
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
dfd = dfd.set_index("When")

dfd = dfd.rename({'SW': 'Shortwave', 'LW': 'Longwave', 'Qs': 'Sensible', 'Ql': 'Latent'}, axis=1)

fig = plt.figure()
y= dfd[['Shortwave','Longwave','Sensible','Latent' ]]
y.plot.bar(stacked=True, edgecolor = dfd['Discharge'], linewidth=0.5)
plt.xlabel('Days')
plt.ylabel('Energy ($W/m^{2}$)')
plt.legend(loc = 'upper left')
plt.ylim(-150, 150)
plt.xticks(rotation=45, fontsize=5)
pp.savefig(bbox_inches="tight")
plt.clf()

y1 = df.h_ice
y2 = df.r_ice

fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.plot(x, y1, "k-")
ax1.set_ylabel("Ice Cone Height ($m$)")
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
ax1.set_ylabel("Ice Volume ($m^3$)")
ax1.set_xlabel("Days")

ax2 = ax1.twinx()
ax2.plot(x, y2, "b-", linewidth=0.5)
ax2.set_ylabel("Energy ($W/m^{2}$)", color="b")
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
ax1.set_ylabel("SA/V ratio ($m^{-1}$)")
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
ax1.plot(x, y1, "k-", linewidth=0.5)
ax1.set_ylabel("Ice Production rate ($l/min$)")
ax1.set_xlabel("Days")

#  format the ticks
ax1.xaxis.set_major_locator(mdates.WeekdayLocator())
ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
ax1.xaxis.set_minor_locator(mdates.DayLocator())
ax1.grid()
fig.autofmt_xdate()
pp.savefig(bbox_inches="tight")
plt.clf()

y1 = df.Discharge

fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.plot(x, y1, "k-")
ax1.set_ylabel("Discharge ($l/min$)")
ax1.set_xlabel("Days")

#  format the ticks
ax1.xaxis.set_major_locator(mdates.WeekdayLocator())
ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
ax1.xaxis.set_minor_locator(mdates.DayLocator())
ax1.grid()
fig.autofmt_xdate()
pp.savefig(bbox_inches="tight")
plt.clf()

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
    ax1.set_ylabel("Ice Volume ($m^3$)")
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
    ax1.set_ylabel("Ice (kg)")
    ax1.grid()
    ax1.xaxis.set_major_locator(mdates.WeekdayLocator())
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax1.xaxis.set_minor_locator(mdates.DayLocator())

    y2 = df.SA
    ax2.plot(x, y2, "b-")
    ax2.set_ylabel("Surface Area ($m^2$)")
    ax2.grid()

    y3 = df.TotalE + df.Ql
    ax3.plot(x, y3, "b-")
    ax3.set_ylabel("E ($W/m^{2}$)")
    ax3.grid()

    # rotates and right aligns the x labels, and moves the bottom of the axes up to make room for them
    fig.autofmt_xdate()
    pp.savefig(bbox_inches="tight")
    plt.clf()

    fig, (ax1, ax2, ax3) = plt.subplots(
        nrows=3, ncols=1, sharex=True, sharey=True, figsize=(10, 5)
    )

    # fig.suptitle("Mass balance", fontsize=14)

    x = df.When
    y1 = df.ice

    ax1.plot(x, y1, "b-")
    ax1.set_ylabel("Ice (kg)")
    ax1.grid()
    ax1.xaxis.set_major_locator(mdates.WeekdayLocator())
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax1.xaxis.set_minor_locator(mdates.DayLocator())

    y2 = df.meltwater
    ax2.plot(x, y2, "b-")
    ax2.set_ylabel("Meltwater (kg)")
    ax2.grid()

    y3 = df.vapour
    ax3.plot(x, y3, "b-")
    ax3.set_ylabel("Vapour (kg)")
    ax3.grid()
    # rotates and right aligns the x labels, and moves the bottom of the axes up to make room for them
    fig.autofmt_xdate()
    pp.savefig(bbox_inches="tight")
    plt.clf()

    y = dfd[['Shortwave', 'Longwave', 'Sensible', 'Latent']]
    y.plot.bar(stacked=True, edgecolor=dfd['Discharge'], linewidth=0.5)
    plt.xlabel('Days')
    plt.ylabel('Energy ($W/m^{2}$)')
    plt.legend(loc='upper left')
    plt.ylim(-150, 150)
    plt.xticks(rotation=45)
    pp.savefig(bbox_inches="tight")
    plt.clf()

    pp.close()
