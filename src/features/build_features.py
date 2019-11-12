import logging
import os
import time
from datetime import datetime
from logging import StreamHandler
from pandas.plotting import register_matplotlib_converters

register_matplotlib_converters()
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from src.data.config import site, option, folders, fountain, surface
from src.models.air_forecast import icestupa

# python -m src.features.build_features

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
filename0 = os.path.join(folders["input_folder"], site + "_" + option + "_input.csv")
df_in = pd.read_csv(filename0, sep=",")
df_in["When"] = pd.to_datetime(df_in["When"], format="%Y.%m.%d %H:%M:%S")

# end
end_date = df_in["When"].iloc[-1]

df = icestupa(df_in, fountain, surface)

total = time.time() - start

print("Total time : ", total / 60)

# Output for manim
filename2 = os.path.join(folders["output_folder"], site + "_model_gif.csv")
cols = ["When", "h_ice", "h_f", "r_ice", "ice", "T_a", "Discharge"]
df[cols].to_csv(filename2, sep=",")

# Output for energy balance
filename3 = os.path.join(folders["output_folder"], site + "_model_energy.csv")
cols = ["When", "SW", "LW", "Qs", "Ql", "SA", "iceV"]
df[cols].to_csv(filename3, sep=",")

# Plots
filename3 = os.path.join(folders["output_folder"], site + "_" + option + "_results.pdf")
pp = PdfPages(filename3)

x = df.When
y1 = df.iceV

fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.plot(x, y1, "k-")
ax1.set_ylabel("Ice Volume[$m^3$]")
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
ax1.set_ylabel("Water used[$l$]")
ax1.set_xlabel("Days")

ax2 = ax1.twinx()
ax2.plot(x, y2, "b-", linewidth=0.5)
ax2.set_ylabel("Water sprayed[$l$]", color="b")
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
ax1.set_ylabel("Surface Area[$m$]")
ax1.set_xlabel("Days")

#  format the ticks
ax1.xaxis.set_major_locator(mdates.WeekdayLocator())
ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
ax1.xaxis.set_minor_locator(mdates.DayLocator())
ax1.grid()
fig.autofmt_xdate()
pp.savefig(bbox_inches="tight")
plt.clf()

df3 = df_in.set_index("When").resample("D").mean().reset_index()
df3["Discharge"] = df3["Discharge"] == 0
df3["Discharge"] = df3["Discharge"].astype(int)
df3["Discharge"] = df3["Discharge"].astype(str)

df2 = df[["When", "SW", "LW", "Qs", "Ql"]]
x3 = df2.set_index("When").resample("D").mean().reset_index()
x3.index = np.arange(1, len(x3) + 1)

fig = plt.figure()
y = x3[["SW", "LW", "Qs", "Ql"]]
y.plot.bar(stacked=True, edgecolor=df3["Discharge"], linewidth=0.5)
plt.xlabel("Days")
plt.ylabel("Energy[$Wm^{-2}]$")
pp.savefig(bbox_inches="tight")
plt.clf()

y1 = df.h_ice
y2 = df.r_ice

fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.plot(x, y1, "k-")
ax1.set_ylabel("Ice Cone Height[$m$]")
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
ax1.plot(x, y1, "k-")
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
ax1.set_ylabel("Ice Volume[$m^3$]")
ax1.set_xlabel("Days")

ax2 = ax1.twinx()
ax2.plot(x, y2, "b-", linewidth=0.5)
ax2.set_ylabel("Energy[$Wm^{-2}$]", color="b")
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
ax1.plot(x, y1, "k-", linewidth=0.5)
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

y1 = df.T_s

fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.plot(x, y1, "k-")
ax1.set_ylabel("Surface Temperature [C]")
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
ax1.plot(x, y1, "k-")
ax1.set_ylabel("Ice Production rate [$LPM$]")
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
ax1.set_ylabel("Discharge [$LPM$]")
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
filename = os.path.join(folders["output_folder"], site + "_" + option + "_plots.pdf")
pp = PdfPages(filename)

x = df.When
y1 = df.iceV
y2 = df.TotalE + df.Ql

fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.plot(x, y1, "k-", lw=1)
ax1.set_ylabel("Ice Volume[$m^3$]")
ax1.set_xlabel("Days")

if site == "schwarzsee":
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

ax1.plot(x, y1, "k-")
ax1.set_ylabel("Ice[kg]")
ax1.grid()
ax1.xaxis.set_major_locator(mdates.WeekdayLocator())
ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
ax1.xaxis.set_minor_locator(mdates.DayLocator())

y2 = df.SA
ax2.plot(x, y2, "k-")
ax2.set_ylabel("Surface Area[$m^2$]")
ax2.grid()

y3 = df.TotalE + df.Ql
ax3.plot(x, y3, "k-")
ax3.set_ylabel("E[$Wm^{-2}$]")
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

ax1.plot(x, y1, "k-")
ax1.set_ylabel("Ice[kg]")
ax1.grid()
ax1.xaxis.set_major_locator(mdates.WeekdayLocator())
ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
ax1.xaxis.set_minor_locator(mdates.DayLocator())

y2 = df.meltwater
ax2.plot(x, y2, "k-")
ax2.set_ylabel("Meltwater[kg]")
ax2.grid()

y3 = df.vapour
ax3.plot(x, y3, "k-")
ax3.set_ylabel("Vapour[kg]")
ax3.grid()
# rotates and right aligns the x labels, and moves the bottom of the axes up to make room for them
fig.autofmt_xdate()
pp.savefig(bbox_inches="tight")
plt.clf()

df3 = df_in.set_index("When").resample("D").mean().reset_index()
df3["Discharge"] = df3["Discharge"] == 0
df3["Discharge"] = df3["Discharge"].astype(int)
df3["Discharge"] = df3["Discharge"].astype(str)

df2 = df[["When", "SW", "LW", "Qs", "Ql"]]
x3 = df2.set_index("When").resample("D").mean().reset_index()
x3.index = np.arange(1, len(x3) + 1)

fig = plt.figure()
y = x3[["SW", "LW", "Qs", "Ql"]]
y.plot.bar(stacked=True, edgecolor=df3["Discharge"], linewidth=0.5)
plt.xlabel("Days")
plt.ylabel("Energy[$Wm^{-2}]$")
plt.ylim(-200, 200)
pp.savefig(bbox_inches="tight")
plt.clf()

pp.close()

#
# ''' PPT FIG'''
#
# x = df.When
# y1 = df.iceV
#
# fig = plt.figure()
# ax1 = fig.add_subplot(111)
# ax1.plot(x, y1, "k-", lw=1)
# ax1.set_ylabel("Ice Volume[$m^3$]")
# ax1.set_xlabel("Days")
# ax1.set_ylim(0,1.2)
#
# #  format the ticks
# ax1.xaxis.set_major_locator(mdates.WeekdayLocator())
# ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
# ax1.xaxis.set_minor_locator(mdates.DayLocator())
# ax1.grid()
# fig.autofmt_xdate()
#
# plt.savefig(
#     os.path.join(folders['output_folder'], site + "ppt1.jpg"), bbox_inches="tight", dpi=300
# )
#
# plt.clf()
#
# x = df.When
# y1 = df.iceV
#
# fig = plt.figure()
# ax1 = fig.add_subplot(111)
# ax1.plot(x, y1, "k-", lw=1)
# ax1.set_ylabel("Ice Volume[$m^3$]")
# ax1.set_xlabel("Days")
# ax1.set_ylim(0,1.2)
#
# if site == "schwarzsee":
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
# #  format the ticks
# ax1.xaxis.set_major_locator(mdates.WeekdayLocator())
# ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
# ax1.xaxis.set_minor_locator(mdates.DayLocator())
# ax1.grid()
# fig.autofmt_xdate()
#
# plt.savefig(
#     os.path.join(folders['output_folder'], site + "ppt1.jpg"), bbox_inches="tight", dpi=300
# )
#
# plt.clf()
#
# df2= df[['When','SW','LW','Qs','Ql', 'SA']]
#
# x= df2.set_index('When').resample('D').mean().reset_index()
# x.index = np.arange(1, len(x) + 1)
# fig, ax = plt.subplots(1)
# y= (x['SW'] + x['LW'] + x['Qs'] + x['Ql']) * x['SA']
# plt.plot(x.index, y)
# y.plot.bar(stacked=False, edgecolor = df3['Discharge'], linewidth=0.5)
# plt.xlabel('Days')
# plt.ylabel('Energy[$Wm^{-2}]$')
# plt.ylim(-150, 150)
# plt.savefig(os.path.join(folders['output_folder'], site + "_energybarfull.jpg"), bbox_inches  =  "tight", dpi=300)
# plt.clf()
#
# # Plots
# filename = os.path.join(folders['output_folder'], site + '_' + option + "_energybar.pdf")
# pp  =  PdfPages(filename)
#
# df2= df[['When','SW','LW','Qs','Ql' ]]
#
# x= df2.set_index('When').resample('D').mean().reset_index()
# x.index = np.arange(1, len(x) + 1)
#
# fig, ax = plt.subplots(1)
# y= x[['SW','LW']]
# y.plot.bar(stacked=True, edgecolor = df3['Discharge'], linewidth=0.5)
# plt.xlabel('Days')
# plt.ylabel('Energy[$Wm^{-2}]$')
# plt.ylim(-150, 150)
# # plt.legend(loc=1, bbox_to_anchor=(0, 1))
# pp.savefig(bbox_inches  =  "tight")
# # plt.savefig(os.path.join(folders['output_folder'], site + "_energybar1.jpg"), bbox_inches  =  "tight", dpi=300)
# plt.clf()
#
# fig, ax = plt.subplots(1)
# y= x[['SW','LW','Qs']]
# y.plot.bar(stacked=True, edgecolor = df3['Discharge'], linewidth=0.5)
# plt.xlabel('Days')
# plt.ylabel('Energy[$Wm^{-2}]$')
# plt.ylim(-150, 150)
# pp.savefig(bbox_inches  =  "tight")
# # plt.savefig(os.path.join(folders['output_folder'], site + "_energybar2.jpg"), bbox_inches  =  "tight", dpi=300)
# plt.clf()
#
# fig, ax = plt.subplots(1)
# y= x[['SW','LW','Qs','Ql' ]]
# y.plot.bar(stacked=True, edgecolor = df3['Discharge'], linewidth=0.5)
# plt.xlabel('Days')
# plt.ylabel('Energy[$Wm^{-2}]$')
# plt.ylim(-150, 150)
# pp.savefig( bbox_inches  =  "tight")
# # plt.savefig(os.path.join(folders['output_folder'], site + "_energybar3.jpg"), bbox_inches  =  "tight", dpi=300)
# plt.clf()
#
# fig, ax = plt.subplots(1)
# y= x[['SW','LW','Qs','Ql' ]]
# y.plot.bar(stacked=True, edgecolor = df3['Discharge'], linewidth=0.5)
# plt.xlabel('Days')
# plt.ylabel('Energy[$Wm^{-2}]$')
# plt.ylim(-150, 150)
# pp.savefig(bbox_inches  =  "tight")
# plt.savefig(os.path.join(folders['output_folder'], site + "_energybar4.jpg"), bbox_inches  =  "tight", dpi=300)
# plt.clf()
#
# pp.close()
