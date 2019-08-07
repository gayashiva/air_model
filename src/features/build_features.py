import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.backends.backend_pdf import PdfPages
import os
from src.models.air_forecast import icestupa
import time

# python -m src.features.build_features

site = input("Input the Field Site Name: ") or 'plaffeien'

dirname = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', '..'))

input_folder = os.path.join(dirname, "data/interim/" )

output_folder = os.path.join(dirname, "data/processed/" )

start = time.time()

#  read files
filename0 = os.path.join(input_folder, site + "_model_input.csv")
df_in = pd.read_csv(filename0, sep=",")
df_in["When"] = pd.to_datetime(df_in["When"], format="%Y.%m.%d %H:%M:%S")

# end
end_date = df_in["When"].iloc[-1]

df = icestupa(df_in)

total = time.time() - start

print("Total time : ", total / 60)

# Output for manim
filename2 = os.path.join(output_folder, site + "_model_gif.csv")
cols = ["When", "h_ice", "h_f", "r_ice", "ice", "T_a", "Discharge"]
df[cols].to_csv(filename2, sep=",")


# Plots
filename3 = os.path.join(output_folder, site + "_model_results.pdf")
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

y1 = df.water
y2 = df.vapour
fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.plot(x, y1, "k-")
ax1.set_ylabel("Water wasted[$kg$]")
ax1.set_xlabel("Days")

ax2 = ax1.twinx()
ax2.plot(x, y2, "b-", linewidth=0.5)
ax2.set_ylabel("Vapour[$kg$]", color="b")
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
ax1.set_ylabel("Active Surface Area [$m^2$]")
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

y1 = df.solid

fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.plot(x, y1, "k-")
ax1.set_ylabel("Ice Production rate [$litres$]")
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
filename = os.path.join(output_folder, site + "_plots.pdf")
pp = PdfPages(filename)

x = df.When
y1 = df.iceV
y2 = df.TotalE + df.Ql

fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.plot(x, y1, "k-", lw=1)
ax1.set_ylabel("Cone Ice Volume[$m^3$]")
ax1.set_xlabel("Days")

#  format the ticks
ax1.xaxis.set_major_locator(mdates.WeekdayLocator())
ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
ax1.xaxis.set_minor_locator(mdates.DayLocator())
ax1.grid()
fig.autofmt_xdate()
pp.savefig(bbox_inches="tight")

plt.savefig(os.path.join(output_folder, site + "_result.jpg"), bbox_inches="tight", dpi=300)
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
