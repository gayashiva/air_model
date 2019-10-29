import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import LightSource
import math
import sys
import os

# python -m src.visualization.ppt

site = input("Input the Field Site Name: ") or "plaffeien"

dirname = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

input_folder = os.path.join(dirname, "data/interim/")

output_folder = os.path.join(dirname, "data/processed/")

#  read files
filename0 = os.path.join(output_folder, site +"_model_gif.csv")
df = pd.read_csv(filename0, sep=",")
df["When"] = pd.to_datetime(df["When"], format="%Y.%m.%d %H:%M:%S")

#  read files
filename1 = os.path.join(output_folder, "guttannen_model_gif.csv")
df1 = pd.read_csv(filename1, sep=",")
df1["When"] = pd.to_datetime(df1["When"], format="%Y.%m.%d %H:%M:%S")

start_date = datetime(2018, 12, 1)
end_date = datetime(2019, 7, 1)
mask = (df["When"] >= start_date) & (df["When"] <= end_date)
df = df.loc[mask]

start_date = datetime(2017, 12, 1)
end_date = datetime(2018, 7, 1)
mask = (df1["When"] >= start_date) & (df1["When"] <= end_date)
df1 = df1.loc[mask]

x = df.When
y1 = df.ice

fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.plot(x, y1, "k-")
ax1.set_ylabel("Ice Volume[$litres$]")
ax1.set_xlabel("Days")

#  format the ticks
ax1.xaxis.set_major_locator(mdates.MonthLocator())
ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
ax1.xaxis.set_minor_locator(mdates.WeekdayLocator())
ax1.grid()
fig.autofmt_xdate()
# pp.savefig(bbox_inches="tight")
plt.savefig(
    os.path.join(output_folder, site + "_ppt.jpg"), bbox_inches="tight", dpi=300
)
plt.clf()

x = df1.When
y1 = df.ice
y2 = df1.ice
fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.plot(x, y1, "k-", label = 'Schwarzsee')
ax1.plot(x, y2, "b-", label = 'Guttannen')
ax1.set_ylabel("Ice Volume[$litres$]")
ax1.set_xlabel("Days")

# ax2 = ax1.twinx()
# ax2.plot(x, y2, "b-", linewidth=0.5)
# ax2.set_ylabel("Vapour[$kg$]", color="b")
# for tl in ax2.get_yticklabels():
#     tl.set_color("b")

#  format the ticks
ax1.xaxis.set_major_locator(mdates.MonthLocator())
ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
ax1.xaxis.set_minor_locator(mdates.WeekdayLocator())
ax1.grid()
fig.autofmt_xdate()
plt.legend()
# pp.savefig(bbox_inches="tight")
plt.savefig(
    os.path.join(output_folder, site + 'guttannen'+ "_ppt.jpg"), bbox_inches="tight", dpi=300
)
plt.clf()

x = df1.When
y1 = df.T_a
y2 = df1.T_a
fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.plot(x, y1, "k-", label = 'Schwarzsee', linewidth = 0.5)
ax1.plot(x, y2, "b-", label = 'Guttannen', linewidth = 0.5)
ax1.set_ylabel("Ice Volume[$litres$]")
ax1.set_xlabel("Days")

# ax2 = ax1.twinx()
# ax2.plot(x, y2, "b-", linewidth=0.5)
# ax2.set_ylabel("Vapour[$kg$]", color="b")
# for tl in ax2.get_yticklabels():
#     tl.set_color("b")

#  format the ticks
ax1.xaxis.set_major_locator(mdates.MonthLocator())
ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
ax1.xaxis.set_minor_locator(mdates.WeekdayLocator())
ax1.grid()
fig.autofmt_xdate()
plt.legend()
# pp.savefig(bbox_inches="tight")
plt.savefig(
    os.path.join(output_folder, site + 'guttannen'+ "_Tppt.jpg"), bbox_inches="tight", dpi=300
)
plt.clf()
