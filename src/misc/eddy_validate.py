import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.backends.backend_pdf import PdfPages
import math
import time
from tqdm import tqdm
import os
import glob
from src.data.config import site, dates, option, folders, fountain, surface

df = pd.read_csv(folders["input_folder"] + "raw_output.csv")
df["TIMESTAMP"] = pd.to_datetime(df["TIMESTAMP"], format='%Y-%m-%d %H:%M:%S')

mask = (df["TIMESTAMP"] >= dates["start_date"]) & (df["TIMESTAMP"] <= dates["end_date"])
df = df.loc[mask]
df = df.reset_index()

df["T_surface"] = 0

# Errors
df['HB'] = -df['HB']
df['H'] = -df['H']
df['HC'] = 0
df['HD'] = 0

df = df.fillna(method='ffill')

c_a = 1.01 * 1000
rho_a = 1.29
p0 = 1013
k = 0.4
z_0_A = 0.0017
df["h_aws_A"] = 1 - df["SnowHeight"]/100

# df['day'] = abs(df.T_probe_Avg) > 1 & df.WS_RSLT > 4
# df['night'] = abs(df.T_probe_Avg) <= 1 & df.WS_RSLT <= 4


df['day'] = df.WS_RSLT > 4
df['night'] = df.WS_RSLT <= 4

for i in range(0,df.shape[0]):

	# Sensible Heat
    df.loc[i, "HC"] = (
            c_a
            * rho_a
            * df.loc[i, "amb_press_Avg"] * 10
            / p0
            * math.pow(k, 2)
            * df.loc[i, "WS_RSLT"]
            * (df.loc[i, "T_probe_Avg"]-df.loc[i, "T_surface"])
            / ((np.log(df.loc[i, "h_aws_A"] / z_0_A)) ** 2)
    )

print(df['H'].corr(df['HC']))

dfd = df.set_index("TIMESTAMP").resample("D").mean().reset_index()
# dfday = df[df.day].set_index("TIMESTAMP").resample("D").mean().reset_index()
# dfnight = df[df.night].set_index("TIMESTAMP").resample("D").mean().reset_index()
dfday = df[df.day]
dfnight = df[df.night]

for i in range(0,df.shape[0]):
    if abs(df.loc[i, "HC"] - df.loc[i, "H"]) > 400:
                print(df.loc[i, "TIMESTAMP"], (df.loc[i, "T_probe_Avg"] - df.loc[i, "T_surface"]), df.loc[i, "WS_RSLT"], df.loc[i, "HC"]/(c_a*rho_a*math.pow(k, 2)))
                # df.loc[i, "HC"] = df.loc[i, "H"]

pp = PdfPages(folders["input_folder"] + site + "_eddy_validate" + ".pdf")

x = df.TIMESTAMP

x1 = dfday.TIMESTAMP

x2 = dfnight.TIMESTAMP


fig = plt.figure()


ax1 = fig.add_subplot(111)
y31 = dfd.H
ax1.plot(dfd.TIMESTAMP, y31, "k-", linewidth=0.5)
ax1.set_ylabel("Sonic A Sensible Heat [$W\\,m^{-2}$]")
ax1.grid()

ax1t = ax1.twinx()
ax1t.plot(dfd.TIMESTAMP, dfd.HC, "b-", linewidth=0.5)
ax1t.set_ylabel("Bulk Sensible Heat [$W\\,m^{-2}$]", color="b")
for tl in ax1t.get_yticklabels():
    tl.set_color("b")

ax1.set_ylim([-100,200])
ax1t.set_ylim([-100,200])
# format the ticks
ax1.xaxis.set_major_locator(mdates.WeekdayLocator())
ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
ax1.xaxis.set_minor_locator(mdates.DayLocator())

fig.autofmt_xdate()
pp.savefig(bbox_inches="tight")
plt.clf()

ax1 = fig.add_subplot(111)
y6 = df.T_probe_Avg
ax1.plot(x, y6, "k-", linewidth=0.5)
ax1.set_ylabel("Temperature Sonic")
ax1.grid()

ax1t = ax1.twinx()
ax1t.plot(x, df.T_surface, "b-", linewidth=0.5)
ax1t.set_ylabel("Temperature surface", color="b")
for tl in ax1t.get_yticklabels():
    tl.set_color("b")

# format the ticks
ax1.xaxis.set_major_locator(mdates.WeekdayLocator())
ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
ax1.xaxis.set_minor_locator(mdates.DayLocator())
fig.autofmt_xdate()
pp.savefig(bbox_inches="tight")
plt.clf()

ax1 = fig.add_subplot(111)
ax1.plot(x, df.HC, "k-", linewidth=0.5)
ax1.set_ylabel("Bulk Sensible Heat [$W\\,m^{-2}$]")
ax1.grid()

# format the ticks
ax1.xaxis.set_major_locator(mdates.WeekdayLocator())
ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
ax1.xaxis.set_minor_locator(mdates.DayLocator())

fig.autofmt_xdate()
pp.savefig(bbox_inches="tight")
plt.clf()

ax1 = fig.add_subplot(111)
ax1.plot(x, df.H, "k-", linewidth=0.5)
ax1.set_ylabel("Meas. Sensible Heat [$W\\,m^{-2}$]")
ax1.grid()

# format the ticks
ax1.xaxis.set_major_locator(mdates.WeekdayLocator())
ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
ax1.xaxis.set_minor_locator(mdates.DayLocator())

fig.autofmt_xdate()
pp.savefig(bbox_inches="tight")
plt.clf()

ax1 = fig.add_subplot(111)
ax1.plot(x, df.WS_RSLT, "k-", linewidth=0.5)
ax1.set_ylabel("Wind [$m\\,s$]")
ax1.grid()

# format the ticks
ax1.xaxis.set_major_locator(mdates.WeekdayLocator())
ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
ax1.xaxis.set_minor_locator(mdates.DayLocator())

fig.autofmt_xdate()
pp.savefig(bbox_inches="tight")
plt.clf()

ax1 = fig.add_subplot(111)
ax1.scatter(dfday.H, dfday.HC, s=2, color = "blue", label ="day")
ax1.scatter(dfnight.H, dfnight.HC, s=2, color = "orange", label ="night")
ax1.set_xlabel("Sonic A Sensible Heat [$W\\,m^{-2}$]")
ax1.set_ylabel("Bulk Sensible Heat [$W\\,m^{-2}$]")
ax1.grid()
ax1.legend()

lims = [
np.min([ax1.get_xlim(), ax1.get_ylim()]),  # min of both axes
np.max([ax1.get_xlim(), ax1.get_ylim()]),  # max of both axes
]

# now plot both limits against eachother
ax1.plot(lims, lims, '--k', alpha=0.25, zorder=0)
ax1.set_aspect('equal')
ax1.set_xlim(lims)
ax1.set_ylim(lims)
# format the ticks

pp.savefig(bbox_inches="tight")
plt.clf()

pp.close()