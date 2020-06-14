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

df = df.fillna(method='ffill')

df.rename(
    columns={
        'Tice_Avg(' + str(2) + ')': "T_surface",
    },
    inplace=True,
)

g=9.81
rho_a = 1.29
p0 = 1013
k = 0.4
L_e = 2514 * 1000  # J/kg Evaporation



z_0_A = 0.05320543621007756

df["h_aws"] = 2 - df["SnowHeight"]/100

df["e_surface"] = 6.112

for i in range(0,df.shape[0]):
			
	df.loc[i,'Ri_A'] = (
	g
	* (df.loc[i, "h_aws"] - z_0_A)
	* (df.loc[i, "T_probe_Avg"])
	/ ((df.loc[i, "T_probe_Avg"]+273) * df.loc[i, "WS_RSLT"] ** 2))

	# Sensible Heat
	df.loc[i, "LE"] = (
	        0.623
	        * L_e
	        * rho_a
	        / p0
	        * math.pow(k, 2)
	        * df.loc[i, "WS_RSLT"]
	        * (df.loc[i, "e_probe"] * 10-df.loc[i, "e_surface"])
	        / ((np.log(df.loc[i, "h_aws"] / z_0_A)) ** 2)
	)


	# if abs(df.loc[i,'Ri_A']) < 0.2:

	#     if df.loc[i,'Ri_A'] > 0:
	#         df.loc[i, "LE"] = df.loc[i, "LE"] * (1- 5 * df.loc[i,'Ri_A']) ** 2
	#     else:
	#         df.loc[i, "LE"] = df.loc[i, "LE"] * math.pow((1- 16 * df.loc[i,'Ri_A']), 0.75)


df["F_surf"] = df["NETRAD"] - df["H"] + df["LE"]


dfd = df.set_index("TIMESTAMP").resample("D").mean().reset_index()

print(df["F_surf"].mean(), df['F_surf'].corr(df['T_surface']))

pp = PdfPages(folders["input_folder"] + "F_surf" + ".pdf")

x = df.TIMESTAMP

fig = plt.figure()

ax1 = fig.add_subplot(111)
y1 = df.F_surf
ax1.plot(x, y1, "k-", linewidth=0.5)
ax1.set_ylabel("F_surf [$W\\,m^{-2}$]")
ax1.grid()

ax1t = ax1.twinx()
ax1t.plot(x, df.T_surface, "b-", linewidth=0.5)
ax1t.set_ylabel("Ground Temperature[$\\degree C$]", color="b")
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
y1 = df.LE
ax1.plot(x, y1, "k-", linewidth=0.5)
ax1.set_ylabel("LE [$W\\,m^{-2}$]")
ax1.grid()

# format the ticks
ax1.xaxis.set_major_locator(mdates.WeekdayLocator())
ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
ax1.xaxis.set_minor_locator(mdates.DayLocator())
fig.autofmt_xdate()
pp.savefig(bbox_inches="tight")
plt.clf()

ax1 = fig.add_subplot(111)
y1 = df["e_probe"] * 10 - df["e_surface"]
ax1.plot(x, y1, "k-", linewidth=0.5)
ax1.set_ylabel("Vapour Gradient [$hPa$]")
ax1.grid()

# format the ticks
ax1.xaxis.set_major_locator(mdates.WeekdayLocator())
ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
ax1.xaxis.set_minor_locator(mdates.DayLocator())
fig.autofmt_xdate()
pp.savefig(bbox_inches="tight")
plt.clf()


ax1 = fig.add_subplot(111)
ax1.scatter(dfd.F_surf, dfd.T_surface, s=2)
ax1.set_xlabel("Surface Energy Flux [$W\\,m^{-2}$]")
ax1.set_ylabel("Ground Temperature [$\\degree C$]")
ax1.grid()

pp.savefig(bbox_inches="tight")
plt.clf()

pp.close()