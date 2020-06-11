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

# df.rename(
#     columns={
#         'Tice_Avg(' + str(2) + ')': "T_surface",
#     },
#     inplace=True,
# )

df["T_surface"] = 0

# Errors
df['HB'] = -df['HB']
df['H'] = -df['H']
df['HC'] = 0
df['HD'] = 0



print(df.info())

df = df.fillna(method='ffill')


g = 9.81

df['z_0_A'] = 0.0017 
df['z_0_B'] = 0.0017

c_a = 1.01 * 1000
rho_a = 1.29
p0 = 1013
k = 0.4
df['Ri_A'] = 0
df['Ri_B'] = 0


df["h_aws_A"] = 1.2 - df["SnowHeight"]/100
df["h_aws_B"] = 3 - df["SnowHeight"]/100

z=[]
z_B=[]

j=0


while (df['HC'].corr(df['H']) < 0.9 or np.isnan(df['HC'].corr(df['H']))) and j < 5:
	j+=1
	z_0_A = df['z_0_A'].mean()
	df['z_0_A'] = z_0_A
	z.append(z_0_A)
	print("Iteration", j, z_0_A, df['HC'].corr(df['H']))

	z_0_B = df['z_0_B'].mean()
	df['z_0_B'] = z_0_B
	z_B.append(z_0_B)
	print("Iteration", j, z_0_B, df['HD'].corr(df['HB']))

	for i in range(0,df.shape[0]):
		
	    df.loc[i,'Ri_A'] = (
	    g
	    * (df.loc[i, "h_aws_A"] - z_0_A)
	    * (df.loc[i, "T_SONIC"]-df.loc[i, "T_surface"])
	    / ((df.loc[i, "T_SONIC"]+273) * df.loc[i, "WS_RSLT"] ** 2))

	    df.loc[i,'Ri_B'] = (
	    g
	    * (df.loc[i, "h_aws_B"] - z_0_B)
	    * (df.loc[i, "TB_SONIC"]-df.loc[i, "T_surface"])
	    / ((df.loc[i, "TB_SONIC"]+273) * df.loc[i, "WSB_RSLT"] ** 2))

	    # Sensible Heat
	    df.loc[i, "HC"] = (
	            c_a
	            * rho_a
	            * df.loc[i, "amb_press_Avg"] * 10
	            / p0
	            * math.pow(k, 2)
	            * df.loc[i, "WS_RSLT"]
	            * (df.loc[i, "T_SONIC"]-df.loc[i, "T_surface"])
	            / ((np.log(df.loc[i, "h_aws_A"] / z_0_A)) ** 2)
	    )

	    # Sensible Heat
	    df.loc[i, "HD"] = (
	            c_a
	            * rho_a
	            * df.loc[i, "amb_press_Avg"] * 10
	            / p0
	            * math.pow(k, 2)
	            * df.loc[i, "WSB_RSLT"]
	            * (df.loc[i, "TB_SONIC"]-df.loc[i, "T_surface"])
	            / ((np.log(df.loc[i, "h_aws_B"] / z_0_B)) ** 2)
	    )



	    if (df.loc[i,'Ri_A']) < 0.2:

	        if df.loc[i,'Ri_A'] > 0:
	            df.loc[i, "HC"] = df.loc[i, "HC"] * (1- 5 * df.loc[i,'Ri_A']) ** 2
	        else:
	            df.loc[i, "HC"] = df.loc[i, "HC"] * math.pow((1- 16 * df.loc[i,'Ri_A']), 0.75)

	    if (df.loc[i,'Ri_B']) < 0.2:

	        if df.loc[i,'Ri_B'] > 0:
	            df.loc[i, "HD"] = df.loc[i, "HD"] * (1- 5 * df.loc[i,'Ri_B']) ** 2
	        else:
	            df.loc[i, "HD"] = df.loc[i, "HD"] * math.pow((1- 16 * df.loc[i,'Ri_B']), 0.75)


	    df.loc[i,'z_0_A'] = (
	                ((np.log(df.loc[i, "h_aws_A"] / df.loc[i,'z_0_A'])) ** 2)
	                * df.loc[i, "HC"]
	                / (df.loc[i, "H"])
	        )

	    df.loc[i,'z_0_B'] = (
	                ((np.log(df.loc[i, "h_aws_B"] / df.loc[i,'z_0_B'])) ** 2)
	                * df.loc[i, "HD"]
	                / (df.loc[i, "HB"])
	        )
	    

	    df.loc[i,'z_0_A'] = math.pow(abs(df.loc[i,'z_0_A']), 1/2)
	    df.loc[i,'z_0_B'] = math.pow(abs(df.loc[i,'z_0_B']), 1/2)


	    df.loc[i,'z_0_A'] = np.log(df.loc[i, "h_aws_A"]) - df.loc[i,'z_0_A']
	    df.loc[i,'z_0_B'] = np.log(df.loc[i, "h_aws_B"]) - df.loc[i,'z_0_B']
	    

	    df.loc[i,'z_0_A'] = math.exp(df.loc[i,'z_0_A'])
	    df.loc[i,'z_0_B'] = math.exp(df.loc[i,'z_0_B'])

	    # if np.isnan(df.loc[i,'z_0_A']):
	    #     print(df.loc[i, "TIMESTAMP"], df.loc[i, "T_SONIC"], df.loc[i, "WS_RSLT"], df.loc[i,'z_0_A'])



print(df['z_0_A'].mean())
# print(df['z_0_B'].mean())


# print(df["amb_press_Avg"].head(), df["T_SONIC"].head(), df["WS"].head(), df["HC"].head() )

dfd = df.set_index("TIMESTAMP").resample("D").mean().reset_index()

for i in range(0,df.shape[0]):
    if abs(df.loc[i, "HC"] - df.loc[i, "H"]) > 1000:
                print(df.loc[i, "TIMESTAMP"], (df.loc[i, "T_SONIC"] - df.loc[i, "T_surface"]), df.loc[i, "WS_RSLT"], df.loc[i, "HC"]/(c_a*rho_a*math.pow(k, 2)))

pp = PdfPages(folders["input_folder"] + site + "_eddy" + ".pdf")

x = df.TIMESTAMP

fig = plt.figure()

ax1 = fig.add_subplot(111)

x1 = dfd.TIMESTAMP

y31 = dfd.H
ax1.plot(x1, y31, "k-", linewidth=0.5)
ax1.set_ylabel("Sonic Sensible Heat [$W\\,m^{-2}$]")
ax1.grid()

ax1t = ax1.twinx()
ax1t.plot(x1, dfd.HC, "b-", linewidth=0.5)
ax1t.set_ylabel("Bulk Sensible Heat [$W\\,m^{-2}$]", color="b")
for tl in ax1t.get_yticklabels():
    tl.set_color("b")

ax1.set_ylim([-100,100])
ax1t.set_ylim([-100,100])
# format the ticks
ax1.xaxis.set_major_locator(mdates.WeekdayLocator())
ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
ax1.xaxis.set_minor_locator(mdates.DayLocator())
ax1.grid()
fig.autofmt_xdate()
pp.savefig(bbox_inches="tight")
plt.clf()

ax1 = fig.add_subplot(111)
y6 = df.T_SONIC - df.T_surface
ax1.plot(x, y6, "k-", linewidth=0.5)
ax1.set_ylabel("Temperature Diff")
ax1.grid()

# ax1t = ax1.twinx()
# ax1t.plot(x, df.T_surface, "b-", linewidth=0.5)
# ax1t.set_ylabel("Temperature surface", color="b")
# for tl in ax1t.get_yticklabels():
#     tl.set_color("b")

# format the ticks
ax1.xaxis.set_major_locator(mdates.WeekdayLocator())
ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
ax1.xaxis.set_minor_locator(mdates.DayLocator())
ax1.grid()
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
ax1.grid()
fig.autofmt_xdate()
pp.savefig(bbox_inches="tight")
plt.clf()


ax1 = fig.add_subplot(111)
ax1.scatter(dfd.H, dfd.HC, s=2)
ax1.set_xlabel("Sonic Sensible Heat [$W\\,m^{-2}$]")
ax1.set_ylabel("Bulk Sensible Heat [$W\\,m^{-2}$]")
ax1.grid()


lims = [
np.min([ax1.get_xlim(), ax1.get_ylim()]),  # min of both axes
np.max([ax1.get_xlim(), ax1.get_ylim()]),  # max of both axes
]
lims = [
np.min([-100, 100]),  # min of both axes
np.max([-100, 100]),  # max of both axes
]

# now plot both limits against eachother
ax1.plot(lims, lims, '--k', alpha=0.25, zorder=0)
ax1.set_aspect('equal')
ax1.set_xlim(lims)
ax1.set_ylim(lims)
# format the ticks
ax1.grid()
pp.savefig(bbox_inches="tight")
plt.clf()


ax1 = fig.add_subplot(111)
y6 = df.WS_RSLT
ax1.plot(x, y6, "k-", linewidth=0.5)
ax1.set_ylabel("Wind Sonic [$m\\,s^{-1}$]")
ax1.grid()

# format the ticks
ax1.xaxis.set_major_locator(mdates.WeekdayLocator())
ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
ax1.xaxis.set_minor_locator(mdates.DayLocator())
ax1.grid()
fig.autofmt_xdate()
pp.savefig(bbox_inches="tight")
plt.clf()

ax1 = fig.add_subplot(111)
y6 = df.h_aws_A
ax1.plot(x, y6, "k-", linewidth=0.5)
ax1.set_ylabel("AWS height above snow [$m$]")
ax1.grid()

# format the ticks
ax1.xaxis.set_major_locator(mdates.WeekdayLocator())
ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
ax1.xaxis.set_minor_locator(mdates.DayLocator())
ax1.grid()
fig.autofmt_xdate()
pp.savefig(bbox_inches="tight")
plt.clf()

ax1 = fig.add_subplot(111)
y7 = df.Ri_A
ax1.plot(x, y7, "k-", linewidth=0.5)
ax1.set_ylabel("Ri")
ax1.grid()
# ax1.set_ylim([-0.2,0.2])


# format the ticks
ax1.xaxis.set_major_locator(mdates.WeekdayLocator())
ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
ax1.xaxis.set_minor_locator(mdates.DayLocator())
ax1.grid()
fig.autofmt_xdate()
pp.savefig(bbox_inches="tight")
plt.clf()

ax1 = fig.add_subplot(111)

ax1.scatter(range(0,j),z, s = 2)
ax1.set_ylabel("z_0_A")
ax1.set_xlabel("Iteration")

ax1.grid()

pp.savefig(bbox_inches="tight")
plt.clf()

pp.close()



pp = PdfPages(folders["input_folder"] + site + "_eddyB" + ".pdf")

x = df.TIMESTAMP

fig = plt.figure()

ax1 = fig.add_subplot(111)

x1 = dfd.TIMESTAMP

y31 = dfd.HB

ax1.plot(x1, y31, "k-", linewidth=0.5)
ax1.set_ylabel("Sonic Sensible Heat [$W\\,m^{-2}$]")
ax1.grid()

ax1t = ax1.twinx()
ax1t.plot(x1, dfd.HD, "b-", linewidth=0.5)
ax1t.set_ylabel("Bulk Sensible Heat [$W\\,m^{-2}$]", color="b")
for tl in ax1t.get_yticklabels():
    tl.set_color("b")

ax1.set_ylim([-100,100])
ax1t.set_ylim([-100,100])
# format the ticks
ax1.xaxis.set_major_locator(mdates.WeekdayLocator())
ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
ax1.xaxis.set_minor_locator(mdates.DayLocator())
ax1.grid()
fig.autofmt_xdate()
pp.savefig(bbox_inches="tight")
plt.clf()

ax1 = fig.add_subplot(111)
ax1.plot(x, df.HD, "k-", linewidth=0.5)
ax1.set_ylabel("Bulk Sensible Heat [$W\\,m^{-2}$]")
ax1.grid()

# format the ticks
ax1.xaxis.set_major_locator(mdates.WeekdayLocator())
ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
ax1.xaxis.set_minor_locator(mdates.DayLocator())
ax1.grid()
fig.autofmt_xdate()
pp.savefig(bbox_inches="tight")
plt.clf()

ax1 = fig.add_subplot(111)
ax1.scatter(df.HB, df.HD, s=2)
ax1.set_xlabel("Sonic Sensible Heat [$W\\,m^{-2}$]")
ax1.set_ylabel("Bulk Sensible Heat [$W\\,m^{-2}$]")
ax1.grid()


lims = [
np.min([ax1.get_xlim(), ax1.get_ylim()]),  # min of both axes
np.max([ax1.get_xlim(), ax1.get_ylim()]),  # max of both axes
]
lims = [
np.min([-100, 100]),  # min of both axes
np.max([-100, 100]),  # max of both axes
]

# now plot both limits against eachother
ax1.plot(lims, lims, '--k', alpha=0.25, zorder=0)
ax1.set_aspect('equal')
ax1.set_xlim(lims)
ax1.set_ylim(lims)
# format the ticks
ax1.grid()
pp.savefig(bbox_inches="tight")
plt.clf()


ax1 = fig.add_subplot(111)
y6 = df.WSB_RSLT
ax1.plot(x, y6, "k-", linewidth=0.5)
ax1.set_ylabel("Wind Sonic [$m\\,s^{-1}$]")
ax1.grid()

# format the ticks
ax1.xaxis.set_major_locator(mdates.WeekdayLocator())
ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
ax1.xaxis.set_minor_locator(mdates.DayLocator())
ax1.grid()
fig.autofmt_xdate()
pp.savefig(bbox_inches="tight")
plt.clf()

ax1 = fig.add_subplot(111)
y6 = df.h_aws_B
ax1.plot(x, y6, "k-", linewidth=0.5)
ax1.set_ylabel("AWS height above snow [$m$]")
ax1.grid()

# format the ticks
ax1.xaxis.set_major_locator(mdates.WeekdayLocator())
ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
ax1.xaxis.set_minor_locator(mdates.DayLocator())
ax1.grid()
fig.autofmt_xdate()
pp.savefig(bbox_inches="tight")
plt.clf()

ax1 = fig.add_subplot(111)
y6 = df.TB_SONIC
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
ax1.grid()
fig.autofmt_xdate()
pp.savefig(bbox_inches="tight")
plt.clf()

ax1 = fig.add_subplot(111)
y7 = df.Ri_B
ax1.plot(x, y7, "k-", linewidth=0.5)
ax1.set_ylabel("Ri")
ax1.grid()
# ax1.set_ylim([-0.2,0.2])


# format the ticks
ax1.xaxis.set_major_locator(mdates.WeekdayLocator())
ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
ax1.xaxis.set_minor_locator(mdates.DayLocator())
ax1.grid()
fig.autofmt_xdate()
pp.savefig(bbox_inches="tight")
plt.clf()

ax1 = fig.add_subplot(111)

ax1.scatter(range(0,j),z_B, s = 2)
ax1.set_ylabel("z_0_B")
ax1.set_xlabel("Iteration")

ax1.grid()

pp.savefig(bbox_inches="tight")
plt.clf()

pp.close()