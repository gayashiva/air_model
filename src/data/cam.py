import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.backends.backend_pdf import PdfPages
import math
import time
from tqdm import tqdm
import shutil, os
import glob
import fnmatch
from src.data.config import site, dates, folders, fountain, surface

dir = "/home/surya/Programs/PycharmProjects/air_model/data/raw/"

df_in1 = pd.read_csv(dir + "Results_1.csv", sep = ',')
df_in2 = pd.read_csv(dir + "Results_2.csv", sep = ',')
df_in3 = pd.read_csv(dir + "Results_rad.csv", sep = ',')


df_in1['Label'] = df_in1['Label'].str.split("m").str[-1]
df_in2['Label'] = df_in2['Label'].str.split("m").str[-1]
df_in3['Label'] = df_in3['Label'].str.split("m").str[-1]


df_in1["Label"] = '2020-' + df_in1["Label"].str[2:4] + "-" + df_in1["Label"].str[4:6] + " " + df_in1["Label"].str[6:8]
df_in2["Label"] = '2020-' + df_in2["Label"].str[2:4] + "-" + df_in2["Label"].str[4:6] + " " + df_in2["Label"].str[6:8]
df_in3["Label"] = '2020-' + df_in3["Label"].str[2:4] + "-" + df_in3["Label"].str[4:6] + " " + df_in3["Label"].str[6:8]


df_in1["When"] = pd.to_datetime(df_in1["Label"], format='%Y-%m-%d %H')
df_in2["When"] = pd.to_datetime(df_in2["Label"], format='%Y-%m-%d %H')
df_in3["When"] = pd.to_datetime(df_in3["Label"], format='%Y-%m-%d %H')


df_in1 = df_in1.set_index("When")
df_in2 = df_in2.set_index("When")

df_in3['Radius'] = df_in3['Length']/(183.85 * 2)

print(df_in3.head())


df_in = df_in1.append(df_in2)

df_in = df_in.reset_index()


# df_in = pd.merge(df_in1, df_in2, how='inner', left_index=True)

print(df_in.head())
print(df_in.tail())

dfd = df_in.set_index("When").resample("D").mean().reset_index()
dfd2 = df_in3.set_index("When").resample("D").mean().reset_index()




pp = PdfPages(folders["input_folder"] + 'guttannen' + "_cam" + ".pdf")

x = dfd.When

fig = plt.figure()

ax1 = fig.add_subplot(111)
y1 = dfd.Area
ax1.plot(x, y1, 'o-')
ax1.set_ylabel("Area [$m^2$]")
ax1.grid()

# format the ticks
ax1.xaxis.set_major_locator(mdates.WeekdayLocator())
ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
ax1.xaxis.set_minor_locator(mdates.DayLocator())
fig.autofmt_xdate()
pp.savefig(bbox_inches="tight")
plt.clf()

ax1 = fig.add_subplot(111)
y1 = dfd2.Radius
ax1.plot(dfd2.When, y1, 'o-')
ax1.set_ylabel("Radius [$m$]")
ax1.grid()

# format the ticks
ax1.xaxis.set_major_locator(mdates.WeekdayLocator())
ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
ax1.xaxis.set_minor_locator(mdates.DayLocator())
fig.autofmt_xdate()
pp.savefig(bbox_inches="tight")
plt.clf()

pp.close()