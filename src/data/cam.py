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
from os import listdir
from os.path import isfile, join


dir = "/home/surya/Programs/PycharmProjects/air_model/data/raw/guttannen/"

def lum2temp(y0):

	df_in = pd.read_csv(dir + "lum_values.csv", sep=",")

	# Correct values
	mask = (df_in["X"]<2000)
	df_in= df_in[mask]

	k = df_in.loc[df_in["Y"] == df_in["Y"].max(), "X"].values

	# Correct values
	mask = (df_in["X"]<k[0])
	df_in= df_in[mask]

	x = df_in.X
	y = df_in.Y

	h = df_in.loc[df_in["Y"] == 200, "X"].values

	x1 = x[:h[0]]
	y1 = y[:h[0]]
	A1 = np.vstack([x1, np.ones(len(x1))]).T
	m1, c1 = np.linalg.lstsq(A1, y1, rcond=None)[0]

	x2 = x[h[0]:]
	y2 = y[h[0]:]
	A2 = np.vstack([x2, np.ones(len(x2))]).T
	m2, c2 = np.linalg.lstsq(A2, y2, rcond=None)[0]

	if y0 >= 200:
		x0 = (y0-c2)/m2
	else:
		x0 = (y0-c1)/m1

	return x0

df_in_section_1 = pd.read_csv(dir + "Results_1.csv", sep=",")
df_in_section_2 = pd.read_csv(dir + "Results_2.csv", sep=",")

df_rad0 = pd.read_csv(dir + "Results_radiuslines0.csv", sep=",")
df_rad1 = pd.read_csv(dir + "Results_radiuslines1.csv", sep=",")
df_rad2 = pd.read_csv(dir + "Results_radiuslines2.csv", sep=",")
df_in5 = pd.read_csv(dir + "Results_dots.csv", sep=",")
df_in6 = pd.read_csv(dir + "Results_dots_2.csv", sep=",")
df_in7 = pd.read_csv(dir + "Results_rest.csv", sep=",")

# Thermal
df_th = pd.read_csv(dir + "Results_full_thermal.csv", sep=",")
df_lum = pd.read_csv(dir + "Results_lum_full.csv", sep=",")
df_err = pd.read_csv(dir + "Results_errors.csv", sep=",")

mypath = "/home/surya/Pictures/Guttannen_Jan"
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
df_names = pd.DataFrame({"col": onlyfiles})

mypath2 = "/home/surya/Pictures/Guttannen_Feb"
onlyfiles2 = [f for f in listdir(mypath2) if isfile(join(mypath2, f))]
df_names2 = pd.DataFrame({"col": onlyfiles2})

mypath3 = "/home/surya/Pictures/Rest"
onlyfiles3 = [f for f in listdir(mypath3) if isfile(join(mypath3, f))]
df_names3 = pd.DataFrame({"col": onlyfiles3})


df_in_section_1["Label"] = df_in_section_1["Label"].str.split("m").str[-1]
df_in_section_2["Label"] = df_in_section_2["Label"].str.split("m").str[-1]
df_rad0["Label"] = df_rad0["Label"].str.split("m").str[-1]
df_rad1["Label"] = df_rad1["Label"].str.split("m").str[-1]
df_rad2["Label"] = df_rad2["Label"].str.split("m").str[-1]
df_th["Label"] = df_th["Label"].str.split("m").str[-1]


df_in5["Label"] = df_in5["Label"].str.split("m").str[-1]
df_in6["Label"] = df_in6["Label"].str.split("m").str[-1]
df_in7["Label"] = df_in7["Label"].str.split("m").str[-1]

df_names["Label"] = df_names["col"].str.split("m").str[-1]
df_names2["Label"] = df_names2["col"].str.split("m").str[-1]
df_names3["Label"] = df_names3["col"].str.split("m").str[-1]


df_in_section_1["Label"] = (
    "2020-"
    + df_in_section_1["Label"].str[2:4]
    + "-"
    + df_in_section_1["Label"].str[4:6]
    + " "
    + df_in_section_1["Label"].str[6:8]
)
df_in_section_2["Label"] = (
    "2020-"
    + df_in_section_2["Label"].str[2:4]
    + "-"
    + df_in_section_2["Label"].str[4:6]
    + " "
    + df_in_section_2["Label"].str[6:8]
)
df_rad0["Label"] = (
    "2020-"
    + df_rad0["Label"].str[2:4]
    + "-"
    + df_rad0["Label"].str[4:6]
    + " "
    + df_rad0["Label"].str[6:8]
)
df_rad1["Label"] = (
    "2020-"
    + df_rad1["Label"].str[2:4]
    + "-"
    + df_rad1["Label"].str[4:6]
    + " "
    + df_rad1["Label"].str[6:8]
)
df_rad2["Label"] = (
    "2020-"
    + df_rad2["Label"].str[2:4]
    + "-"
    + df_rad2["Label"].str[4:6]
    + " "
    + df_rad2["Label"].str[6:8]
)
df_in5["Label"] = (
    "2020-"
    + df_in5["Label"].str[2:4]
    + "-"
    + df_in5["Label"].str[4:6]
    + " "
    + df_in5["Label"].str[6:8]
)
df_in6["Label"] = (
    "2020-"
    + df_in6["Label"].str[2:4]
    + "-"
    + df_in6["Label"].str[4:6]
    + " "
    + df_in6["Label"].str[6:8]
)
df_in7["Label"] = (
    "2020-"
    + df_in7["Label"].str[2:4]
    + "-"
    + df_in7["Label"].str[4:6]
    + " "
    + df_in7["Label"].str[6:8]
)

df_names["Label"] = (
    "2020-"
    + df_names["Label"].str[2:4]
    + "-"
    + df_names["Label"].str[4:6]
    + " "
    + df_names["Label"].str[6:8]
)
df_names2["Label"] = (
    "2020-"
    + df_names2["Label"].str[2:4]
    + "-"
    + df_names2["Label"].str[4:6]
    + " "
    + df_names2["Label"].str[6:8]
)
df_names3["Label"] = (
    "2020-"
    + df_names3["Label"].str[2:4]
    + "-"
    + df_names3["Label"].str[4:6]
    + " "
    + df_names3["Label"].str[6:8]
)
df_th["Label"] = (
    "2020-"
    + df_th["Label"].str[2:4]
    + "-"
    + df_th["Label"].str[4:6]
    + " "
    + df_th["Label"].str[6:8]
)


df_in_section_1["When"] = pd.to_datetime(df_in_section_1["Label"], format="%Y-%m-%d %H")
df_in_section_2["When"] = pd.to_datetime(df_in_section_2["Label"], format="%Y-%m-%d %H")
df_rad0["When"] = pd.to_datetime(df_rad0["Label"], format="%Y-%m-%d %H")
df_rad1["When"] = pd.to_datetime(df_rad1["Label"], format="%Y-%m-%d %H")
df_rad2["When"] = pd.to_datetime(df_rad2["Label"], format="%Y-%m-%d %H")
df_th["When"] = pd.to_datetime(df_th["Label"], format="%Y-%m-%d %H")


df_in5["When"] = pd.to_datetime(df_in5["Label"], format="%Y-%m-%d %H")
df_in6["When"] = pd.to_datetime(df_in6["Label"], format="%Y-%m-%d %H")
df_in7["When"] = pd.to_datetime(df_in7["Label"], format="%Y-%m-%d %H")

df_names["When"] = pd.to_datetime(df_names["Label"], format="%Y-%m-%d %H")
df_names2["When"] = pd.to_datetime(df_names2["Label"], format="%Y-%m-%d %H")
df_names3["When"] = pd.to_datetime(df_names3["Label"], format="%Y-%m-%d %H")


df_names = df_names.set_index("When").sort_index()
df_names = df_names.reset_index()
df_names["Slice"] = df_names.index + 1
df_lum["Slice"] = df_lum.index + 1

df_lum = df_lum[:320]

#Remove errors
for i in range(0, df_lum.shape[0]):
    for j in range(0, df_err.shape[0]):
        if df_err.loc[j, "Slice"] == df_lum.loc[i, "Slice"]:
            df_lum.loc[i, "Mean1"] = np.NaN


df_names2 = df_names2.set_index("When").sort_index()
df_names2 = df_names2.reset_index()
df_names2["Slice"] = df_names2.index + 1

df_names3 = df_names3.set_index("When").sort_index()
df_names3 = df_names3.reset_index()
df_names3["Slice"] = df_names3.index + 1

for i in range(0, df_names.shape[0]):
    for j in range(0, df_in5.shape[0]):
        if df_in5.loc[j, "Slice"] == df_names.loc[i, "Slice"]:
            df_in5.loc[j, "When"] = df_names.loc[i, "When"]

for i in range(0, df_names2.shape[0]):
    for j in range(0, df_in6.shape[0]):
        if df_in6.loc[j, "Slice"] == df_names2.loc[i, "Slice"]:
            df_in6.loc[j, "When"] = df_names2.loc[i, "When"]

for i in range(0, df_names3.shape[0]):
    for j in range(0, df_in7.shape[0]):
        if df_in7.loc[j, "Slice"] == df_names3.loc[i, "Slice"]:
            df_in7.loc[j, "When"] = df_names3.loc[i, "When"]

df_in_section_1 = df_in_section_1.set_index("When")
df_in_section_2 = df_in_section_2.set_index("When")


df_rad0["Radius"] = df_rad0["Length"] / 2
df_rad1["Radius"] = df_rad1["Length"] / 2
df_rad2["Radius"] = df_rad2["Length"] / 2

df_in = df_in_section_1.append(df_in_section_2)
df_in = df_in.reset_index()

df_in_rad = df_rad0.append(df_rad1)
df_in_rad = df_rad0.append(df_rad2)
df_in_rad = df_in_rad.reset_index()


left = df_in5.index % 3 == 0
centre = df_in5.index % 3 == 1
right = df_in5.index % 3 == 2

df_in5["right"] = df_in5.loc[right, "X"]
df_in5["left"] = df_in5.loc[left, "X"]
df_in5["height"] = df_in5.loc[centre, "Y"]

df_in5.right = df_in5.sort_values(["Slice", "right"]).right.ffill()
df_in5.left = df_in5.sort_values(["Slice", "left"]).left.ffill()
df_in5.height = df_in5.sort_values(["Slice", "height"]).height.ffill()

left = df_in6.index % 3 == 0
centre = df_in6.index % 3 == 1
right = df_in6.index % 3 == 2

df_in6["right"] = df_in6.loc[right, "X"]
df_in6["left"] = df_in6.loc[left, "X"]
df_in6["height"] = df_in6.loc[centre, "Y"] + 106

df_in6.right = df_in6.sort_values(["Slice", "right"]).right.ffill()
df_in6.left = df_in6.sort_values(["Slice", "left"]).left.ffill()
df_in6.height = df_in6.sort_values(["Slice", "height"]).height.ffill()

left = df_in7.index % 3 == 0
centre = df_in7.index % 3 == 1
right = df_in7.index % 3 == 2

df_in7["right"] = df_in7.loc[right, "X"]
df_in7["left"] = df_in7.loc[left, "X"]
df_in7["height"] = df_in7.loc[centre, "Y"] + 1436 - 1156

df_in7.right = df_in7.sort_values(["Slice", "right"]).right.ffill()
df_in7.left = df_in7.sort_values(["Slice", "left"]).left.ffill()
df_in7.height = df_in7.sort_values(["Slice", "height"]).height.ffill()

df_in_dot = df_in5.append(df_in6)

df_in_dot = df_in_dot.append(df_in7)

df_in_dot = df_in_dot.drop(["X", "Y", "S.No.", "Label", " "], axis=1)

# Correct thermal values
mask = (df_th["Mean"] > 180)
mask_index = df_th[mask].index
df_th.loc[mask_index, "Mean"] = np.NaN

# Correct thermal values
mask = (df_th["Mean"] < 165.876)
mask_index = df_th[mask].index
df_th.loc[mask_index, "Mean"] = np.NaN

m = 2.4/(df_th["Mean"].max()-168.419)
c = -m*df_th["Mean"].max()
df_th["Temp"] = m*df_th["Mean"] + c


df_lum = df_lum.set_index("Slice")
df_th = df_th.set_index("Slice")

df_lum["When"] = df_th["When"]
df_th = df_th.reset_index()
df_lum = df_lum.reset_index()

df_lum["Mean"] = df_lum["Mean1"].apply(lum2temp)

T_0 = df_lum.loc[df_lum["When"] == datetime(2020, 1, 26,9), "Mean"].values
T_measured = df_lum.loc[df_lum["When"] == datetime(2020, 1, 24,16), "Mean"].values

m = 2.4/(T_0- T_measured)
c = -m*T_0
df_lum["Temp"] = m*df_lum["Mean"] + c


dfd = df_in.set_index("When").resample("D").mean().reset_index()
dfd2 = df_in_rad.set_index("When").resample("D").mean().reset_index()
dfd3 = pd.merge(dfd, dfd2, how="inner", on=["When"])
dfd4 = df_in_dot.set_index("When").resample("D").mean().reset_index()
dfd_th = df_th.set_index("When").resample("D").mean().reset_index()
dfd_lum = df_lum.set_index("When").resample("D").mean().reset_index()

dfd3["Height"] = dfd3["Area"] / dfd3["Radius"]

dfd4["Radius"] = (dfd4["right"] - dfd4["left"]) / (183.85 * 2)
dfd4["Height"] = -(dfd4["height"] - dfd4.loc[0, "height"]) / (183.85)

dfd4["Volume"] = abs(1 / 3 * math.pi * dfd4["Radius"] ** 2 * dfd4["Height"])

dfd4["SA"] = abs(
    math.pi
    * dfd4["Radius"]
    * np.power((dfd4["Radius"] **2 + (dfd4["Height"]) ** 2), 1 / 2)
)

# Correct Hollow Volume
mask = dfd4["When"] > datetime(2020, 2, 10)
mask_index = dfd4[mask].index
dfd4.loc[mask_index,"Volume"] += 1/3* math.pi * fountain["tree_height"] * 6**2

dfd4 = dfd4.set_index("When")
dfd_th = dfd_th.set_index("When")

dfd4["Temp"] = dfd_th["Temp"]
dfd4 = dfd4.reset_index()
dfd_th = dfd_th.reset_index()

# dfd4.merge(dfd_th[["When", "Temp"]],on='When', how='left')


df_out = dfd4[["When", "Radius", "Height", "SA", "Volume", "Temp"]]

df_out.to_csv(folders["input_folder"] + "cam.csv")

df_out2 = df_th[["When", "Mean", "Temp"]]
df_out2.to_csv(folders["input_folder"] + "temp.csv")

df_out2 = df_lum[["When", "Mean1", "Temp"]]
df_out2.to_csv(folders["input_folder"] + "lumtemp.csv")

pp = PdfPages(folders["input_folder"] + site + "_cam" + ".pdf")

x = dfd.When

fig = plt.figure()

ax1 = fig.add_subplot(111)
ax1.plot(dfd_lum.When, dfd_lum.Temp, 'o-', color="k")
ax1.set_ylabel("Temp ")
ax1.grid()

# format the ticks
ax1.xaxis.set_major_locator(mdates.WeekdayLocator())
ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
ax1.xaxis.set_minor_locator(mdates.DayLocator())
fig.autofmt_xdate()
pp.savefig(bbox_inches="tight")
plt.clf()

ax1 = fig.add_subplot(111)
ax1.scatter(df_lum.When, df_lum.Temp, color="b", alpha=0.5, s = 1)
ax1.set_ylabel("Temp [$m^2$]")
ax1.grid()

# format the ticks
ax1.xaxis.set_major_locator(mdates.WeekdayLocator())
ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
ax1.xaxis.set_minor_locator(mdates.DayLocator())
fig.autofmt_xdate()
pp.savefig(bbox_inches="tight")
plt.clf()

ax1 = fig.add_subplot(111)
y1 = dfd4.Height
ax1.plot(dfd4.When, y1, "o-", color="k")
ax1.set_ylabel("Height [$m$]")
ax1.grid()

ax1t = ax1.twinx()
ax1t.plot(dfd4.When, dfd4.Radius, "o-", color="b", alpha=0.5, linewidth=0.5)
ax1t.set_ylabel("Radius [$m$]", color="b")
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
ax1.plot(dfd4.When, dfd4.SA, "o-", color="k")
ax1.set_ylabel("SA [$m^2$]")
ax1.grid()

# format the ticks
ax1.xaxis.set_major_locator(mdates.WeekdayLocator())
ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
ax1.xaxis.set_minor_locator(mdates.DayLocator())
fig.autofmt_xdate()
pp.savefig(bbox_inches="tight")
plt.clf()

ax1 = fig.add_subplot(111)
ax1.plot(dfd4.When, dfd4.Volume, "o-", color="k")
ax1.set_ylabel("Volume [$m^3$]")
ax1.grid()

# format the ticks
ax1.xaxis.set_major_locator(mdates.WeekdayLocator())
ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
ax1.xaxis.set_minor_locator(mdates.DayLocator())
fig.autofmt_xdate()
pp.savefig(bbox_inches="tight")
plt.clf()

pp.close()
