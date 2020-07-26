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


dir = "/home/surya/Programs/PycharmProjects/air_model/data/raw/"

df_in1 = pd.read_csv(dir + "Results_1.csv", sep=",")
df_in2 = pd.read_csv(dir + "Results_2.csv", sep=",")
df_in3 = pd.read_csv(dir + "Results_rad.csv", sep=",")
df_in4 = pd.read_csv(dir + "Results_rad2.csv", sep=",")
df_in5 = pd.read_csv(dir + "Results_dots.csv", sep=",")
df_in6 = pd.read_csv(dir + "Results_dots_2.csv", sep=",")
df_in7 = pd.read_csv(dir + "Results_rest.csv", sep=",")


mypath = "/home/surya/Pictures/Guttannen_Jan"
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
df_names = pd.DataFrame({"col": onlyfiles})

mypath2 = "/home/surya/Pictures/Guttannen_Feb"
onlyfiles2 = [f for f in listdir(mypath2) if isfile(join(mypath2, f))]
df_names2 = pd.DataFrame({"col": onlyfiles2})

mypath3 = "/home/surya/Pictures/Rest"
onlyfiles3 = [f for f in listdir(mypath3) if isfile(join(mypath3, f))]
df_names3 = pd.DataFrame({"col": onlyfiles3})


df_in1["Label"] = df_in1["Label"].str.split("m").str[-1]
df_in2["Label"] = df_in2["Label"].str.split("m").str[-1]
df_in3["Label"] = df_in3["Label"].str.split("m").str[-1]
df_in4["Label"] = df_in4["Label"].str.split("m").str[-1]
df_in5["Label"] = df_in5["Label"].str.split("m").str[-1]
df_in6["Label"] = df_in6["Label"].str.split("m").str[-1]
df_in7["Label"] = df_in7["Label"].str.split("m").str[-1]

df_names["Label"] = df_names["col"].str.split("m").str[-1]
df_names2["Label"] = df_names2["col"].str.split("m").str[-1]
df_names3["Label"] = df_names3["col"].str.split("m").str[-1]


df_in1["Label"] = (
    "2020-"
    + df_in1["Label"].str[2:4]
    + "-"
    + df_in1["Label"].str[4:6]
    + " "
    + df_in1["Label"].str[6:8]
)
df_in2["Label"] = (
    "2020-"
    + df_in2["Label"].str[2:4]
    + "-"
    + df_in2["Label"].str[4:6]
    + " "
    + df_in2["Label"].str[6:8]
)
df_in3["Label"] = (
    "2020-"
    + df_in3["Label"].str[2:4]
    + "-"
    + df_in3["Label"].str[4:6]
    + " "
    + df_in3["Label"].str[6:8]
)
df_in4["Label"] = (
    "2020-"
    + df_in4["Label"].str[2:4]
    + "-"
    + df_in4["Label"].str[4:6]
    + " "
    + df_in4["Label"].str[6:8]
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


df_in1["When"] = pd.to_datetime(df_in1["Label"], format="%Y-%m-%d %H")
df_in2["When"] = pd.to_datetime(df_in2["Label"], format="%Y-%m-%d %H")
df_in3["When"] = pd.to_datetime(df_in3["Label"], format="%Y-%m-%d %H")
df_in4["When"] = pd.to_datetime(df_in4["Label"], format="%Y-%m-%d %H")
df_in5["When"] = pd.to_datetime(df_in5["Label"], format="%Y-%m-%d %H")
df_in6["When"] = pd.to_datetime(df_in6["Label"], format="%Y-%m-%d %H")
df_in7["When"] = pd.to_datetime(df_in7["Label"], format="%Y-%m-%d %H")

df_names["When"] = pd.to_datetime(df_names["Label"], format="%Y-%m-%d %H")
df_names2["When"] = pd.to_datetime(df_names2["Label"], format="%Y-%m-%d %H")
df_names3["When"] = pd.to_datetime(df_names3["Label"], format="%Y-%m-%d %H")


df_names = df_names.set_index("When").sort_index()
df_names = df_names.reset_index()
df_names["Slice"] = df_names.index + 1

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

df_in1 = df_in1.set_index("When")
df_in2 = df_in2.set_index("When")


df_in3["Radius"] = df_in3["Length"] / (183.85 * 2)
df_in4["Radius"] = df_in4["Length"] / (183.85 * 2)


df_in = df_in1.append(df_in2)
df_in = df_in.reset_index()

df_in_rad = df_in3.append(df_in4)
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

print(df_in_dot.columns)
print(df_in_dot.tail())


dfd = df_in.set_index("When").resample("D").mean().reset_index()
dfd2 = df_in_rad.set_index("When").resample("D").mean().reset_index()
dfd3 = pd.merge(dfd, dfd2, how="inner", on=["When"])
dfd4 = df_in_dot.set_index("When").resample("D").mean().reset_index()

dfd3["Height"] = dfd3["Area"] / dfd3["Radius"]

dfd4["Radius"] = (dfd4["right"] - dfd4["left"]) / (183.85 * 2)
dfd4["Height"] = -(dfd4["height"] - dfd4.loc[0, "height"]) / (183.85)

dfd4["Volume"] = abs(1 / 3 * math.pi * dfd4["Radius"] ** 2 * dfd4["Height"])

dfd4["SA"] = abs(
    math.pi
    * dfd4["Radius"]
    * np.power((dfd4["Radius"] **2 + (dfd4["Height"]) ** 2), 1 / 2)
)

print(dfd4["Radius"].mean())

# Correct Hollow Volume
mask = dfd4["When"] > datetime(2020, 2, 10)
mask_index = dfd4[mask].index
dfd4.loc[mask_index,"Volume"] += 1/3* math.pi * fountain["tree_height"] * 6**2


df_out = dfd4[["When", "Radius", "Height", "SA", "Volume"]]

df_out.to_csv(folders["input_folder"] + "cam.csv")

pp = PdfPages(folders["input_folder"] + site + "_cam" + ".pdf")

x = dfd.When

fig = plt.figure()

# ax1 = fig.add_subplot(111)
# y1 = dfd.Area
# ax1.plot(x, y1, 'o-', color="k")
# ax1.set_ylabel("Area [$m^2$]")
# ax1.grid()

# # format the ticks
# ax1.xaxis.set_major_locator(mdates.WeekdayLocator())
# ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
# ax1.xaxis.set_minor_locator(mdates.DayLocator())
# fig.autofmt_xdate()
# pp.savefig(bbox_inches="tight")
# plt.clf()

# ax1 = fig.add_subplot(111)
# y1 = dfd.Area
# ax1.plot(dfd2.When, dfd2.Radius, 'o-', color="k")
# ax1.set_ylabel("Radius [$m$]")
# ax1.grid()

# # format the ticks
# ax1.xaxis.set_major_locator(mdates.WeekdayLocator())
# ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
# ax1.xaxis.set_minor_locator(mdates.DayLocator())
# fig.autofmt_xdate()
# pp.savefig(bbox_inches="tight")
# plt.clf()

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
