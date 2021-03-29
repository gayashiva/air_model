import matplotlib.pyplot as plt
import sys
import os

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
)
import numpy as np
import pandas as pd
import math
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.colors
import uncertainpy as un
import statistics as st
from datetime import datetime, timedelta
from src.data.config import SITE, FOUNTAIN, FOLDERS
import matplotlib.dates as mdates

input = FOLDERS["sim_folder"] + "/"
figures = FOLDERS["sim_folder"] + "/"

names = "full"
variance = []
mean = []
evaluations = []

data = un.Data()
filename1 = input + names + ".h5"
data.load(filename1)
# print(data)

eval = data["max_volume"].evaluations
print(
    f"95 percent confidence interval caused by {names} is {round(2 * st.stdev(eval),2)}"
)

print(data["max_volume"].mean)

df = pd.read_hdf(FOLDERS["output_folder"] + "model_output.h5", "df")

df = df.set_index("When").resample("1H").mean().reset_index()

# start=datetime(2019, 1, 30, 20)
# end= start + timedelta(hours = 65*24)
days = pd.date_range(
    start=datetime(2019, 1, 30, 20),
    end=datetime(2019, 1, 30, 20) + timedelta(hours=75 * 24 - 1),
    freq="1H",
)

data = data["UQ_Icestupa"]
# print(len(data.percentile_5))

data["When"] = days

# data['When'] = data['When'] +  pd.to_timedelta(data.time, unit='D')

# data["IceV"] = df["IceV"]

# data = data.set_index("When").resample("5T").mean()

CB91_Blue = "#2CBDFE"
CB91_Green = "#47DBCD"
CB91_Pink = "#F3A0F2"
CB91_Purple = "#9D2EC5"
CB91_Violet = "#661D98"
CB91_Amber = "#F5B14C"

fig, ax = plt.subplots()


# ax.plot(data.When, data.mean, color = 'black', label="Mean", linewidth=0.5)
# ax.plot(data.time, data.percentile_5, color = 'blue')
# ax.set_xlabel("Time [Days]")
ax.set_ylabel("Ice Volume[$m^3$]")


ax.fill_between(
    data["When"],
    data.percentile_5,
    data.percentile_95,
    color="xkcd:gray",
    alpha=0.2,
    label="90% prediction interval",
)


x = df.When
y1 = df.iceV

# ax1.set_xlabel("Days")

# Include Validation line segment 1
ax.plot(
    [datetime(2019, 2, 14, 16), datetime(2019, 2, 14, 16)],
    [0.67115, 1.042],
    color="green",
    lw=1,
    label="Validation Measurement",
)
ax.scatter(datetime(2019, 2, 14, 16), 0.856575, color="green", marker="o")

# Include Validation line segment 2
ax.plot(
    [datetime(2019, 3, 10, 18), datetime(2019, 3, 10, 18)],
    [0.037, 0.222],
    color="green",
    lw=1,
)
ax.scatter(datetime(2019, 3, 10, 18), 0.1295, color="green", marker="o")

ax.plot(x, y1, "b-", label="Modelled Ice Volume", linewidth=1, color=CB91_Blue)
ax.set_ylim(bottom=0)
plt.legend()
# plt.legend(["Mean", "90% prediction interval", "Std", "Validation Measurement"], loc="best")

ax.xaxis.set_major_locator(mdates.WeekdayLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
ax.xaxis.set_minor_locator(mdates.DayLocator())
fig.autofmt_xdate()
plt.savefig(figures + "uncertainty.jpg", bbox_inches="tight", dpi=300)
plt.savefig(FOLDERS["output_folder"] + "jpg/Figure_8.jpg", dpi=300, bbox_inches="tight")
# plt.show()
