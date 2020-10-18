import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.colors
import uncertainpy as un
import statistics as st
import sys
sys.path.append('/home/surya/Programs/PycharmProjects/air_model')
from src.data.config import site, dates, folders
from datetime import datetime
import matplotlib.dates as mdates


input = "/home/surya/Programs/PycharmProjects/air_model/data/processed/schwarzsee/simulations/data/"
figures = "/home/surya/Programs/PycharmProjects/air_model/data/processed/schwarzsee/simulations/figures/"

names = "full"
variance = []
mean = []
evaluations = []

data = un.Data()
filename1 = input + names + ".h5"
data.load(filename1)
# print(data)

eval = data["max_volume"].evaluations
print(f"95 percent confidence interval caused by {names} is {round(2 * st.stdev(eval),2)}")

print(data["max_volume"].mean)

df = pd.read_hdf(folders["output_folder"] + "model_output.h5", "df")

df = df.set_index('When').resample('1H').mean().reset_index()

days = pd.date_range(start=datetime(2019, 1, 30, 20), end=datetime(2019, 4, 8,1), freq="1H")

data = data["UQ_Icestupa"]

data['When'] = days

# data['When'] = data['When'] +  pd.to_timedelta(data.time, unit='D')

# data["IceV"] = df["IceV"]

# data = data.set_index("When").resample("5T").mean()




fig, ax = plt.subplots()


# ax.plot(data.When, data.mean, color = 'black', label="Mean", linewidth=0.5)
# ax.plot(data.time, data.percentile_5, color = 'blue')
# ax.set_xlabel("Time [Days]")
ax.set_ylabel("Ice Volume[$m^3$]")


ax.fill_between(data["When"],
                         data.percentile_5,
                         data.percentile_95, color='gray', alpha=0.2, label = "90% prediction interval")


x = df.When
y1 = df.iceV

ax.plot(x, y1, "b-", label = "Estimation", linewidth=0.5)
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

ax.set_ylim(bottom=0)
plt.legend()
# plt.legend(["Mean", "90% prediction interval", "Std", "Validation Measurement"], loc="best")

ax.xaxis.set_major_locator(mdates.WeekdayLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
ax.xaxis.set_minor_locator(mdates.DayLocator())
fig.autofmt_xdate()
plt.savefig(figures + "uncertainty.jpg", bbox_inches="tight", dpi=300)
# plt.show()
