import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.backends.backend_pdf import PdfPages

dir = "/home/surya/Programs/PycharmProjects/air_model/data/raw/"

df = pd.read_csv(dir + "April-June2017_Meltwater.csv", sep = ',')

df["When"] = pd.to_datetime(df["Date"], format='%d/%m/%y')

print(df["Meltwater"].mean())

print(df.head())

pp = PdfPages("/home/surya/Programs/PycharmProjects/air_model/data/interim/ladakh_melt.pdf")

x = df.When

fig = plt.figure()

ax1 = fig.add_subplot(111)
y1 = df.Meltwater
ax1.bar(x, y1)
ax1.set_ylabel("Daily Meltwater [$m^3$]")
ax1.grid(axis="y")

# format the ticks
ax1.xaxis.set_major_locator(mdates.WeekdayLocator())
ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
ax1.xaxis.set_minor_locator(mdates.DayLocator())
fig.autofmt_xdate()
pp.savefig(bbox_inches="tight")
plt.clf()
pp.close()