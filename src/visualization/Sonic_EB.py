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

df = df.fillna(method='ffill')

df["Balance"] = df["NETRAD"] - df["H"]

print(df["Balance"].mean())

dfd = df.set_index("TIMESTAMP").resample("D").mean().reset_index()

pp = PdfPages(folders["input_folder"] + "EB" + ".pdf")

x = dfd.TIMESTAMP

fig = plt.figure()
ax1 = fig.add_subplot(111)

y1 = dfd.Balance
ax1.plot(x, y1, "k-", linewidth=0.5)
ax1.set_ylabel("Temperature [$\\degree C$]")
ax1.grid()

# format the ticks
ax1.xaxis.set_major_locator(mdates.WeekdayLocator())
ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
ax1.xaxis.set_minor_locator(mdates.DayLocator())
ax1.grid()
fig.autofmt_xdate()
pp.savefig(bbox_inches="tight")
plt.clf()

pp.close()