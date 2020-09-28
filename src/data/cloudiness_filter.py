import pandas as pd
import math
import os

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.dates as mdates
from src.data.config import site, dates, fountain, folders

df = pd.read_csv(
    os.path.join(folders["input_folder"] + "solar_output.csv"), sep=",", header=0, parse_dates=["When"]
)

print(df.head())

r = df["cld"].rolling(window=11)
mps = r.mean() + 0.1
print(mps)
df['cld'] = df['cld'].where(df.cld < mps, np.nan)

df['cld'] = df['cld'].interpolate(method='linear')
df['cld'] = df['cld'].fillna(method='bfill')



pp = PdfPages(folders["input_folder"] + "cloudiness_filter.pdf")
fig, ax1 = plt.subplots()
x = df.When
y1 = df.cld
ax1.plot(df.When, y1, linewidth=0.5)
ax1.set_ylabel("Cloudiness")
ax1.grid()
ax1.xaxis.set_major_locator(mdates.WeekdayLocator())
ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
ax1.xaxis.set_minor_locator(mdates.DayLocator())

fig.autofmt_xdate()
pp.savefig(bbox_inches="tight")
plt.clf()

y1 = mps
ax1 = fig.add_subplot(111)
ax1.scatter(df.When, y1, s=0.1)
ax1.set_ylabel("Peaks")
ax1.grid()
ax1.xaxis.set_major_locator(mdates.WeekdayLocator())
ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
ax1.xaxis.set_minor_locator(mdates.DayLocator())

fig.autofmt_xdate()
pp.savefig(bbox_inches="tight")
plt.clf()

# y1 = df.cld_11
# ax1 = fig.add_subplot(111)
# ax1.plot(df.When, y1, "k-", linewidth=0.5)
# ax1.set_ylabel("Cloudiness")
# ax1.grid()
# ax1.xaxis.set_major_locator(mdates.WeekdayLocator())
# ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
# ax1.xaxis.set_minor_locator(mdates.DayLocator())
#
# fig.autofmt_xdate()
# pp.savefig(bbox_inches="tight")
# plt.clf()

pp.close()
