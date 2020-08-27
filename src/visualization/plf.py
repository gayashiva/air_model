import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.backends.backend_pdf import PdfPages
import os

dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
# read files
df_in = pd.read_csv(
    os.path.join(dir, "data/raw/PLF_Feb20.txt"),
    encoding="latin-1",
    skiprows=2,
    sep=";",
)

# Drop
df_in=df_in.drop(['stn'],axis=1)

df_in["When"] = pd.to_datetime(df_in["time"], format="%Y%m%d%H%M")  # Datetime

df_in["Prec"] = pd.to_numeric(
    df_in["rre150z0"], errors="coerce"
)  # Add Precipitation data

df_in["Temperature"] = pd.to_numeric(
    df_in["tre200s0"], errors="coerce"
)  # Air temperature

df_in["WindSpeed"] = pd.to_numeric(
    df_in["fkl010z0"], errors="coerce"
)  # Wind speed

df_in["Humidity"] = pd.to_numeric(
    df_in["ure200s0"], errors="coerce"
)

mask = (df_in["When"] >= datetime(2020, 2, 15)) & (df_in["When"] <= datetime(2020, 2, 18))
df_in = df_in.loc[mask]
df_in = df_in.reset_index()

print(df_in.info())

pp = PdfPages(os.path.join(dir, "data/processed/PLF.pdf"))

x = df_in.When

fig = plt.figure()

ax1 = fig.add_subplot(111)
y1 = df_in.Temperature
ax1.plot(x, y1)
ax1.set_ylabel("Temperature [$C$]")
ax1.grid()

# format the ticks
ax1.xaxis.set_major_locator(mdates.DayLocator())
ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
ax1.xaxis.set_minor_locator(mdates.HourLocator())
fig.autofmt_xdate()
pp.savefig(bbox_inches="tight")
plt.clf()

ax1 = fig.add_subplot(111)
y1 = df_in.Humidity
ax1.plot(x, y1)
ax1.set_ylabel("Humidity [$\\%$]")
ax1.grid()

# format the ticks
ax1.xaxis.set_major_locator(mdates.DayLocator())
ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
ax1.xaxis.set_minor_locator(mdates.HourLocator())
fig.autofmt_xdate()
pp.savefig(bbox_inches="tight")
plt.clf()

ax1 = fig.add_subplot(111)
y1 = df_in.Prec
ax1.plot(df_in.When, y1)
ax1.set_ylabel("Precipitation [$mm$]")
ax1.grid()

# Add labels to the plot
style = dict(size=10, color='gray', rotation=90)

ax1.text(datetime(2020, 2, 17, 13,35), 0, "Sonic Lower(A) failed", **style)
ax1.text(datetime(2020, 2, 16, 2,34), 0, "Sonic Upper(B) failed", **style)

# format the ticks
ax1.xaxis.set_major_locator(mdates.DayLocator())
ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
ax1.xaxis.set_minor_locator(mdates.HourLocator())
fig.autofmt_xdate()
pp.savefig(bbox_inches="tight")
plt.clf()

ax1 = fig.add_subplot(111)
y1 = df_in.WindSpeed
ax1.plot(x, y1)
ax1.set_ylabel("Wind Speed [$ms^{-1}$]")
ax1.grid()

# Add labels to the plot
style = dict(size=10, color='gray', rotation=90)

ax1.text(datetime(2020, 2, 17, 13,35), 0, "Sonic Lower(A) failed", **style)
ax1.text(datetime(2020, 2, 16, 2,34), 0, "Sonic Upper(B) failed", **style)

# format the ticks
ax1.xaxis.set_major_locator(mdates.DayLocator())
ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
ax1.xaxis.set_minor_locator(mdates.HourLocator())
fig.autofmt_xdate()
pp.savefig(bbox_inches="tight")
plt.clf()

pp.close()



