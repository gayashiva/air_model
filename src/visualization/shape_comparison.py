import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
import matplotlib.ticker as mticker
import matplotlib.dates as mdates
from matplotlib.backends.backend_pdf import PdfPages
import os

filename = "/home/surya/Programs/PycharmProjects/air_model/data/processed/schwarzsee/"
df_cyl = pd.read_csv(filename + "cylinder_model_results.csv", sep=",")
df_cyl = df_cyl[['When', 'iceV', 'SA']]
df_cyl = df_cyl.rename(
            {
                "iceV": "iceV_cyl",
                "SA": "SA_cyl",
            },
            axis=1,
        )
df_cone = pd.read_csv(filename + "model_results.csv", sep=",")
df_cone = df_cone[['When', 'iceV', 'SA']]

df_cyl['When'] = pd.to_datetime(df_cyl['When'])
df_cone['When'] = pd.to_datetime(df_cone['When'])

df=pd.merge(df_cone,df_cyl, how='outer', on = 'When')

df = df.set_index('When').resample('1H').mean().reset_index()

pp = PdfPages(filename + "shape_results.pdf")

x= df.When
y1 = df.iceV
y2 = df.iceV_cyl
fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.plot(x, y1, "k-")
ax1.set_ylabel("Cone Ice Volume [$m^3$]")
# ax1.set_xlabel("Days")

ax2 = ax1.twinx()
ax2.plot(x, y2, "b-", linewidth=0.5)
ax2.set_ylabel("Cylinder Ice Volume [$m^3$]", color="b")
for tl in ax2.get_yticklabels():
    tl.set_color("b")

ax2.set_ylim(ax1.get_ylim())

#  format the ticks
ax1.xaxis.set_major_locator(mdates.WeekdayLocator())
ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
ax1.xaxis.set_minor_locator(mdates.DayLocator())
ax1.grid()
fig.autofmt_xdate()
pp.savefig(bbox_inches="tight")
plt.clf()

y1 = df.SA
y2 = df.SA_cyl
fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.plot(x, y1, "k-")
ax1.set_ylabel("Cone Area [$m^2$]")
# ax1.set_xlabel("Days")

ax2 = ax1.twinx()
ax2.plot(x, y2, "b-", linewidth=0.5)
ax2.set_ylabel("Cylinder Area [$m^2$]", color="b")
for tl in ax2.get_yticklabels():
    tl.set_color("b")

ax2.set_ylim(ax1.get_ylim())
#  format the ticks
ax1.xaxis.set_major_locator(mdates.WeekdayLocator())
ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
ax1.xaxis.set_minor_locator(mdates.DayLocator())
ax1.grid()
fig.autofmt_xdate()
pp.savefig(bbox_inches="tight")
pp.close()