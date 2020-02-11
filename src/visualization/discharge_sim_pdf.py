import os
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from src.data.config import site, folders


filename2 = os.path.join(
    folders['sim_folder'], site + "_simulations_discharge.csv"
)
dfx = pd.read_csv(filename2, sep=",")



# Plots
filename3 = os.path.join(folders["sim_folder"], site + "_discharge_analysis.pdf")
pp = PdfPages(filename3)
x = dfx['Discharge']
y1 = dfx['r']
y2 = dfx['Max SA']
y3 = dfx.MaxV
y4 = (dfx['Meltwater']+ dfx['Endice'])/dfx['Water used'] *100
y5 = dfx['Avg Freeze Rate']
y6 = dfx['Meltwater']


fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.plot(x, y1, 'bo', markersize=3)
ax1.set_ylabel("Active Radius")
ax1.set_xlabel("Discharge")
ax1.grid()
pp.savefig(bbox_inches="tight")
plt.clf()

fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.plot(x, y2, 'bo', markersize=3)
ax1.set_ylabel("Max Surface Area")
ax1.set_xlabel("Discharge")
ax1.grid()
pp.savefig(bbox_inches="tight")
plt.clf()

fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.plot(x, y3, 'bo', markersize=3)
ax1.set_ylabel("Max Ice Volume [$m^3$]")
ax1.set_xlabel("Discharge")
ax1.grid()
pp.savefig(bbox_inches="tight")
plt.clf()

fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.plot(x, y4, 'bo', markersize=3)
ax1.set_ylabel("Efficiency[%]")
ax1.set_xlabel("Discharge")
ax1.grid()
pp.savefig(bbox_inches="tight")
plt.clf()

fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.plot(x, y3, 'ko', markersize=3)
ax1.set_ylabel("Max Ice Volume [$m^3$]")

ax2 = ax1.twinx()
ax2.plot(x, y5, "bo", markersize=3)
ax2.set_ylabel("Avg Freeze Rate [$l\,min^{-1}$]",  color="b")
for tl in ax2.get_yticklabels():
    tl.set_color("b")

ax1.set_xlabel("Discharge")
ax1.grid()
pp.savefig(bbox_inches="tight")
plt.clf()


fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.plot(x, y6, 'bo', markersize=3)
ax1.set_ylabel("Endice")
ax1.set_xlabel("Discharge")
ax1.grid()
pp.savefig(bbox_inches="tight")
plt.clf()

# x = dfx.r
y1 = dfx.MaxV
y2 = dfx['Water used']

fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.plot(x, y5, "ko", markersize=3)
ax1.set_ylabel("Avg Freeze Rate [$l\,min^{-1}$]")
ax1.set_xlabel("Discharge")
#
ax2 = ax1.twinx()
ax2.plot(x, y4, "bo", markersize=3)
ax2.set_ylabel("Efficiency[%]", color="b")
for tl in ax2.get_yticklabels():
    tl.set_color("b")

ax1.grid()
pp.savefig(bbox_inches="tight")
plt.clf()

pp.close()