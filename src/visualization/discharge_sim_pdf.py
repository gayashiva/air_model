import logging
import os
import time
from datetime import datetime
from logging import StreamHandler
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from src.data.config import site, option, folders, fountain, surface
from src.models.air_forecast import icestupa

from SALib.sample import saltelli
from SALib.analyze import sobol
import matplotlib.colors

# Create the Logger
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Create the Handler for logging data to a file
logger_handler = logging.FileHandler(
    os.path.join(os.path.join(folders["dirname"], "data/logs/"), site + "_site.log"),
    mode="w",
)
logger_handler.setLevel(logging.DEBUG)

# Create the Handler for logging data to console.
console_handler = StreamHandler()
console_handler.setLevel(logging.CRITICAL)

# Create a Formatter for formatting the log messages
logger_formatter = logging.Formatter("%(name)s - %(levelname)s - %(message)s")

# Add the Formatter to the Handler
logger_handler.setFormatter(logger_formatter)
console_handler.setFormatter(logger_formatter)

# Add the Handler to the Logger
logger.addHandler(logger_handler)
logger.addHandler(console_handler)

filename2 = os.path.join(
    folders['output_folder'], site + "_simulations_discharge.csv"
)
dfx = pd.read_csv(filename2, sep=",")



# Plots
filename3 = os.path.join(folders["output_folder"], site + "_shapeanalysis.pdf")
pp = PdfPages(filename3)
x = dfx.Discharge
y1 = dfx['r']
y2 = dfx['Max SA']
y3 = dfx.MaxV
y4 = dfx['Max SA']/dfx['MaxV']
y5 = dfx['Max Growthrate']/ 5
y6 = dfx['Endice']


fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.plot(x, y1, 'bo', markersize=3)
ax1.set_ylabel("Active Radius")
ax1.set_xlabel("Discharge ($l/min$)")
ax1.grid()
pp.savefig(bbox_inches="tight")
plt.clf()

fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.plot(x, y2, 'bo', markersize=3)
ax1.set_ylabel("Max Surface Area")
ax1.set_xlabel("Discharge ($l/min$)")
ax1.grid()
pp.savefig(bbox_inches="tight")
plt.clf()

fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.plot(x, y3, 'bo', markersize=3)
ax1.set_ylabel("Max Ice Volume ($m^3$)")
ax1.set_xlabel("Discharge ($l/min$)")
ax1.grid()
pp.savefig(bbox_inches="tight")
plt.clf()

fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.plot(x, y4, 'bo', markersize=3)
ax1.set_ylabel("Max SA/ Max V")
ax1.set_xlabel("Discharge ($l/min$)")
ax1.grid()
pp.savefig(bbox_inches="tight")
plt.clf()

fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.plot(x, y5, 'bo', markersize=3)
ax1.set_ylabel("Max Growth Rate ($l/min$)")
ax1.set_xlabel("Discharge ($l/min$)")
ax1.grid()
pp.savefig(bbox_inches="tight")
plt.clf()

fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.plot(x, y6, 'bo', markersize=3)
ax1.set_ylabel("Endice")
ax1.set_xlabel("Discharge ($l/min$)")
ax1.grid()
pp.savefig(bbox_inches="tight")
plt.clf()

# x = dfx.r
y1 = dfx.MaxV
y2 = dfx['Max Growthrate']/ 5

fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.plot(x, y1, "ko", markersize=3)
ax1.set_ylabel("Max Ice Volume ($m^3$)")
ax1.set_xlabel("Discharge ($l/min$)")

ax2 = ax1.twinx()
ax2.plot(x, y2, "bo", markersize=3)
ax2.set_ylabel("Max Growth Rate ($l/min$)", color="b")
for tl in ax2.get_yticklabels():
    tl.set_color("b")

ax1.grid()
pp.savefig(bbox_inches="tight")
plt.clf()

pp.close()