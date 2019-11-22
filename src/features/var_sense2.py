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

# problem = {"num_vars": 1, "names": ["discharge"], "bounds": [[9, 12]]}

# # Generate samples
# param_values = saltelli.sample(problem, 3, calc_second_order=False)

param_values = [4,6,8,10,12]

filename = os.path.join(
    folders['output_folder'], site + "_simulations_" + str(param_values) + ".csv"
)

# Plots
plt.rcParams["figure.figsize"] = (10,7)
matplotlib.rc('xtick', labelsize=5)
fig = plt.figure()
cmap = plt.cm.rainbow
norm = matplotlib.colors.Normalize(
    vmin=param_values[0], vmax=param_values[-1]
)

# Output file Initialise
columns = ["Ice", "IceV"]
index = range(0, len(param_values))
dfo = pd.DataFrame(index=index, columns=columns)
dfo = dfo.fillna(0)

for i, X in enumerate(param_values):

    filename = os.path.join(
        folders['output_folder'], site + "_simulations_" + str(X) + '_' + str(param_values) + ".csv"
    )

    #  read files
    dfd = pd.read_csv(filename, sep=",")
    dfd["When"] = pd.to_datetime(dfd["When"], format="%Y.%m.%d")
    dfd['When'] = dfd['When'].dt.strftime("%b %d")
    dfd = dfd.set_index("When")

    y1 = dfd['iceV']
    y2 = dfd['SW'] + dfd['LW'] + dfd['Qs'] + dfd['Ql']
    y3 = dfd['SA']

    y1 = dfd['SA']/dfd['iceV']
    y2 = dfd['r_ice']
    y3 = dfd['h_ice']

    ax1 = fig.add_subplot(3, 1, 1)
    if (X==10) or (X==12):
        ax1.plot(y1, linewidth=1, color=cmap(norm(X)))
    else:
        ax1.plot(y1, linewidth=0.5, color=cmap(norm(X)))
    # ax1.set_ylabel("Ice ($m^3$)")
    ax1.set_ylabel("Surface Area/Volume Ratio")
    ax1.set_ylim(0,200)
    x_axis = ax1.axes.get_xaxis()
    x_axis.set_visible(False)


    ax2 = fig.add_subplot(3, 1, 2)
    if (X==10) or (X==12):
        ax2.plot(y2, linewidth=1, color=cmap(norm(X)))
    else:
        ax2.plot(y2, linewidth=0.5, color=cmap(norm(X)))
    # ax2.set_ylabel("Energy ($W/m^{2}$)")
    ax2.set_ylabel("Ice Radius ($m$)")
    x_axis = ax2.axes.get_xaxis()
    x_axis.set_visible(False)


    ax3 = fig.add_subplot(3, 1, 3)
    if (X==10) or (X==12):
        ax3.plot(y3, linewidth=1, color=cmap(norm(X)))
    else:
        ax3.plot(y3, linewidth=0.5, color=cmap(norm(X)))
    # ax3.set_ylim(0, 200)
    ax3.set_ylabel("Ice Height ($m$)")
    ax3.set_xlabel("Days")

ax1.grid()
ax2.grid()
ax3.grid(axis = 'y')
plt.xticks(rotation=45)
plt.tight_layout()
fig.subplots_adjust(right=0.8)
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
cbar = fig.colorbar(sm, cax=cbar_ax)
cbar.set_label("Fountain Discharge ($l/min$)")

# rotates and right aligns the x labels, and moves the bottom of the axes up to make room for them
fig.autofmt_xdate()
plt.savefig(
    os.path.join(
        folders['output_folder'], site + "_simulations_" + str(param_values) + ".jpg"
    ),
    bbox_inches="tight",
    dpi=300,
)
plt.clf()

# dfo = dfo.round(4)
# filename2 = os.path.join(
#     folders['output_folder'], site + "_simulations_" + str(param_values) + ".csv"
# )
# dfo.to_csv(filename2, sep=",")
