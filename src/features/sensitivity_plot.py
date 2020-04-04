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

problem = {"num_vars": 1, "names": ["dx"],
           "bounds": [[1e-04, 1e-03]]}
param_values = saltelli.sample(problem, 5, calc_second_order=False)

filename = os.path.join(
        folders['sim_folder'], site + "_simulations_" + str(problem["names"]) + str(param_values.shape[0]) + ".csv"
    )

#  read files
dfo = pd.read_csv(filename, sep=",")

print(dfo)

x = dfo.dx
y1 = dfo.Max_IceV

fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.scatter(x, y1)
plt.show()

# # Plots
# fig = plt.figure()
# cmap = plt.cm.rainbow
# norm = matplotlib.colors.Normalize(
#     vmin=problem["bounds"][0][0], vmax=problem["bounds"][0][1]
# )
#
# for i in range(dfo.shape[0]):
#     x = dfo.Efficiency[i]
#     y1 = dfo.Max_IceV[i]
#
#     ax1 = fig.add_subplot(111)
#     ax1.scatter(x, y1, color=cmap(norm(dfo.dx[i])))
#     ax1.set_ylabel("Max_IceV[$m^3$]")
#     ax1.set_xlabel("Efficiency")
#
# # format the ticks
# ax1.grid()
#
# sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
# sm.set_array([])
# cbar = fig.colorbar(sm)
# cbar.set_label("Layer Thickness[$m$]")
# # plt.show()
#
# plt.savefig(
#     os.path.join(
#         folders['sim_folder'], site + "_simulations_" + str(problem["names"]) + str(problem["bounds"])  + ".jpg"
#     ),
#     bbox_inches="tight",
#     dpi=300,
# )


