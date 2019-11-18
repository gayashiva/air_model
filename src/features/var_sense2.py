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

# python -m src.features.build_features

# Create the Logger
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Create the Handler for logging data to a file
logger_handler = logging.FileHandler(
    os.path.join(os.path.join(folders['dirname'], "data/logs/"), site + "_site.log"), mode="w"
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

#  read files
filename0 = os.path.join(folders['input_folder'], site + "_" + option + "_input.csv")
df_in = pd.read_csv(filename0, sep=",")
df_in["When"] = pd.to_datetime(df_in["When"], format="%Y.%m.%d %H:%M:%S")

# end
end_date = df_in["When"].iloc[-1]


problem = {"num_vars": 1, "names": ["discharge"], "bounds": [[7, 8]]}

# # Generate samples
# param_values = saltelli.sample(problem, 3, calc_second_order=False)

param_values = [[7], [8]]


# Plots
fig = plt.figure()
cmap = plt.cm.rainbow
norm = matplotlib.colors.Normalize(
    vmin=problem["bounds"][0][0], vmax=problem["bounds"][0][1]
)

# Output file Initialise
columns = ["Ice", "IceV"]
index = range(0, len(param_values))
dfo = pd.DataFrame(index=index, columns=columns)
dfo = dfo.fillna(0)

for i, X in enumerate(param_values):
    #  read files
    filename0 = os.path.join(folders['input_folder'], site + "_" + option + "_input.csv")
    df_in = pd.read_csv(filename0, sep=",")
    df_in["When"] = pd.to_datetime(df_in["When"], format="%Y.%m.%d %H:%M:%S")

    print("Discharge", X[0])
    fountain['discharge'] = X[0]
    df = icestupa(df_in, fountain, surface)
    dfo.loc[i, "discharge"] = X[0]
    dfo.loc[i, "Ice"] = float(df["ice"].tail(1))
    dfo.loc[i, "Meltwater"] = float(df["meltwater"].tail(1))
    dfo.loc[i, "Vapour"] = float(df["vapour"].tail(1))
    dfo.loc[i, "Ice Max"] = df["ice"].max()
    dfo.loc[i, "Runtime"] = df["When"].iloc[-1]

    x = df.set_index('When').resample('D').mean().reset_index()
    x.index = np.arange(1, len(x) + 1)
    y1 = x['iceV']
    y2 = x['SW'] + x['LW'] + x['Qs'] + x['Ql']
    y3 = x['SA']/ x['iceV']
    y4 = x['h_ice']

    ax1 = fig.add_subplot(2, 2, 1)
    ax1.plot(y1, linewidth=0.5, color=cmap(norm(X[0])))
    ax1.set_ylabel("Ice ($m^3$)")
    ax1.set_xlabel("Days")


    ax2 = fig.add_subplot(2, 2, 2)
    ax2.plot(y2, linewidth=0.5, color=cmap(norm(X[0])))
    ax2.set_ylabel("Energy ($W/m^{2}$)")
    ax2.set_xlabel("Days")


    ax3 = fig.add_subplot(2, 2, 3)
    ax3.plot(y3, linewidth=0.5, color=cmap(norm(X[0])))
    ax3.set_ylim(0, 100)
    ax3.set_ylabel("SA/V ($m^{-1}$)")
    ax3.set_xlabel("Days")


    ax4 = fig.add_subplot(2, 2, 4)
    ax4.plot(y4, linewidth=0.5, color=cmap(norm(X[0])))
    ax4.set_ylabel("Height ($m$)")
    ax4.set_xlabel("Days")


ax1.grid()
ax2.grid()
ax3.grid()
ax4.grid()
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
        folders['output_folder'], site + "_simulations_" + str(problem["names"][0]) + ".jpg"
    ),
    bbox_inches="tight",
    dpi=300,
)
plt.clf()

dfo = dfo.round(4)
filename2 = os.path.join(
    folders['output_folder'], site + "_simulations_" + str(problem["names"][0]) + ".csv"
)
dfo.to_csv(filename2, sep=",")
