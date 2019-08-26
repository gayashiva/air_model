import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib.colors
import matplotlib.dates as mdates
from matplotlib.backends.backend_pdf import PdfPages
import os
from src.models.air_forecast import icestupa
import time
from SALib.sample import saltelli
from SALib.analyze import sobol

# python -m src.features.var_sense


site = input("Input the Field Site Name: ") or "plaffeien"

dirname = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

input_folder = os.path.join(dirname, "data/interim/")

output_folder = os.path.join(dirname, "data/processed/")

start = time.time()

#  read files
filename0 = os.path.join(input_folder, site + "_model_input.csv")
df_in = pd.read_csv(filename0, sep=",")
df_in["When"] = pd.to_datetime(df_in["When"], format="%Y.%m.%d %H:%M:%S")

# end
end_date = df_in["When"].iloc[-1]

problem = {"num_vars": 1, "names": ["dx"], "bounds": [[0.001, 0.01]]}

# Generate samples
param_values = saltelli.sample(problem, 10)

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
    print(X)
    df = icestupa(df_in, dx=X[0])
    dfo.loc[i, "Ice"] = float(df["ice"].tail(1))
    dfo.loc[i, "Meltwater"] = float(df["meltwater"].tail(1))
    dfo.loc[i, "Vapour"] = float(df["vapour"].tail(1))
    dfo.loc[i, "Ice Max"] = df["ice"].max()
    dfo.loc[i, "Runtime"] = df["When"].iloc[-1]

    x = df["When"]
    y1 = df["iceV"]

    ax1 = fig.add_subplot(111)
    ax1.plot(x, y1, linewidth=0.5, color=cmap(norm(X[0])))
    ax1.set_ylabel("Ice Volume[$m^3$]")
    ax1.set_xlabel("Days")

# format the ticks
ax1.xaxis.set_major_locator(mdates.WeekdayLocator())
ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
ax1.xaxis.set_minor_locator(mdates.DayLocator())
ax1.grid()

sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = fig.colorbar(sm)
cbar.set_label("Ice layer thickness[$m$]")

# rotates and right aligns the x labels, and moves the bottom of the axes up to make room for them
fig.autofmt_xdate()
plt.savefig(
    os.path.join(
        output_folder, site + "_simulations__" + str(problem["names"][0]) + ".jpg"
    ),
    bbox_inches="tight",
    dpi=300,
)
plt.clf()

dfo = dfo.round(4)
filename2 = os.path.join(
    output_folder, site + "_simulations__" + str(problem["names"][0]) + ".csv"
)
dfo.to_csv(filename2, sep=",")
