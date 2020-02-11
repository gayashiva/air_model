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
import math
from matplotlib.backends.backend_pdf import PdfPages
from src.data.config import site, option, folders, fountain, surface
from src.models.air_forecast import fountain_runtime, albedo, projectile_xy, getSEA

from SALib.sample import saltelli
from SALib.analyze import sobol
import matplotlib.colors

problem = {"num_vars": 2, "names": ["discharge", "crit_temp"], "bounds": [[8, 14], [-8, 2]]}

filename2 = os.path.join(
    folders['sim_folder'], site + "_simulations_" + str(problem["names"]) + ".csv"
)

df = pd.read_csv(filename2, sep=",")
df["Efficiency"] = z = df["Efficiency"] * 100
df =df.round(1)

# Plots
pp = PdfPages(folders["sim_folder"] + site + "_optimize" + ".pdf")
fig = plt.figure()
cmap = plt.cm.rainbow_r
norm = matplotlib.colors.Normalize(
    vmin=df["Efficiency"].min() , vmax=df["Efficiency"].max()
)

x = df["crit_temp"]
y = df["discharge"]
z = df["Efficiency"]


ax1 = fig.add_subplot(111)
ax1.scatter(x, y, c=z)
ax1.set_ylabel("Discharge[$l\,min^{-1}$]")
ax1.set_xlabel("Critical Temperature [$\degree C$]")

sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = fig.colorbar(sm)
cbar.set_label('Efficiency[$\%$]')

for i in df.index:
    ax1.annotate(str(df.loc[i, "Max IceV"])+ '$m^3$', (df.loc[i, "crit_temp"], df.loc[i, "discharge"]))

plt.savefig(
    os.path.join(folders["sim_folder"], site + "_optimize.jpg"),
    bbox_inches="tight",
    dpi=300,
)

ax1.grid()

pp.savefig()

plt.clf()
pp.close()

# + ',' + str(df.loc[i, "Efficiency"]) + '$\%$' + ')'