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

param_values = np.arange(8, 12.5, 0.5).tolist()


dfx = pd.DataFrame({'Discharge': [], 'Runtime': [], 'MaxV': []})
for i, X in enumerate(param_values):

    #  read files
    filename0 = os.path.join(folders['input_folder'], site + "_" + option + "_input.csv")
    df_in = pd.read_csv(filename0, sep=",")
    df_in["When"] = pd.to_datetime(df_in["When"], format="%Y.%m.%d %H:%M:%S")

    print(X)
    fountain['discharge'] = X
    df = pd.DataFrame({'A': []})
    df = icestupa(df_in, fountain, surface)
    print("Model runtime", df["When"].iloc[-1] - df["When"].iloc[0])

    # dfd = df.set_index("When").resample("D").mean().reset_index()
    # filename1 = os.path.join(
    #     folders['output_folder'], site + "_simulations_" + str(X) + '_' + str(param_values) + ".csv"
    # )
    # dfd.to_csv(filename1, sep =',')

    dfx = dfx.append({'Discharge': X, 'Max Growthrate' : df["solid"].max(), 'Max SA' : df["SA"].max(), 'MaxV': df["iceV"].max(), 'h/r': df["h_r"].iloc[-1], 'r': df["r_ice"].max(), 'Runtime': df["When"].iloc[-1] - df["When"].iloc[0],  'SA/V': (df['SA']/df['iceV']).min()}, ignore_index=True)
    print(dfx)
    # 'Endice': float(df.loc[df["When"] == "2019-03-05", "iceV"]),


filename2 = os.path.join(
    folders['output_folder'], site + "_simulations_discharge.csv"
)
print(dfx)
dfx.to_csv(filename2, sep=',')
#


filename2 = os.path.join(
    folders['output_folder'], site + "_simulations_discharge.csv"
)
dfx = pd.read_csv(filename2, sep=",")



# Plots
filename3 = os.path.join(folders["output_folder"], site + "_shapeanalysis.pdf")
pp = PdfPages(filename3)
x = dfx.Discharge
y1 = dfx['h/r']
y12 = dfx['Max SA']
y2 = dfx['SA/V']
y3 = dfx.MaxV
y4 = dfx['Max Growthrate']/ 5


fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.plot(x, y1, 'ro', markersize=3)
ax1.set_ylabel("Max h/r")
ax1.set_xlabel("Discharge ($l/min$)")
ax1.grid()
pp.savefig(bbox_inches="tight")
plt.clf()

fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.plot(x, y12, 'ro', markersize=3)
ax1.set_ylabel("SA")
ax1.set_xlabel("Discharge ($l/min$)")
ax1.grid()
pp.savefig(bbox_inches="tight")
plt.clf()

fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.plot(x, y2, 'ro', markersize=3)
ax1.set_ylabel("Min SA/V")
ax1.set_xlabel("Discharge ($l/min$)")
ax1.grid()
pp.savefig(bbox_inches="tight")
plt.clf()

fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.plot(x, y3, 'ro', markersize=3)
ax1.set_ylabel("Max Ice Volume ($m^3$)")
ax1.set_xlabel("Discharge ($l/min$)")
ax1.grid()
pp.savefig(bbox_inches="tight")
plt.clf()

fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.plot(x, y4, 'ro', markersize=3)
ax1.set_ylabel("Max Growthrate ($l/min$)")
ax1.set_xlabel("Discharge ($l/min$)")
ax1.grid()
pp.savefig(bbox_inches="tight")
plt.clf()

pp.close()