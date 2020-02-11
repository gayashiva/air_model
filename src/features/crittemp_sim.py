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

param_values = np.arange(-9, 5, 0.5).tolist()

dfx = pd.DataFrame({'MaxV': []})
for i, X in enumerate(param_values):

    print(X)
    fountain['crit_temp'] = X

    #  read files
    filename0 = os.path.join(folders["input_folder"] + site + "_input.csv")
    df_in = pd.read_csv(filename0, sep=",")
    df_in["When"] = pd.to_datetime(df_in["When"], format="%Y.%m.%d %H:%M:%S")

    # Remove Precipitation
    df_in["Prec"] = 0

    df = pd.DataFrame({'A': []})
    df = icestupa(df_in, fountain, surface)
    print("Model runtime", df["When"].iloc[-1] - df["When"].iloc[0])

    dfx = dfx.append({'Critical Temp': X, 'Max Growthrate' : df[df["solid"]>0]["solid"].max()/5, 'Avg Freeze Rate' : df[df["solid"]>0]["solid"].mean()/5, 'Water used' : df["sprayed"].iloc[-1], 'Endice' : df["iceV"].iloc[-1], 'Max SA' : df["SA"].max(), 'MaxV': df["iceV"].max(), 'h/r': df["h_r"].iloc[-1], 'r': df["r_ice"].max(), 'Meltwater': df["meltwater"].iloc[-1], 'Runtime': df["When"].iloc[-1] - df["When"].iloc[0]}, ignore_index=True)
    print(dfx)


filename2 = os.path.join(
    folders['sim_folder'], site + "_simulations_crittemp.csv"
)

dfx.to_csv(filename2, sep=',')