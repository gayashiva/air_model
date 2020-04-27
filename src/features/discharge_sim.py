import logging
import os
import math
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
from src.data.make_dataset import projectile_xy, discharge_rate

from SALib.sample import saltelli
from SALib.analyze import sobol
import matplotlib.colors

from src.models.air import Icestupa

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


param_values = np.arange(9, 15, 0.5).tolist()

dfx = pd.DataFrame({'MaxV': []})

for i, X in enumerate(param_values):

    print(X)
    fountain['discharge'] = X

    #  read files
    filename0 = os.path.join(folders["input_folder"] + site + "_raw_input.csv")
    df_in = pd.read_csv(filename0, sep=",")
    df_in["When"] = pd.to_datetime(df_in["When"], format="%Y.%m.%d %H:%M:%S")

    """ Derived Parameters"""

    l = [
        "a",
        "r_f",
        "Fountain",
    ]
    for col in l:
        df_in[col] = 0

    """Discharge Rate"""
    df_in["Fountain"], df_in["Discharge"] = discharge_rate(df_in, fountain)
    df_in["Discharge"] = fountain["discharge"] * df_in["Fountain"]

    """Albedo Decay"""
    surface["decay_t"] = (
            surface["decay_t"] * 24 * 60 / 5
    )  # convert to 5 minute time steps
    s = 0
    f = 0

    """ Fountain Spray radius """
    Area = math.pi * math.pow(fountain["aperture_f"], 2) / 4

    for i in range(1, df_in.shape[0]):

        if option == "schwarzsee":

            ti = surface["decay_t"]
            a_min = surface["a_i"]

            # Precipitation
            if (df_in.loc[i, "Fountain"] == 0) & (df_in.loc[i, "Prec"] > 0):
                if df_in.loc[i, "T_a"] < surface["rain_temp"]:  # Snow
                    s = 0
                    f = 0

            if df_in.loc[i, "Fountain"] > 0:
                f = 1
                s = 0

            if f == 0:  # last snowed
                df_in.loc[i, "a"] = a_min + (surface["a_s"] - a_min) * math.exp(-s / ti)
                s = s + 1
            else:  # last sprayed
                df_in.loc[i, "a"] = a_min
                s = s + 1
        else:
            df_in.loc[i, "a"] = surface["a_i"]

        """ Fountain Spray radius """
        v_f = df_in.loc[i, "Discharge"] / (60 * 1000 * Area)
        df_in.loc[i, "r_f"] = projectile_xy(
            v_f, fountain["h_f"]
        )


    df = icestupa(df_in, fountain, surface)
    print("Model runtime", df["When"].iloc[-1] - df["When"].iloc[0])

    dfx = dfx.append({'Discharge': X, 'Max Growthrate' : df[df["solid"] > 0]["solid"].max() / 5, 'Avg Freeze Rate' : df[df["solid"] > 0]["solid"].max() / 5, 'Water used' : df["sprayed"].iloc[-1], 'Endice' : df["iceV"].iloc[-1], 'Max SA' : df["SA"].max(), 'MaxV': df["iceV"].max(), 'h/r': df["h_r"].iloc[-1], 'r': df["r_ice"].mean(), 'Meltwater': df["meltwater"].iloc[-1], 'Runtime': df["When"].iloc[-1] - df["When"].iloc[0]}, ignore_index=True)
    print(dfx)


filename2 = os.path.join(
    folders['sim_folder'], site + "_simulations_discharge.csv"
)

dfx.to_csv(filename2, sep=',')
