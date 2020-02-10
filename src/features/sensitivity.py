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
import uncertainpy as un
import chaospy as cp

if __name__ == '__main__':
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

    #  read files
    filename0 = os.path.join(folders['input_folder'], site + "_input.csv")
    df_in = pd.read_csv(filename0, sep=",")
    df_in["When"] = pd.to_datetime(df_in["When"], format="%Y.%m.%d %H:%M:%S")

    model = un.Model(
        run = icestupa,
        labels = ["Ice Emissivity"]
    )

    # Create distribution
    ie_dist = cp.Uniform(0.81,0.99)

    parameters = {"ie": ie_dist}

    # Uncertainty Quantification
    UQ = un.UncertaintyQuantification(
        model = model,
        parameters = parameters
    )

    data = UQ.quantify(seed = 10)

# problem = {"num_vars": 4, "names": ["ie", "a_i", "a_s", "decay_t"], "bounds": [[0.81, 0.99], [0.36, 0.44], [0.77, 0.93], [9, 11]]}
#
# # Generate samples
# param_values = saltelli.sample(problem, 3)
#
# # Plots
# fig = plt.figure()
# cmap = plt.cm.rainbow
# norm = matplotlib.colors.Normalize(
#     vmin=problem["bounds"][0][0], vmax=problem["bounds"][0][1]
# )
#
# # Output file Initialise
# columns = ["Ice", "IceV"]
# index = range(0, len(param_values))
# dfo = pd.DataFrame(index=index, columns=columns)
# dfo = dfo.fillna(0)
#
# for i, X in enumerate(param_values):
#
#     #  read files
#     filename0 = os.path.join(folders['input_folder'], site + "_input.csv")
#     df_in = pd.read_csv(filename0, sep=",")
#     df_in["When"] = pd.to_datetime(df_in["When"], format="%Y.%m.%d %H:%M:%S")
#
#     print(X)
#     surface['ie'] = X[0]
#     surface['a_i'] = X[1]
#     surface['a_s'] = X[2]
#     surface['decay_t'] = X[3]
#     df = icestupa(df_in, fountain, surface)
#     dfo.loc[i, "ie"] = X[0]
#     dfo.loc[i, "a_i"] = X[1]
#     dfo.loc[i, "a_s"] = X[2]
#     dfo.loc[i, "decay_t"] = X[3]
#     dfo.loc[i, "Ice"] = float(df["ice"].tail(1))
#     dfo.loc[i, "Meltwater"] = float(df["meltwater"].tail(1))
#     dfo.loc[i, "Vapour"] = float(df["vapour"].tail(1))
#     dfo.loc[i, "Ice Max"] = df["ice"].max()
#     dfo.loc[i, "Runtime"] = df["When"].iloc[-1]
#
# dfo = dfo.round(4)
# filename2 = os.path.join(
#     folders['sim_folder'], site + "_simulations__" + str(problem["names"]) + ".csv"
# )
# dfo.to_csv(filename2, sep=",")
