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
from SALib.sample import saltelli
from SALib.analyze import sobol
import matplotlib.colors

problem = {"num_vars": 4, "names": ["ie", "a_i", "a_s", "decay_t"], "bounds": [[0.9025, 0.9975], [0.3325, 0.36175], [0.8075, 0.8925], [9.5, 10.5]]}


filename2 = os.path.join(
    folders['sim_folder'], site + "_simulations_" + str(problem["names"]) + ".csv"
)

df = pd.read_csv(filename2, sep=",")

Y = df["Max IceV"].values
Z = df["Efficiency"].values
Si = sobol.analyze(problem, Y, print_to_console=True)

print(problem["names"])
print(Si['S1_conf'])

plt.bar(problem["names"], Si['S1'])
plt.show()

plt.bar(problem["names"], Si['ST'])
plt.show()

Si = sobol.analyze(problem, Z, print_to_console=True)

plt.bar(problem["names"], Si['S1'])
plt.show()

plt.bar(problem["names"], Si['ST'])
plt.show()

filename = os.path.join(
    folders['sim_folder'], site + 'salib' + ".csv"
)