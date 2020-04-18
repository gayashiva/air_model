import logging
import os
import time
from datetime import datetime
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.colors
import uncertainpy as un

dirname = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

filename1 = os.path.join(dirname, "models/data/Icestupa_ie.h5")
filename2 = os.path.join(dirname, "models/data/Icestupa_full.h5")

data = un.Data()
data.load(filename2)

# print(data["Icestupa"])

# plot1 = un.plotting.PlotUncertainty(filename1)
# plot1.mean_variance_1d(show = True)

# plot2 = un.plotting.PlotUncertainty(filename2)
# plot2.mean_variance_1d(show = True)

# fig, ax = plt.subplots()
# ax.plot(data["Icestupa"].time, data["Icestupa"].mean)
# ax.set_xlabel(data["Icestupa"].labels[0])
# ax.set_ylabel(data["Icestupa"].labels[1])
# plt.show()


