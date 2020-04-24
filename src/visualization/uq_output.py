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
import statistics as st


input = "/home/surya/Programs/PycharmProjects/air_model/data/processed/schwarzsee/simulations/data/"
figures = "/home/surya/Programs/PycharmProjects/air_model/data/processed/schwarzsee/simulations/figures/"

names = ["Fount", "Surface", "Meteo"]
variance = []
mean = []
evaluations = []

# for name in names:
#     data = un.Data()
#     filename1 = input + name + ".h5"
#     filename2 = input + name + "_corrected.h5"
#     data.load(filename1)
#     data.model_name = name
#     data.save(filename2)

for name in names:
    data = un.Data()
    filename1 = input + name + ".h5"
    data.load(filename1)
    variance.append(data["max_volume"].variance)
    mean.append(data["max_volume"].mean)
    evaluations.extend(data["max_volume"].evaluations)

    # plot1 = un.plotting.PlotUncertainty(filename1)
    # plot1.prediction_interval_1d(show = True)

    if len(data.uncertain_parameters) > 1:
        plt.bar(data.uncertain_parameters, data["max_volume"].sobol_first * 100)
        plt.ylabel("Sensitivity of variance(%)")
        plt.savefig(figures + name + "_sobol_first.jpg", bbox_inches="tight", dpi=300)
        plt.clf()

# print(st.mean(evaluations))
# print(st.variance(evaluations))
# print(2*st.stdev(evaluations))

fig, ax = plt.subplots()
ax.bar(names, variance)
ax.set_xlabel("Type")
ax.set_ylabel("Variance")
plt.savefig(figures + "error_type.jpg", bbox_inches="tight", dpi=300)

# data = un.Data()
# filename1 = input + name + ".h5"
# data.load(filename1)
# variance.append(math.sqrt(data["max_volume"].variance))

# plot2 = un.plotting.PlotUncertainty(filename2)
# plot2.mean_variance_1d(show = True)



