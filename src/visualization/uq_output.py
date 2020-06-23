import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.colors
import uncertainpy as un
import statistics as st

def draw_plot(data, edge_color, fill_color, labels):
    bp = ax.boxplot(data, patch_artist=True, labels = labels)

    for element in ['boxes', 'whiskers', 'fliers', 'means', 'medians', 'caps']:
        plt.setp(bp[element], color=edge_color)

    for patch in bp['boxes']:
        patch.set(facecolor=fill_color)

input = "/home/surya/Programs/PycharmProjects/air_model/data/processed/schwarzsee/simulations/data/"
figures = "/home/surya/Programs/PycharmProjects/air_model/data/processed/schwarzsee/simulations/figures/"

names = ["T_rain", "k_i", "d_ppt", "ie", "a_i", "a_s", "t_decay",  "dia_f",  "h_f", "h_aws", "T_w", "dx"]
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
    evaluations.append(data["max_volume"].evaluations)

    eval = data["max_volume"].evaluations

    print(f"95 percent confidence interval caused by {name} is {round(2 * st.stdev(eval),2)}")

    # plot1 = un.plotting.PlotUncertainty(filename1)
    # plot1.prediction_interval_1d(show = True)

    # if len(data.uncertain_parameters) > 1:
    #     plt.bar(data.uncertain_parameters, data["max_volume"].sobol_first * 100)
    #     plt.ylabel("Sensitivity of variance(%)")
    #     plt.savefig(figures + name + "_sobol_first.jpg", bbox_inches="tight", dpi=300)
    #     plt.clf()


names = ["$T_{rain}$", "$k_{ice}$", r'$\rho_{ppt}$', "$\\epsilon_{ice}$", r'$\alpha_{ice}$', r'$\alpha_{snow}$', "$t_{decay}$",  "$dia_{F}$",  "$h_F$", "$h_{AWS}$", "$T_{water}$",  "$\\Delta x$"]

fig, ax = plt.subplots()
draw_plot(evaluations, 'k', 'xkcd:grey', names)
ax.set_xlabel("Parameter")
ax.set_ylabel("Sensitivity of Maximum Ice Volume ($m^3$)")
ax.grid(axis = "y")
plt.savefig(figures + "sensitivities.jpg", bbox_inches="tight", dpi=300)



# data = un.Data()
# filename1 = input + name + ".h5"
# data.load(filename1)
# variance.append(math.sqrt(data["max_volume"].variance))

# plot2 = un.plotting.PlotUncertainty(filename2)
# plot2.mean_variance_1d(show = True)



