import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.colors
import uncertainpy as un
import statistics as st
import sys

sys.path.append("/home/surya/Programs/Github/air_model")
from src.data.config import SITE, FOUNTAIN, FOLDERS


def draw_plot(data, edge_color, fill_color, labels):
    bp = ax.boxplot(data, patch_artist=True, labels=labels, sym="o")

    for element in ["boxes", "whiskers", "fliers", "means", "medians", "caps"]:
        plt.setp(bp[element], color=edge_color)

    for patch in bp["boxes"]:
        patch.set(facecolor=fill_color)


input = FOLDERS["sim_folder"] + "/"
output = FOLDERS["sim_folder"] + "/"

names = [
    "full",
    "T_RAIN",
    "IE",
    "A_I",
    "A_S",
    "T_DECAY",
    "dia_f",
    "h_f",
    "h_aws",
    "T_w",
    "DX",
]
variance = []
mean = []
evaluations = []

for name in names:
    data = un.Data()
    filename1 = input + name + ".h5"
    data.load(filename1)
    variance.append(data["max_volume"].variance)
    mean.append(data["max_volume"].mean)
    evaluations.append(data["max_volume"].evaluations)

    eval = data["max_volume"].evaluations

    print(
        f"95 percent confidence interval caused by {name} is {round(st.mean(eval),2)} and {round(2 * st.stdev(eval),2)}"
    )

names = [
    "$T_{ppt}$",
    "$\\epsilon_{ice}$",
    r"$\alpha_{ice}$",
    r"$\alpha_{snow}$",
    "$t_{decay}$",
    "$dia_{F}$",
    "$h_F$",
    "$h_{AWS}$",
    "$T_{water}$",
    "$\\Delta x$",
]

fig, ax = plt.subplots()
draw_plot(evaluations, "k", "xkcd:grey", names)
ax.set_xlabel("Parameter")
ax.set_ylabel("Sensitivity of Maximum Ice Volume [$m^3$]")
ax.grid(axis="y")
plt.savefig(output + "sensitivities.jpg", bbox_inches="tight", dpi=300)
# plt.savefig(
#     "/home/surya/Documents/AIR_Manuscript/Figures/Figure_9.jpg",
#     bbox_inches="tight",
#     dpi=300,
# )
