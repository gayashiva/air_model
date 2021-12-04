import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from matplotlib.backends.backend_pdf import PdfPages

import sys
import os

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
)
from src.data.config import SITE, FOUNTAIN, FOLDERS

plt.rcParams["figure.figsize"] = (10, 7)
# mpl.rc('xtick', labelsize=5)
plt.gca().spines["top"].set_visible(False)
plt.gca().spines["right"].set_visible(False)

filename = FOLDERS["sim_folder"] + "/dis_sim.csv"
df = pd.read_csv(filename, sep=",")

filename2 = FOLDERS["sim_folder"] + "/dis_sim.h5"
data_store = pd.HDFStore(filename2)
dfd = data_store["dfd"]
data_store.close()

figures = FOLDERS["sim_folder"] + "/dis_sim.pdf"

dfd = dfd.fillna(0)

print(df["Duration"].tail())

print(df[df["dia_f"] < 0.0051]["water_stored"])
print(df[df["dia_f"] < 0.0051]["avg_freeze_rate"])

print(df[df["dia_f"] < 0.0051]["Duration"])
print(df[df["dia_f"] < 0.0051]["Efficiency"])

cmap = plt.cm.rainbow  # define the colormap
norm = mpl.colors.Normalize(vmin=0, vmax=3.6)
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])

pp = PdfPages(figures)
x = df["dia_f"] * 1000
y = df["Efficiency"]
y1 = df["avg_freeze_rate"]
y2 = df.Duration

fig, ax = plt.subplots()
ax.scatter(x, y, color="k")
ax.set_ylabel("Storage Efficiency [%]")
ax.set_xlabel("Nozzle Diameter [$mm$]")
ax.set_xlim([1, 15])
ax.set_ylim([0, 100])
# Add labels to the plot
style = dict(size=10, color="gray", rotation=90)
ax.text(5.01, 9, "Schwarzsee Experiment", **style)


ax1t = ax.twinx()
for i in range(0, df.shape[0]):
    ax1t.scatter(x[i], y2[i], color=cmap(norm(y1[i])))
# ax1t.scatter(x, y2, color = "b", alpha=0.5)
ax1t.set_ylabel("Survival Duration [Days]", color="b")
ax1t.set_ylim([0, 100])
for tl in ax1t.get_yticklabels():
    tl.set_color("b")
ax.grid()

fig.subplots_adjust(right=0.78)
cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
cbar = fig.colorbar(sm, cax=cbar_ax)
cbar.set_label("Average Freeze Rate [$l\\, min^{-1}$]")

pp.savefig(bbox_inches="tight")
plt.savefig(
    FOLDERS["output_folder"] + "jpg/Figure_10.jpg", dpi=300, bbox_inches="tight"
)
plt.clf()

# fig, ax = plt.subplots()
# for i in range(0,df.shape[0]):
#     ax.scatter(x[i], y1[i], color=cmap(norm(y1[i])))
# ax.set_ylabel("Storage Efficiency(%)")
# ax.set_xlabel("Diameter($m$)")
# ax.grid()
#
# fig.subplots_adjust(right=0.8)
# cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
# cbar = fig.colorbar(sm, cax=cbar_ax)
# cbar.set_label("Fountain Spray Radius ($m$)")
#
# pp.savefig(bbox_inches="tight")
# plt.clf()

plt.close()
pp.close()
