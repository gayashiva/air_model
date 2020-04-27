import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
import matplotlib.ticker as mticker
import matplotlib.dates as mdates
from matplotlib.backends.backend_pdf import PdfPages

plt.rcParams["figure.figsize"] = (10,7)
matplotlib.rc('xtick', labelsize=5)

filename = "/home/surya/Programs/PycharmProjects/air_model/data/processed/schwarzsee/simulations/results.csv"
df = pd.read_csv(filename, sep=",")

filename2 = "/home/surya/Programs/PycharmProjects/air_model/data/processed/schwarzsee/simulations/discharge.csv"
dfd = pd.read_csv(filename2, sep=",")

param_values = np.arange(3, 15, 1).tolist()


x = dfd.index.values / 24
fig = plt.figure()
cmap = plt.cm.rainbow
norm = matplotlib.colors.Normalize(
    vmin=param_values[0], vmax=param_values[-1]
)


# Plots
figures = "/home/surya/Programs/PycharmProjects/air_model/data/processed/schwarzsee/simulations/figures/"
pp = PdfPages(figures + "discharge_sims.pdf")

ax1 = fig.add_subplot(111)
for i in param_values:
    ax1.plot(x, dfd[str(i) + "_iceV"], lw=1, color=cmap(norm(i)))

ax1.set_ylabel("Ice Volume ($m^3$)")

fig.subplots_adjust(right=0.8)
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
cbar = fig.colorbar(sm, cax=cbar_ax)
cbar.set_label("Fountain Discharge ($l/min$)")
pp.savefig(bbox_inches="tight")
plt.clf()

fig = plt.figure()
ax1 = fig.add_subplot(111)
for i in param_values:
    ax1.plot(x, dfd[str(i) + "_SA"], lw=1, color=cmap(norm(i)))

ax1.set_ylabel("Surface Area ($m^2$)")
fig.subplots_adjust(right=0.8)
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
cbar = fig.colorbar(sm, cax=cbar_ax)
cbar.set_label("Fountain Discharge ($l/min$)")
pp.savefig(bbox_inches="tight")
plt.clf()

fig = plt.figure()
ax1 = fig.add_subplot(111)
for i in param_values:
    ax1.scatter(i, dfd[dfd[str(i) + "_solid"] > 0][str(i) + "_solid"].mean() / 5, color=cmap(norm(i)))

ax1.set_ylabel("Average Freeze Rate ($l/min$)")

fig.subplots_adjust(right=0.8)
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
cbar = fig.colorbar(sm, cax=cbar_ax)
cbar.set_label("Fountain Discharge ($l/min$)")
pp.savefig(bbox_inches="tight")
plt.clf()

# mask = df["Efficiency"]>30
# df = df[mask]
x = df["Discharge"]
y1 = df["Max_IceV"]
y2 = df.Efficiency

fig, ax = plt.subplots()
ax.scatter(x, y1, color = "k")
# ax1.scatter(x='Efficiency', y='Max_IceV', s="Discharge", data=df, alpha=0.3)
ax.set_ylabel("Max_IceV")
ax.grid()

ax1t = ax.twinx()
ax1t.scatter(x, y2, color = "b")
ax1t.set_ylabel("Efficiency", color="b")
for tl in ax1t.get_yticklabels():
    tl.set_color("b")

# fig.subplots_adjust(right=0.8)
# sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
# sm.set_array([])
# cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
# cbar = fig.colorbar(sm, cax=cbar_ax)
# cbar.set_label("Fountain Discharge ($l/min$)")

pp.savefig(bbox_inches="tight")
pp.close()