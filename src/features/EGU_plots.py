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
# dfd = dfd.fillna(0)

param_values = np.arange(5, 18, 1).tolist()



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

fig = plt.figure()
for j in range(1,35):

    ax1 = fig.add_subplot(111)
    for i in param_values:
        ax1.scatter(i, dfd[ dfd.index == j * 24 ][str(i) + "_iceV"])

    ax1.set_ylabel("Day" + str(j) + " ice ($l/min$)")
    ax1.set_ylim(0,80)
    pp.savefig(bbox_inches="tight")
    plt.clf()



y3 = []
for i in param_values:
    col = str(i) + "_iceV"
    y3.append(dfd[col].iat[-1])


x = df["Discharge"]
y1 = df["Max_IceV"]
y2 = df.Efficiency
y4 = df["4"]

fig, ax = plt.subplots()
# ax.scatter(x, y1, color = "k")
ax.scatter(x='Discharge', y='Max_IceV', s="Efficiency", data=df, color = "k", alpha=0.3)
ax.set_ylabel("Max_IceV")
ax.grid()

ax1t = ax.twinx()
ax1t.scatter(x, y2, color = "b")
ax1t.set_ylabel("Efficiency", color="b")
for tl in ax1t.get_yticklabels():
    tl.set_color("b")
pp.savefig(bbox_inches="tight")
plt.clf()

fig, ax = plt.subplots()
ax.scatter(x, df.Duration, color = "k")
ax.set_ylabel("Duration")
ax.grid()
pp.savefig(bbox_inches="tight")
plt.clf()

fig = plt.figure()

for i in param_values:

    col = str(i) + "_thickness"
    dfds = dfd[[str(i) + "_When", str(i) + "_thickness", str(i) + "_SA"]]
    dfds.loc[:,str(i) + "_When"] = pd.to_datetime(dfds[str(i) + "_When"], format="%Y.%m.%d %H:%M:%S")

    dfds.loc[:,"melted"] = 0
    dfds.loc[:,"solid"] = 0
    for row in dfds.itertuples():
        if dfds.loc[row.Index, col] < 0:
            dfds.loc[row.Index, "melted"] = dfds.loc[row.Index, col]
        else:
            dfds.loc[row.Index, "solid"] = dfds.loc[row.Index, col]

    dfds = dfds.set_index(str(i) + "_When").resample("D").sum().reset_index()
    dfds2 = dfds.set_index(str(i) + "_When").resample("D").mean().reset_index()

    y1 = dfds[["solid",'melted']]
    y3 = dfds2[str(i) + "_SA"]

    ax1 = fig.add_subplot(2, 1, 1)
    y1.plot(kind='bar', stacked=True, edgecolor='black', linewidth=0.5, color=['#D9E9FA', '#0C70DE'], ax=ax1)
    plt.xlabel('Days')
    plt.ylabel('Thickness ($m$)' + str(i))
    plt.xticks(rotation=45)
    plt.legend(loc='upper right')
    ax1.set_ylim(-0.0025, 0.0025)
    ax1.yaxis.set_minor_locator(AutoMinorLocator())
    ax1.grid(axis="y", color="black", alpha=.3, linewidth=.5, which="major")
    x_axis = ax1.axes.get_xaxis()
    x_axis.set_visible(False)

    ax3 = fig.add_subplot(2, 1, 2)
    y3.plot.bar(y='SA', linewidth=0.5, ax=ax3)
    ax3.set_ylim(0, 10000)
    ax3.set_ylabel("Surface Area ($m^2$)")
    ax3.set_xlabel("Days")
    ax1.grid()
    ax3.grid(axis='y')
    plt.xticks(rotation=45)
    # filename = os.path.join(folders['output_folder'], site + '_' + option + "_energySA2.jpg")
    # plt.savefig(filename, bbox_inches  =  "tight", dpi=300)

    pp.savefig(bbox_inches="tight")
    plt.clf()

pp.close()