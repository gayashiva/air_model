import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
import matplotlib.ticker as mticker
import matplotlib.dates as mdates
from matplotlib.backends.backend_pdf import PdfPages

from src.features.discharge_sim_class import param_values, variables

plt.rcParams["figure.figsize"] = (10,7)
mpl.rc('xtick', labelsize=5)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)

filename = "/home/surya/Programs/PycharmProjects/air_model/data/processed/schwarzsee/simulations/results.csv"
df = pd.read_csv(filename, sep=",")

filename2 = "/home/surya/Programs/PycharmProjects/air_model/data/processed/schwarzsee/simulations/discharge.h5"
data_store = pd.HDFStore(filename2)
dfd = data_store['dfd']
data_store.close()

figures = "/home/surya/Programs/PycharmProjects/air_model/data/processed/schwarzsee/simulations/figures/"

"""Equate column lengths"""
for i in param_values:
    iterables = [[i], variables]
    index = pd.MultiIndex.from_product(iterables, names=['discharge_rate', 'variables'])

    days = pd.date_range(start=dfd.loc[0,(param_values[0], "When")], end=dfd.loc[len(dfd[param_values[0]])-1,(param_values[0], "When")], freq="H")

    days = pd.DataFrame({"When": days})

    dfd[i] = pd.merge(days, dfd[i], on="When",how='outer')

dfd = dfd.fillna(0)

cmap = plt.cm.rainbow  # define the colormap
cmaplist = [cmap(i) for i in range(cmap.N)]
cmap = mpl.colors.LinearSegmentedColormap.from_list(
    'Custom cmap', cmaplist, cmap.N)
# define the bins and normalize
bounds = np.linspace(param_values[0], param_values[-1], 8)
norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])

# Plots
pp = PdfPages(figures + "showing_why_11.pdf")
x = dfd.index.values / 24
for i in param_values:
    dfd.loc[0,(i, "solid")] = 0
    # y1 = dfd[(i, "Discharge")].cumsum(axis = 0) * 60
    y1 = dfd[(i, "input")].cumsum(axis = 0) * 12
    y2 = dfd[(i, "iceV")]*916
    # dfd["solid"] = dfd[(i, "solid")].abs()
    # y3 = y2 + dfd["solid"].cumsum(axis = 0)
    fig, ax = plt.subplots()
    ax.plot(x, y1, color = 'xkcd:blue')
    ax.plot(x, y2, color= 'k' )
    # ax.plot(x, y3, color= None )
    # ax.plot(x, y3, color= cmap(norm(i)) )
    ax.fill_between(x, y1, y2, alpha = 0.5, color = 'xkcd:lightish blue')
    # ax.fill_between(x, y2, y3, alpha = 0.5, color = 'xkcd:lightish blue')
    # ax.fill_between(x, y2, 0, alpha = 0.5, color='white')
    ax.axhline(color = 'k')
    ax.set_ylim(0)
    ax.set_ylabel("Mass ($litres$)")
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label("Fountain Discharge" +str(i)+ "($l/min$)")
    pp.savefig(bbox_inches="tight")
    plt.clf()

x = df["Discharge"]
y1 = df["Max_IceV"]
y2 = df.Efficiency
y4 = df["h_r"]

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

pp.close()

# Plots

pp = PdfPages(figures + "discharge_sims.pdf")

fig = plt.figure()
ax1 = fig.add_subplot(111)
x = dfd.index.values / 24
for i in param_values:

    ax1.plot(x, dfd[(i,"iceV")], lw=1, color=cmap(norm(i)))

ax1.set_ylabel("Ice Volume ($m^3$)")

fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
cbar = fig.colorbar(sm, cax=cbar_ax)
cbar.set_label("Fountain Discharge ($l/min$)")
pp.savefig(bbox_inches="tight")
plt.clf()

fig = plt.figure()
ax1 = fig.add_subplot(111)
for i in param_values:
    ax1.plot(x, dfd[(i,"SA")], lw=1, color=cmap(norm(i)))

ax1.set_ylabel("Surface Area ($m^2$)")
fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
cbar = fig.colorbar(sm, cax=cbar_ax)
cbar.set_label("Fountain Discharge ($l/min$)")
pp.savefig(bbox_inches="tight")
plt.clf()


fig = plt.figure()
ax1 = fig.add_subplot(111)
for i in param_values:
    ax1.scatter(i, dfd[dfd[(i,"solid")] > 0][(i,"solid")].mean() / 5, color=cmap(norm(i)))

ax1.set_ylabel("Average Freeze Rate ($l/min$)")

fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
cbar = fig.colorbar(sm, cax=cbar_ax)
cbar.set_label("Fountain Discharge ($l/min$)")
pp.savefig(bbox_inches="tight")
plt.clf()
pp.close()

pp = PdfPages(figures + "Dot_plot.pdf")

fig = plt.figure()
for j in range(1,35):

    ax1 = fig.add_subplot(111)
    for i in param_values:
        ax1.scatter(i, dfd[ dfd.index == j * 24 ][(i,"iceV")])

    ax1.set_ylabel("Day" + str(j) + " ice ($l/min$)")
    ax1.set_ylim(0,80)
    pp.savefig(bbox_inches="tight")
    plt.clf()
pp.close()




pp = PdfPages(figures + "thicknessandSA.pdf")

fig = plt.figure()

for i in param_values:

    dfds = dfd[[(i,"When"), (i,"thickness"), (i,"SA")]]
    dfds.loc[:,(i,"When")] = pd.to_datetime(dfds[(i,"When")], format="%Y.%m.%d %H:%M:%S")

    dfds.loc[:,(i,"melted")] = 0
    dfds.loc[:,(i,"solid")] = 0
    for row in dfds.itertuples():
        if dfds.loc[row.Index, (i,"thickness")] < 0:
            dfds.loc[row.Index, (i,"negative")] = dfds.loc[row.Index, (i,"thickness")]
        else:
            dfds.loc[row.Index, (i,"positive")] = dfds.loc[row.Index, (i,"thickness")]

    dfds = dfds.set_index((i,"When")).resample("D").sum().reset_index()
    dfds2 = dfds.set_index((i,"When")).resample("D").mean().reset_index()

    dfds = dfds[i]
    dfds2 = dfds2[i]

    y1 = dfds[["positive",'negative']]
    y3 = dfds2["SA"]

    ax1 = fig.add_subplot(2, 1, 1)
    y1.plot(kind='bar', stacked=True, edgecolor='black', linewidth=0.5, color=['#D9E9FA', '#0C70DE'], ax=ax1)
    plt.xlabel('Days')
    plt.ylabel('Thickness ($m$)' + str(i))
    plt.xticks(rotation=45)
    # plt.legend(loc='upper right')
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