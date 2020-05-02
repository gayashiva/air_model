import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from matplotlib.backends.backend_pdf import PdfPages

from src.features.discharge_sim_class import param_values, variables
from celluloid import Camera


plt.rcParams["figure.figsize"] = (10,7)
# mpl.rc('xtick', labelsize=5)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)

filename = "/home/surya/Programs/PycharmProjects/air_model/data/processed/schwarzsee/simulations/spray_radius_results.csv"
df = pd.read_csv(filename, sep=",")

# df = df[:param_values[-1]]

filename2 = "/home/surya/Programs/PycharmProjects/air_model/data/processed/schwarzsee/simulations/spray_radius.h5"
data_store = pd.HDFStore(filename2)
dfd = data_store['dfd']
data_store.close()

figures = "/home/surya/Programs/PycharmProjects/air_model/data/processed/schwarzsee/simulations/figures/spray_radius_"

"""Equate column lengths"""
for i in param_values:
    iterables = [[i], variables]
    index = pd.MultiIndex.from_product(iterables, names=['spray_radius', 'variables'])

    days = pd.date_range(start=dfd.loc[0,(param_values[0], "When")], end=dfd.loc[len(dfd[param_values[0]])-1,(param_values[0], "When")], freq="H")

    days = pd.DataFrame({"When": days})

    dfd[i] = pd.merge(days, dfd[i], on="When",how='outer')

dfd = dfd.fillna(0)

cmap = plt.cm.rainbow  # define the colormap
norm = mpl.colors.Normalize(vmin=param_values[0], vmax=param_values[-1])

# cmaplist = [cmap(i) for i in range(cmap.N)]
# cmap = mpl.colors.LinearSegmentedColormap.from_list(
#     'Custom cmap', cmaplist, cmap.N)
# # define the bins and normalize
# bounds = np.linspace(param_values[0], param_values[-1], param_values[-1] - param_values[0] + 1 )
# norm = mpl.colors.BoundaryNorm(bounds, cmap.N)


sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])


# Plots
pp = PdfPages(figures + "showing_why_6.pdf")
x = dfd.index.values / 24

end_ice = []
for i in param_values:
    end_ice.append(dfd[(i, "iceV")].iloc[-1]*916)
    # y1 = dfd[(i, "input")]
    # y2 = dfd[(i, "meltwater")]
    # y3 = dfd[(i, "iceV")]*916
    #
    # fig, ax = plt.subplots()
    # fig.suptitle("Fountain Spray Radius :", fontsize=16)
    # ax.set_title(str(i) + " $m$", fontsize=16)
    # plt.gca().spines['top'].set_visible(False)
    # plt.gca().spines['right'].set_visible(False)
    # p1 = ax.plot(x, y1, color = 'xkcd:royal blue', label= 'Water Sprayed')
    # p2 = ax.plot(x, y2 + y3, color= 'xkcd:blue', label = 'Water Stored' )
    # p3 = ax.plot(x, y3, color= 'k' , label = 'Ice')
    # ax.fill_between(x, y1, y2+y3, alpha = 0.5, color = 'xkcd:royal blue')
    # ax.fill_between(x, y2+y3, y3, alpha = 0.5, color = 'xkcd:lightish blue')
    # ax.fill_between(x, y3, 0, alpha = 0.5, color = 'white')
    # ax.legend()
    # ax.axhline(color = 'k')
    # ax.set_ylim(0, 28000)
    # ax.set_ylabel("Mass ($litres$)")
    # ax.grid(axis="x", color="black", alpha=.3, linewidth=.5, which="major")
    # ax.set_xlabel("Day Number")
    #
    # pp.savefig(bbox_inches="tight")
    # plt.clf()

fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(param_values, end_ice, color = "k")
ax.set_ylabel("Ice Mass left ($kg$)")
ax.set_xlabel("Spray Radius($m$)")
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
ax.grid(axis="x", color="black", alpha=.3, linewidth=.5, which="major")
pp.savefig(bbox_inches="tight")
plt.clf()

x = df["spray_radius"]
y1 = df["Max_IceV"] * 916
y2 = df.Duration
y4 = df["h_r"]

fig, ax = plt.subplots()
ax.plot(param_values, end_ice, color = "k")
ax.set_ylabel("Ice Mass left ($kg$)")
ax.set_xlabel("Spray Radius($m$)")
ax.grid()

ax1t = ax.twinx()
ax1t.plot(df['spray_radius'], df['Efficiency'], color = "b", alpha=0.5)
ax1t.set_ylabel("Storage Efficiency(%)", color="b")
for tl in ax1t.get_yticklabels():
    tl.set_color("b")
pp.savefig(bbox_inches="tight")
plt.clf()



plt.close()
pp.close()

# Plots

pp = PdfPages(figures + "discharge_sims.pdf")
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
fig = plt.figure()
ax1 = fig.add_subplot(111)
x = dfd.index.values / 24


# for i in param_values:
#
#     ax1.plot(x, dfd[(i,"iceV")]*916, color=cmap(norm(i)))
#     ax1.set_ylabel("Ice Mass ($kg$)")
#     ax1.set_xlabel("Day Number")
#     ax1.grid(axis="both", color="black", alpha=.3, linewidth=.5, which="major")
#     ax1.set_ylim(0,14000)
#     fig.subplots_adjust(right=0.8)
#     cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
#     cbar = fig.colorbar(sm, cax=cbar_ax)
#     cbar.set_label("Fountain Spray Radius ($m$)")
#     pp.savefig(bbox_inches="tight")
#
# plt.clf()

ax1 = fig.add_subplot(111)
for i in param_values:
    if i == 2:
        ax1.plot(x, dfd[(i,"iceV")]*916, color='k', label = "Experiment")
        ax1.set_ylabel("Ice Mass ($kg$)")
        ax1.set_xlabel("Day Number")
        ax1.grid(axis="y", color="black", alpha=.3, linewidth=.5, which="major")
        ax1.set_ylim(0,15000)


    if i == 6:
        ax1.plot(x, dfd[(i,"iceV")]*916, color='xkcd:lightish blue', label = "Simulation")
        ax1.set_ylabel("Ice Mass ($kg$)")
        ax1.set_xlabel("Day Number")
        ax1.grid(axis="y", color="black", alpha=.3, linewidth=.5, which="major")
        ax1.set_ylim(0,15000)

ax1.legend()
pp.savefig(bbox_inches="tight")
plt.clf()

ax1 = fig.add_subplot(111)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
for i in param_values:
    ax1.plot(x, dfd[(i,"SA")], lw=1, color=cmap(norm(i)))

ax1.set_ylabel("Surface Area ($m^2$)")
ax1.set_xlabel("Day Number")
ax1.grid(axis="x", color="black", alpha=.3, linewidth=.5, which="major")


fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
cbar = fig.colorbar(sm, cax=cbar_ax)
cbar.set_label("Fountain Spray Radius ($m$)")
pp.savefig(bbox_inches="tight")
plt.clf()

discharge = dfd[(5,'Discharge')].replace(0, np.NaN).mean()

ax1 = fig.add_subplot(111)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
freeze_rate = []
for i in param_values:
    ax1.scatter(i, dfd[dfd[(i,"solid")] > 0][(i,"solid")].mean() / 5, color ='k')
    freeze_rate.append(dfd[dfd[(i,"solid")] > 0][(i,"solid")].mean() / 5)


ax1.set_ylabel("Average Freeze Rate ($l/min$)")
ax1.set_xlabel("Spray Radius ($m$)")
ax1.grid(axis='both', color="black", alpha=.3, linewidth=.5, which="major")
pp.savefig(bbox_inches="tight")
plt.clf()

ax1 = fig.add_subplot(111)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
ax1.plot(param_values, freeze_rate, color ='k', label = "Average Freeze Rate")

ax1.axhline(y=discharge, color='xkcd:royal blue', label = "Average Discharge Rate")
ax1.legend()
ax1.set_ylabel("Average Freeze Rate ($l/min$)")
ax1.set_xlabel("Spray Radius ($m$)")
ax1.grid(axis='both', color="black", alpha=.3, linewidth=.5, which="major")
pp.savefig(bbox_inches="tight")
plt.clf()
pp.close()

# r = np.arange(1, param_values[-1] + 1, 1).tolist()
# totals = [i + j + k for i, j, k in zip(df['water_stored'], df['water_lost'], df['unfrozen_water'])]
# water_stored = [i / j * 100 for i, j in zip(df['water_stored'], totals)]
# water_lost = [i / j * 100 for i, j in zip(df['water_lost'], totals)]
# unfrozen_water = [i / j * 100 for i, j in zip(df['unfrozen_water'], totals)]

# fig,ax = plt.subplots()
# barWidth = 0.85
# ax.bar(r, water_stored, color='xkcd:lightish blue', edgecolor='white', width=barWidth, label='% Water Stored')
# ax.bar(r, water_lost, bottom=water_stored, color='#f9bc86', edgecolor='white', width=barWidth, label='% Water Vapour Lost')
# ax.bar(r, unfrozen_water, bottom=[i + j for i, j in zip(water_stored, water_lost)], color='xkcd:royal blue', edgecolor='white',
#         width=barWidth, label ='% Unused Fountain Water')
#
# ax.set_xlabel("Spray Radius ($m$)")
# ax.set_ylabel("Percentage")
# ax.grid(axis="y", color="black", alpha=.3, linewidth=.5, which="major")
# ax.legend(loc = 'upper left')
# plt.gca().spines['top'].set_visible(False)
# plt.gca().spines['right'].set_visible(False)
# pp.savefig(bbox_inches="tight")
# plt.clf()




# pp = PdfPages(figures + "thicknessandSA.pdf")
# fig = plt.figure()
# camera = Camera(fig)
# for i in param_values:
#     dfds = dfd[[(i,"When"), (i,"thickness"), (i,"SA")]]
#     dfds.loc[:,(i,"When")] = pd.to_datetime(dfds[(i,"When")], format="%Y.%m.%d %H:%M:%S")
#
#     dfds.loc[:,(i,"melted")] = 0
#     dfds.loc[:,(i,"solid")] = 0
#     for row in dfds.itertuples():
#         if dfds.loc[row.Index, (i,"thickness")] < 0:
#             dfds.loc[row.Index, (i,"negative")] = dfds.loc[row.Index, (i,"thickness")]
#         else:
#             dfds.loc[row.Index, (i,"positive")] = dfds.loc[row.Index, (i,"thickness")]
#
#     dfds1 = dfds.set_index((i,"When")).resample("D").sum().reset_index()
#     dfds2 = dfds.set_index((i,"When")).resample("D").mean().reset_index()
#
#     dfds1 = dfds1[i]
#     dfds2 = dfds2[i]
#
#     dfds1 = dfds1.rename(columns={"positive": "Ice thickness", 'negative': 'Meltwater thickness'})
#     y1 = dfds1[["Ice thickness",'Meltwater thickness']] * 1000
#     y3 = dfds2["SA"]
#
#     ax1 = fig.add_subplot(2, 1, 1)
#     # fig.suptitle("Fountain Spray Radius :", fontsize=16)
#     # ax1.set_title(str(i) + " $m$", fontsize=16)
#     y1.plot(kind='bar', stacked=True, edgecolor='black', linewidth=0.5, color=['#D9E9FA', '#0C70DE'], ax=ax1)
#     plt.xlabel('Days')
#     plt.ylabel('Thickness ($mm$)')
#     plt.xticks(rotation=45)
#     # plt.legend(loc='upper right')
#     ax1.set_ylim(-2.5, 2.5)
#     ax1.yaxis.set_minor_locator(AutoMinorLocator())
#     ax1.grid(axis="y", color="black", alpha=.3, linewidth=.5, which="major")
#     x_axis = ax1.axes.get_xaxis()
#     x_axis.set_visible(False)
#
#     ax3 = fig.add_subplot(2, 1, 2)
#     y3.plot.bar(y='SA', linewidth=0.5, ax=ax3)
#     # ax3.set_ylim(0, 10000)
#     ax3.set_ylabel("Surface Area ($m^2$)")
#     ax3.set_xlabel("Days")
#     ax1.grid()
#     ax3.grid(axis='y')
#     plt.xticks(rotation=45)
#     pp.savefig(bbox_inches="tight")
#     plt.clf()
#
# pp.close()