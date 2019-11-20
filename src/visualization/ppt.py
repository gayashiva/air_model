import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.dates as mdates
from matplotlib.backends.backend_pdf import PdfPages
import os
from src.models.air_forecast import icestupa
import time
import logging
from logging import StreamHandler
from src.data.config import site, option, folders, fountain, surface

# python -m src.visualization.ppt

#  read files
filename0 = os.path.join(folders['output_folder'], site +"_model_results.csv")
df = pd.read_csv(filename0, sep=",")
df["When"] = pd.to_datetime(df["When"], format="%Y.%m.%d %H:%M:%S")

# Day melt and Night freeze Plots

for i in range(0, df.shape[0]):
    if df.loc[i,'solid'] < 0:
        df.loc[i,'solid'] = 0


dfd = df.set_index("When").resample("D").mean().reset_index()
dfd['When'] = dfd['When'].dt.strftime("%b %d")
dfd["Discharge"] = dfd["Discharge"] == 0
dfd["Discharge"] = dfd["Discharge"].astype(int)
dfd["Discharge"] = dfd["Discharge"].astype(str)


x = dfd.When
y1 = dfd.iceV

fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.plot(x, y1, "k-", lw=1, color ='#0C70DE')
ax1.set_ylabel("Ice Volume ($m^3$)")
ax1.set_xlabel("Days")
ax1.set_ylim(0,1.2)

plt.xticks(rotation=45)
ax1.yaxis.grid(which="major")
plt.show()
plt.savefig(
    os.path.join(folders['output_folder'], site + "iceV.jpg"), bbox_inches="tight", dpi=300
)

plt.clf()

x = dfd.When
y1 = dfd.iceV

fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.plot(x, y1, "k-", lw=1, color ='#0C70DE')
ax1.set_ylabel("Ice Volume ($m^3$)")
ax1.set_xlabel("Days")
ax1.set_ylim(0,1.2)
ax1.yaxis.grid(which="major")

if site == "schwarzsee":
    # Include Validation line segment 1
    ax1.plot(
        ['Feb 14', 'Feb 14'],
        [0.67115, 1.042],
        color="green",
        lw=1,
    )
    ax1.scatter('Feb 14', 0.856575, color="green", marker="o")

    # Include Validation line segment 2
    ax1.plot(
        ['Mar 10', 'Mar 10'],
        [0.037, 0.222],
        color="green",
        lw=1,
    )
    ax1.scatter('Mar 10', 0.1295, color="green", marker="o")

plt.xticks(rotation=45)

plt.savefig(
    os.path.join(folders['output_folder'], site + "iceV2.jpg"), bbox_inches="tight", dpi=300
)

plt.clf()

dfds = df.set_index("When").resample("D").sum().reset_index()
dfds['melted'] = dfds['melted'] * -1 / 1000
dfds['solid'] = dfds['solid'] / 1000
dfds["When"] = pd.to_datetime(dfds["When"], format="%Y.%m.%d %H:%M:%S")

dfds['When'] = dfds['When'].dt.strftime("%b %d")
dfds = dfds.set_index("When")
dfds = dfds[['solid','melted']]
dfds = dfds.rename({'solid': 'ice', 'melted': 'meltwater'}, axis=1)  # new method

# fig = plt.figure()

dfds.plot( kind = 'bar', stacked=True, edgecolor = 'black', linewidth=0.5, color = ['#D9E9FA', '#0C70DE'])
plt.xlabel('Days')
plt.ylabel('Water ($m^{3}$)')
plt.xticks(rotation=45)
plt.legend(loc = 'upper right')
filename = os.path.join(folders['output_folder'], site + '_' + option + "_daynight.jpg")
plt.savefig(filename, bbox_inches  =  "tight", dpi=300)

# Energy Plots
y= dfd[['SW','LW']]
y.plot.bar(stacked=True, edgecolor = dfd['Discharge'], linewidth=0.5)
plt.xlabel('Days')
plt.ylabel('Energy ($W/m^{2}$)')
plt.ylim(-150, 150)
plt.legend(loc = 'upper right')
plt.savefig(os.path.join(folders['output_folder'], site + "_energybar1.jpg"), bbox_inches  =  "tight", dpi=300)
plt.clf()

y= dfd[['SW','LW','Qs' ]]
y.plot.bar(stacked=True, edgecolor = dfd['Discharge'], linewidth=0.5)
plt.xlabel('Days')
plt.ylabel('Energy ($W/m^{2}$)')
plt.legend(loc = 'upper right')
plt.ylim(-150, 150)
plt.savefig(os.path.join(folders['output_folder'], site + "_energybar2.jpg"), bbox_inches  =  "tight", dpi=300)
plt.clf()

y= dfd[['SW','LW','Qs','Ql' ]]
y.plot.bar(stacked=True, edgecolor = dfd['Discharge'], linewidth=0.5)
plt.xlabel('Days')
plt.ylabel('Energy ($W/m^{2}$)')
plt.legend(loc = 'upper right')
plt.ylim(-150, 150)
plt.savefig(os.path.join(folders['output_folder'], site + "_energybar3.jpg"), bbox_inches  =  "tight", dpi=300)
plt.clf()

# y1 = dfd['iceV']
# y2 = (dfd['SW'] + dfd['LW'] + dfd['Qs'] + dfd['Ql']) * dfd['SA']
#
# fig = plt.figure()
# ax1 = fig.add_subplot(111)
# ax1.plot( y1, "k-", lw=1)
# ax1.set_ylabel("Ice Volume ($m^3$)")
# ax1.set_xlabel("Days")
# ax1.set_ylim(0,1.2)
#
# ax2 = ax1.twinx()
# ax2.plot(y2, "r-", linewidth=0.5)
# ax2.set_ylabel("Energy[$W$]", color="r")
# for tl in ax2.get_yticklabels():
#     tl.set_color("r")
#
# ax1.grid(axis = 'x')
#
# plt.savefig(os.path.join(folders['output_folder'], site + "_energybarppt1.jpg"), bbox_inches  =  "tight", dpi=300)
#
# plt.show()
# plt.clf()
#
#
# y1 = (dfd['SW'] + dfd['LW'] + dfd['Qs'] + dfd['Ql'])
# y2 = dfd['SA']
# fig = plt.figure()
# ax1 = fig.add_subplot(111)
# ax1.plot( y1, "r-")
# plt.ylabel('Energy ($W/m^{2}$)$')
# ax1.set_xlabel("Days")
#
# ax2 = ax1.twinx()
# ax2.plot( y2, "b-", linewidth=0.5)
# ax2.set_ylabel("Surface Area ($m^{-2}$)", color="b")
# for tl in ax2.get_yticklabels():
#     tl.set_color("b")
#
# ax1.grid(axis = 'x')
#
# plt.show()
# plt.savefig(os.path.join(folders['output_folder'], site + "_energybarppt2.jpg"), bbox_inches  =  "tight", dpi=300)
# plt.clf()




# dff = pd.DataFrame([[df['SW'].sum(), df['LW'].sum(), df['Qs'].sum(), df['Ql'].sum()]], columns=['SW', 'LW', 'Qs', 'Ql'])
#
# # Plot the figure.
# ax = dff.plot(kind='bar', stacked=True, legend=None, width = 0.05)
# # ax.set_title( site + ' Icestupa Energy Balance')
# ax.set_ylabel('Energy ($W/m^{2}$)')
# plt.axis('off')
#
# plt.savefig(
#     os.path.join(folders['output_folder'], site + "ppt_ful.jpg"), bbox_inches="tight", dpi=300
# )

# positive_energy = 0
# negative_energy = 0
# for j in dff.columns:
#     x = dff[j].values[0]
#     if x > 0:
#         positive_energy = positive_energy + x
#     else:
#         negative_energy = negative_energy + x
#
# rects = ax.patches
#
#
# # For each bar: Place a label
# for rect in rects:
#     # Get X and Y placement of label from rect.
#     x_value = rect.get_width()/2
#     y_value = rect.get_y() + rect.get_height()
#     y_value_pos = rect.get_y() + rect.get_height()/2
#
#     # Number of points between bar and label. Change to your liking.
#     space = 5
#     # Vertical alignment for positive values
#     ha = 'left'
#     va = 'top'
#
#     # If value of bar is negative: Place label left of bar
#     if x_value < 0:
#         # Invert space to place label to the left
#         space *= -1
#         # Horizontally align label at right
#         va = 'bottom'
#
#     # Use X value as label and format number with one decimal place
#     if y_value > 0:
#         percent = y_value/positive_energy
#     else:
#         percent = y_value / positive_energy
#
#     label = "{:.2%}".format(percent)
#
#     # Create annotation
#     plt.annotate(
#         label,                      # Use `label` as label
#         (x_value, y_value_pos),         # Place label at end of the bar
#         xytext=(space, 0),          # Horizontally shift label by `space`
#         textcoords="offset points", # Interpret `xytext` as offset in points
#         va='center',                # Vertically center label
#         ha='left')                      # Horizontally align label differently for
#                                     # positive and negative values.
#
# f = mticker.ScalarFormatter(useOffset=False, useMathText=True)
# g = lambda x,pos : "${}$".format(f._formatSciNotation('%1.10e' % x))
# plt.gca().yaxis.set_major_formatter(mticker.FuncFormatter(g))
#
# plt.savefig(
#     os.path.join(folders['output_folder'], site + "ppt_full.jpg"), bbox_inches="tight", dpi=300
# )