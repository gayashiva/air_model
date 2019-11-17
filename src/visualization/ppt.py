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

# #  read files
# filename0 = os.path.join(folders['output_folder'], site +"_model_gif.csv")
# df = pd.read_csv(filename0, sep=",")
# df["When"] = pd.to_datetime(df["When"], format="%Y.%m.%d %H:%M:%S")
#
# #  read files
# filename1 = os.path.join(folders['output_folder'], "guttannen_model_gif.csv")
# df1 = pd.read_csv(filename1, sep=",")
# df1["When"] = pd.to_datetime(df1["When"], format="%Y.%m.%d %H:%M:%S")
#
# start_date = datetime(2018, 12, 1)
# end_date = datetime(2019, 7, 1)
# mask = (df["When"] >= start_date) & (df["When"] <= end_date)
# df = df.loc[mask]
#
# start_date = datetime(2017, 12, 1)
# end_date = datetime(2018, 7, 1)
# mask = (df1["When"] >= start_date) & (df1["When"] <= end_date)
# df1 = df1.loc[mask]

#  read files
filename0 = os.path.join(folders['output_folder'], site +"_model_energy.csv")
df = pd.read_csv(filename0, sep=",")
df["When"] = pd.to_datetime(df["When"], format="%Y.%m.%d %H:%M:%S")

#  read files
filename0 = os.path.join(folders['input_folder'], site + "_" + option + "_input.csv")
df_in = pd.read_csv(filename0, sep=",")
df_in["When"] = pd.to_datetime(df_in["When"], format="%Y.%m.%d %H:%M:%S")

# end
end_date = df_in["When"].iloc[-1]

df3 = df_in.set_index("When").resample("D").mean().reset_index()
df3["Discharge"] = df3["Discharge"] == 0
df3["Discharge"] = df3["Discharge"].astype(int)
df3["Discharge"] = df3["Discharge"].astype(str)

dff = pd.DataFrame([[df['SW'].sum(), df['LW'].sum(), df['Qs'].sum(), df['Ql'].sum()]], columns=['SW', 'LW', 'Qs', 'Ql'])

# Plot the figure.
ax = dff.plot(kind='bar', stacked=True, legend=None, width = 0.05)
# ax.set_title( site + ' Icestupa Energy Balance')
ax.set_ylabel('Energy ($W/m^{2}$)')
plt.axis('off')

plt.savefig(
    os.path.join(folders['output_folder'], site + "ppt_ful.jpg"), bbox_inches="tight", dpi=300
)

positive_energy = 0
negative_energy = 0
for j in dff.columns:
    x = dff[j].values[0]
    if x > 0:
        positive_energy = positive_energy + x
    else:
        negative_energy = negative_energy + x

rects = ax.patches


# For each bar: Place a label
for rect in rects:
    # Get X and Y placement of label from rect.
    x_value = rect.get_width()/2
    y_value = rect.get_y() + rect.get_height()
    y_value_pos = rect.get_y() + rect.get_height()/2

    # Number of points between bar and label. Change to your liking.
    space = 5
    # Vertical alignment for positive values
    ha = 'left'
    va = 'top'

    # If value of bar is negative: Place label left of bar
    if x_value < 0:
        # Invert space to place label to the left
        space *= -1
        # Horizontally align label at right
        va = 'bottom'

    # Use X value as label and format number with one decimal place
    if y_value > 0:
        percent = y_value/positive_energy
    else:
        percent = y_value / positive_energy

    label = "{:.2%}".format(percent)

    # Create annotation
    plt.annotate(
        label,                      # Use `label` as label
        (x_value, y_value_pos),         # Place label at end of the bar
        xytext=(space, 0),          # Horizontally shift label by `space`
        textcoords="offset points", # Interpret `xytext` as offset in points
        va='center',                # Vertically center label
        ha='left')                      # Horizontally align label differently for
                                    # positive and negative values.

f = mticker.ScalarFormatter(useOffset=False, useMathText=True)
g = lambda x,pos : "${}$".format(f._formatSciNotation('%1.10e' % x))
plt.gca().yaxis.set_major_formatter(mticker.FuncFormatter(g))

plt.savefig(
    os.path.join(folders['output_folder'], site + "ppt_full.jpg"), bbox_inches="tight", dpi=300
)

''' PPT FIG'''

x = df.When
y1 = df.iceV

fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.plot(x, y1, "k-", lw=1)
ax1.set_ylabel("Ice Volume ($m^3$)")
ax1.set_xlabel("Days")
ax1.set_ylim(0,1.2)

#  format the ticks
ax1.xaxis.set_major_locator(mdates.WeekdayLocator())
ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
ax1.xaxis.set_minor_locator(mdates.DayLocator())
ax1.grid(axis = 'x')
fig.autofmt_xdate()

plt.savefig(
    os.path.join(folders['output_folder'], site + "ppt0.jpg"), bbox_inches="tight", dpi=300
)

plt.clf()

x = df.When
y1 = df.iceV

fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.plot(x, y1, "k-", lw=1)
ax1.set_ylabel("Ice Volume ($m^3$)")
ax1.set_xlabel("Days")
ax1.set_ylim(0,1.2)

if site == "schwarzsee":
    # Include Validation line segment 1
    ax1.plot(
        [datetime(2019, 2, 14, 16), datetime(2019, 2, 14, 16)],
        [0.67115, 1.042],
        color="green",
        lw=1,
    )
    ax1.scatter(datetime(2019, 2, 14, 16), 0.856575, color="green", marker="o")

    # Include Validation line segment 2
    ax1.plot(
        [datetime(2019, 3, 10, 18), datetime(2019, 3, 10, 18)],
        [0.037, 0.222],
        color="green",
        lw=1,
    )
    ax1.scatter(datetime(2019, 3, 10, 18), 0.1295, color="green", marker="o")

#  format the ticks
ax1.xaxis.set_major_locator(mdates.WeekdayLocator())
ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
ax1.xaxis.set_minor_locator(mdates.DayLocator())
ax1.grid(axis = 'x')
fig.autofmt_xdate()

plt.savefig(
    os.path.join(folders['output_folder'], site + "ppt1.jpg"), bbox_inches="tight", dpi=300
)

plt.clf()



x= df.set_index('When').resample('D').mean().reset_index()
x.index = np.arange(1, len(x) + 1)

y1 = x['iceV']
y2 = (x['SW'] + x['LW'] + x['Qs'] + x['Ql']) * x['SA']

fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.plot( y1, "k-", lw=1)
ax1.set_ylabel("Ice Volume ($m^3$)")
ax1.set_xlabel("Days")
ax1.set_ylim(0,1.2)

ax2 = ax1.twinx()
ax2.plot(y2, "r-", linewidth=0.5)
ax2.set_ylabel("Energy[$W$]", color="r")
for tl in ax2.get_yticklabels():
    tl.set_color("r")

ax1.grid(axis = 'x')

plt.savefig(os.path.join(folders['output_folder'], site + "_energybarppt1.jpg"), bbox_inches  =  "tight", dpi=300)

plt.clf()


y1 = (x['SW'] + x['LW'] + x['Qs'] + x['Ql'])
y2 = x['SA']
fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.plot( y1, "r-")
plt.ylabel('Energy ($W/m^{2}$)$')
ax1.set_xlabel("Days")

ax2 = ax1.twinx()
ax2.plot( y2, "b-", linewidth=0.5)
ax2.set_ylabel("Surface Area ($m^{-2}$)", color="b")
for tl in ax2.get_yticklabels():
    tl.set_color("b")

ax1.grid(axis = 'x')

plt.savefig(os.path.join(folders['output_folder'], site + "_energybarppt2.jpg"), bbox_inches  =  "tight", dpi=300)
plt.clf()


# Plots
filename = os.path.join(folders['output_folder'], site + '_' + option + "_energybar.pdf")
pp  =  PdfPages(filename)

x= df.set_index('When').resample('D').mean().reset_index()
x.index = np.arange(1, len(x) + 1)

fig, ax = plt.subplots(1)
y= x[['SW','LW']]
y.plot.bar(stacked=True, edgecolor = df3['Discharge'], linewidth=0.5)
plt.xlabel('Days')
plt.ylabel('Energy ($W/m^{2}$)')
plt.ylim(-150, 150)
# plt.legend(loc=1, bbox_to_anchor=(0, 1))
pp.savefig(bbox_inches  =  "tight")
# plt.savefig(os.path.join(folders['output_folder'], site + "_energybar1.jpg"), bbox_inches  =  "tight", dpi=300)
plt.clf()

fig, ax = plt.subplots(1)
y= x[['SW','LW']]
y.plot.bar(stacked=True, edgecolor = df3['Discharge'], linewidth=0.5)
plt.xlabel('Days')
plt.ylabel('Energy ($W/m^{2}$)')
plt.ylim(-150, 150)
plt.legend(loc = 'upper right')
pp.savefig(bbox_inches  =  "tight")
plt.savefig(os.path.join(folders['output_folder'], site + "_energybar2.jpg"), bbox_inches  =  "tight", dpi=300)
plt.clf()

fig, ax = plt.subplots(1)
y= x[['SW','LW','Qs' ]]
y.plot.bar(stacked=True, edgecolor = df3['Discharge'], linewidth=0.5)
plt.xlabel('Days')
plt.ylabel('Energy ($W/m^{2}$)')
plt.legend(loc = 'upper right')
plt.ylim(-150, 150)
pp.savefig( bbox_inches  =  "tight")
plt.savefig(os.path.join(folders['output_folder'], site + "_energybar3.jpg"), bbox_inches  =  "tight", dpi=300)
plt.clf()

fig, ax = plt.subplots(1)
y= x[['SW','LW','Qs','Ql' ]]
y.plot.bar(stacked=True, edgecolor = df3['Discharge'], linewidth=0.5)
plt.xlabel('Days')
plt.ylabel('Energy ($W/m^{2}$)')
plt.legend(loc = 'upper right')
plt.ylim(-150, 150)
pp.savefig(bbox_inches  =  "tight")
plt.savefig(os.path.join(folders['output_folder'], site + "_energybar4.jpg"), bbox_inches  =  "tight", dpi=300)
plt.clf()

pp.close()


# x = df1.When
# y1 = df.ice
# y2 = df1.ice
# fig = plt.figure()
# ax1 = fig.add_subplot(111)
# ax1.plot(x, y1, "k-", label = 'Schwarzsee')
# ax1.plot(x, y2, "b-", label = 'Guttannen')
# ax1.set_ylabel("Ice Volume[$litres$]")
# ax1.set_xlabel("Days")
#
# # ax2 = ax1.twinx()
# # ax2.plot(x, y2, "b-", linewidth=0.5)
# # ax2.set_ylabel("Vapour[$kg$]", color="b")
# # for tl in ax2.get_yticklabels():
# #     tl.set_color("b")
#
# #  format the ticks
# ax1.xaxis.set_major_locator(mdates.MonthLocator())
# ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
# ax1.xaxis.set_minor_locator(mdates.WeekdayLocator())
# ax1.grid()
# fig.autofmt_xdate()
# plt.legend()
# # pp.savefig(bbox_inches="tight")
# plt.savefig(
#     os.path.join(folders['output_folder'], site + 'guttannen'+ "_ppt.jpg"), bbox_inches="tight", dpi=300
# )
# plt.clf()
#
# x = df1.When
# y1 = df.T_a
# y2 = df1.T_a
# fig = plt.figure()
# ax1 = fig.add_subplot(111)
# ax1.plot(x, y1, "k-", label = 'Schwarzsee', linewidth = 0.5)
# ax1.plot(x, y2, "b-", label = 'Guttannen', linewidth = 0.5)
# ax1.set_ylabel("Ice Volume[$litres$]")
# ax1.set_xlabel("Days")
#
# #  format the ticks
# ax1.xaxis.set_major_locator(mdates.MonthLocator())
# ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
# ax1.xaxis.set_minor_locator(mdates.WeekdayLocator())
# ax1.grid()
# fig.autofmt_xdate()
# plt.legend()
# # pp.savefig(bbox_inches="tight")
# plt.savefig(
#     os.path.join(output_folder, site + 'guttannen'+ "_Tppt.jpg"), bbox_inches="tight", dpi=300
# )
# plt.clf()
