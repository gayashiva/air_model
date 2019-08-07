import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.backends.backend_pdf import PdfPages
import scipy.integrate as spi
import math
from pathlib import Path
import os

site = input("Input the Field Site Name: ")

dirname = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', '..'))

data_file = os.path.join(dirname, "data/raw/" + site + "_aws.txt")

interim_folder = os.path.join(dirname, "data/interim/")


# settings
start_date = datetime(2019, 2, 1)
end_date = datetime(2019, 5, 1)

# read files
df_in = pd.read_csv(
    data_file, encoding="latin-1", skiprows=2, sep=";"
)

df_in["When"] = pd.to_datetime(df_in["time"], format="%Y%m%d%H%M")  # Datetime

if site == 'plaffeien':

    """
    Parameter
    ---------
              Unit                                 Description
    pva200s0  hPa                                  Vapour pressure 2 m above ground; current value
    prestas0  hPa                                  Pressure at station level (QFE); current value
    ods000z0  W/m²                                 diffuse radiation, average 10 minutes
    gre000z0  W/m²                                 Global radiation; ten minutes mean
    tre200s0  °C                                   Air temperature 2 m above ground; current value
    rre150z0  mm                                   Precipitation; ten minutes total
    ure200s0  %                                    Relative air humidity 2 m above ground; current value
    fkl010z0  m/s                                  Wind speed scalar; ten minutes mean
    """

    # Model Time Window
    start_date = datetime(2019, 1, 29)
    end_date = datetime(2019, 5, 1)
    mask = (df_in["When"] >= start_date) & (df_in["When"] <= end_date)
    df_in = df_in.loc[mask]
    df_in = df_in.reset_index()

    # Add Radiation data
    df_in["ods000z0"] = pd.to_numeric(df_in["ods000z0"], errors="coerce")
    df_in["gre000z0"] = pd.to_numeric(df_in["gre000z0"], errors="coerce")
    df_in["Rad"] = df_in["gre000z0"] - df_in["ods000z0"]
    df_in["DRad"] = df_in["ods000z0"]
    df_in["T_a"] = pd.to_numeric(df_in["tre200s0"], errors="coerce")  # Add Temperature data
    df_in["Prec"] = pd.to_numeric(
        df_in["rre150z0"], errors="coerce"
    )  # Add Precipitation data
    df_in["RH"] = pd.to_numeric(df_in["ure200s0"], errors="coerce")  # Add Humidity data
    df_in["v_a"] = pd.to_numeric(df_in["fkl010z0"], errors="coerce")  # Add wind speed data
    df_in["p_a"] = pd.to_numeric(df_in["prestas0"], errors="coerce")  # Air pressure
    df_in["vp_a"] = pd.to_numeric(
        df_in["pva200s0"], errors="coerce"
    )  # Vapour pressure over air

    df_in["Prec"] = df_in["Prec"] / 1000

    df_in["Fountain"] = 0
    df_in["Discharge"] = 4  # litres per minute

    # Fill nans
    df_in = df_in.fillna(method="ffill")


    df_out = df_in[
        [
            "When",
            "T_a",
            "RH",
            "v_a",
            "Rad",
            "DRad",
            "Prec",
            "p_a",
            "vp_a",
            "Fountain",
            "Discharge",
        ]
    ]


    # 5 minute sum
    cols = ["T_a", "RH", "v_a", "Rad", "DRad", "Prec", "p_a", "vp_a"]
    df_out[cols] = df_out[cols] / 2
    df_out = df_out.set_index("When").resample("5T").ffill().reset_index()



    '''Fountain Runtime'''

    """Settings"""
    z = 2  # m height of AWS

    """Material Properties"""
    a_w = 0.6
    we = 0.95
    z0mi=0.001
    z0ms=0.0015
    z0hi=0.0001
    c = 0.5
    Lf = 334 * 1000  #  J/kg Fusion
    cw = 4.186 * 1000  # J/kg Specific heat water
    rho_w = 1000  # Density of water
    rho_a = 1.29  # kg/m3 air density at mean sea level
    p0 = 1013  # hPa
    k = 0.4  # Van Karman constant
    bc = 5.670367 * math.pow(10, -8)  # Stefan Boltzman constant

    """Initialise"""
    df_out["TotalE"] = 0
    df_out["SW"] = 0
    df_out["LW"] = 0
    df_out["Qs"] = 0

    """ Simulation """
    for i in range(1, df_out.shape[0]):

        # Vapor Pressure empirical relations
        if "vp_a" not in list(df_out.columns):
            Ea = (
                6.11
                * math.pow(
                    10, 7.5 * df_out.loc[i - 1, "T_a"] / (df_out.loc[i - 1, "T_a"] + 237.3)
                )
                * df_out.loc[i, "RH"]
                / 100
            )
        else:
            Ea = df_out.loc[i, "vp_a"]

        df_out.loc[i, "e_a"] = (
            1.24
            * math.pow(abs(Ea / (df_out.loc[i, "T_a"] + 273.15)), 1 / 7)
            * (1 + 0.22 * math.pow(c, 2))
        )

        # Short Wave Radiation SW
        df_out.loc[i, "SW"] = (1 - a_w) * (
            df_out.loc[i, "Rad"] + df_out.loc[i, "DRad"]
        )

        # Long Wave Radiation LW
        if "oli000z0" not in list(df_out.columns):
            df_out.loc[i, "LW"] = df_out.loc[i, "e_a"] * bc * math.pow(
                df_out.loc[i, "T_a"] + 273.15, 4
            ) - we * bc * math.pow(0 + 273.15, 4)
        else:
            df_out.loc[i, "LW"] = df_out.loc[i, "oli000z0"] - we * bc * math.pow(0 + 273.15, 4)

        # Sensible Heat Qs
        df_out.loc[i, "Qs"] = (
            cw
            * rho_a
            * df_out.loc[i, "p_a"]
            / p0
            * math.pow(k, 2)
            * df_out.loc[i, "v_a"]
            * (df_out.loc[i, "T_a"])
            / (np.log(z / z0mi) * np.log(z / z0hi))
        )

        # Total Energy W/m2
        df_out.loc[i, "TotalE"] = df_out.loc[i, "SW"] + df_out.loc[i, "LW"] + df_out.loc[i, "Qs"]

    # df_in.Fountain[df_in.T_a < -5] = 1
    df_out.Fountain[df_out.TotalE < -100] = 1

    df_out = df_out.round(5)

if site == 'guttannen':

    """
    Parameter
    ---------
              Unit                                 Description
    pva200s0  hPa                                  Vapour pressure 2 m above ground; current value
    prestas0  hPa                                  Pressure at station level (QFE); current value
    gre000z0  W/m²                                 Global radiation; ten minutes mean
    oli000z0  W/m²                                 Longwave incoming radiation; ten minute average
    tre200s0  °C                                   Air temperature 2 m above ground; current value
    rre150z0  mm                                   Precipitation; ten minutes total
    ure200s0  %                                    Relative air humidity 2 m above ground; current value
    fkl010z0  m/s                                  Wind speed scalar; ten minutes mean
    """

    # Model Time Window
    start_date = datetime(2017, 11, 1)
    end_date = datetime(2018, 11, 1)
    mask = (df_in["When"] >= start_date) & (df_in["When"] <= end_date)
    df_in = df_in.loc[mask]
    df_in = df_in.reset_index()

    # Convert to int
    df_in["oli000z0"] = pd.to_numeric(
        df_in["oli000z0"], errors="coerce"
    )  # Add Longwave Radiation data
    df_in["gre000z0"] = pd.to_numeric(
        df_in["gre000z0"], errors="coerce"
    )  # Add Radiation data
    df_in["T_a"] = pd.to_numeric(df_in["tre200s0"], errors="coerce")  # Add Temperature data
    df_in["Prec"] = pd.to_numeric(
        df_in["rre150z0"], errors="coerce"
    )  # Add Precipitation data
    df_in["RH"] = pd.to_numeric(df_in["ure200s0"], errors="coerce")  # Add Humidity data
    df_in["v_a"] = pd.to_numeric(df_in["fkl010z0"], errors="coerce")  # Add wind speed data
    df_in["p_a"] = pd.to_numeric(df_in["prestas0"], errors="coerce")  # Air pressure
    df_in["vp_a"] = pd.to_numeric(
        df_in["pva200s0"], errors="coerce"
    )  # Vapour pressure over air

    df_in["Rad"] = df_in["gre000z0"] - df_in["gre000z0"] * 0.1
    df_in["DRad"] = df_in["gre000z0"] * 0.1
    df_in["LW"] = df_in["oli000z0"]
    df_in["Prec"] = df_in["Prec"] / 1000

    df_in["Fountain"] = 0
    df_in.Fountain[df_in.T_a < -5] = 1
    df_in["Discharge"] = 1  # litres per minute

    # Fill nans
    df_in = df_in.fillna(method="ffill")

    df_out = df_in[
        [
            "When",
            "T_a",
            "RH",
            "v_a",
            "Rad",
            "DRad",
            "oli000z0",
            "Prec",
            "p_a",
            "vp_a",
            "Fountain",
            "Discharge",
        ]
    ]
    df_out = df_out.round(5)

    # 5 minute sum
    cols = ["T_a", "RH", "v_a", "Rad", "DRad", "Prec", "p_a", "vp_a", "oli000z0"]
    df_out[cols] = df_out[cols] / 2
    df_out = df_out.set_index("When").resample("5T").ffill().reset_index()

df_out.to_csv(interim_folder + site + "_model_input.csv", sep=",")

# Plots
filename = interim_folder + site + "_all_data_" + str(end_date.day) + ".pdf"
pp = PdfPages(filename)

x = df_out["When"]
y1 = df_out["T_a"]
y2 = df_out["Prec"]

fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.plot(x, y1, "k-", linewidth=0.5)
ax1.set_ylabel("Temperature[C]")
ax1.set_xlabel("Days")

ax2 = ax1.twinx()
ax2.plot(x, y2, "b-", linewidth=0.5)
ax2.set_ylabel("Prec[m]", color="b")
for tl in ax2.get_yticklabels():
    tl.set_color("b")

# format the ticks
ax1.xaxis.set_major_locator(mdates.WeekdayLocator())
ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
ax1.xaxis.set_minor_locator(mdates.DayLocator())
ax1.grid()

# rotates and right aligns the x labels, and moves the bottom of the axes up to make room for them
fig.autofmt_xdate()
pp.savefig(bbox_inches="tight")
plt.clf()

y1 = df_out["v_a"]

fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.plot(x, y1, "k-", linewidth=0.5)
ax1.set_ylabel("Wind Speed[$ms^{-1}$]")
ax1.set_xlabel("Days")

# format the ticks
ax1.xaxis.set_major_locator(mdates.WeekdayLocator())
ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
ax1.xaxis.set_minor_locator(mdates.DayLocator())
ax1.grid()

fig.autofmt_xdate()
pp.savefig(bbox_inches="tight")
plt.clf()

y1 = df_out["p_a"]
fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.plot(x, y1, "k-", linewidth=0.5)
ax1.set_ylabel("Pressure[hPa]")
ax1.set_xlabel("Days")

# format the ticks
ax1.xaxis.set_major_locator(mdates.WeekdayLocator())
ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
ax1.xaxis.set_minor_locator(mdates.DayLocator())
ax1.grid()

fig.autofmt_xdate()
pp.savefig(bbox_inches="tight")
plt.clf()

y1 = df_out["RH"]

fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.plot(x, y1, "k-", linewidth=0.5)
ax1.set_ylabel("Relative Humidity")
ax1.set_xlabel("Days")

# format the ticks
ax1.xaxis.set_major_locator(mdates.WeekdayLocator())
ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
ax1.xaxis.set_minor_locator(mdates.DayLocator())
ax1.grid()

fig.autofmt_xdate()
pp.savefig(bbox_inches="tight")
plt.clf()


y1 = df_out["vp_a"]

fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.plot(x, y1, "k-", linewidth=0.5)
ax1.set_ylabel("Vapour Pressure air")
ax1.set_xlabel("Days")

# format the ticks
ax1.xaxis.set_major_locator(mdates.WeekdayLocator())
ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
ax1.xaxis.set_minor_locator(mdates.DayLocator())
ax1.grid()

# rotates and right aligns the x labels, and moves the bottom of the axes up to make room for them
fig.autofmt_xdate()
pp.savefig(bbox_inches="tight")
plt.clf()

y1 = df_out["Rad"]
y2 = df_out["DRad"]

fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.plot(x, y1, "k-", linewidth=0.5)
ax1.set_ylabel("SWR")
ax1.set_xlabel("Days")

ax2 = ax1.twinx()
ax2.plot(x, y2, "b-", linewidth=0.5)
ax2.set_ylabel("DR", color="b")
for tl in ax2.get_yticklabels():
    tl.set_color("b")

# format the ticks
ax1.xaxis.set_major_locator(mdates.WeekdayLocator())
ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
ax1.xaxis.set_minor_locator(mdates.DayLocator())
ax1.grid()

# rotates and right aligns the x labels, and moves the bottom of the axes up to make room for them
fig.autofmt_xdate()
pp.savefig(bbox_inches="tight")
plt.clf()

y1 = df_out["Discharge"]
y2 = df_out["Fountain"]

fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.plot(x, y1, "k-", linewidth=0.5)
ax1.set_ylabel("Discharge[$lmin^{-1}$]")
ax1.set_xlabel("Days")

ax2 = ax1.twinx()
ax2.plot(x, y2, "b-", linewidth=0.5)
ax2.set_ylabel("Fountain On/Off", color="b")
for tl in ax2.get_yticklabels():
    tl.set_color("b")

# format the ticks
ax1.xaxis.set_major_locator(mdates.WeekdayLocator())
ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
ax1.xaxis.set_minor_locator(mdates.DayLocator())
ax1.grid()

# rotates and right aligns the x labels, and moves the bottom of the axes up to make room for them
fig.autofmt_xdate()
pp.savefig(bbox_inches="tight")
plt.clf()

pp.close()


# Plots
filename = interim_folder + site + "_data" + ".pdf"
pp = PdfPages(filename)

fig, (ax1, ax2, ax3, ax4, ax5, ax6) = plt.subplots(
    nrows=6, ncols=1, sharex="col", sharey="row", figsize=(15, 10)
)

# fig.suptitle("Field Data", fontsize=14)
x = df_out.When

y1 = df_out.T_a
ax1.plot(x, y1, "k-", linewidth=0.5)
ax1.set_ylabel("T[$C$]")
ax1.grid()
ax1.xaxis.set_major_locator(mdates.WeekdayLocator())
ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
ax1.xaxis.set_minor_locator(mdates.DayLocator())

y2 = df_out.Discharge * df_out.Fountain * 5
ax2.plot(x, y2, "k-", linewidth=0.5)
ax2.set_ylabel("Discharge[$kg$]")
ax2.grid()

y3 = df_out.Rad
ax3.plot(x, y3, "k-", linewidth=0.5)
ax3.set_ylabel("SWR[$Wm^{-2}$]")
ax3.set_ylim([0, 600])
ax3.grid()

ax3t = ax3.twinx()
ax3t.plot(x, df_out.DRad, "b-", linewidth=0.5)
ax3t.set_ylabel("Diffused[$Wm^{-2}$]", color="b")
ax3t.set_ylim([0, 600])
for tl in ax3t.get_yticklabels():
    tl.set_color("b")

y4 = df_out.Prec * 1000
ax4.plot(x, y4, "k-", linewidth=0.5)
ax4.set_ylabel("Ppt[$mm$]")
ax4.grid()

y5 = df_out.p_a
ax5.plot(x, y5, "k-", linewidth=0.5)
ax5.set_ylabel("Pressure[$hpa$]")
ax5.grid()

y6 = df_out.v_a
ax6.plot(x, y6, "k-", linewidth=0.5)
ax6.set_ylabel("Wind[$ms^{-1}$]")
ax6.grid()

# rotates and right aligns the x labels, and moves the bottom of the axes up to make room for them
fig.autofmt_xdate()
pp.savefig(bbox_inches="tight")
plt.clf()

fig = plt.figure()
ax1 = fig.add_subplot(111)

y1 = df_out.T_a
ax1.plot(x, y1, "k-", linewidth=0.5)
ax1.set_ylabel("T[$C$]")
ax1.grid()

# format the ticks
ax1.xaxis.set_major_locator(mdates.WeekdayLocator())
ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
ax1.xaxis.set_minor_locator(mdates.DayLocator())
ax1.grid()
fig.autofmt_xdate()
pp.savefig(bbox_inches="tight")
plt.clf()

fig = plt.figure()
ax1 = fig.add_subplot(111)

y2 = df_out.Discharge * df_out.Fountain * 10
ax1.plot(x, y2, "k-", linewidth=0.5)
ax1.set_ylabel("Discharge[$kg$]")
ax1.grid()

# format the ticks
ax1.xaxis.set_major_locator(mdates.WeekdayLocator())
ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
ax1.xaxis.set_minor_locator(mdates.DayLocator())
ax1.grid()
fig.autofmt_xdate()
pp.savefig(bbox_inches="tight")
plt.clf()

fig = plt.figure()
ax1 = fig.add_subplot(111)

y3 = df_out.Rad
ax1.plot(x, y3, "k-", linewidth=0.5)
ax1.set_ylabel("SWR[$Wm^{-2}$]")
ax1.set_ylim([0, 600])
ax1.grid()

# format the ticks
ax1.xaxis.set_major_locator(mdates.WeekdayLocator())
ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
ax1.xaxis.set_minor_locator(mdates.DayLocator())
ax1.grid()
fig.autofmt_xdate()
pp.savefig(bbox_inches="tight")
plt.clf()

fig = plt.figure()
ax1 = fig.add_subplot(111)

y4 = df_out.Prec * 1000
ax1.plot(x, y4, "k-", linewidth=0.5)
ax1.set_ylabel("Ppt[$mm$]")
ax1.grid()

# format the ticks
ax1.xaxis.set_major_locator(mdates.WeekdayLocator())
ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
ax1.xaxis.set_minor_locator(mdates.DayLocator())
ax1.grid()
fig.autofmt_xdate()
pp.savefig(bbox_inches="tight")
plt.clf()

fig = plt.figure()
ax1 = fig.add_subplot(111)

y5 = df_out.p_a
ax1.plot(x, y5, "k-", linewidth=0.5)
ax1.set_ylabel("Pressure[$hpa$]")
ax1.grid()

# format the ticks
ax1.xaxis.set_major_locator(mdates.WeekdayLocator())
ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
ax1.xaxis.set_minor_locator(mdates.DayLocator())
ax1.grid()
fig.autofmt_xdate()
pp.savefig(bbox_inches="tight")
plt.clf()

fig = plt.figure()
ax1 = fig.add_subplot(111)

y6 = df_out.v_a
ax1.plot(x, y6, "k-", linewidth=0.5)
ax1.set_ylabel("Wind[$ms^{-1}$]")
ax1.grid()

# format the ticks
ax1.xaxis.set_major_locator(mdates.WeekdayLocator())
ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
ax1.xaxis.set_minor_locator(mdates.DayLocator())
ax1.grid()
fig.autofmt_xdate()
pp.savefig(bbox_inches="tight")
plt.clf()

pp.close()
