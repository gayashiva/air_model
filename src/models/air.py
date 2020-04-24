import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from tqdm import tqdm
import os
import math
import time
from pandas.plotting import register_matplotlib_converters

register_matplotlib_converters()
import matplotlib
from matplotlib.offsetbox import AnchoredText
import math
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import scipy.stats as stats


class Icestupa: #todo create subclass

    """Physical Constants"""

    L_s = 2848 * 1000  # J/kg Sublimation
    L_e = 2514 * 1000  # J/kg Evaporation
    L_f = 334 * 1000  # J/kg Fusion
    c_w = 4.186 * 1000  # J/kgC Specific heat water
    c_i = 2.108 * 1000  # J/kgC Specific heat ice
    rho_w = 1000  # Density of water
    rho_i = 916  # Density of Ice rho_i
    rho_a = 1.29  # kg/m3 air density at mean sea level
    k = 0.4  # Van Karman constant
    bc = 5.670367 * math.pow(10, -8)  # Stefan Boltzman constant
    p0 = 1013  # Standard air pressure hPa
    vp_w = 6.112  # Saturation Water vapour pressure

    """Model constants"""
    time_steps = 5 * 60  # s Model time steps
    dirname = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

    """Surface"""
    ie = 0.95  # Ice Emissivity ie
    a_i = 0.35  # Albedo of Ice a_i
    a_s = 0.85  # Albedo of Fresh Snow a_s
    decay_t = 10  # Albedo decay rate decay_t_d
    dx = 1e-02  # Ice layer thickness

    """Meteorological"""
    z0mi = 0.0017  # Ice Momentum roughness length
    z0hi = 0.0017  # Ice Scalar roughness length
    snow_fall_density = 250  # Snowfall density
    rain_temp = 1  # Temperature condition for liquid precipitation

    """Fountain"""
    aperture_f = 0.005  # Fountain aperture diameter
    h_f = 1.35  # Fountain steps h_f

    """Site constants"""
    latitude = 46.693723
    longitude = 7.297543
    utc_offset = 1

    """Miscellaneous"""
    ftl = 0  # Fountain flight time loss ftl,
    h_aws = 3  # m height of AWS
    theta_f = 45  # Fountain aperture diameter

    def __init__(self, site="schwarzsee"):

        self.site = site

        self.folders = dict(
            input_folder=os.path.join(self.dirname, "data/interim/" + site + "/"),
            output_folder=os.path.join(self.dirname, "data/processed/" + site + "/"),
            sim_folder=os.path.join(
                self.dirname, "data/processed/" + site + "/simulations"
            ),
        )

        input_file = self.folders["input_folder"] + "raw_input.csv"

        self.state = 0
        self.df = pd.read_csv(input_file, sep=",", header=0, parse_dates=["When"])

        i = 0
        start = 0
        while self.df.loc[i, "Discharge"] == 0:
            start = i
            i += 1

        i = start
        fountain_off = 0
        while self.df.Discharge[i:].any() > 0:
            fountain_off = i
            i += 1

        self.dates = dict(
            start_date=self.df.loc[start, "When"],
            fountain_off_date=self.df.loc[fountain_off, "When"],
        )

        self.df = self.df[start:]
        self.df = self.df.reset_index(drop=True)

    def SEA(self, date):

        latitude = self.latitude
        longitude = self.longitude
        utc_offset = self.utc_offset
        hour = date.hour
        minute = date.minute
        # Check your timezone to add the offset
        hour_minute = (hour + minute / 60) - utc_offset
        day_of_year = date.timetuple().tm_yday

        g = (360 / 365.25) * (day_of_year + hour_minute / 24)

        g_radians = math.radians(g)

        declination = (
            0.396372
            - 22.91327 * math.cos(g_radians)
            + 4.02543 * math.sin(g_radians)
            - 0.387205 * math.cos(2 * g_radians)
            + 0.051967 * math.sin(2 * g_radians)
            - 0.154527 * math.cos(3 * g_radians)
            + 0.084798 * math.sin(3 * g_radians)
        )

        time_correction = (
            0.004297
            + 0.107029 * math.cos(g_radians)
            - 1.837877 * math.sin(g_radians)
            - 0.837378 * math.cos(2 * g_radians)
            - 2.340475 * math.sin(2 * g_radians)
        )

        SHA = (hour_minute - 12) * 15 + longitude + time_correction

        if SHA > 180:
            SHA_corrected = SHA - 360
        elif SHA < -180:
            SHA_corrected = SHA + 360
        else:
            SHA_corrected = SHA

        lat_radians = math.radians(latitude)
        d_radians = math.radians(declination)
        SHA_radians = math.radians(SHA)

        SZA_radians = math.acos(
            math.sin(lat_radians) * math.sin(d_radians)
            + math.cos(lat_radians) * math.cos(d_radians) * math.cos(SHA_radians)
        )

        SZA = math.degrees(SZA_radians)

        SEA = 90 - SZA

        if SEA < 0:  # Before Sunrise or after sunset
            SEA = 0

        return math.radians(SEA)

    def projectile_xy(self, v):
        hs = self.h_f
        g = 9.81
        data_xy = []
        t = 0.0
        theta_f = math.radians(self.theta_f)
        while True:
            # now calculate the height y
            y = hs + (t * v * math.sin(theta_f)) - (g * t * t) / 2
            # projectile has hit ground level
            if y < 0:
                break
            # calculate the distance x
            x = v * math.cos(theta_f) * t
            # append the (x, y) tuple to the list
            data_xy.append((x, y))
            # use the time in increments of 0.1 seconds
            t += 0.01
        return x

    def albedo(self, row, s=0, f=0):

        i = row.Index

        a_min = self.a_i

        """Albedo"""
        # Precipitation
        if (row.Discharge == 0) & (row.Prec > 0):
            if row.T_a < self.rain_temp:  # Snow
                s = 0
                f = 0

        if row.Discharge > 0:
            f = 1
            s = 0

        if f == 0:  # last snowed
            self.df.loc[i, "a"] = self.a_i + (self.a_s - self.a_i) * math.exp(
                -s / self.decay_t
            )
            s = s + 1
        else:  # last sprayed
            self.df.loc[i, "a"] = self.a_i

        return s, f

    def derive_parameters(self):

        missing = ["a", "cld", "SEA", "e_a", "vp_a", "r_f", "LW_in"]
        for col in missing:
            if col in list(self.df.columns):
                missing = missing.remove(col)
            else:
                self.df[col] = 0

        """ Fountain Spray radius """
        Area = math.pi * math.pow(self.aperture_f, 2) / 4

        """Albedo Decay"""
        self.decay_t = (
            self.decay_t * 24 * 60 * 60 / self.time_steps
        )  # convert to 5 minute time steps
        s = 0
        f = 0

        for row in tqdm(self.df[1:].itertuples(), total=self.df.shape[0]):

            """Solar Elevation Angle"""
            self.df.loc[row.Index, "SEA"] = self.SEA(row.When)

            """ Vapour Pressure"""
            if "vp_a" in missing:
                self.df.loc[row.Index, "vp_a"] = (
                    6.11
                    * math.pow(
                        10,
                        7.5
                        * self.df.loc[row.Index - 1, "T_a"]
                        / (self.df.loc[row.Index - 1, "T_a"] + 237.3),
                    )
                    * row.RH
                    / 100
                )

            """LW incoming"""
            if "LW_in" in missing:

                # Cloudiness from diffuse fraction
                if row.SW_direct + row.SW_diffuse > 1:
                    self.df.loc[row.Index, "cld"] = row.SW_diffuse / (
                        row.SW_direct + row.SW_diffuse
                    )
                else:
                    # Night Cloudiness average of last 8 hours
                    if row.Index - 96 > 0:
                        for j in range(row.Index - 96, row.Index):
                            self.df.loc[row.Index, "cld"] += self.df.loc[j, "cld"]
                        self.df.loc[row.Index, "cld"] = (
                            self.df.loc[row.Index, "cld"] / 96
                        )
                    else:
                        for j in range(0, row.Index):
                            self.df.loc[row.Index, "cld"] += self.df.loc[j, "cld"]
                        self.df.loc[row.Index, "cld"] = (
                            self.df.loc[row.Index, "cld"] / row.Index
                        )

                self.df.loc[row.Index, "e_a"] = (
                    1.24
                    * math.pow(
                        abs(self.df.loc[row.Index, "vp_a"] / (row.T_a + 273.15)), 1 / 7
                    )
                ) * (1 + 0.22 * math.pow(self.df.loc[row.Index, "cld"], 2))

                self.df.loc[row.Index, "LW_in"] = (
                    self.df.loc[row.Index, "e_a"]
                    * self.bc
                    * math.pow(row.T_a + 273.15, 4)
                )

            """ Fountain Spray radius """
            v_f = row.Discharge / (60 * 1000 * Area)
            self.df.loc[row.Index, "r_f"] = self.projectile_xy(v_f)

            s, f = self.albedo(row, s, f)

        self.df = self.df.round(5)

        self.df = self.df.drop(["e_a", "cld", "Unnamed: 0"], axis=1)

        data_store = pd.HDFStore(self.folders["input_folder"] + "model_input.h5")
        data_store["df"] = self.df
        data_store.close()

        self.print_input()

    def print_input(self):

        pp = PdfPages(self.folders["input_folder"] + "derived_parameters.pdf")

        x = self.df.When

        fig = plt.figure()
        ax1 = fig.add_subplot(111)

        y1 = self.df.Discharge
        ax1.plot(x, y1, "k-", linewidth=0.5)
        ax1.set_ylabel("Discharge [$l\, min^{-1}$]")
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

        y1 = self.df.r_f
        ax1.plot(x, y1, "k-", linewidth=0.5)
        ax1.set_ylabel("Spray Radius [$m$]")
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
        y1 = self.df.vp_a
        ax1.plot(x, y1, "k-", linewidth=0.5)
        ax1.set_ylabel("Vapour Pressure")
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
        y1 = self.df.a
        ax1.plot(x, y1, "k-", linewidth=0.5)
        ax1.set_ylabel("Albedo")
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

        """Input Plots"""

        pp = PdfPages(self.folders["input_folder"] +"data.pdf")

        fig, (ax1, ax2, ax3, ax4, ax5, ax6) = plt.subplots(
            nrows=6, ncols=1, sharex="col", sharey="row", figsize=(15, 12)
        )

        # fig.suptitle("Field Data", fontsize=14)
        # Remove horizontal space between axes
        # fig.subplots_adjust(hspace=0)

        x = self.df.When

        y1 = self.df.Discharge
        ax1.plot(x, y1, "k-", linewidth=0.5)
        ax1.set_ylabel("Discharge [$l\, min^{-1}$]")
        ax1.grid()

        ax1t = ax1.twinx()
        ax1t.plot(x, self.df.Prec * 1000, "b-", linewidth=0.5)
        ax1t.set_ylabel("Precipitation [$mm$]", color="b")
        for tl in ax1t.get_yticklabels():
            tl.set_color("b")

        y2 = self.df.T_a
        ax2.plot(x, y2, "k-", linewidth=0.5)
        ax2.set_ylabel("Temperature [$\degree C$]")
        ax2.grid()

        y3 = self.df.SW_direct + self.df.SW_diffuse
        ax3.plot(x, y3, "k-", linewidth=0.5)
        ax3.set_ylabel("Global [$W\,m^{-2}$]")
        ax3.grid()

        ax3t = ax3.twinx()
        ax3t.plot(x, self.df.SW_diffuse, "b-", linewidth=0.5)
        ax3t.set_ylim(ax3.get_ylim())
        ax3t.set_ylabel("Diffuse [$W\,m^{-2}$]", color="b")
        for tl in ax3t.get_yticklabels():
            tl.set_color("b")

        y4 = self.df.RH
        ax4.plot(x, y4, "k-", linewidth=0.5)
        ax4.set_ylabel("Humidity [$\%$]")
        ax4.grid()

        y5 = self.df.p_a
        ax5.plot(x, y5, "k-", linewidth=0.5)
        ax5.set_ylabel("Pressure [$hPa$]")
        ax5.grid()

        y6 = self.df.v_a
        ax6.plot(x, y6, "k-", linewidth=0.5)
        ax6.set_ylabel("Wind [$m\,s^{-1}$]")
        ax6.grid()

        ax1.xaxis.set_major_locator(mdates.WeekdayLocator())
        ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
        ax1.xaxis.set_minor_locator(mdates.DayLocator())

        # rotates and right aligns the x labels, and moves the bottom of the axes up to make room for them
        fig.autofmt_xdate()
        pp.savefig(bbox_inches="tight")

        plt.savefig("data.jpg", bbox_inches="tight", dpi=300)

        plt.clf()

        fig = plt.figure()
        ax1 = fig.add_subplot(111)

        y1 = self.df.T_a
        ax1.plot(x, y1, "k-", linewidth=0.5)
        ax1.set_ylabel("Temperature [$\degree C$]")
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

        y2 = self.df.Discharge
        ax1.plot(x, y2, "k-", linewidth=0.5)
        ax1.set_ylabel("Discharge Rate ")
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

        y3 = self.df.SW_direct
        ax1.plot(x, y3, "k-", linewidth=0.5)
        ax1.set_ylabel("Direct SWR [$W\,m^{-2}$]")
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

        y31 = self.df.SW_diffuse
        ax1.plot(x, y31, "k-", linewidth=0.5)
        ax1.set_ylabel("Diffuse SWR [$W\,m^{-2}$]")
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

        y4 = self.df.Prec * 1000
        ax1.plot(x, y4, "k-", linewidth=0.5)
        ax1.set_ylabel("Ppt [$mm$]")
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

        y5 = self.df.p_a
        ax1.plot(x, y5, "k-", linewidth=0.5)
        ax1.set_ylabel("Pressure [$hPa$]")
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

        y6 = self.df.v_a
        ax1.plot(x, y6, "k-", linewidth=0.5)
        ax1.set_ylabel("Wind [$m\,s^{-1}$]")
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

    def cylinder_surface_area(self, i):
        # Ice Radius
        self.df.loc[i, "r_ice"] = self.r_mean

        # Ice Height
        self.df.loc[i, "h_ice"] = (
                self.df.loc[i - 1, "iceV"]
                / (math.pi * self.df.loc[i, "r_ice"] ** 2)
        )

        # Area of Conical Ice Surface
        self.df.loc[i, "SA"] = (
                math.pi
                * math.pow(self.df.loc[i, "r_ice"], 2)
                + math.pi * self.df.loc[i, "r_ice"] * self.df.loc[i, "h_ice"]
        )

        self.SRf = (
                           self.df.loc[i, "h_ice"]
                           * 2 * self.df.loc[i, "r_ice"]
                           * math.cos(self.df.loc[i, "SEA"])
                           + math.pi
                           * math.pow(self.df.loc[i, "r_ice"], 2)
                           * math.sin(self.df.loc[i, "SEA"])
                   ) / self.df.loc[i, "SA"]

    def surface_area(self, i):

        if (self.df.Discharge[i] > 0) & (self.df.loc[i - 1, "r_ice"] >= self.r_mean):
            # Ice Radius
            self.df.loc[i, "r_ice"] = self.df.loc[i - 1, "r_ice"]

            # Ice Height
            self.df.loc[i, "h_ice"] = (
                3
                * self.df.loc[i - 1, "iceV"]
                / (math.pi * self.df.loc[i, "r_ice"] ** 2)
            )

            # Height by Radius ratio
            self.df.loc[i, "h_r"] = (
                self.df.loc[i - 1, "h_ice"] / self.df.loc[i - 1, "r_ice"]
            )

            # Area of Conical Ice Surface
            self.df.loc[i, "SA"] = (
                math.pi
                * self.df.loc[i, "r_ice"]
                * math.pow(
                    (
                        math.pow(self.df.loc[i, "r_ice"], 2)
                        + math.pow((self.df.loc[i, "h_ice"]), 2)
                    ),
                    1 / 2,
                )
            )

        else:

            # Height to radius ratio
            self.df.loc[i, "h_r"] = self.df.loc[i - 1, "h_r"]

            # Ice Radius
            self.df.loc[i, "r_ice"] = math.pow(
                self.df.loc[i - 1, "iceV"] / math.pi * (3 / self.df.loc[i, "h_r"]),
                1 / 3,
            )

            # Ice Height
            self.df.loc[i, "h_ice"] = self.df.loc[i, "h_r"] * self.df.loc[i, "r_ice"]

            # Area of Conical Ice Surface
            self.df.loc[i, "SA"] = (
                math.pi
                * self.df.loc[i, "r_ice"]
                * math.pow(
                    (
                        math.pow(self.df.loc[i, "r_ice"], 2)
                        + math.pow(self.df.loc[i, "r_ice"] * self.df.loc[i, "h_r"], 2)
                    ),
                    1 / 2,
                )
            )

        self.SRf = (
            0.5
            * self.df.loc[i, "h_ice"]
            * self.df.loc[i, "r_ice"]
            * math.cos(self.df.loc[i, "SEA"])
            + math.pi
            * math.pow(self.df.loc[i, "r_ice"], 2)
            * 0.5
            * math.sin(self.df.loc[i, "SEA"])
        ) / self.df.loc[i, "SA"]

    def energy_balance(self, row):
        i = row.Index

        self.vp_ice = 6.112 * np.exp(
            22.46 * (self.df.loc[i - 1, "T_s"]) / ((self.df.loc[i - 1, "T_s"]) + 272.62)
        )

        # Water Boundary
        if (row.Discharge > 0) or (row.T_a > self.rain_temp and row.Prec > 0):
            self.df.loc[i, "vp_s"] = self.vp_w
            self.L = self.L_e
            self.c_s = self.c_w

        else:
            self.df.loc[i, "vp_s"] = self.vp_ice
            self.L = self.L_s
            self.c_s = self.c_i

        self.df.loc[i, "Ql"] = (
            0.623
            * self.L
            * self.rho_a
            / self.p0
            * math.pow(self.k, 2)
            * self.df.loc[i, "v_a"]
            * (row.vp_a - self.df.loc[i, "vp_s"])
            / (np.log(self.h_aws / self.z0mi) * np.log(self.h_aws / self.z0hi))
        )

        if self.df.loc[i, "Ql"] < 0:
            self.gas -= (
                self.df.loc[i, "Ql"] * self.df.loc[i, "SA"] * self.time_steps
            ) / self.L

            # Removing gas quantity generated from previous ice
            self.solid += (
                self.df.loc[i, "Ql"] * (self.df.loc[i, "SA"]) * self.time_steps
            ) / self.L

            # Ice Temperature
            self.delta_T_s += (self.df.loc[i, "Ql"] * self.time_steps) / (
                self.rho_i * self.dx * self.c_s
            )

        else:  # Deposition

            self.df.loc[i, "deposition"] += (
                self.df.loc[i, "Ql"] * self.df.loc[i, "SA"] * self.time_steps
            ) / self.L

        # Sensible Heat Qs
        self.df.loc[i, "Qs"] = (
            self.c_s
            * self.rho_a
            * row.p_a
            / self.p0
            * math.pow(self.k, 2)
            * self.df.loc[i, "v_a"]
            * (self.df.loc[i, "T_a"] - self.df.loc[i - 1, "T_s"])
            / (np.log(self.h_aws / self.z0mi) * np.log(self.h_aws / self.z0hi))
        )

        # Short Wave Radiation SW
        self.df.loc[i, "SW"] = (1 - row.a) * (row.SW_direct * self.SRf + row.SW_diffuse)

        # Long Wave Radiation LW
        self.df.loc[i, "LW"] = row.LW_in - self.ie * self.bc * math.pow(
            self.df.loc[i - 1, "T_s"] + 273.15, 4
        )

        # Conduction Freezing
        if (self.liquid > 0) & (self.df.loc[i - 1, "T_s"] < 0):
            self.df.loc[i, "Qc"] = (
                self.rho_i
                * self.dx
                * self.c_i
                * (-self.df.loc[i - 1, "T_s"])
                / self.time_steps
            )
            self.delta_T_s = -self.df.loc[i - 1, "T_s"]

        # Total Energy W/m2
        self.df.loc[i, "TotalE"] = (
            self.df.loc[i, "SW"]
            + self.df.loc[i, "LW"]
            + self.df.loc[i, "Qs"]
            + self.df.loc[i, "Qc"]
        )

        # Total Energy Joules
        self.EJoules = self.df.loc[i, "TotalE"] * self.time_steps * self.df.loc[i, "SA"]

    def summary(self, i):

        self.df = self.df[:i]
        Efficiency = float(
            (self.df["meltwater"].tail(1) + self.df["ice"].tail(1))
            / (
                self.df["Discharge"].sum() * self.time_steps / 60
                + self.df["ppt"].sum()
                + self.df["deposition"].sum()
            )
            * 100
        )

        print("\nIce Volume Max", float(self.df["iceV"].max()))
        print("Fountain efficiency", Efficiency)
        print("Ice Mass Remaining", float(self.df["ice"].tail(1)))
        print("Meltwater", float(self.df["meltwater"].tail(1)))
        print("Ppt", self.df["ppt"].sum())
        print("Model runtime", self.df.loc[i - 1, "When"] - self.df.loc[0, "When"])

        # Full Output
        filename4 = self.folders["output_folder"] + "model_results.csv"
        self.df.to_csv(filename4, sep=",")

    def print_output(self):

        self.df = self.df.rename(
            {
                "SW": "$SW_{net}$",
                "LW": "$LW_{net}$",
                "Qs": "$Q_S$",
                "Ql": "$Q_L$",
                "Qc": "$Q_C$",
            },
            axis=1,
        )

        # Plots
        pp = PdfPages(self.folders["output_folder"] + "model_results.pdf")

        x = self.df.When
        y1 = self.df.iceV

        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        ax1.plot(x, y1, "k-")
        ax1.set_ylabel("Ice Volume [$m^3$]")
        ax1.set_xlabel("Days")

        #  format the ticks
        ax1.xaxis.set_major_locator(mdates.WeekdayLocator())
        ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
        ax1.xaxis.set_minor_locator(mdates.DayLocator())
        ax1.grid()
        fig.autofmt_xdate()
        pp.savefig(bbox_inches="tight")
        plt.clf()

        y1 = self.df.SA

        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        ax1.plot(x, y1, "k-")
        ax1.set_ylabel("Surface Area [$m^2$]")
        ax1.set_xlabel("Days")

        #  format the ticks
        ax1.xaxis.set_major_locator(mdates.WeekdayLocator())
        ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
        ax1.xaxis.set_minor_locator(mdates.DayLocator())
        ax1.grid()
        fig.autofmt_xdate()
        pp.savefig(bbox_inches="tight")
        plt.clf()

        y1 = self.df.h_ice
        y2 = self.df.r_ice

        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        ax1.plot(x, y1, "k-")
        ax1.set_ylabel("Ice Cone Height [$m$]")
        ax1.set_xlabel("Days")

        ax2 = ax1.twinx()
        ax2.plot(x, y2, "b-", linewidth=0.5)
        ax2.set_ylabel("Ice Radius", color="b")
        for tl in ax2.get_yticklabels():
            tl.set_color("b")

        #  format the ticks
        ax1.xaxis.set_major_locator(mdates.WeekdayLocator())
        ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
        ax1.xaxis.set_minor_locator(mdates.DayLocator())
        ax1.grid()
        fig.autofmt_xdate()
        pp.savefig(bbox_inches="tight")
        plt.clf()

        y1 = self.df.iceV
        y2 = self.df["TotalE"] + self.df["$Q_L$"]

        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        ax1.plot(x, y1, "k-")
        ax1.set_ylabel("Ice Volume [$m^3$]")
        ax1.set_xlabel("Days")

        ax2 = ax1.twinx()
        ax2.plot(x, y2, "b-", linewidth=0.5)
        ax2.set_ylabel("Energy [$W\,m^{-2}$]", color="b")
        for tl in ax2.get_yticklabels():
            tl.set_color("b")

        #  format the ticks
        ax1.xaxis.set_major_locator(mdates.WeekdayLocator())
        ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
        ax1.xaxis.set_minor_locator(mdates.DayLocator())
        ax1.grid()
        fig.autofmt_xdate()
        pp.savefig(bbox_inches="tight")
        plt.clf()

        y1 = self.df.T_s

        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        ax1.plot(x, y1, "k-", linewidth=0.5)
        ax1.set_ylabel("Surface Temperature [$\degree C$]")
        ax1.set_xlabel("Days")

        #  format the ticks
        ax1.xaxis.set_major_locator(mdates.WeekdayLocator())
        ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
        ax1.xaxis.set_minor_locator(mdates.DayLocator())
        ax1.grid()
        fig.autofmt_xdate()
        pp.savefig(bbox_inches="tight")
        plt.close("all")

        pp.close()

    def run(self, **parameters):

        self.set_parameters(**parameters)
        print(parameters.values())

        if 'aperture_f' in parameters.keys():  # todo change to general
            """ Fountain Spray radius """
            Area = math.pi * math.pow(self.aperture_f, 2) / 4

            for row in self.df[1:].itertuples():
                v_f = row.Discharge / (60 * 1000 * Area)
                self.df.loc[row.Index, "r_f"] = self.projectile_xy(v_f)

        if 'a_i' or 'rain_temp' in parameters.keys():
            """Albedo Decay"""
            self.decay_t = (
                    self.decay_t * 24 * 60 * 60 / self.time_steps
            )  # convert to 5 minute time steps
            s = 0
            f = 0

            for row in self.df[1:].itertuples():
                s, f = self.albedo(row, s, f)

        self.melt_freeze()

        Efficiency = float(
            (self.df["meltwater"].tail(1) + self.df["ice"].tail(1))
            / (self.df["Discharge"].sum() * self.time_steps / 60 + self.df["ppt"].sum() + self.df["deposition"].sum())
            * 100
        )

        print("\nIce Volume Max", float(self.df["iceV"].max()))
        print("Fountain efficiency", Efficiency)
        print("Ice Mass Remaining", float(self.df["ice"].tail(1)))
        print("Meltwater", float(self.df["meltwater"].tail(1)))
        print("Ppt", self.df["ppt"].sum())
        print("Model runtime", self.df.loc[i - 1, "When"] - self.df.loc[0, "When"])

        self.df = self.df.set_index('When').resample('1H').mean().reset_index()

        return self.df.index.values / 24, self.df["iceV"].values

    def melt_freeze(self):

        l = [
            "T_s",  # Surface Temperature
            "ice",
            "iceV",
            "vapour",
            "water",
            "TotalE",
            "SW",
            "LW",
            "Qs",
            "Ql",
            "Qc",
            "meltwater",
            "SA",
            "h_ice",
            "r_ice",
            "ppt",
            "deposition",
        ]
        for col in l:
            self.df[col] = 0

        self.delta_T_s, self.solid, self.liquid, self.gas, self.melted, self.SRf, self.vp_ice, self.EJoules = (
            [0] * 8
        )

        for row in tqdm(self.df[1:].itertuples(), total=self.df.shape[0]):
            i = row.Index

            if i == 1 :
                """Initialize"""
                self.r_mean = self.df['r_f'].replace(0, np.NaN).mean()
                self.df.loc[i, "r_ice"] = self.r_mean
                self.df.loc[i, "h_ice"] = self.dx
                self.df.loc[i, "iceV"] = self.dx * math.pi * self.df.loc[0, "r_ice"] ** 2
                self.df.loc[i, "ice"] = self.df.loc[0, "iceV"] * self.rho_i

            else:

                # Ice Melted
                if self.df.loc[i - 1, "iceV"] <= 0:
                    self.df.loc[i - 1, "solid"] = 0
                    self.df.loc[i - 1, "ice"] = 0
                    self.df.loc[i - 1, "iceV"] = 0
                    if self.df.Discharge[i:].sum() == 0:  # If ice melted after fountain run
                        break
                    else:  # If ice melted in between fountain run
                        self.state = 0

                self.surface_area(i)

                # Precipitation to ice quantity
                if row.T_a < self.rain_temp and row.Prec > 0:

                    self.df.loc[i, "ppt"] = (
                        self.snow_fall_density
                        * row.Prec
                        * math.pi
                        * math.pow(self.df.loc[i, "r_ice"], 2)
                    )

                # Fountain water output
                self.liquid = row.Discharge * (1 - self.ftl) * self.time_steps / 60

                self.energy_balance(row)

                if self.EJoules < 0:

                    """ And fountain on """
                    if self.df.loc[i - 1, "Discharge"] > 0:

                        """Freezing water"""

                        self.liquid -= (self.EJoules) / (-self.L_f)

                        if self.liquid < 0:
                            self.liquid += (self.EJoules) / (-self.L_f)
                            self.solid += self.liquid
                            self.liquid = 0
                        else:
                            self.solid += (self.EJoules) / (-self.L_f)

                    else:
                        """ When fountain off and energy negative """
                        # Cooling Ice
                        self.delta_T_s += (self.df.loc[i, "TotalE"] * self.time_steps) / (
                            self.rho_i * self.dx * self.c_s
                        )

                else:
                    # Heating Ice
                    self.delta_T_s += (self.df.loc[i, "TotalE"] * self.time_steps) / (
                        self.rho_i * self.dx * self.c_s
                    )

                    """Hot Ice"""
                    if (self.df.loc[i - 1, "T_s"] + self.delta_T_s) > 0:
                        # Melting Ice by Temperature
                        self.solid -= (
                            (self.rho_i * self.dx * self.c_s * self.df.loc[i, "SA"])
                            * (-(self.df.loc[i - 1, "T_s"] + self.delta_T_s))
                            / (-self.L_f)
                        )

                        self.melted += (
                            (self.rho_i * self.dx * self.c_s * self.df.loc[i, "SA"])
                            * (-(self.df.loc[i - 1, "T_s"] + self.delta_T_s))
                            / (-self.L_f)
                        )

                        self.df.loc[i, "thickness"] = self.melted / (
                            self.df.loc[i, "SA"] * self.rho_i
                        )

                        self.df.loc[i - 1, "T_s"] = 0
                        self.delta_T_s = 0

                """ Quantities of all phases """
                self.df.loc[i, "T_s"] = self.df.loc[i - 1, "T_s"] + self.delta_T_s
                self.df.loc[i, "meltwater"] = self.df.loc[i - 1, "meltwater"] + self.melted
                self.df.loc[i, "ice"] = (
                    self.df.loc[i - 1, "ice"]
                    + self.solid
                    + self.df.loc[i, "ppt"]
                    + self.df.loc[i, "deposition"]
                )
                self.df.loc[i, "vapour"] = self.df.loc[i - 1, "vapour"] + self.gas
                self.df.loc[i, "water"] = self.df.loc[i - 1, "water"] + self.liquid
                self.df.loc[i, "iceV"] = self.df.loc[i, "ice"] / self.rho_i

                self.delta_T_s, self.solid, self.liquid, self.gas, self.melted, self.SRf, self.vp_ice, self.EJoules = (
                    [0] * 8
                )



if __name__ == "__main__":

    start = time.time()

    schwarzsee = Icestupa()

    # schwarzsee.derive_parameters()

    # schwarzsee.run()

    schwarzsee.melt_freeze()

    total = time.time() - start

    print("Total time : ", total / 60)
