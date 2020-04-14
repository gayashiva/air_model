import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from tqdm import tqdm
import os
import math

class Icestupa:

    """Constants"""
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
    vp_w = 6.112

    """Miscellaneous"""
    time_steps = 5 * 60  # s Model time steps
    p0 = 1013  # Standard air pressure hPa
    ftl = 0  # Fountain flight time loss ftl

    surface = dict(
        ie=0.95,  # Ice Emissivity ie
        a_i=0.35,  # Albedo of Ice a_i
        a_s=0.85,  # Albedo of Fresh Snow a_s
        decay_t=10,  # Albedo dry decay rate decay_t_d
        dx=1e-02,  # Ice layer thickness
        z0mi=0.0017,  # Ice Momentum roughness length
        z0hi=0.0017,  # Ice Scalar roughness length
        snow_fall_density=250,  # Snowfall density
        rain_temp=1,  # Temperature condition for liquid precipitation
        h_aws=3,  # m height of AWS
    )

    fountain = dict(
        aperture_f=0.005,  # Fountain aperture diameter
        h_f=1.35,  # Fountain steps h_f
        discharge=3.58,  # Fountain on discharge
        crit_temp=-5,  # Fountain runtime temperature
        latitude=46.693723,
        longitude=7.297543,
        utc_offset=1,
    )

    dirname = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


    def __init__(self, site ='schwarzsee'):

        folders = dict(
            input_folder=os.path.join(self.dirname, "data/interim/" + site),
            output_folder=os.path.join(self.dirname, "data/processed/" + site),
            sim_folder=os.path.join(self.dirname, "data/processed/" + site + "/simulations"),
        )
        input_file = folders["input_folder"] + "/input.csv"
        output_file = folders["output_folder"] + "/output.csv"

        self.state = 0
        self.df = pd.read_csv(input_file, sep=",", header=0, parse_dates= ["When"])

        i = 0
        start = 0
        while self.df.loc[i, "Discharge"] == 0:
            start = i
            i += 1

        i = start
        fountain_off = 0
        while self.df.Discharge[i:] > 0:
            fountain_off = i
            i += 1

        dates = dict(
            start_date=self.df.loc[start, "When"],
            end_date=datetime(2019, 3, 20, 18),
            fountain_off_date=self.df.loc[fountain_off, "When"],
        )

        self.df = self.df[start:]

    def SEA(self, date):

        latitude = self.fountain['latitude']
        longitude = self.fountain['longitude']
        utc_offset = self.fountain['utc_offset']
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

    def projectile_xy(self, v, g =9.81 ):
        hs = self.fountain["h_f"]
        data_xy = []
        t = 0.0
        theta_f = math.radians(45)
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

    def derive_parameters(self):

        missing = [
            "a",
            "cld",
            "SEA",
            "vp_a",
            "r_f",
            "LW_in",
        ]
        for col in missing:
            if col in list(self.df.columns):
                missing = missing.remove(col)
            else:
                self.df[col] = 0

        """Albedo Decay"""
        self.surface["decay_t"] = (
                self.surface["decay_t"] * 24 * 60 / self.time_steps
        )  # convert to 5 minute time steps
        s = 0
        f = 0
        ti = self.surface["decay_t"]
        a_min = self.surface["a_i"]

        """ Fountain Spray radius """
        Area = math.pi * math.pow(self.fountain["aperture_f"], 2) / 4

        for i in tqdm(range(1, self.df.shape[0])):

            """Solar Elevation Angle"""
            self.df.loc[i, "SEA"] = self.SEA(self.df.loc[i, "When"])

            """ Vapour Pressure"""
            if "vp_a" in missing:
                self.df.loc[i, "vp_a"] = (6.11 * math.pow(10, 7.5 * self.df.loc[i - 1, "T_a"] / (self.df.loc[i - 1, "T_a"] + 237.3)) * self.df.loc[i, "RH"] / 100)

            """LW incoming"""
            if "LW_in" in missing:

                # Cloudiness from diffuse fraction
                if self.df.loc[i, "SW_direct"] + self.df.loc[i, "SW_diffuse"] > 1:
                    self.df.loc[i, "cld"] = self.df.loc[i, "SW_diffuse"] / (
                            self.df.loc[i, "SW_direct"] + self.df.loc[i, "SW_diffuse"]
                    )
                else:
                    # Night Cloudiness average of last 8 hours
                    if i - 96 > 0:
                        for j in range(i - 96, i):
                            self.df.loc[i, "cld"] += self.df.loc[j, "cld"]
                        self.df.loc[i, "cld"] = self.df.loc[i, "cld"] / 96
                    else:
                        for j in range(0, i):
                            self.df.loc[i, "cld"] += self.df.loc[j, "cld"]
                        self.df.loc[i, "cld"] = self.df.loc[i, "cld"] / i

                self.df.loc[i, "e_a"] = ( 1.24 * math.pow(abs(self.df.loc[i, "vp_a"] / (self.df.loc[i, "T_a"] + 273.15)), 1 / 7)
                                   ) * (1 + 0.22 * math.pow(self.df.loc[i, "cld"], 2))

                self.df.loc[i, "LW_in"] = self.df.loc[i, "e_a"] * self.bc * math.pow(
                        self.df.loc[i, "T_a"] + 273.15, 4
                    )

            """Albedo"""
            # Precipitation
            if (self.df.loc[i, "Discharge"] == 0) & (self.df.loc[i, "Prec"] > 0):
                if self.df.loc[i, "T_a"] < self.surface["rain_temp"]:  # Snow
                    s = 0
                    f = 0

            if self.df.loc[i, "Discharge"] > 0:
                f = 1
                s = 0

            if f == 0:  # last snowed
                self.df.loc[i, "a"] = a_min + (surface["a_s"] - a_min) * math.exp(-s / ti)
                s = s + 1
            else:  # last sprayed
                self.df.loc[i, "a"] = a_min
                s = s + 1

            """ Fountain Spray radius """
            v_f = self.df.loc[i, "Discharge"] / (60 * 1000 * Area)
            self.df.loc[i, "r_f"] = self.projectile_xy(v_f)

        self.r_mean = (
            df['r_f'].replace(0, np.NaN).mean()
        )

    def surface_area(self, i):

        if (self.df.Discharge[i] > 0) & (self.df.loc[i - 1, "r_ice"] >= self.r_mean):
            # Ice Radius
            self.df.loc[i, "r_ice"] = self.df.loc[i - 1, "r_ice"]

            # Ice Height
            self.df.loc[i, "h_ice"] = (
                    3 * self.df.loc[i - 1, "iceV"] / (math.pi * self.df.loc[i, "r_ice"] ** 2)
            )

            # Height by Radius ratio
            self.df.loc[i, "h_r"] = self.df.loc[i - 1, "h_ice"] / self.df.loc[i - 1, "r_ice"]

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
                self.df.loc[i - 1, "iceV"] / math.pi * (3 / self.df.loc[i, "h_r"]), 1 / 3
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

        self.df.loc[i, "SRf"] = (
                                   0.5
                                   * self.df.loc[i, "h_ice"]
                                   * self.df.loc[i, "r_ice"]
                                   * math.cos(self.df.loc[i, "SEA"])
                                   + math.pi
                                   * math.pow(self.df.loc[i, "r_ice"], 2)
                                   * 0.5
                                   * math.sin(self.df.loc[i, "SEA"])
                           ) / (
                                   math.pi
                                   * math.pow(
                               (math.pow(self.df.loc[i, "h_ice"], 2) + math.pow(self.df.loc[i, "r_ice"], 2)),
                               1 / 2,
                           )
                                   * self.df.loc[i, "r_ice"]
                           )

schwarzsee = Icestupa()

schwarzsee.derive_parameters()