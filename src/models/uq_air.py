import uncertainpy as un
import chaospy as cp
import pandas as pd
import numpy as np
import os
import math

def uniform(parameter, interval):
    """
    A closure that creates a function that takes a `parameter` as input and
    returns a uniform distribution with `interval` around `parameter`.
    Parameters
    ----------
    interval : int, float
        The interval of the uniform distribution around `parameter`.
    Returns
    -------
    distribution : function
        A function that takes `parameter` as input and returns a
        uniform distribution with `interval` around this `parameter`.
    Notes
    -----
    This function ultimately calculates:
    .. code-block:: Python
        cp.Uniform(parameter - abs(interval/2.*parameter),
                   parameter + abs(interval/2.*parameter)).
    """
    if parameter == 0:
        raise ValueError("Creating a percentage distribution around 0 does not work")

    return cp.Uniform(parameter - abs(interval/2.*parameter),
                      parameter + abs(interval/2.*parameter))

    return distribution

def max_volume(time, values):
    # Calculate the feature using time, values and info.
    icev_max = values.max()
    # Return the feature times and values.
    return None, icev_max #todo include efficiency

class UQ_Icestupa(Icestupa, un.Model):

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

    def __init__(self, site='schwarzsee'):

        super(UQ_Icestupa, self).__init__(labels=["Time (days)", "Ice Volume (m3)"], interpolate=False)

        """Surface"""
        self.ie = 0.95  # Ice Emissivity ie
        self.a_i = 0.35  # Albedo of Ice a_i
        self.a_s = 0.85  # Albedo of Fresh Snow a_s
        self.decay_t = 10  # Albedo decay rate decay_t_d
        self.rain_temp = 1  # Temperature condition for self.liquid precipitation

        """Meteorological"""
        self.z0mi = 0.0017  # Ice Momentum roughness length
        self.z0hi = 0.0017  # Ice Scalar roughness length
        self.snow_fall_density = 250  # Snowfall density


        """Fountain"""
        self.aperture_f = 0.005  # Fountain aperture diameter
        self.h_f = 1.35  # Fountain steps h_f

        """Site constants"""
        self.latitude = 46.693723
        self.longitude = 7.297543
        self.utc_offset = 1

        """Miscellaneous"""
        self.ftl = 0  # Fountain flight time loss ftl,
        self.dx = 1e-02  # Ice layer thickness
        self.h_aws = 3  # m height of AWS
        self.theta_f = 45  # Fountain aperture diameter

        self.site = site

        self.state = 0

        data_store = pd.HDFStore("/home/surya/Programs/PycharmProjects/air_model/data/interim/schwarzsee/model_input.h5")
        self.df = data_store['df']
        data_store.close()

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

    def projectile_xy(self, v):
        hs = self.h_f
        data_xy = []
        g = 9.81
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
                -s / self.decay_t)
            s = s + 1
        else:  # last sprayed
            self.df.loc[i, "a"] = self.a_i

        return s, f

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

        self.SRf = (
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
                / (
                        np.log(self.h_aws / self.z0mi)
                        * np.log(self.h_aws / self.z0hi)
                )
        )

        if self.df.loc[i, "Ql"] < 0:
            self.gas -= (self.df.loc[i, "Ql"] * self.df.loc[i, "SA"] * self.time_steps) / self.L

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
                / (
                        np.log(self.h_aws / self.z0mi)
                        * np.log(self.h_aws / self.z0hi)
                )
        )

        # Short Wave Radiation SW
        self.df.loc[i, "SW"] = (1 - row.a) * (
                row.SW_direct * self.SRf + row.SW_diffuse
        )

        # Long Wave Radiation LW
        self.df.loc[i, "LW"] = row.LW_in - self.ie * self.bc * math.pow(
            self.df.loc[i - 1, "T_s"] + 273.15, 4
        )

        # Conduction Freezing
        if (self.liquid > 0) & (self.df.loc[i - 1, "T_s"] < 0):
            self.df.loc[i, "Qc"] = (
                    self.rho_i * self.dx * self.c_i * (
                -self.df.loc[i - 1, "T_s"]) / self.time_steps
            )
            self.delta_T_s = -self.df.loc[i - 1, "T_s"]

        # Total Energy W/m2
        self.df.loc[i, "TotalE"] = (
                self.df.loc[i, "SW"] + self.df.loc[i, "LW"] + self.df.loc[i, "Qs"] + self.df.loc[i, "Qc"]
        )

        # Total Energy Joules
        self.EJoules = self.df.loc[i, "TotalE"] * self.time_steps * self.df.loc[i, "SA"]

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

        self.delta_T_s, self.solid, self.liquid, self.gas, self.melted, self.SRf, self.vp_ice, self.EJoules = [0] * 8

        """Initialize"""
        self.r_mean = self.df['r_f'].replace(0, np.NaN).mean()
        self.df.loc[0, "r_ice"] = self.r_mean
        self.df.loc[0, "h_ice"] = self.dx
        self.df.loc[0, "iceV"] = self.dx * math.pi * self.df.loc[0, "r_ice"] ** 2

        for row in self.df[1:].itertuples():
            i = row.Index

            # Ice Melted
            if self.df.loc[i - 1, "iceV"] <= 0:
                self.df.loc[i - 1, "solid"] = 0
                self.df.loc[i - 1, "ice"] = 0
                self.df.loc[i - 1, "iceV"] = 0
                if self.df.Discharge[i:].sum() == 0:  # If ice melted after fountain run
                    break
                else:  # If ice melted in between fountain run
                    state = 0

            self.surface_area(i)

            # Precipitation to ice quantity
            if (row.T_a < self.rain_temp) and row.Prec > 0:

                if self.df.loc[i, 'When'] <= self.dates['fountain_off_date']:
                    self.df.loc[i, "ppt"] = (
                            self.snow_fall_density
                            * row.Prec
                            * math.pi
                            * self.r_mean ** 2)
                else:

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
                            self.df.loc[i, 'SA'] * self.rho_i)

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

            self.delta_T_s, self.solid, self.liquid, self.gas, self.melted, self.SRf, self.vp_ice, self.EJoules = [
                                                                                                                      0] * 8

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

        return self.df.index.values / 24, self.df["iceV"].valuesz

list_of_feature_functions = [max_volume]

features = un.Features(new_features=list_of_feature_functions,
                       features_to_run=["max_volume"])


# # Set all parameters to have a uniform distribution
# # within a 20% interval around their fixed value
# parameters.set_all_distributions(un.uniform(0.05))

ie = 0.95
a_i =0.35
a_s = 0.85
decay_t = 10

interval = 0.05

ie_dist = uniform(ie, interval)
a_i_dist = uniform(a_i, interval)
a_s_dist = uniform(a_s, interval)
decay_t_dist = uniform(decay_t, interval)

rain_temp_dist = cp.Uniform(0, 2)
z0mi_dist = cp.Uniform(0.0007, 0.0027)
z0hi_dist = cp.Uniform(0.0007, 0.0027)
snow_fall_density_dist = cp.Uniform(200, 300)

interval = 0.01

aperture_f_dist = uniform(0.005, interval)
height_f_dist = uniform(1.35, interval)

dx_dist = cp.Uniform(0.0001, 0.01)

parameters = {
                "ie": ie_dist,
                "a_i": a_i_dist,
                "a_s": a_s_dist,
                "decay_t": decay_t_dist,
                "dx": dx_dist
}

# parameters = {
#               "rain_temp": rain_temp_dist,
#               "z0mi": z0mi_dist,
#               "z0hi": z0hi_dist,
#               "snow_fall_density": snow_fall_density_dist
#               }

# parameters = {
#               "aperture_f": aperture_f_dist,
#               "height_f": height_f_dist,
#               }


# Create the parameters
parameters = un.Parameters(parameters)

# Initialize the model
model = UQ_Icestupa()

# Set up the uncertainty quantification
UQ = un.UncertaintyQuantification(model=model,
                                  parameters=parameters,
                                  features=features,
                                  )

# Perform the uncertainty quantification using
# polynomial chaos with point collocation (by default)
data = UQ.quantify(data_folder = "/home/surya/Programs/PycharmProjects/air_model/data/processed/schwarzsee/simulations/data/",
                    figure_folder="/home/surya/Programs/PycharmProjects/air_model/data/processed/schwarzsee/simulations/figures/",
                    filename="Surface2")

# data = UQ.quantify(filename="Meteorological")

