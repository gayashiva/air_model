import pandas as pd
import streamlit as st
import sys
from datetime import datetime
from tqdm import tqdm
import os
import math
import time
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText
from matplotlib.ticker import AutoMinorLocator
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
from pvlib import location

dirname = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

sys.path.append(dirname)
from src.data.settings import config
import logging
import coloredlogs

# Required for colored logging statements
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
coloredlogs.install(
    # fmt="%(name)s %(levelname)s %(message)s",
    fmt="%(levelname)s %(message)s",
    logger=logger,
)
logger.debug("Model begins")

class Icestupa:
    """Physical Constants"""
    L_S = 2848 * 1000  # J/kg Sublimation
    L_E = 2514 * 1000  # J/kg Evaporation
    L_F = 334 * 1000  # J/kg Fusion
    C_A = 1.01 * 1000  # J/kgC Specific heat air
    C_W = 4.186 * 1000  # J/kgC Specific heat water
    C_I = 2.097 * 1000  # J/kgC Specific heat ice
    RHO_W = 1000  # Density of water
    RHO_I = 917  # Density of Ice RHO_I
    RHO_A = 1.29  # kg/m3 air density at mean sea level
    VAN_KARMAN = 0.4  # Van Karman constant
    K_I = 2.123  # Thermal Conductivity Waite et al. 2006
    STEFAN_BOLTZMAN = 5.670367 * math.pow(10, -8)  # Stefan Boltzman constant
    P0 = 1013  # Standard air pressure hPa
    G = 9.81 #Gravitational acceleration

    """Surface Properties"""
    IE = 0.95  # Ice Emissivity IE
    A_I = 0.35  # Albedo of Ice A_I
    A_S = 0.85  # Albedo of Fresh Snow A_S
    T_DECAY = 10  # Albedo decay rate decay_t_d
    Z_I = 0.0017  # Ice Momentum and Scalar roughness length
    T_RAIN = 1  # Temperature condition for liquid precipitation

    """Model constants"""
    DX = 5e-03  # Initial Ice layer thickness

    def __init__(self, *initial_data, **kwargs):
        # Initialise all variables of dictionary
        for dictionary in initial_data:
            for key in dictionary:
                setattr(self, key, dictionary[key])
                logger.info(f"%s -> %s" % (key, str(dictionary[key])))
        # Initialise other variables
        for key in kwargs:
            setattr(self, key, kwargs[key])

        #Define directory structure
        self.raw_folder = os.path.join(dirname, "data/" + self.name + "/raw/")
        self.input_folder = os.path.join(dirname, "data/" + self.name + "/interim/")
        self.output_folder = os.path.join(dirname, "data/" + self.name + "/processed/")
        self.sim_folder = os.path.join(
            dirname, "data/" + self.name + "/processed/simulations"
        )

        #Initialize input dataset
        input_file = self.input_folder + self.name + "_input_model.csv"
        self.df = pd.read_csv(input_file, sep=",", header=0, parse_dates=["When"])
        self.TIME_STEP = int(pd.infer_freq(self.df["When"])[:-1]) * 60 # Extract time step from datetime column
        logger.debug(f"Time steps -> %s minutes" % (str(self.TIME_STEP / 60)))
        mask = self.df["When"] >= self.start_date
        self.df = self.df.loc[mask]
        self.df = self.df.reset_index(drop=True)

        # Add Validation data to input
        if self.name in ['guttannen']:
            df_v = pd.read_csv(self.input_folder + self.name + "_drone.csv", sep=",", header=0, parse_dates=["When"])
            df_v = df_v.set_index('When')
            # df_v = df_v.astype(float)
            self.df = self.df.set_index('When')
            self.df['DroneV'] = df_v['DroneV']
            self.df = self.df.reset_index()

    @st.cache
    def get_parameter_metadata(self, parameter): # Provides Metadata of all input and Output variables
        return {
            "When": {
                "name": "Timestamp",
                "kind": "Misc",
                "units": "()",
            },
            "DroneV": {
                "name": "Drone Validation",
                "kind": "Derived",
                "units": "($m^3$)",
            },
            "cld": {
                "name": "Cloudiness",
                "kind": "Derived",
                "units": "()",
            },
            "missing": {
                "name": "Filled from ERA5",
                "kind": "Derived",
                "units": "()",
            },
            "e_a": {
                "name": "Atmospheric Emissivity",
                "kind": "Derived",
                "units": "()",
            },
            "vp_a": {
                "name": "Air Vapour Pressure",
                "kind": "Derived",
                "units": "($hPa$)",
            },
            "vp_ice": {
                "name": "Ice Vapour Pressure",
                "kind": "Derived",
                "units": "($hPa$)",
            },
            "solid": {
                "name": "Ice per time step",
                "kind": "Derived",
                "units": "($kg$)",
            },
            "melted": {
                "name": "Melt per time step",
                "kind": "Derived",
                "units": "($kg$)",
            },
            "gas": {
                "name": "Vapour per time step",
                "kind": "Derived",
                "units": "($kg$)",
            },
            "thickness": {
                "name": "Ice thickness",
                "kind": "Derived",
                "units": "($m$)",
            },
            "Discharge": {
                "name": "Fountain Spray",
                "kind": "Input",
                "units": "($l\\, min^{-1}$)",
            },
            "T_a": {
                "name": "Temperature",
                "kind": "Input",
                "units": "($\\degree C$)",
            },
            "delta_T_s": {
                "name": "Temperature change per time step",
                "kind": "Derived",
                "units": "($\\degree C$)",
            },
            "RH": {
                "name": "Relative Humidity",
                "kind": "Input",
                "units": "($\\%$)",
            },
            "p_a": {
                "name": "Pressure",
                "kind": "Input",
                "units": "($hPa$)",
            },
            "SW_direct": {
                "name": "Shortwave Direct",
                "kind": "Input",
                "units": "($W\\,m^{-2}$)",
            },
            "SW": {
                "name": "Shortwave Radiation",
                "kind": "Output",
                "units": "($W\\,m^{-2}$)",
            },
            "SW_diffuse": {
                "name": "Shortwave Diffuse",
                "kind": "Input",
                "units": "($W\\,m^{-2}$)",
            },
            "LW_in": {
                "name": "Longwave",
                "kind": "Input",
                "units": "($W\\,m^{-2}$)",
            },
            "LW": {
                "name": "Longwave Radiation",
                "kind": "Output",
                "units": "($W\\,m^{-2}$)",
            },
            "Qs": {
                "name": "Sensible Heat flux",
                "kind": "Output",
                "units": "($W\\,m^{-2}$)",
            },
            "Ql": {
                "name": "Latent Heat flux",
                "kind": "Output",
                "units": "($W\\,m^{-2}$)",
            },
            "Qf": {
                "name": "Fountain water heat flux",
                "kind": "Output",
                "units": "($W\\,m^{-2}$)",
            },
            "Qg": {
                "name": "Bulk Icestupa heat flux",
                "kind": "Output",
                "units": "($W\\,m^{-2}$)",
            },
            "$q_{T}$": {
                "name": "Temperature flux",
                "kind": "Derived",
                "units": "($W\\,m^{-2}$)",
            },
            "$q_{melt}$": {
                "name": "Melt energy",
                "kind": "Derived",
                "units": "($W\\,m^{-2}$)",
            },
            "Prec": {
                "name": "Precipitation",
                "kind": "Input",
                "units": "($mm$)",
            },
            "v_a": {
                "name": "Wind Speed",
                "kind": "Input",
                "units": "($m\\,s^{-1}$)",
            },
            "iceV": {
                "name": "Ice Volume",
                "kind": "Output",
                "units": "($m^3$)",
            },
            "ice": {
                "name": "Ice Mass",
                "kind": "Output",
                "units": "($kg$)",
            },
            "a": {
                "name": "Albedo",
                "kind": "Derived",
                "units": "()",
            },
            "f_cone": {
                "name": "Solar Surface Area Fraction",
                "kind": "Derived",
                "units": "()",
            },
            "s_cone": {
                "name": "Ice Cone Slope",
                "kind": "Derived",
                "units": "()",
            },
            "h_ice": {
                "name": "Ice Cone Height",
                "kind": "Output",
                "units": "($m$)",
            },
            "r_ice": {
                "name": "Ice Cone Radius",
                "kind": "Output",
                "units": "($m$)",
            },
            "T_s": {
                "name": "Surface Temperature",
                "kind": "Output",
                "units": "($\\degree C$)",
            },
            "T_bulk": {
                "name": "Bulk Temperature",
                "kind": "Output",
                "units": "($\\degree C$)",
            },
            "sea": {
                "name": "Solar Elevation Angle",
                "kind": "Derived",
                "units": "($\\degree$)",
            },
            "TotalE": {
                "name": "Net Energy",
                "kind": "Output",
                "units": "($W\\,m^{-2}$)",
            },
            "ppt": {
                "name": "Snow Accumulation",
                "kind": "Output",
                "units": "($kg$)",
            },
            "cdt": {
                "name": "Condensation",
                "kind": "Output",
                "units": "($kg$)",
            },
            "dpt": {
                "name": "Deposition",
                "kind": "Output",
                "units": "($kg$)",
            },
            "vapour": {
                "name": "Vapour loss",
                "kind": "Output",
                "units": "($kg$)",
            },
            "meltwater": {
                "name": "Meltwater",
                "kind": "Output",
                "units": "($kg$)",
            },
            "Input": {
                "name": "Water Input",
                "kind": "Output",
                "units": "($kg$)",
            },
            "unfrozen_water": {
                "name": "Water Runoff",
                "kind": "Output",
                "units": "($kg$)",
            },
            "SA": {
                "name": "Surface Area",
                "kind": "Output",
                "units": "($m^2$)",
            },
            "input": {
                "name": "Mass Input",
                "kind": "Misc",
                "units": "($kg$)",
            },
        }[parameter]

    def get_solar(self): # Provides solar angle for each time step

        site_location = location.Location(self.latitude, self.longitude)

        times = pd.date_range(
            start=self.start_date,
            end=self.df["When"].iloc[-1],
            freq=(str(int(self.TIME_STEP / 60)) + "T"),
        )

        # Get solar azimuth and zenith to pass to the transposition function
        solar_position = site_location.get_solarposition(
            times=times, method="ephemeris"
        )

        solar_df = pd.DataFrame(
            {
                # "ghics": clearsky["ghi"],
                # "difcs": clearsky["dhi"],
                # "zen": solar_position["zenith"],
                "sea": np.radians(solar_position["elevation"]),
            }
        )
        solar_df.loc[solar_df["sea"] < 0, "sea"] = 0
        solar_df.index = solar_df.index.set_names(["When"])
        solar_df = solar_df.reset_index()

        self.df = pd.merge(solar_df, self.df, on="When")

    def projectile_xy(self, v, h=0): # standard projectile equation function that returns range of projectile
        if h == 0:
            hs = self.h_f
        else:
            hs = h
        data_xy = []
        t = 0.0
        theta_f = math.radians(self.theta_f)
        while True:
            # now calculate the height y
            y = hs + (t * v * math.sin(theta_f)) - (self.G * t * t) / 2
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

    def albedo(self, row, s=0, f=0): # Albedo Scheme described in Section 3.2.1

        i = row.Index

        """Albedo"""
        # Precipitation event
        if (row.Discharge == 0) & (row.Prec > 0):
            if row.T_a < self.T_RAIN:  # Snow event
                s = 0
                f = 0

        if row.Discharge > 0: # Spray event
            f = 1
            s = 0

        if f == 0:  # last snowed
            self.df.loc[i, "a"] = self.A_I + (self.A_S - self.A_I) * math.exp(
                -s / self.T_DECAY
            )
            s = s + 1
        else:  # last sprayed

            self.df.loc[i, "a"] = self.A_I

        return s, f

    def spray_radius(self, r_mean=0): # Provides spray radius assuming projectile motion of water droplets
        Area = math.pi * math.pow(self.dia_f, 2) / 4

        if r_mean != 0:
            self.r_mean = r_mean
        else:
            self.v = self.discharge / (60 * 1000 * Area)
            self.r_mean = self.projectile_xy(v=self.v)

        logger.info("Spray radius %s" % (self.r_mean))
        return self.r_mean

    def height_steps(self, i): # Updates discharge based on new fountain height
        h_steps = 1
        self.df.loc[i:, "Discharge"] /= self.discharge
        if self.name != 'guttannen':
            if self.discharge != 0:
                Area = math.pi * math.pow(self.dia_f, 2) / 4
                if self.discharge < 6:
                    discharge = 0
                    self.v = 0
                else:
                    self.v = np.sqrt(self.v ** 2 - 2 * self.G * h_steps)
                    discharge = self.v * (60 * 1000 * Area)
                logger.warning(
                    "Discharge changed from %.2f to %.2f" % (self.discharge, discharge)
                )
                self.discharge = discharge

        self.df.loc[i:, "Discharge"] *= self.discharge

    def discharge_rate(self): # Provides discharge info based on trigger setting

        self.df["Discharge"] = 0

        if self.trigger == "Temperature":
            self.df["Prec"] = 0
            mask = (self.df["T_a"] < self.crit_temp) & (self.df["SW_direct"] < 100)
            mask_index = self.df[mask].index
            self.df.loc[mask_index, "Discharge"] = 1 * self.discharge

            logger.debug(
                f"Hours of spray : %.2f"
                % (self.df.Discharge.astype(bool).sum(axis=0) * self.TIME_STEP / 3600)
            )

        if self.trigger == "NetEnergy":

            col = [
                "T_s",  # Surface Temperature
                "T_bulk",  # Bulk Temperature
                "f_cone",
                "TotalE",
                "SW",
                "LW",
                "Qs",
                "Ql",
                "Qf",
                "Qg",
                "ppt",
                "dpt",
                "cdt",
            ]

            for column in col:
                self.df[column] = 0

            self.df.Discharge = 0
            logger.debug("Calculating discharge from energy trigger ...")
            for row in tqdm(self.df[1:-1].itertuples(), total=self.df.shape[0]):
                self.energy_balance(row, mode="trigger")
            mask = self.df["TotalE"] < 0
            mask_index = self.df[mask].index
            self.df.loc[mask_index, "Discharge"] = 1 * self.discharge

            col = [
                "T_s",  # Surface Temperature
                "T_bulk",  # Bulk Temperature
                "f_cone",
                "TotalE",
                "SW",
                "LW",
                "Qs",
                "Ql",
                "Qf",
                "Qg",
                "ppt",
                "dpt",
                "cdt",
            ]
            self.df.drop(columns=col)

            logger.debug(
                f"Hours of spray : %.2f"
                % (self.df.Discharge.astype(bool).sum(axis=0) * self.TIME_STEP / 3600)
            )

        if self.trigger == "Manual" and self.name == "guttannen":
            self.df["Discharge"] = self.discharge

        if self.trigger == "Manual" and self.name == "schwarzsee":

            # self.start_date=datetime(2021, 1, 30, 17)
            # self.end_date=self.df.When.tail(1).values[0]
            mask = self.df["When"] >= self.start_date
            self.df = self.df.loc[mask]
            self.df = self.df.reset_index(drop=True)
            logger.warning(f"Start date changed to %s" % (self.start_date))

            df_f = pd.read_csv(
                os.path.join(dirname, "data/" + "schwarzsee" + "/interim/")
                + "schwarzsee_input_field.csv"
            )
            df_f["When"] = pd.to_datetime(df_f["When"], format="%Y.%m.%d %H:%M:%S")
            # df_f['When'] = df_f['When'].mask(df_f['When'].dt.year == 2019, df_f['When'] + pd.offsets.DateOffset(year=2021))
            # df_f['When'] = df_f['When'] - pd.DateOffset(10)
            # mask = (df_f["When"] >= self.df.When[0]) & (
            #     df_f["When"] <= self.end_date
            # )
            # df_f = df_f.loc[mask]
            # df_f = df_f.reset_index(drop = True)
            df_f = (
                df_f.set_index("When")
                .resample(str(int(self.TIME_STEP / 60)) + "T")
                .mean()
            )
            self.df = self.df.set_index("When")
            mask = df_f["Discharge"] != 0
            mask_index = df_f[mask].index
            self.df.loc[mask_index, "Discharge"] = df_f["Discharge"]
            self.df = self.df.reset_index()
            self.df["Discharge"] = self.df.Discharge.replace(np.nan, 0)
            logger.debug(
                f"Hours of spray : %.2f"
                % (self.df.Discharge.astype(bool).sum(axis=0) * self.TIME_STEP / 3600)
            )

        mask = (self.df["When"] > self.fountain_off_date)
        mask_index = self.df[mask].index
        self.df.loc[mask_index, "Discharge"] = 0
        # if self.trigger == "Manual" and self.name != "schwarzsee":
        #     logger.error("Manual discharge information does not exist")
        #     st.write("Manual discharge information does not exist")
        #     sys.exit()

    def derive_parameters(self): # Derives additional parameters required for simulation
        unknown = ["a", "vp_a", "LW_in", "cld"] # Possible unknown variables
        for i in range(len(unknown)):
            if unknown[i] in list(self.df.columns):
                unknown[i] = np.NaN # Removes known variable
            else:
                logger.warning("%s is unknown" % (unknown[i]))
                self.df[unknown[i]] = 0

        logger.debug("Creating model input file...")
        for row in tqdm(self.df[1:].itertuples(), total=self.df.shape[0]):
            i = row.Index

            """ Vapour Pressure"""
            if "vp_a" in unknown:
                self.df.loc[row.Index, "vp_a"] = (
                    6.107
                    * math.pow(
                        10,
                        7.5
                        * self.df.loc[row.Index, "T_a"]
                        / (self.df.loc[row.Index, "T_a"] + 237.3),
                    )
                    * row.RH
                    / 100
                )

            """LW incoming"""
            if "LW_in" in unknown:

                self.df.loc[row.Index, "e_a"] = (
                    1.24
                    * math.pow(
                        abs(self.df.loc[row.Index, "vp_a"] / (row.T_a + 273.15)), 1 / 7
                    )
                ) * (1 + 0.22 * math.pow(self.df.loc[row.Index, "cld"], 2))

                self.df.loc[row.Index, "LW_in"] = (
                    self.df.loc[row.Index, "e_a"]
                    * self.STEFAN_BOLTZMAN
                    * math.pow(row.T_a + 273.15, 4)
                )

        self.discharge_rate()
        self.f_on = self.df.When[self.df.Discharge.astype(bool)].tolist() # List of all timesteps when fountain on
        self.start_date = self.f_on[0]
        logger.warning("Fountain ends %s" % self.f_on[-1])
        # self.end_date =  self.f_on[-1] #+ timedelta(days=30)
        logger.info("Model starts at %s" % (self.start_date))
        # logger.info("Model ends at %s" % (self.end_date))

        mask = self.df["When"] >= self.start_date
        self.df = self.df.loc[mask]
        self.df = self.df.reset_index(drop=True)

        self.get_solar()
        self.df.Prec = self.df.Prec * self.TIME_STEP #mm
        if self.name == 'guttannen':
            self.df.a = self.A_I
        else:
            """Albedo Decay parameters initialized"""
            self.T_DECAY = self.T_DECAY * 24 * 60 * 60 / self.TIME_STEP
            s = 0
            f = 0
            for row in tqdm(self.df[1:].itertuples(), total=self.df.shape[0]):
                s, f = self.albedo(row, s, f)

        self.df = self.df.round(3)
        self.df = self.df[self.df.columns.drop(list(self.df.filter(regex="Unnamed")))] # Remove junk columns

        self.df.to_hdf(
            self.input_folder + "model_input_" + self.trigger + ".h5",
            key="df",
            mode="w",
        )
        self.df.to_csv(self.input_folder + "model_input_" + self.trigger + ".csv")

    def surface_area(self, i):

        if (
            self.df.solid[i - 1]
            - self.df.melted[i - 1]
            > 0
        ) & (self.df.loc[i - 1, "r_ice"] > self.r_mean): # Growth rate positive and radius goes beyond spray radius
            self.df.loc[i, "r_ice"] = self.df.loc[i - 1, "r_ice"]

            self.df.loc[i, "h_ice"] = (
                3 * self.df.loc[i, "iceV"] / (math.pi * self.df.loc[i, "r_ice"] ** 2)
            )

            self.df.loc[i, "s_cone"] = (
                self.df.loc[i - 1, "h_ice"] / self.df.loc[i - 1, "r_ice"]
            )

        else:

            # Maintain constant Height to radius ratio 
            self.df.loc[i, "s_cone"] = self.df.loc[i - 1, "s_cone"]
            # self.df.loc[i, "s_cone"] = self.h_f/self.r_mean

            # Ice Radius
            # logger.warning("%s,%s" %(self.df.loc[i, "iceV"], self.df.loc[i, "s_cone"]))
            self.df.loc[i, "r_ice"] = math.pow(
                self.df.loc[i, "iceV"] / math.pi * (3 / self.df.loc[i, "s_cone"]), 1 / 3
            )

            # Ice Height
            self.df.loc[i, "h_ice"] = self.df.loc[i, "s_cone"] * self.df.loc[i, "r_ice"]

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

        self.df.loc[i, "f_cone"] = (
            0.5
            * self.df.loc[i, "h_ice"]
            * self.df.loc[i, "r_ice"]
            * math.cos(self.df.loc[i, "sea"])
            + math.pi
            * math.pow(self.df.loc[i, "r_ice"], 2)
            * 0.5
            * math.sin(self.df.loc[i, "sea"])
        ) / self.df.loc[i, "SA"]

    def energy_balance(self, row, mode="normal"):
        i = row.Index

        if mode == "trigger": # Used while deriving discharge rate
            self.df.loc[i, "T_s"] = 0
            self.df.loc[i, "Qf"] = 0
            self.df.loc[i, "Qg"] = 0
            self.liquid = 0
            self.df.loc[i, "f_cone"] = 1

        self.df.loc[i, "vp_ice"] = (
            (
                1.0016
                + 3.15 * math.pow(10, -6) * self.df.loc[i, "p_a"]
                - 0.074 * math.pow(self.df.loc[i, "p_a"], -1)
            )
            * 6.112
            * np.exp(
                22.46 * (self.df.loc[i, "T_s"]) / ((self.df.loc[i, "T_s"]) + 272.62)
            )
        )

        if mode != "trigger":
            self.df.loc[i, "Ql"] = (
                0.623
                * self.L_S
                * self.RHO_A
                / self.P0
                * math.pow(self.VAN_KARMAN, 2)
                * self.df.loc[i, "v_a"]
                * (row.vp_a - self.df.loc[i, "vp_ice"])
                / ((np.log(self.h_aws / self.Z_I)) ** 2)
            )

        # Sensible Heat Qs
        self.df.loc[i, "Qs"] = (
            self.C_A
            * self.RHO_A
            * row.p_a
            / self.P0
            * math.pow(self.VAN_KARMAN, 2)
            * self.df.loc[i, "v_a"]
            * (self.df.loc[i, "T_a"] - self.df.loc[i, "T_s"])
            / ((np.log(self.h_aws / self.Z_I)) ** 2)
        )

        # Short Wave Radiation SW
        if mode != "trigger":
            self.df.loc[i, "SW"] = (1 - row.a) * (
                row.SW_direct * self.df.loc[i, "f_cone"] + row.SW_diffuse
            )
        else:
            self.df.loc[i, "SW"] = (1 - self.A_I) * (row.SW_direct + row.SW_diffuse)

        # Long Wave Radiation LW
        self.df.loc[i, "LW"] = row.LW_in - self.IE * self.STEFAN_BOLTZMAN * math.pow(
            self.df.loc[i, "T_s"] + 273.15, 4
        )

        if np.isnan(self.df.loc[i, "LW"]):
            logger.error(
                f"When {self.df.When[i]},LW {self.df.LW[i]}, LW_in {self.df.LW_in[i]}, T_s {self.df.T_s[i - 1]}"
            )
            sys.exit("LW nan")

        if self.liquid > 0:
            self.df.loc[i, "Qf"] = (
                (self.df.loc[i - 1, "solid"])
                * self.C_W
                * self.T_w
                / (self.TIME_STEP * self.df.loc[i, "SA"])
            )

            self.df.loc[i, "Qf"] += (
                self.RHO_I
                * self.DX
                * self.C_I
                * (self.df.loc[i, "T_s"])
                / self.TIME_STEP
            )

        if mode == "normal":
            self.df.loc[i, "Qg"] = (
                self.K_I
                * (self.df.loc[i, "T_bulk"] - self.df.loc[i, "T_s"])
                / (self.df.loc[i, "r_ice"] / 2)
            )

            # Bulk Temperature
            self.df.loc[i + 1, "T_bulk"] = self.df.loc[i, "T_bulk"] - self.df.loc[
                i, "Qg"
            ] * self.TIME_STEP * self.df.loc[i, "SA"] / (
                self.df.loc[i, "ice"] * self.C_I
            )

        # Total Energy W/m2
        self.df.loc[i, "TotalE"] = (
            self.df.loc[i, "SW"]
            + self.df.loc[i, "LW"]
            + self.df.loc[i, "Qs"]
            + self.df.loc[i, "Qf"]
            + self.df.loc[i, "Qg"]
        )

        if np.isnan(self.df.loc[i, "TotalE"]):
            logger.error(
                f"When {self.df.When[i]}, SW {self.df.SW[i]}, LW {self.df.LW[i]}, Qs {self.df.Qs[i]}, Qf {self.df.Qf[i]}, Qg {self.df.Qg[i]}"
            )
            sys.exit("Energy nan")

    def summary(self): # Summaries results and saves output

        self.df = self.df[self.df.columns.drop(list(self.df.filter(regex="Unnamed")))] # Drops garbage columns
        Efficiency = (
            (self.df["meltwater"].iloc[-1] + self.df["ice"].iloc[-1])
            / self.df["input"].iloc[-1]
            * 100
        )

        Duration = self.df.index[-1] * 5 / (60 * 24)

        print("\nIce Volume Max", float(self.df["iceV"].max()))
        print("Fountain efficiency", Efficiency)
        print("Ice Mass Remaining", self.df["ice"].iloc[-1])
        print("Meltwater", self.df["meltwater"].iloc[-1])
        print("Ppt", self.df["ppt"].sum())
        print("Deposition", self.df["dpt"].sum())
        print("Duration", Duration)

        # Full Output
        filename4 = self.output_folder + "model_output_" + self.trigger + ".csv"
        self.df.to_csv(filename4, sep=",")
        self.df.to_hdf(
            self.output_folder + "model_output_" + self.trigger + ".h5",
            key="df",
            mode="w",
        )

    def read_input(self): # Use processed input dataset

        self.df = pd.read_hdf(
            self.input_folder + "model_input_" + self.trigger + ".h5", "df"
        )

        logger.debug(self.df.head())

        if self.df.isnull().values.any():
            logger.warning("Warning: Null values present")

    def read_output(self): # Reads output and Displays outputs useful for manuscript

        self.df = pd.read_hdf(
            self.output_folder + "model_output_" + self.trigger + ".h5", "df"
        )

        if self.df.isnull().values.any():
            logger.warning("Warning: Null values present")

        # Table outputs
        # f_off = self.df.index[self.df.Discharge.astype(bool)].tolist()
        # f_off = f_off[-1] + 1
        # logger.debug(
        #     "When %s \n M_U %.2f\n M_solid %.2f\n M_gas %.2f\n M_liquid %.2f\n M_R %.2f\n M_D %.2f\n M_C %.2f\n M_F %.2f\n"
        #     % (
        #         self.df.When.iloc[f_off],
        #         self.df.unfrozen_water.iloc[f_off],
        #         self.df.ice.iloc[f_off],
        #         self.df.vapour.iloc[f_off],
        #         self.df.meltwater.iloc[f_off],
        #         self.df.ppt[: f_off + 1].sum(),
        #         self.df.dpt[: f_off + 1].sum(),
        #         self.df.cdt[: f_off + 1].sum(),
        #         self.df.Discharge[: f_off + 1].sum() * 5
        #         + self.df.iceV.iloc[0] * self.RHO_I,
        #     )
        # )
        logger.debug(
            f"M_input {self.df.input.iloc[-1]}\n M_R {self.df.ppt.sum()}\n M_D {self.df.dpt.sum()}\n M_C {self.df.cdt.sum()}\n M_F {self.df.Discharge.sum() * 5 + self.df.iceV.iloc[0] * self.RHO_I}\n"
        )
        logger.debug(
            f"When {self.df.When.iloc[-1]}\n M_U {self.df.unfrozen_water.iloc[-1]},\n M_solid {self.df.ice.iloc[-1]},\n M_gas {self.df.vapour.iloc[-1]},\n M_liquid {self.df.meltwater.iloc[-1]}"
        )
        logger.debug(
            f"Temperature of spray : %.2f"
            % (self.df.loc[self.df["TotalE"] < 0, "T_a"].mean())
        )
        logger.debug(f"Temperature minimum : {self.df.T_a.min()}")
        logger.debug(f"Energy mean : {self.df.TotalE.mean()}")

        dfd = self.df.set_index("When").resample("D").mean().reset_index(drop=True)
        dfh = self.df.set_index("When").resample("H").mean().reset_index(drop=True)

        dfd["Global"] = dfd["SW_diffuse"] + dfd["SW_direct"]
        logger.debug(f"Global max: {dfd.Global.max()}\n SW max: {dfd.SW.max()}")
        logger.debug(f"Nonzero Qf: {self.df.Qf.astype(bool).sum(axis=0)}")
        Total = (
            dfd.SW.abs().sum()
            + dfd.LW.abs().sum()
            + dfd.Qs.abs().sum()
            + dfd.Ql.abs().sum()
            + dfd.Qf.abs().sum()
            + dfd.Qg.abs().sum()
            + dfd["$q_{melt}$"].abs().sum()
            + dfd["$q_{T}$"].abs().sum()
        )
        logger.debug(
            "Qmelt: %.2f, Qt: %.2f" % (dfd["$q_{melt}$"].mean(), dfd["$q_{T}$"].mean())
        )
        logger.debug(
            "Percent of Qmelt: %.2f \n Qt: %.8f"
            % (
                dfd["$q_{melt}$"].abs().sum() / Total,
                dfd["$q_{T}$"].abs().sum() / Total,
            )
        )
        logger.debug(
            f"% of SW {dfd.SW.abs().sum()/Total}, LW {dfd.LW.abs().sum()/Total}, Qs {dfd.Qs.abs().sum()/Total}, Ql {dfd.Ql.abs().sum()/Total}, Qf {dfd.Qf.abs().sum()/Total}, Qg {dfd.Qg.abs().sum()/Total}"
        )

        logger.debug(
            f"Mean of SW {dfd.SW.mean()}, LW {dfd.LW.mean()}, Qs {dfd.Qs.mean()}, Ql {dfd.Ql.mean()}, Qf {dfd.Qf.mean()}, Qg {dfd.Qg.mean()}"
        )
        logger.debug(
            f"Range of SW {dfd.SW.min()}-{dfd.SW.max()}, LW {dfd.LW.min()}-{dfd.LW.max()}, Qs {dfd.Qs.min()}-{dfd.Qs.max()}, Ql {dfd.Ql.min()}-{dfd.Ql.max()}, Qf {dfd.Qf.min()}-{dfd.Qf.max()}, Qg {dfd.Qg.min()}-{dfd.Qg.max()}"
        )
        logger.debug(f"Max SA {self.df.SA.max()}")
        logger.debug(
            f"M_input {self.df.input.iloc[-1]}, M_R {self.df.ppt.sum()}, M_D {self.df.dpt.sum()}, M_F {self.df.Discharge.sum() * 5 + self.df.iceV.iloc[0] * self.RHO_I}"
        )
        logger.debug(
            f"Max_growth {self.df.solid.max() / 5}, Mean_growth {self.df.solid.replace(0,np.NaN).mean() / 5},average_discharge {self.df.Discharge.replace(0, np.NaN).mean()}"
        )

        logger.debug(f"Duration {self.df.index[-1] * 5 / (60 * 24)}")
        logger.debug(f"Ended {self.df.When.iloc[-1]}")
        logger.debug(
            f"Evaporation/Condensation: %.2f, Sublimation/Deposition: %.2f"
            % (
                self.df.loc[self.df["RH"] > 60, "RH"].count() / self.df.index[-1],
                self.df.loc[self.df["RH"] <= 60, "RH"].count() / self.df.index[-1],
            )
        )
        logger.warning(
            f"Q_net below zero time: %.2f, Fountain time: %.2f"
            % (
                dfh.loc[dfh["TotalE"] < 0, "TotalE"].count(),
                dfh.loc[dfh["Discharge"] > 0, "Discharge"].count(),
            )
        )

        # # Output for manim
        # filename2 = os.path.join(self.output_folder, self.name + "_model_gif.csv")
        # self.df["h_f"] = self.h_f
        # cols = ["When", "h_ice", "h_f", "r_ice", "ice", "T_a", "Discharge"]
        # self.df[cols].to_csv(filename2, sep=",")

    def melt_freeze(self): # Main function

        col = [
            "T_s",
            "T_bulk",
            "f_cone",
            "ice",
            "iceV",
            "solid",
            "gas",
            "vapour",
            "melted",
            "delta_T_s",
            "unfrozen_water",
            "TotalE",
            "SW",
            "LW",
            "Qs",
            "Ql",
            "Qf",
            "Qg",
            "meltwater",
            "SA",
            "h_ice",
            "r_ice",
            "ppt",
            "dpt",
            "cdt",
            "$q_{T}$",
            "$q_{melt}$",
        ]

        for column in col:
            self.df[column] = 0

        self.liquid = [0] * 1
        STATE = 0
        self.start = 0
        ctr = 0

        logger.debug("AIR simulation begins...")
        for row in tqdm(self.df[1:-1].itertuples(), total=self.df.shape[0]):
            i = row.Index
            ice_melted = self.df.loc[i, "ice"] < 0.001

            if ice_melted and STATE == 1: # Break loop when ice melted and simulation done
                self.df.loc[i - 1, "meltwater"] += self.df.loc[i - 1, "ice"]
                self.df.loc[i - 1, "ice"] = 0
                logger.info("Model ends at %s" % (self.df.When[i]))
                self.df = self.df[self.start : i - 1]
                self.df = self.df.reset_index(drop=True)
                break

            if self.df.Discharge[i] > 0 and STATE == 0:
                STATE = 1

                # Special Initialisaton for specific sites
                if self.name == "schwarzsee":
                    self.df.loc[i - 1, "r_ice"] = self.spray_radius()
                    self.df.loc[i - 1, "h_ice"] = self.DX

                if self.name == "guttannen":
                    self.spray_radius()
                    if self.h_i !=0:
                        self.df.loc[i - 1, "h_ice"] = self.h_i
                        self.df.loc[i - 1, "r_ice"] = self.r_i
                    else:
                        self.df.loc[i - 1, "h_ice"] = self.DX
                        self.df.loc[i - 1, "r_ice"] = self.spray_radius()

                new_sites = ["hial", "gangles", "secmol", "leh"]
                if self.name in new_sites:
                    self.df.loc[i - 1, "r_ice"] = self.spray_radius()
                    self.df.loc[i - 1, "h_ice"] = self.DX

                self.df.loc[i - 1, "s_cone"] = (
                    self.df.loc[i - 1, "h_ice"] / self.df.loc[i - 1, "r_ice"]
                )
                self.df.loc[i, "iceV"] = (
                    math.pi
                    / 3
                    * self.df.loc[i - 1, "r_ice"] ** 2
                    * self.df.loc[i - 1, "h_ice"]
                )
                self.df.loc[i, "ice"] = (
                    math.pi / 3 * self.df.loc[i - 1, "r_ice"] ** 2
                    * self.df.loc[i - 1, "h_ice"] * self.RHO_I
                )
                self.df.loc[i, "input"] = self.df.loc[i, "ice"]
                logger.warning("Initialise: radius %s, height %s, iceV %s" % (self.df.loc[i - 1, "r_ice"], self.df.loc[i - 1, "h_ice"], self.df.loc[i , "iceV"]))

                self.start = i - 1

            if STATE == 1:

                # if self.name == "guttannen" and i == self.start + 1 and self.hollow_V != 0:
                #     self.df.loc[i, "iceV"] += self.hollow_V
                #     self.df.loc[i, "ice"] += self.hollow_V * self.RHO_I

                self.surface_area(i)

                #Change in fountain height
                if self.df.h_ice[i] > self.h_f and self.discharge != 0:
                    self.h_f += 1
                    # ctr += 1
                    logger.warning("Height increased to %s on %s" % (self.h_f, self.df.When[i]))
                    self.height_steps(i)

                # Precipitation to ice quantity
                if row.T_a < self.T_RAIN and row.Prec > 0:
                    self.df.loc[i, "ppt"] = (
                        self.RHO_W
                        * row.Prec
                        / 1000
                        * math.pi
                        * math.pow(self.df.loc[i, "r_ice"], 2)
                    )

                # Fountain water output
                self.liquid = (
                    self.df.Discharge.loc[i] * (1 - self.ftl) * self.TIME_STEP / 60
                )

                if self.df.loc[i, "SA"]:
                    self.energy_balance(row)
                else:
                    logger.error("SA zero")
                    sys.exit("SA zero")

                # Latent Heat
                self.df.loc[i, "$q_{T}$"] = self.df.loc[i, "Ql"]

                if self.df.loc[i, "Ql"] < 0:
                    # Sublimation
                    if self.df.loc[i, "RH"] < 60:
                        L = self.L_S
                        self.df.loc[i, "gas"] -= (
                            self.df.loc[i, "$q_{T}$"]
                            * self.TIME_STEP
                            * self.df.loc[i, "SA"]
                            / L
                        )

                        # Removing gas quantity generated from ice
                        self.df.loc[i, "solid"] += (
                            self.df.loc[i, "$q_{T}$"]
                            * self.TIME_STEP
                            * self.df.loc[i, "SA"]
                            / L
                        )
                    # Evaporation
                    else:
                        L = self.L_E
                        self.df.loc[i, "gas"] -= (
                            self.df.loc[i, "$q_{T}$"]
                            * self.TIME_STEP
                            * self.df.loc[i, "SA"]
                            / L
                        )

                        # Removing gas quantity generated from meltwater
                        self.df.loc[i, "meltwater"] += (
                            self.df.loc[i, "$q_{T}$"]
                            * self.TIME_STEP
                            * self.df.loc[i, "SA"]
                            / L
                        )

                else:
                    # Deposition
                    if self.df.loc[i, "RH"] < 60:
                        L = self.L_S
                        self.df.loc[i, "dpt"] += (
                            self.df.loc[i, "$q_{T}$"]
                            * self.TIME_STEP
                            * self.df.loc[i, "SA"]
                            / self.L_S
                        )
                    # Condensation
                    else:
                        L = self.L_E
                        self.df.loc[i, "cdt"] += (
                            self.df.loc[i, "$q_{T}$"]
                            * self.TIME_STEP
                            * self.df.loc[i, "SA"]
                            / self.L_S
                        )

                if self.df.loc[i, "TotalE"] < 0 and self.liquid > 0:
                    """Freezing water"""
                    self.liquid += (
                        self.df.loc[i, "TotalE"] * self.TIME_STEP * self.df.loc[i, "SA"]
                    ) / (self.L_F)

                    # DUE TO qF force surface temperature zero
                    self.df.loc[i, "$q_{T}$"] += (
                        -self.df.loc[i, "T_s"]
                        * self.RHO_I
                        * self.DX
                        * self.C_I
                        / self.TIME_STEP
                        - self.df.loc[i, "Ql"]
                    )
                    # self.df.loc[i, "delta_T_s"] = -self.df.loc[i, "T_s"]

                    if self.liquid < 0:
                        # Cooling Ice
                        self.df.loc[i, "$q_{T}$"] += (self.liquid * self.L_F) / (
                            self.TIME_STEP * self.df.loc[i, "SA"]
                        )
                        self.liquid -= (
                            self.df.loc[i, "TotalE"]
                            * self.TIME_STEP
                            * self.df.loc[i, "SA"]
                        ) / (self.L_F)
                        self.df.loc[i, "$q_{melt}$"] += (-self.liquid * self.L_F) / (
                            self.TIME_STEP * self.df.loc[i, "SA"]
                        )
                        self.liquid = 0
                        logger.warning("Discharge froze completely")
                    else:
                        self.df.loc[i, "$q_{melt}$"] += self.df.loc[i, "TotalE"]

                else:
                    # Heating Ice
                    self.df.loc[i, "$q_{T}$"] += self.df.loc[i, "TotalE"]

                self.df.loc[i, "delta_T_s"] += (
                    self.df.loc[i, "$q_{T}$"]
                    * self.TIME_STEP
                    / (self.RHO_I * self.DX * self.C_I)
                )

                """Ice temperature above zero"""
                if (self.df.loc[i, "T_s"] + self.df.loc[i, "delta_T_s"]) > 0:
                    self.df.loc[i, "$q_{melt}$"] += (
                        (self.RHO_I * self.DX * self.C_I)
                        * (self.df.loc[i, "T_s"] + self.df.loc[i, "delta_T_s"])
                        / self.TIME_STEP
                    )

                    self.df.loc[i, "$q_{T}$"] -= (
                        (self.RHO_I * self.DX * self.C_I)
                        * (self.df.loc[i, "T_s"] + self.df.loc[i, "delta_T_s"])
                        / self.TIME_STEP
                    )

                    self.df.loc[i, "delta_T_s"] = -self.df.loc[i, "T_s"]
                    logger.debug("Hot Ice")
                    if np.isnan(self.df.loc[i, "delta_T_s"]):
                        logger.error(
                            f"When {self.df.When[i]},LW {self.df.LW[i]}, LW_in {self.df.LW_in[i]}, T_s {self.df.T_s[i - 1]}"
                        )
                        sys.exit("Ice Temperature nan")

                if self.df.loc[i, "$q_{melt}$"] < 0:
                    self.df.loc[i, "solid"] -= (
                        self.df.loc[i, "$q_{melt}$"]
                        * self.TIME_STEP
                        * self.df.loc[i, "SA"]
                        / (self.L_F)
                    )
                else:
                    self.df.loc[i, "melted"] += (
                        self.df.loc[i, "$q_{melt}$"]
                        * self.TIME_STEP
                        * self.df.loc[i, "SA"]
                        / (self.L_F)
                    )

                """ Quantities of all phases """
                self.df.loc[i + 1, "T_s"] = (
                    self.df.loc[i, "T_s"] + self.df.loc[i, "delta_T_s"]
                )
                self.df.loc[i + 1, "meltwater"] = (
                    self.df.loc[i, "meltwater"]
                    + self.df.loc[i, "melted"]
                    + self.df.loc[i, "cdt"]
                )
                self.df.loc[i + 1, "ice"] = (
                    self.df.loc[i, "ice"]
                    + self.df.loc[i, "solid"]
                    + self.df.loc[i, "dpt"]
                    - self.df.loc[i, "melted"]
                    + self.df.loc[i, "ppt"]
                )
                self.df.loc[i + 1, "vapour"] = (
                    self.df.loc[i, "vapour"] + self.df.loc[i, "gas"]
                )
                self.df.loc[i + 1, "unfrozen_water"] = (
                    self.df.loc[i, "unfrozen_water"] + self.liquid
                )
                self.df.loc[i + 1, "iceV"] = (
                    self.df.loc[i + 1, "ice"] / self.RHO_I
                    + self.df.loc[self.start, "iceV"]
                )
                self.df.loc[i + 1, "input"] = (
                    self.df.loc[i, "input"]
                    + self.df.loc[i, "ppt"]
                    + self.df.loc[i, "dpt"]
                    + self.df.loc[i, "cdt"]
                    + self.df.loc[i, "Discharge"] * 5
                )
                self.df.loc[i + 1, "thickness"] = (
                    self.df.loc[i, "solid"]
                    + self.df.loc[i, "dpt"]
                    - self.df.loc[i, "melted"]
                    + self.df.loc[i, "ppt"]
                ) / (self.df.loc[i, "SA"] * self.RHO_I)

                # logger.warning("%s, %s"%(self.df.loc[i, "When"], self.df.loc[i, "Discharge"]))

                self.liquid = [0] * 1

    def corr_plot(self): # Produces correlation plot of all variables

        data = self.df

        data = data[data.columns.drop(list(data.filter(regex="Unnamed")))]

        data["$q_{net}$"] = data["TotalE"] + data["Ql"]

        data["$\\Delta M_{input}$"] = data["Discharge"] * 5 + data["dpt"] + data["ppt"]

        data["$SW_{in}$"] = data["SW_direct"] + data["SW_diffuse"]

        data["$\\Delta M_{ice}$"] = (
            self.df["solid"] + self.df["dpt"] - self.df["melted"] + self.df["ppt"]
        )

        data = data.rename(
            {
                "delta_T_s": "$\\Delta T_{ice}$",
                "SA": "A",
                "T_a": "$T_a$",
                "v_a": "$v_a$",
                "p_a": "$p_a$",
            },
            axis=1,
        )

        data = data[
            [
                "$q_{net}$",
                "$T_a$",
                "$v_a$",
                "$p_a$",
                "RH",
                "$SW_{in}$",
                "$\\Delta M_{ice}$",
                "A",
            ]
        ]

        logger.debug(
            data.drop("$q_{net}$", axis=1).apply(
                lambda x: x.corr(data["$q_{net}$"]) ** 2
            )
        )
        logger.debug(
            data.drop("$\\Delta M_{ice}$", axis=1).apply(
                lambda x: x.corr(data["$\\Delta M_{ice}$"]) ** 2
            )
        )

        # corr = data.corr()
        # ax = sns.heatmap(
        #     corr,
        #     vmin=-1,
        #     vmax=1,
        #     center=0,
        #     cmap=sns.diverging_palette(20, 220, n=200),
        #     square=True,
        # )
        # ax.set_xticklabels(
        #     ax.get_xticklabels(), rotation=45, horizontalalignment="right"
        # )
        # plt.show()

    def paper_figures(self, output="none"):
        CB91_Blue = "#2CBDFE"
        CB91_Green = "#47DBCD"
        CB91_Pink = "#F3A0F2"
        CB91_Purple = "#9D2EC5"
        CB91_Violet = "#661D98"
        CB91_Amber = "#F5B14C"
        # grey = '#ced4da'
        self.df = self.df.rename(
            {
                "SW": "$q_{SW}$",
                "LW": "$q_{LW}$",
                "Qs": "$q_S$",
                "Ql": "$q_L$",
                "Qf": "$q_{F}$",
                "Qg": "$q_{G}$",
            },
            axis=1,
        )

        mask = self.df.missing != 1
        nmask = self.df.missing != 0
        ymask = self.df.missing == 1
        pmask = self.df.missing != 2
        df_ERA5 = self.df.copy()
        df_ERA52 = self.df.copy()
        df_SZ = self.df.copy()
        df_SZ.loc[
            ymask, ["When", "T_a", "SW_direct", "SW_diffuse", "v_a", "p_a", "RH"]
        ] = np.NaN
        df_ERA5.loc[
            mask, ["When", "T_a", "SW_direct", "SW_diffuse", "v_a", "p_a", "RH"]
        ] = np.NaN
        df_ERA52.loc[
            pmask, ["When", "T_a", "SW_direct", "SW_diffuse", "v_a", "p_a", "RH"]
        ] = np.NaN

        events = np.split(df_ERA5.When, np.where(np.isnan(df_ERA5.When.values))[0])
        # removing NaN entries
        events = [
            ev[~np.isnan(ev.values)] for ev in events if not isinstance(ev, np.ndarray)
        ]
        # removing empty DataFrames
        events = [ev for ev in events if not ev.empty]
        events2 = np.split(df_ERA52.When, np.where(np.isnan(df_ERA52.When.values))[0])
        # removing NaN entries
        events2 = [
            ev[~np.isnan(ev.values)] for ev in events2 if not isinstance(ev, np.ndarray)
        ]
        # removing empty DataFrames
        events2 = [ev for ev in events2 if not ev.empty]

        fig, (ax1, ax2, ax3, ax4, ax5, ax6) = plt.subplots(
            nrows=6, ncols=1, sharex="col", sharey="row", figsize=(12, 18)
        )

        x = self.df.When

        y1 = self.df.Discharge
        ax1.plot(x, y1, linestyle="-", color="#284D58", linewidth=1)
        ax1.set_ylabel("Fountain Spray [$l\\, min^{-1}$]")

        ax1t = ax1.twinx()
        ax1t.plot(
            x,
            self.df.Prec * 1000,
            linestyle="-",
            color=CB91_Blue,
            label="Plaffeien",
        )
        ax1t.set_ylabel("Precipitation [$mm$]", color=CB91_Blue)
        for tl in ax1t.get_yticklabels():
            tl.set_color(CB91_Blue)

        y2 = df_SZ.T_a
        y2_ERA5 = df_ERA5.T_a
        ax2.plot(x, y2, linestyle="-", color="#284D58", linewidth=1)
        for ev in events:
            ax2.axvspan(
                ev.head(1).values, ev.tail(1).values, facecolor="xkcd:grey", alpha=0.25
            )
        ax2.plot(x, y2_ERA5, linestyle="-", color="#284D58")
        ax2.set_ylabel("Temperature [$\\degree C$]")

        y3 = self.df.SW_direct
        lns2 = ax3.plot(
            x, y3, linestyle="-", label="Shortwave Direct", color=CB91_Amber
        )
        lns1 = ax3.plot(
            x,
            self.df.SW_diffuse,
            linestyle="-",
            label="Shortwave Diffuse",
            color=CB91_Blue,
        )
        lns3 = ax3.plot(
            x, self.df.LW_in, linestyle="-", label="Longwave", color=CB91_Green
        )
        ax3.axvspan(
            self.df.When.head(1).values,
            self.df.When.tail(1).values,
            facecolor="grey",
            alpha=0.25,
        )
        ax3.set_ylabel("Radiation [$W\\,m^{-2}$]")

        # added these three lines
        lns = lns1 + lns2 + lns3
        labs = [l.get_label() for l in lns]
        ax3.legend(lns, labs, ncol=3, loc="best")

        y4 = df_SZ.RH
        y4_ERA5 = df_ERA5.RH
        ax4.plot(x, y4, linestyle="-", color="#284D58", linewidth=1)
        ax4.plot(x, y4_ERA5, linestyle="-", color="#284D58")
        for ev in events:
            ax4.axvspan(
                ev.head(1).values, ev.tail(1).values, facecolor="grey", alpha=0.25
            )
        ax4.set_ylabel("Humidity [$\\%$]")

        y5 = df_SZ.p_a
        y5_ERA5 = df_ERA5.p_a
        ax5.plot(x, y5, linestyle="-", color="#264653", linewidth=1)
        ax5.plot(x, y5_ERA5, linestyle="-", color="#284D58")
        for ev in events:
            ax5.axvspan(
                ev.head(1).values, ev.tail(1).values, facecolor="grey", alpha=0.25
            )
        ax5.set_ylabel("Pressure [$hPa$]")

        y6 = df_SZ.v_a
        y6_ERA5 = df_ERA5.v_a
        y6_ERA52 = df_ERA52.v_a
        ax6.plot(x, y6, linestyle="-", color="#264653", linewidth=1, label="Schwarzsee")
        ax6.plot(x, y6_ERA5, linestyle="-", color="#284D58")
        ax6.plot(x, y6_ERA52, linestyle="-", color="#284D58")
        for ev in events:
            ax6.axvspan(
                ev.head(1).values, ev.tail(1).values, facecolor="grey", alpha=0.25
            )
        for ev in events2:
            ax6.axvspan(
                ev.head(1).values, ev.tail(1).values, facecolor="grey", alpha=0.25
            )
        ax6.set_ylabel("Wind speed [$m\\,s^{-1}$]")

        ax1.xaxis.set_major_locator(mdates.WeekdayLocator())
        ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
        ax1.xaxis.set_minor_locator(mdates.DayLocator())
        fig.autofmt_xdate()
        plt.savefig(
            # self.output_folder + "paper_figures/Figure_3.jpg", dpi=300, bbox_inches="tight"
            self.output_folder + "paper_figures/Model_Input_" + self.trigger + ".jpg", dpi=300, bbox_inches="tight"
        )
        plt.clf()

        fig = plt.figure(figsize=(12, 14))
        dfds = self.df[
            [
                "When",
                "solid",
                "ppt",
                "dpt",
                "cdt",
                "melted",
                "gas",
                "SA",
                "iceV",
                "Discharge",
            ]
        ]

        with pd.option_context("mode.chained_assignment", None):
            for i in range(1, dfds.shape[0]):
                dfds.loc[i, "solid"] = dfds.loc[i, "solid"] / (
                    self.df.loc[i, "SA"] * self.RHO_I
                )
                dfds["solid"] = dfds.loc[dfds.solid >= 0, "solid"]
                dfds.loc[i, "melted"] *= -1 / (self.df.loc[i, "SA"] * self.RHO_I)
                dfds.loc[i, "gas"] *= -1 / (self.df.loc[i, "SA"] * self.RHO_I)
                dfds.loc[i, "ppt"] *= 1 / (self.df.loc[i, "SA"] * self.RHO_I)
                dfds.loc[i, "dpt"] *= 1 / (self.df.loc[i, "SA"] * self.RHO_I)
                dfds.loc[i, "cdt"] *= 1 / (self.df.loc[i, "SA"] * self.RHO_I)

        dfds = dfds.set_index("When").resample("D").sum().reset_index()
        dfds["When"] = dfds["When"].dt.strftime("%b %d")

        # dfds["label"] = " "
        # labels = [
        #     "Jan 30",
        #     "Feb 05",
        #     "Feb 12",
        #     "Feb 19",
        #     "Feb 26",
        #     "Mar 05",
        #     "Mar 12",
        #     "Mar 19",
        #     "Mar 26",
        #     "Apr 02",
        # ]
        # for i in range(0, dfds.shape[0]):
        #     for item in labels:
        #         if dfds.When[i] == item:
        #             dfds.loc[i, "label"] = dfds.When[i]

        # dfds = dfds.set_index("label")
        dfds = dfds.rename(
            columns={
                "solid": "Ice",
                "ppt": "Snow",
                "melted": "Melt",
                "gas": "Vapour sub./evap.",
            }
        )
        dfds["Vapour cond./dep."] = dfds["dpt"] + dfds["cdt"]

        y2 = dfds[
            [
                "Ice",
                "Snow",
                "Vapour cond./dep.",
                "Vapour sub./evap.",
                "Melt",
            ]
        ]

        dfd = self.df.set_index("When").resample("D").mean().reset_index()
        dfd["When"] = dfd["When"].dt.strftime("%b %d")

        # dfd["label"] = " "
        # labels = [
        #     "Jan 30",
        #     "Feb 05",
        #     "Feb 12",
        #     "Feb 19",
        #     "Feb 26",
        #     "Mar 05",
        #     "Mar 12",
        #     "Mar 19",
        #     "Mar 26",
        #     "Apr 02",
        # ]
        # for i in range(0, dfd.shape[0]):
        #     for item in labels:
        #         if dfd.When[i] == item:
        #             dfd.loc[i, "label"] = dfd.When[i]

        # dfd = dfd.set_index("label")

        dfds2 = self.df.set_index("When").resample("D").mean().reset_index()
        dfds2["When"] = dfds2["When"].dt.strftime("%b %d")
        # dfds2["label"] = " "
        # labels = [
        #     "Jan 30",
        #     "Feb 05",
        #     "Feb 12",
        #     "Feb 19",
        #     "Feb 26",
        #     "Mar 05",
        #     "Mar 12",
        #     "Mar 19",
        #     "Mar 26",
        #     "Apr 02",
        # ]
        # for i in range(0, dfds2.shape[0]):
        #     for item in labels:
        #         if dfds2.When[i] == item:
        #             dfds2.loc[i, "label"] = dfds2.When[i]
        # dfds2 = dfds2.set_index("label")
        y3 = dfds2["SA"]
        y4 = dfds2["iceV"]
        y0 = dfds["Discharge"] * 5 / 1000

        z = dfd[["$q_{SW}$", "$q_{LW}$", "$q_S$", "$q_L$", "$q_{F}$", "$q_{G}$"]]

        ax0 = fig.add_subplot(5, 1, 1)
        ax0 = y0.plot.bar(
            y="Discharge", linewidth=0.5, edgecolor="black", color="#0C70DE", ax=ax0
        )
        ax0.xaxis.set_label_text("")
        ax0.set_ylabel("Discharge ($m^3$)")
        ax0.grid(axis="y", color="#0C70DE", alpha=0.3, linewidth=0.5, which="major")
        at = AnchoredText("(a)", prop=dict(size=10), frameon=True, loc="upper left")
        at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
        x_axis = ax0.axes.get_xaxis()
        x_axis.set_visible(False)
        ax0.add_artist(at)

        ax1 = fig.add_subplot(5, 1, 2)
        ax1 = z.plot.bar(stacked=True, edgecolor="black", linewidth=0.5, ax=ax1)
        ax1.xaxis.set_label_text("")
        ax1.grid(color="black", alpha=0.3, linewidth=0.5, which="major")
        plt.ylabel("Energy Flux [$W\\,m^{-2}$]")
        plt.legend(loc="upper center", ncol=6)
        # plt.ylim(-125, 125)
        x_axis = ax1.axes.get_xaxis()
        x_axis.set_visible(False)
        at = AnchoredText("(b)", prop=dict(size=10), frameon=True, loc="upper left")
        at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
        ax1.add_artist(at)

        ax2 = fig.add_subplot(5, 1, 3)
        y2.plot(
            kind="bar",
            stacked=True,
            edgecolor="black",
            linewidth=0.5,
            color=["#D9E9FA", CB91_Blue, "#EA9010", "#006C67", "#0C70DE"],
            ax=ax2,
        )
        plt.ylabel("Thickness ($m$ w. e.)")
        plt.xticks(rotation=45)
        plt.legend(loc="upper center", ncol=6)
        # ax2.set_ylim(-0.03, 0.03)
        ax2.yaxis.set_minor_locator(AutoMinorLocator())
        ax2.grid(axis="y", color="black", alpha=0.3, linewidth=0.5, which="major")
        x_axis = ax2.axes.get_xaxis()
        x_axis.set_visible(False)
        at = AnchoredText("(c)", prop=dict(size=10), frameon=True, loc="upper left")
        at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
        ax2.add_artist(at)

        ax3 = fig.add_subplot(5, 1, 4)
        ax3 = y3.plot.bar(
            y="SA", linewidth=0.5, edgecolor="black", color="xkcd:grey", ax=ax3
        )
        ax3.xaxis.set_label_text("")
        ax3.set_ylabel("Surface Area ($m^2$)")
        ax3.grid(axis="y", color="black", alpha=0.3, linewidth=0.5, which="major")
        x_axis = ax3.axes.get_xaxis()
        x_axis.set_visible(False)
        at = AnchoredText("(d)", prop=dict(size=10), frameon=True, loc="upper left")
        at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
        ax3.add_artist(at)

        ax4 = fig.add_subplot(5, 1, 5)
        ax4 = y4.plot.bar(
            y="iceV", linewidth=0.5, edgecolor="black", color="#D9E9FA", ax=ax4
        )
        ax4.xaxis.set_label_text("")
        ax4.set_ylabel("Ice Volume($m^3$)")
        ax4.grid(axis="y", color="black", alpha=0.3, linewidth=0.5, which="major")
        at = AnchoredText("(e)", prop=dict(size=10), frameon=True, loc="upper left")
        at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
        ax4.add_artist(at)
        ax4.xaxis.set_major_locator(mdates.WeekdayLocator())
        ax4.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
        ax4.xaxis.set_minor_locator(mdates.DayLocator())
        fig.autofmt_xdate()
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(
            # self.output_folder + "paper_figures/Figure_6.jpg", dpi=300, bbox_inches="tight"
            self.output_folder + "paper_figures/Model_Output_" + self.trigger + ".jpg", dpi=300, bbox_inches="tight"
        )
        plt.clf()

        fig, (ax1, ax2) = plt.subplots(
            nrows=2, ncols=1, sharex="col", sharey="row", figsize=(15, 12)
        )

        x = self.df.When

        y1 = self.df.a
        y2 = self.df.f_cone
        ax1.plot(x, y1, color="#16697a")
        ax1.set_ylabel("Albedo")
        ax1t = ax1.twinx()
        ax1t.plot(x, y2, color="#ff6d00", linewidth=0.5)
        ax1t.set_ylabel("$f_{cone}$", color="#ff6d00")
        for tl in ax1t.get_yticklabels():
            tl.set_color("#ff6d00")
        ax1.set_ylim([0, 1])
        ax1t.set_ylim([0, 1])

        y1 = self.df.T_s
        y2 = self.df.T_bulk
        ax2.plot(
            x, y1, "k-", linestyle="-", color="#00b4d8", linewidth=0.5, label="Surface"
        )
        ax2.set_ylabel("Temperature [$\\degree C$]")
        ax2.plot(x, y2, linestyle="-", color="#023e8a", linewidth=1, label="Bulk")
        ax2.set_ylim([-20, 1])
        ax2.legend()

        ax1.xaxis.set_major_locator(mdates.WeekdayLocator())
        ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
        ax1.xaxis.set_minor_locator(mdates.DayLocator())
        fig.autofmt_xdate()
        plt.savefig(
                # self.output_folder + "paper_figures/Figure_7.jpg", dpi=300, bbox_inches="tight"
            self.output_folder + "paper_figures/albedo_temperature.jpg", dpi=300, bbox_inches="tight"
        )
        plt.close("all")

        self.df = self.df.rename(
            {
                "$q_{SW}$": "SW",
                "$q_{LW}$": "LW",
                "$q_S$": "Qs",
                "$q_L$": "Ql",
                "$q_{F}$": "Qf",
                "$q_{G}$": "Qg",
            },
            axis=1,
        )

if __name__ == "__main__":
    start = time.time()

    SITE, FOUNTAIN = config("Guttannen")

    icestupa = Icestupa(SITE, FOUNTAIN)

    # icestupa.derive_parameters()

    # icestupa.read_input()

    # icestupa.melt_freeze()

    icestupa.read_output()

    # icestupa.corr_plot()

    # icestupa.summary()

    # icestupa.print_input()
    icestupa.paper_figures()

    # icestupa.print_output()

    total = time.time() - start

    logger.debug("Total time  : %.2f", total / 60)
