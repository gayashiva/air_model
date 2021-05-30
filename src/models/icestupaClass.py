"""Icestupa class object definition
"""

# External modules
import pickle
pickle.HIGHEST_PROTOCOL = 4 # For python version 2.7
import pandas as pd
import sys, os, math
from datetime import datetime, timedelta
import numpy as np
from functools import lru_cache
import logging
from stqdm import stqdm
from codetiming import Timer

# Locals
dirname = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.append(dirname)
from src.models.methods.solar import get_solar
from src.utils.settings import config
from src.utils import setup_logger
from src.models.methods.calibration import get_calibration
from src.models.methods.droplet import get_droplet_projectile

# Module logger
logger = logging.getLogger(__name__)


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
    G = 9.81  # Gravitational acceleration

    """Surface Properties"""
    IE = 0.95  # Ice Emissivity IE
    A_I = 0.35  # Albedo of Ice A_I
    A_S = 0.85  # Albedo of Fresh Snow A_S
    A_DECAY = 10  # Albedo decay rate decay_t_d
    Z = 0.0017  # Ice Momentum and Scalar roughness length
    T_RAIN = 1  # Temperature condition for liquid precipitation

    """Fountain constants"""
    T_W = 5  # FOUNTAIN Water temperature

    """Simulation constants"""
    trigger = "Manual"

    """Model constants"""
    # DX = 10e-03  # Initial Ice layer thickness
    # TIME_STEP = 30*60 # Model time step
    DX = 20e-03  # Initial Ice layer thickness
    TIME_STEP = 60*60 # Model time step


    def __init__(self, location = "Guttannen 2021"):

        SITE, FOLDER, df_h = config(location)
        initial_data = [SITE, FOLDER]

        # Initialise all variables of dictionary
        for dictionary in initial_data:
            for key in dictionary:
                setattr(self, key, dictionary[key])
                logger.info(f"%s -> %s" % (key, str(dictionary[key])))


        # Initialize input dataset
        input_file = self.input + self.name + "_input_model.csv"
        self.df = pd.read_csv(input_file, sep=",", header=0, parse_dates=["When"])
        mask = self.df["When"] >= self.start_date
        mask &= self.df["When"] <= self.end_date
        self.df = self.df.loc[mask]
        self.df = self.df.reset_index(drop=True)

        self.df = self.df[
            self.df.columns.drop(list(self.df.filter(regex="Unnamed")))
        ]  # Drops garbage columns

        """Fountain height"""
        df_h = df_h.set_index("When")
        self.df = self.df.set_index("When")
        logger.debug(df_h.head())
        self.df["h_f"] = df_h
        self.df.loc[:,"h_f"] = self.df.loc[:,"h_f"].ffill()
        self.df = self.df.reset_index()

        logger.debug(self.df.head())
        logger.debug(self.df.tail())

    # Imported methods
    from src.models.methods._freq import change_freq
    from src.models.methods._self_attributes import self_attributes
    from src.models.methods._albedo import get_albedo
    from src.models.methods._height_steps import get_height_steps
    from src.models.methods._discharge import get_discharge
    from src.models.methods._area import get_area
    from src.models.methods._temp import get_temp, test_get_temp
    from src.models.methods._energy import get_energy, test_get_energy
    from src.models.methods._figures import summary_figures

    @Timer(text="Preprocessed data in {:.2f} seconds" , logger = logging.warning)
    def derive_parameters(self):  # Derives additional parameters required for simulation

        self.change_freq()
        self.self_attributes(save=True)

        unknown = ["a", "vp_a", "LW_in", "cld"]  # Possible unknown variables
        for i in range(len(unknown)):
            if unknown[i] in list(self.df.columns):
                unknown[i] = np.NaN  # Removes known variable
            else:
                logger.info(" %s is unknown\n" % (unknown[i]))
                self.df[unknown[i]] = 0

        for row in stqdm(
            self.df[1:].itertuples(),
            # range(1,self.df.shape[0]),
            total=self.df.shape[0],
            desc="Creating AIR input...",
        ):
            i = row.Index

            """ Vapour Pressure"""
            if "vp_a" in unknown:
                self.df.loc[i, "vp_a"] = (
                    6.107
                    * math.pow(
                        10,
                        7.5
                        * row.T_a
                        / (row.T_a + 237.3),
                    )
                    * row.RH
                    / 100
                )

            """LW incoming"""
            if "LW_in" in unknown:

                self.df.loc[i, "e_a"] = (
                    1.24
                    * math.pow(
                        abs(row.vp_a / (row.T_a + 273.15)), 1 / 7
                    )
                ) * (1 + 0.22 * math.pow(row.cld, 2))

                self.df.loc[i, "LW_in"] = (
                    self.df.loc[i, "e_a"]
                    * self.STEFAN_BOLTZMAN
                    * math.pow(row.T_a + 273.15, 4)
                )

        self.get_discharge()

        solar_df = get_solar(
            latitude=self.latitude,
            longitude=self.longitude,
            start=self.start_date,
            end=self.df["When"].iloc[-1],
            TIME_STEP=self.TIME_STEP,
        )
        self.df = pd.merge(solar_df, self.df, on="When")
        self.df.Prec = self.df.Prec * self.TIME_STEP  # mm

        """Albedo"""

        if "a" in unknown:
            """Albedo Decay parameters initialized"""
            self.A_DECAY = self.A_DECAY * 24 * 60 * 60 / self.TIME_STEP
            s = 0
            f = 1
            if self.name in ["schwarzsee19", "guttannen20"]:
                f = 0  # Start with snow event

            for row in self.df.itertuples():
                i=row.Index
                s, f = self.get_albedo(i, s, f, site=self.name)

        self.df = self.df.round(3)

        if self.df.isnull().values.any():
            # print(self.df[self.df.columns].isna().sum())
            for column in self.df.columns:
                if self.df[column].isna().sum() > 0: 
                    logger.warning(" Null values interpolated in %s" %column)
                    self.df.loc[:, column] = self.df[column].interpolate()

        self.df.to_hdf(
            self.input + "model_input_" + self.trigger + ".h5",
            key="df",
            mode="a",
        )
        self.df.to_csv(self.input + "model_input_" + self.trigger + ".csv")
        logger.debug(self.df.head())
        logger.debug(self.df.tail())

    def summary(self):  # Summarizes results and saves output

        f_efficiency = 100 - (
            (
                self.df["unfrozen_water"].iloc[-1]
                / (self.df["Discharge"].sum() * self.TIME_STEP / 60)
                * 100
            )
        )

        Duration = self.df.index[-1] * self.TIME_STEP / (60 * 60 * 24)

        print("\nIce Volume Max", float(round(self.df["iceV"].max(), 2)))
        print("Fountain efficiency", round(f_efficiency, 1))
        print("Ice Mass Remaining", round(self.df["ice"].iloc[-1], 2))
        print("Meltwater", round(self.df["meltwater"].iloc[-1], 2))
        print("Ppt", round(self.df["ppt"].sum(), 2))
        print("Duration", round(Duration, 2))

        # Full Output
        filename4 = self.output + "model_output_" + self.trigger + ".csv"
        self.df.to_csv(filename4, sep=",")
        self.df.to_hdf(
            self.output + "model_output_" + self.trigger + ".h5",
            key="df",
            mode="a",
        )

        # # Output for manim
        # filename2 = os.path.join(self.output, self.name + "_manim_" + self.trigger + ".csv")
        # df = self.df.copy()
        # cols = ["When", "h_ice", "h_s", "r_ice", "ice", "T_a", "Discharge"]
        # df = df[cols]
        # df.set_index('When').to_csv(filename2, sep=",")
        # logger.info("Manim output produced")

    def read_input(self):  # Use processed input dataset

        self.df = pd.read_hdf(self.input + "model_input_" + self.trigger + ".h5", "df")

        self.change_freq()

        df_c = pd.read_hdf(self.input + "model_input_" + self.trigger + ".h5", "df_c")

        if self.df.isnull().values.any():
            logger.warning("\n Null values present\n")

    def read_output( self ):  # Reads output

        self.df = pd.read_hdf(self.output + "model_output_" + self.trigger + ".h5", "df")

        self.change_freq()
        self.self_attributes()


    @Timer(text="Simulation executed in {:.2f} seconds", logger = logging.warning)
    def melt_freeze(self, test=False):

        # Initialisaton for sites
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
            "Qsurf",
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
            "thickness",
            "fountain_runoff",
            "fountain_froze",
            "Qt",
            "Qmelt",
        ]

        for column in col:
            self.df[column] = 0

        self.start = self.df.index[self.df.Discharge > 0][0]

        if self.start == 0:
            self.start+=1

        if hasattr(self, "h_i"):
            self.df.loc[self.start - 1, "h_ice"] = self.h_i
        else:
            self.df.loc[self.start - 1, "h_ice"] = self.DX

        self.df.loc[self.start - 1, "r_ice"] = self.r_spray

        self.df.loc[self.start - 1, "s_cone"] = (
            self.df.loc[self.start - 1, "h_ice"] / self.df.loc[self.start - 1, "r_ice"]
        )
        self.initial_vol= (
            math.pi
            / 3
            * self.df.loc[self.start - 1, "r_ice"] ** 2
            * self.df.loc[self.start - 1, "h_ice"]
        )
        self.df.loc[self.start, "ice"] = (
            self.initial_vol
            * self.RHO_I
        )
        self.df.loc[self.start, "iceV"] = self.initial_vol
        self.df.loc[self.start, "input"] = self.df.loc[self.start, "ice"]
        logger.warning(
            "Initialise: When %s, radius %.1f, height %.1f, iceV %.1f\n"
            % (
                self.df.loc[self.start - 1, "When"],
                self.df.loc[self.start - 1, "r_ice"],
                self.df.loc[self.start - 1, "h_ice"],
                self.df.loc[self.start, "iceV"],
            )
        )

        t = stqdm(
            self.df[self.start:-1].itertuples(),
            total=self.df.shape[0],
        )

        t.set_description("Simulating %s Icestupa" % self.name)

        for row in t:
            i = row.Index

            ice_melted = self.df.loc[i, "iceV"] < self.initial_vol - 1 

            if ice_melted and i != self.start:   
                logger.error("Simulation ends %s %0.1f "%(self.df.When[i], self.df.iceV[i]))

                if self.df.loc[i-1, "When"] < self.fountain_off_date and self.df.loc[i-1, "solid"] <= 0:
                    self.df.loc[i-1, "iceV"] = self.dome_vol
                    self.df.loc[i, "T_s"] = 0 
                    self.df.loc[i, "thickness"] = 0 
                    col_list = ["meltwater", "ice", "vapour", "unfrozen_water", "iceV", "input"]
                    logger.error("Skipping %s"%self.df.loc[i, "When"])
                    for column in col_list:
                        self.df.loc[i, column] = self.df.loc[i-1, column]
                    continue

                self.df.loc[i - 1, "meltwater"] += self.df.loc[i - 1, "ice"]
                self.df.loc[i - 1, "ice"] = 0
                logger.error("Model ends at %s" % (self.df.When[i]))
                self.df = self.df[self.start : i - 1]
                self.df = self.df.reset_index(drop=True)
                break

            # Fountain water output
            self.df.loc[i,"fountain_runoff"]= (
                self.df.Discharge.loc[i] * self.TIME_STEP / 60
            )

            self.get_area(i)

            if test:
                self.test_get_energy(i)
            else:
                self.get_energy(i)

            if test:
                self.test_get_temp(i)
            else:
                self.get_temp(i)

            if self.df.loc[i, "Qmelt"] < 0:
                self.df.loc[i, "solid"] -= (
                    self.df.loc[i, "Qmelt"]
                    * self.TIME_STEP
                    * self.df.loc[i, "SA"]
                    / (self.L_F)
                )
            else:
                self.df.loc[i, "melted"] += (
                    self.df.loc[i, "Qmelt"]
                    * self.TIME_STEP
                    * self.df.loc[i, "SA"]
                    / (self.L_F)
                )

            # Precipitation to ice quantity
            if self.df.loc[i, "T_a"] < self.T_RAIN and self.df.loc[i, "Prec"] > 0:
                self.df.loc[i, "ppt"] = (
                    self.RHO_W
                    * self.df.loc[i, "Prec"]
                    / 1000
                    * math.pi
                    * math.pow(self.df.loc[i, "r_ice"], 2)
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
                self.df.loc[i, "unfrozen_water"] 
                + self.df.loc[i,"fountain_runoff"] 
            )
            self.df.loc[i + 1, "iceV"] = (
                self.df.loc[i + 1, "ice"] / self.RHO_I
            )

            self.df.loc[i + 1, "input"] = (
                self.df.loc[i, "input"]
                + self.df.loc[i, "ppt"]
                + self.df.loc[i, "dpt"]
                + self.df.loc[i, "cdt"]
                + self.df.loc[i,"fountain_runoff"]
            )
            self.df.loc[i + 1, "thickness"] = (
                self.df.loc[i+1, "iceV"]
                - self.df.loc[i, "iceV"]
            ) / (self.df.loc[i, "SA"])

            if test:
                logger.info(
                    f" When {self.df.When[i]},iceV {self.df.iceV[i+1]}, thickness  {self.df.thickness[i]}"
                )
