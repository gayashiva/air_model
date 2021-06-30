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
    L_F = 334 * 1000  # J/kg Fusion
    C_A = 1.01 * 1000  # J/kgC Specific heat air
    C_I = 2.097 * 1000  # J/kgC Specific heat ice
    C_W = 4.186 * 1000  # J/kgC Specific heat water
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
    T_PPT = 1  # Temperature condition for liquid precipitation
    MU_CONE = 0.5 # Turbulence of cone
    H_AWS = 3 # Height of AWS
    # H_PPT = 1 # mm Precipitaion height that changes albedo
    # DX_DT = 5.5556e-06 #m/s Surface layer thickness growth rate

    """Fountain constants"""
    T_W = 1  # FOUNTAIN Water temperature

    """Model constants"""
    DT = 60*60 # Model time step
    DX = 20e-03 #m Surface layer thickness growth rate
    
    def __init__(self, location = "Guttannen 2021"):

        SITE, FOLDER = config(location)
        initial_data = [SITE, FOLDER]

        # Initialise all variables of dictionary
        for dictionary in initial_data:
            for key in dictionary:
                setattr(self, key, dictionary[key])
                logger.info(f"%s -> %s" % (key, str(dictionary[key])))

        # Initialize input dataset
        input_file = self.input + self.name + "_input_model.csv"
        self.df = pd.read_csv(input_file, sep=",", header=0, parse_dates=["When"])

        # Drops garbage columns
        self.df = self.df[
            self.df.columns.drop(list(self.df.filter(regex="Unnamed")))
        ]  

        logger.debug(self.df.head())
        logger.debug(self.df.tail())

    # Imported methods
    from src.models.methods._freq import change_freq
    from src.models.methods._self_attributes import self_attributes
    from src.models.methods._albedo import get_albedo
    from src.models.methods._discharge import get_discharge
    from src.models.methods._area import get_area
    from src.models.methods._temp import get_temp, test_get_temp
    from src.models.methods._energy import get_energy, test_get_energy
    from src.models.methods._figures import summary_figures

    @Timer(text="Preprocessed data in {:.2f} seconds" , logger = logging.warning)
    def derive_parameters(self):  # Derives additional parameters required for simulation

        self.change_freq()

        unknown = ["a", "vp_a", "LW_in", "cld"]  # Possible unknown variables
        for i in range(len(unknown)):
            if unknown[i] in list(self.df.columns):
                unknown[i] = np.NaN  # Removes known variable
            else:
                logger.error(" %s is unknown\n" % (unknown[i]))
                self.df[unknown[i]] = 0

        for row in stqdm(
            self.df[1:].itertuples(),
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
                        abs(self.df.loc[i, "vp_a"] / (row.T_a + 273.15)), 1 / 7
                    )
                ) * (1 + 0.22 * math.pow(row.cld, 2))

                self.df.loc[i, "LW_in"] = (
                    self.df.loc[i, "e_a"]
                    * self.STEFAN_BOLTZMAN
                    * math.pow(row.T_a + 273.15, 4)
                )

        self.get_discharge()
        self.self_attributes(save=True)

        solar_df = get_solar(
            latitude=self.latitude,
            longitude=self.longitude,
            start=self.start_date,
            end=self.df["When"].iloc[-1],
            DT=self.DT,
        )
        self.df = pd.merge(solar_df, self.df, on="When")

        """Albedo"""
        if "a" in unknown:
            """Albedo Decay parameters initialized"""
            self.A_DECAY = self.A_DECAY * 24 * 60 * 60 / self.DT
            s = 0
            f = 1
            for row in self.df.itertuples():
                i=row.Index
                s, f = self.get_albedo(i, s, f)

        self.df = self.df.round(3)

        if self.df.isnull().values.any():
            for column in self.df.columns:
                if self.df[column].isna().sum() > 0: 
                    logger.warning(" Null values interpolated in %s" %column)
                    self.df.loc[:, column] = self.df[column].interpolate()

        self.df.to_hdf(
            self.input + "model_input.h5",
            key="df",
            mode="a",
        )
        self.df.to_csv(self.input + "model_input.csv")
        logger.debug(self.df.head())
        logger.debug(self.df.tail())

    def summary(self):  # Summarizes results and saves output

        f_efficiency = 100 - (
            (
                self.df["unfrozen_water"].iloc[-1]
                / (self.df["Discharge"].sum() * self.DT / 60)
                * 100
            )
        )

        Duration = self.df.index[-1] * self.DT / (60 * 60 * 24)

        print("\nIce Volume Max", float(round(self.df["iceV"].max(), 2)))
        print("Fountain efficiency", round(f_efficiency, 1))
        print("Ice Mass Remaining", round(self.df["ice"].iloc[-1], 2))
        print("Meltwater", round(self.df["meltwater"].iloc[-1], 2))
        print("Ppt", round(self.df["ppt"].sum(), 2))
        print("Duration", round(Duration, 2))

        # Full Output
        filename4 = self.output + "model_output.csv"
        self.df.to_csv(filename4, sep=",")
        self.df.to_hdf(
            self.output + "model_output.h5",
            key="df",
            mode="a",
        )

    def read_input(self):  # Use processed input dataset

        self.df = pd.read_hdf(self.input + "model_input.h5", "df")

        self.change_freq()

        if self.df.isnull().values.any():
            logger.warning("\n Null values present\n")

    def read_output( self ):  # Reads output

        self.df = pd.read_hdf(self.output + "model_output.h5", "df")

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
            "sub",
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
            "dep",
            "thickness",
            "fountain_runoff",
            "fountain_froze",
            "Qt",
            "Qmelt",
            "input",
        ]

        for column in col:
            self.df[column] = 0

        # Initialise first model time step
        self.df.loc[0, "h_ice"] = self.h_i
        self.df.loc[0, "r_ice"] = self.r_spray
        self.df.loc[0, "s_cone"] = (
            self.df.loc[0, "h_ice"] / self.df.loc[0, "r_ice"]
        )
        V_initial = (
            math.pi
            / 3
            * self.r_spray ** 2
            * self.h_i
        )
        self.df.loc[1, "ice"] = (V_initial* self.RHO_I)
        self.df.loc[1, "iceV"] = V_initial
        self.df.loc[1, "input"] = self.df.loc[1, "ice"]

        logger.warning(
            "Initialise: When %s, radius %.3f, height %.3f, iceV %.3f\n"
            % (
                self.df.loc[0, "When"],
                self.df.loc[0, "r_ice"],
                self.df.loc[0, "h_ice"],
                self.df.loc[1, "iceV"],
            )
        )

        t = stqdm(
            self.df[1:-1].itertuples(),
            total=self.df.shape[0] - 1,
        )

        t.set_description("Simulating %s Icestupa" % self.name)

        for row in t:
            i = row.Index

            ice_melted = self.df.loc[i, "iceV"] < self.V_dome

            if ice_melted:
                logger.error("Simulation ends %s %0.1f "%(self.df.When[i], self.df.iceV[i]))

                if self.df.loc[i-1, "When"] < self.fountain_off_date and self.df.loc[i-1, "melted"] > 0:
                    self.df.loc[i, "T_s"] = 0 
                    self.df.loc[i, "thickness"] = 0 
                    col_list = ["meltwater", "ice", "vapour", "unfrozen_water", "iceV", "input"]
                    logger.error("Skipping %s"%self.df.loc[i, "When"])
                    for column in col_list:
                        self.df.loc[i, column] = self.df.loc[i-1, column]
                    continue

                col_list = ["dep", "ppt", "fountain_froze", "fountain_runoff", "sub", "melted"]
                for column in col_list:
                    self.df.loc[i-1, column] = 0

                self.df = self.df[1 : i]
                self.df = self.df.reset_index(drop=True)
                break

            self.get_area(i)

            if test:
                self.test_get_energy(i)
            else:
                self.get_energy(i)

            if test:
                self.test_get_temp(i)
            else:
                self.get_temp(i)

            # Sublimation and deposition
            if self.df.loc[i, "Ql"] < 0:
                L = self.L_S
                self.df.loc[i, "sub"] = -(
                    self.df.loc[i, "Ql"]
                    * self.DT
                    * self.df.loc[i, "SA"]
                    / L
                )
            else:
                L = self.L_S
                self.df.loc[i, "dep"] = (
                    self.df.loc[i, "Ql"]
                    * self.DT
                    * self.df.loc[i, "SA"]
                    / self.L_S
                )

            # Precipitation to ice quantity
            if self.df.loc[i, "T_a"] < self.T_PPT and self.df.loc[i, "Prec"] > 0:
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
            )
            self.df.loc[i + 1, "ice"] = (
                self.df.loc[i, "ice"]
                + self.df.loc[i, "fountain_froze"]
                + self.df.loc[i, "dep"]
                + self.df.loc[i, "ppt"]
                - self.df.loc[i, "sub"]
                - self.df.loc[i, "melted"]
            )
            self.df.loc[i + 1, "vapour"] = (
                self.df.loc[i, "vapour"] + self.df.loc[i, "sub"]
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
                + self.df.loc[i, "dep"]
                + self.df.loc[i,"Discharge"] * self.DT / 60
            )
            self.df.loc[i + 1, "thickness"] = (
                self.df.loc[i+1, "iceV"]
                - self.df.loc[i, "iceV"]
            ) / (self.df.loc[i, "SA"])

            if test:
                output =self.df.loc[i+1,"ice"] + self.df.loc[i+1,"unfrozen_water"] +self.df.loc[i+1,"vapour"]+self.df.loc[i+1,"meltwater"] 
                input = self.df.loc[i+1,"input"]
                input2 = self.df.loc[1,"input"] + self.df.Discharge[1:i+1].sum() * self.DT/60 +self.df["dep"].sum() + self.df["ppt"].sum()

                # Check mass conservation
                if round(input2,2) != round(input,2):
                    logger.error("Not equal When %s input %.1f input2 %.1f" %(self.df.loc[i,"When"], input, input2))
                    logger.error("input default%.1f Discharge %.1f"%(self.df.loc[1,"input"],self.df.Discharge[1:i+1].sum() * self.DT/60))
                    sys.exit()
                if round(input,2) != round(output,2):
                    logger.error("Not equal When %s input %.1f output %.1f" %(self.df.loc[i,"When"], input, output))
                    logger.error("fountain froze %.1f Discharge %.1f fountain_runoff%.1f"%(self.df.loc[i,"fountain_froze"],self.df.loc[i,"Discharge"] *self.DT/60,self.df.loc[i,"fountain_runoff"])) 
                    logger.error("ppt %.1f dep %.1f sub %.1f melted %.1f"%(self.df.loc[i,"ppt"],self.df.loc[i,"dep"],self.df.loc[i,"sub"],self.df.loc[i,"melted"])) 
                    sys.exit()

                logger.info(
                    f" When {self.df.When[i]},iceV {self.df.iceV[i+1]}, thickness  {self.df.thickness[i]}"
                )
