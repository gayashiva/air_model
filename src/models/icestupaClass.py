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
from src.models.methods.calibration import get_calibration
from src.models.methods.solar import get_solar
from src.models.methods.droplet import get_droplet_projectile
from src.utils.settings import config
from src.utils import setup_logger

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
    T_DECAY = 10  # Albedo decay rate decay_t_d
    Z_I = 0.0017  # Ice Momentum and Scalar roughness length
    T_RAIN = 1  # Temperature condition for liquid precipitation

    """Fountain constants"""
    # theta_f = 45  # FOUNTAIN angle
    T_w = 5  # FOUNTAIN Water temperature

    """Simulation constants"""
    trigger = "Manual"
    # crit_temp = 0  # FOUNTAIN runtime temperature

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
    from src.models.methods._albedo import get_albedo
    from src.models.methods._height_steps import get_height_steps
    from src.models.methods._discharge import get_discharge
    from src.models.methods._area import get_area
    from src.models.methods._energy import get_energy
    from src.models.methods._figures import summary_figures

    @Timer(text="Preprocessed data in {:.2f} seconds" , logger = logging.warning)
    def derive_parameters(
        self,
    ):  # Derives additional parameters required for simulation

        self.change_freq()

        if self.name in ["guttannen21", "guttannen20"]:
            df_c, df_cam = get_calibration(site=self.name, input=self.raw)
        else:
            df_c = get_calibration(site=self.name, input=self.raw)

        df_c.to_hdf(
            self.input + "model_input_" + self.trigger + ".h5",
            key="df_c",
            mode="w",
        )
        # df_c.to_csv(self.input + "measured_vol.csv")

        if self.name in ["guttannen21", "guttannen20"]:
            df_cam.to_hdf(
                self.input + "model_input_" + self.trigger + ".h5",
                key="df_cam",
                mode="a",
            )
            df_cam.to_csv(self.input + "measured_temp.csv")

        if self.name == "schwarzsee19":
            self.r_spray = get_droplet_projectile(
                dia=self.dia_f, h=self.df.loc[0,"h_f"], d=self.discharge
            )
            self.dome_vol=0
            logger.warning("Measured spray radius from fountain parameters %0.1f"%self.r_spray)
        else:
            if hasattr(self, "perimeter"):
                self.r_spray = self.perimeter/(math.pi *2)
                logger.warning("Measured spray radius from perimeter %0.1f"%self.r_spray)
            else:
                self.r_spray= df_c.loc[df_c.When < self.fountain_off_date, "dia"].mean() / 2
                logger.warning("Measured spray radius from drone %0.1f"%self.r_spray)
            # Get initial height
            if hasattr(self, "dome_rad"):
                self.dome_vol = 2/3 * math.pi * self.dome_rad ** 3 # Volume of dome
                self.h_i = 3 * self.dome_vol/ (math.pi * self.r_spray ** 2)
                logger.warning("Initial height estimated from dome %0.1f"%self.h_i)
            else:
                self.h_i = 3 * df_c.loc[0, "DroneV"] / (math.pi * self.r_spray ** 2)
                self.dome_vol = df_c.loc[0, "DroneV"]
                logger.warning("Initial height estimated from drone %0.1f"%self.h_i)


        unknown = ["a", "vp_a", "LW_in", "cld"]  # Possible unknown variables
        for i in range(len(unknown)):
            if unknown[i] in list(self.df.columns):
                unknown[i] = np.NaN  # Removes known variable
            else:
                logger.info(" %s is unknown\n" % (unknown[i]))
                self.df[unknown[i]] = 0

        for row in stqdm(
            self.df[1:].itertuples(),
            total=self.df.shape[0],
            desc="Creating AIR input...",
        ):
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
            self.T_DECAY = self.T_DECAY * 24 * 60 * 60 / self.TIME_STEP
            s = 0
            f = 1
            if self.name in ["schwarzsee19", "guttannen20"]:
                f = 0  # Start with snow event

            for i, row in self.df.iterrows():
                s, f = self.get_albedo(i, s, f, site=self.name)

        self.df = self.df.round(3)
        self.df = self.df[
            self.df.columns.drop(list(self.df.filter(regex="Unnamed")))
        ]  # Remove junk columns

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

        # TODO
        self.df = self.df[
            self.df.columns.drop(list(self.df.filter(regex="Unnamed")))
        ]  # Drops garbage columns

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

    def read_input(self):  # Use processed input dataset

        self.df = pd.read_hdf(self.input + "model_input_" + self.trigger + ".h5", "df")

        self.change_freq()

        df_c = pd.read_hdf(self.input + "model_input_" + self.trigger + ".h5", "df_c")

        if self.name == "schwarzsee19":
            self.r_spray = get_droplet_projectile(
                dia=self.dia_f, h=self.df.loc[0,"h_f"], d=self.discharge
            )
            self.dome_vol=0
            logger.warning("Measured spray radius from fountain parameters %0.1f"%self.r_spray)
        else:
            if hasattr(self, "perimeter"):
                self.r_spray = self.perimeter/(math.pi *2)
                logger.warning("Measured spray radius from perimeter %0.1f"%self.r_spray)
            else:
                self.r_spray= df_c.loc[df_c.When < self.fountain_off_date, "dia"].mean() / 2
                logger.warning("Measured spray radius from drone %0.1f"%self.r_spray)
            # Get initial height
            if hasattr(self, "dome_rad"):
                self.dome_vol = 2/3 * math.pi * self.dome_rad ** 3 # Volume of dome
                self.h_i = 3 * self.dome_vol/ (math.pi * self.r_spray ** 2)
                logger.warning("Initial height estimated from dome %0.1f"%self.h_i)
            else:
                self.h_i = 3 * df_c.loc[0, "DroneV"] / (math.pi * self.r_spray ** 2)
                self.dome_vol = df_c.loc[0, "DroneV"]
                logger.warning("Initial height estimated from drone %0.1f"%self.h_i)



        if self.df.isnull().values.any():
            logger.warning("\n Null values present\n")

    def read_output( self ):  # Reads output

        self.df = pd.read_hdf(self.output + "model_output_" + self.trigger + ".h5", "df")

        self.change_freq()

    def manim_output(self):
        # Output for manim
        filename2 = os.path.join(self.output, self.name + "_manim_" + self.trigger + ".csv")
        df = self.df.copy()
        cols = ["When", "h_ice", "h_s", "r_ice", "ice", "T_a", "Discharge"]
        df = df[cols]
        df.set_index('When').to_csv(filename2, sep=",")
        logger.info("Manim output produced")

    @Timer(text="Simulation executed in {:.2f} seconds", logger = logging.warning)
    def melt_freeze(self):

        # Initialise required columns
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
            "wind_loss",
            "Qt",
            "Qmelt",
            "freezing_discharge_fraction",
        ]

        for column in col:
            if column not in ["freezing_discharge_fraction"]:
                self.df[column] = 0
            else:
                self.df[column] = np.NaN 

        STATE = 0
        self.start = 0

        t = stqdm(
            self.df[1:-1].itertuples(),
            total=self.df.shape[0],
        )
        t.set_description("Simulating %s Icestupa" % self.name)
        for row in t:
            i = row.Index

            ice_melted = self.df.loc[i, "ice"] < 1 or self.df.loc[i, "T_bulk"] < -50 or self.df.loc[i, "T_s"] < -200 #or i==self.df.shape[0] - 1
            

            if (
                ice_melted and STATE == 1
            ):  # Break loop when ice melted and simulation done
                logger.error("Ice %0.1f, T_s %0.1f" %(self.df.loc[i, "ice"], self.df.loc[i, "T_s"]))
                self.df.loc[i - 1, "meltwater"] += self.df.loc[i - 1, "ice"]
                self.df.loc[i - 1, "ice"] = 0
                logger.info("Model ends at %s" % (self.df.When[i]))
                self.df = self.df[self.start : i - 1]
                self.df = self.df.reset_index(drop=True)
                break

            if self.df.Discharge[i] > 0 and STATE == 0:
                STATE = 1

                # Initialisaton for sites
                if hasattr(self, "h_i"):
                    self.df.loc[i - 1, "h_ice"] = self.h_i
                else:
                    self.df.loc[i - 1, "h_ice"] = self.DX

                self.df.loc[i - 1, "r_ice"] = self.r_spray

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
                    math.pi
                    / 3
                    * self.df.loc[i - 1, "r_ice"] ** 2
                    * self.df.loc[i - 1, "h_ice"]
                    * self.RHO_I
                )
                self.df.loc[i, "input"] = self.df.loc[i, "ice"]
                logger.info(
                    "Initialise: radius %.1f, height %.1f, iceV %.1f\n"
                    % (
                        self.df.loc[i - 1, "r_ice"],
                        self.df.loc[i - 1, "h_ice"],
                        self.df.loc[i, "iceV"],
                    )
                )

                self.start = i - 1

            if STATE == 1:
#                 # Change in fountain height
#                 if self.df.loc[i, "h_f"] != self.df.loc[i-1, "h_f"]:
# 
#                     # Area = math.pi * math.pow(self.dia_f, 2) / 4
#                     # logger.warning(
#                     #     "Old mean discharge %s on %s" % (self.discharge, self.df.When[i])
#                     # )
#                     # self.discharge = math.sqrt((self.discharge/ (60 * 1000))**2-2*self.G*Area**2 * (self.df.loc[i, "h_f"] - self.df.loc[i-1, "h_f"]))*60*1000
#                     # logger.warning(
#                     #     "New mean discharge %s on %s" % (self.discharge, self.df.When[i])
#                     # )
# 
#                     # Maintains velocity of spray
#                     logger.warning(
#                         "Height increased to %s on %s" % (self.df.loc[i, "h_f"], self.df.When[i])
#                     )
#                     logger.warning(
#                         "Old spray radius %s on %s" % (self.r_spray, self.df.When[i])
#                     )
#                     self.r_spray = get_droplet_projectile(
#                         dia=self.dia_f, h=self.df.loc[i, "h_f"], d=self.discharge
#                     )
#                     logger.warning(
#                         "New spray radius %s on %s" % (self.r_spray, self.df.When[i])
#                     )
#                     self.df.loc[i - 1, "r_ice"] = self.r_spray
#                     self.df.loc[i - 1, "h_ice"] = (
#                         3 * self.df.loc[i, "iceV"] / (math.pi * self.r_spray ** 2)
#                     )
# 
                self.get_area(i)

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
                self.df.loc[i,"fountain_runoff"]= (
                    self.df.Discharge.loc[i] * self.TIME_STEP / 60
                )

                # Energy Flux
                self.get_energy(row)

                # Latent Heat
                if self.df.loc[i, "Ql"] < 0:
                    # Sublimation
                    L = self.L_S
                    self.df.loc[i, "gas"] -= (
                        self.df.loc[i, "Ql"]
                        * self.TIME_STEP
                        * self.df.loc[i, "SA"]
                        / L
                    )

                    # Removing gas quantity generated from ice
                    self.df.loc[i, "solid"] += (
                        self.df.loc[i, "Ql"]
                        * self.TIME_STEP
                        * self.df.loc[i, "SA"]
                        / L
                    )

                else:
                    # Deposition
                    L = self.L_S
                    self.df.loc[i, "dpt"] += (
                        self.df.loc[i, "Ql"]
                        * self.TIME_STEP
                        * self.df.loc[i, "SA"]
                        / self.L_S
                    )

                self.df.loc[i, "Qt"] += self.df.loc[i, "Ql"]
                freezing_energy = (self.df.loc[i, "Qsurf"] - self.df.loc[i, "Ql"])

                self.df.loc[i,"freezing_discharge_fraction"] = -(
                    self.df.loc[i, "fountain_runoff"]* self.L_F
                    # / (self.df.loc[i,"Qsurf"] * self.TIME_STEP * self.df.loc[i, "SA"])
                    / (freezing_energy * self.TIME_STEP * self.df.loc[i, "SA"])
                )

                if self.df.loc[i,"Qsurf"] > 0 and freezing_energy < 0:
                    self.df.loc[i,"freezing_discharge_fraction"] = 1
                elif freezing_energy > 0:
                    self.df.loc[i,"freezing_discharge_fraction"] = 0
                else:
                    if self.df.loc[i,"freezing_discharge_fraction"] > 1: # Enough water available
                        self.df.loc[i,"freezing_discharge_fraction"] = 1
                        self.df.loc[i,"fountain_runoff"] += (
                            self.df.loc[i,"freezing_discharge_fraction"] *freezing_energy* self.TIME_STEP * self.df.loc[i, "SA"]
                        ) / (self.L_F)
                    if self.df.loc[i,"freezing_discharge_fraction"] < 1 : # Not Enough water available
                        self.df.loc[i,"fountain_runoff"] = 0
                        # logger.warning("Discharge froze completely with freezing_discharge_fraction %.2f" %self.df[i,"freezing_discharge_fraction"])

                self.df.loc[i, "Qmelt"] += self.df.loc[i,"freezing_discharge_fraction"] * freezing_energy
                self.df.loc[i, "Qt"] += (1- self.df.loc[i,"freezing_discharge_fraction"]) * freezing_energy
                # self.df.loc[i,"freezing_discharge_fraction"] = self.df.loc[i, "Qmelt"] / self.df.loc[i, "Qsurf"]

                if self.df.loc[i,"Qt"] * self.df.loc[i,"Qmelt"] < 0:
                    self.df.loc[i,"freezing_discharge_fraction"] = np.NaN
                else:
                    self.df.loc[i,"freezing_discharge_fraction"] = self.df.loc[i, "Qmelt"] / self.df.loc[i, "Qsurf"]

                self.df.loc[i, "delta_T_s"] = (
                    self.df.loc[i, "Qt"]
                    * self.TIME_STEP
                    / (self.RHO_I * self.DX * self.C_I)
                )

                # TODO Add to paper
                """Ice temperature above zero"""
                if (self.df.loc[i, "T_s"] + self.df.loc[i, "delta_T_s"]) > 0:
                    self.df.loc[i, "Qmelt"] += (
                        (self.df.loc[i, "T_s"] + self.df.loc[i, "delta_T_s"])
                        * self.RHO_I
                        * self.DX
                        * self.C_I
                        / self.TIME_STEP
                    )

                    self.df.loc[i, "Qt"] -= (
                        (self.df.loc[i, "T_s"] + self.df.loc[i, "delta_T_s"])
                        * self.RHO_I
                        * self.DX
                        * self.C_I
                        / self.TIME_STEP
                    )
                    # self.df.loc[i, "Qt"] += self.df.loc[i, "Ql"]
                    # self.df.loc[i, "Qmelt"] -= self.df.loc[i, "Ql"]

                    # self.df.loc[i, "delta_T_s"] = (
                    #     self.df.loc[i, "Qt"]
                    #     * self.TIME_STEP
                    #     / (self.RHO_I * self.DX * self.C_I)
                    # )

                    self.df.loc[i, "delta_T_s"] = -self.df.loc[i, "T_s"]

                    # self.df.loc[i,"freezing_discharge_fraction"] = self.df.loc[i, "Qmelt"] / (self.df.loc[i, "Qsurf"] - self.df.loc[i, "Ql"])
                    # self.df.loc[i,"freezing_discharge_fraction"] = -1 * math.fabs(self.df.loc[i,"freezing_discharge_fraction"])

                    if self.df.loc[i,"Qt"] * self.df.loc[i,"Qmelt"] < 0:
                        self.df.loc[i,"freezing_discharge_fraction"] = np.NaN
                    else:
                        self.df.loc[i,"freezing_discharge_fraction"] = -self.df.loc[i, "Qmelt"] / self.df.loc[i, "Qsurf"]


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


                """ Unit tests """
                if self.df.loc[i, "T_s"] < -100:
                    self.df.loc[i, "T_s"] = -100
                    logger.error(
                        f"Surface temp rest to 100 When {self.df.When[i]},LW {self.df.LW[i]}, LW_in {self.df.LW_in[i]}, T_s {self.df.T_s[i - 1]}"
                    )

                if self.df.loc[i, "delta_T_s"] > 1 * self.TIME_STEP/60:
                    logger.warning("Too much fountain energy %s causes temperature change of %0.1f on %s" %(self.df.loc[i, "Qf"],self.df.loc[i, "delta_T_s"],self.df.loc[i, "When"]))
                    if math.fabs(self.df.delta_T_s[i]) > 50:
                        logger.error(
                            "%s,Surface Temperature %s,Mass %s"
                            % (
                                self.df.loc[i, "When"],
                                self.df.loc[i, "T_s"],
                                self.df.loc[i, "ice"],
                            )
                        )

                if np.isnan(self.df.loc[i, "delta_T_s"]):
                    logger.error(
                        f"When {self.df.When[i]},LW {self.df.LW[i]}, LW_in {self.df.LW_in[i]}, T_s {self.df.T_s[i - 1]}"
                    )
                    sys.exit("Ice Temperature nan")

                if np.isnan(self.df.loc[i, "Qsurf"]):
                    logger.error(
                        f"When {self.df.When[i]}, SW {self.df.SW[i]}, LW {self.df.LW[i]}, Qs {self.df.Qs[i]}, Qf {self.df.Qf[i]}, Qg {self.df.Qg[i]}"
                    )
                    sys.exit("Energy nan")

                if self.df.loc[i,'fountain_runoff'] - self.df.loc[i, 'Discharge'] * self.TIME_STEP / 60 > 2:

                    logger.error(
                        f"Discharge exceeded When {self.df.When[i]}, Fountain in {self.df.fountain_runoff[i]}, Discharge in {self.df.Discharge[i]* self.TIME_STEP / 60}"
                    )

                if math.fabs(self.df.loc[i, "Qsurf"]) > 800:
                    logger.warning(
                        "Energy above 800 %s,Fountain water %s,Sensible %s, SW %s, LW %s, Qg %s"
                        % (
                            self.df.loc[i, "When"],
                            self.df.loc[i, "Qf"],
                            self.df.loc[i, "Qs"],
                            self.df.loc[i, "SW"],
                            self.df.loc[i, "LW"],
                            self.df.loc[i, "Qg"],
                        )
                    )

                if math.fabs(self.df.loc[i, "delta_T_s"]) > 20:
                    logger.warning(
                        "Temperature change above 20C %s,Surface temp %i,Bulk temp %i"
                        % (
                            self.df.loc[i, "When"],
                            self.df.loc[i, "T_s"],
                            self.df.loc[i, "T_bulk"],
                        )
                    )

                if math.fabs(self.df.loc[i,"freezing_discharge_fraction"]) > 1.01: 
                    logger.error(
                        "%s,temp flux %.1f,melt flux %.1f,total %.1f, Ql %.1f,freezing_discharge_fraction %.2f"
                        % (
                            self.df.loc[i, "When"],
                            self.df.loc[i, "Qt"],
                            self.df.loc[i, "Qmelt"],
                            self.df.loc[i, "Qsurf"], 
                            self.df.loc[i, "Ql"], 
                            self.df.loc[i, "freezing_discharge_fraction"],
                        )
                    )

                if self.df.loc[i, "iceV"] <= self.dome_vol and self.df.loc[i, "When"] < self.fountain_off_date and self.df.loc[i, "solid"] < 0:
                    self.df.loc[i, "iceV"] = self.dome_vol
                    self.df.loc[i + 1, "T_s"] = 0 
                    self.df.loc[i + 1, "thickness"] = 0 
                    col_list = ["meltwater", "ice", "vapour", "unfrozen_water", "iceV", "input"]
                    logger.error("Skipping %s"%self.df.loc[i, "When"])
                    for column in col_list:
                        self.df.loc[i+1, column] = self.df.loc[i, column]
                    continue

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
                    + self.df.loc[self.start, "iceV"]
                )

                self.df.loc[i + 1, "input"] = (
                    self.df.loc[i, "input"]
                    + self.df.loc[i, "ppt"]
                    + self.df.loc[i, "dpt"]
                    + self.df.loc[i, "cdt"]
                    + self.df.loc[i,"fountain_runoff"]
                )
                self.df.loc[i + 1, "thickness"] = (
                    self.df.loc[i, "solid"]
                    + self.df.loc[i, "dpt"]
                    - self.df.loc[i, "melted"]
                    + self.df.loc[i, "ppt"]
                ) / (self.df.loc[i, "SA"] * self.RHO_I)

