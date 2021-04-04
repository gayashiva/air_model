"""Icestupa class object definition
"""

# External modules
import pandas as pd
import sys, os, math, time
from datetime import datetime
from tqdm import tqdm
import numpy as np
from functools import lru_cache
from pandas_profiling import ProfileReport
import logging

# Locals
dirname = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.append(dirname)
from src.models.methods.calibration import get_calibration
from src.models.methods.metadata import get_parameter_metadata
from src.models.methods.solar import get_solar
from src.models.methods.droplet import get_droplet_projectile
from src.data.settings import config

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

    """Model constants"""
    # DX = 5e-03  # Initial Ice layer thickness
    DX = 5e-02  # Initial Ice layer thickness
    theta_f = 45  # FOUNTAIN angle
    ftl = 0  # FOUNTAIN flight time loss ftl
    T_w = 5  # FOUNTAIN Water temperature
    crit_temp = 0  # FOUNTAIN runtime temperature

    def __init__(self, *initial_data, **kwargs):
        # Initialise all variables of dictionary
        for dictionary in initial_data:
            for key in dictionary:
                setattr(self, key, dictionary[key])
                logger.info(f"%s -> %s" % (key, str(dictionary[key])))
        # Initialise other variables
        for key in kwargs:
            setattr(self, key, kwargs[key])
            logger.info(f"%s -> %s" % (key, str(dictionary[key])))

        # Initialize input dataset
        input_file = self.input + self.name + "_input_model.csv"
        self.df = pd.read_csv(input_file, sep=",", header=0, parse_dates=["When"])
        self.TIME_STEP = (
            int(pd.infer_freq(self.df["When"])[:-1]) * 60
        )  # Extract time step from datetime column
        logger.info(f"Time steps -> %s minutes" % (str(self.TIME_STEP / 60)))
        mask = self.df["When"] >= self.start_date
        self.df = self.df.loc[mask]
        self.df = self.df.reset_index(drop=True)
        logger.debug(self.df.head())
        logger.debug(self.df.tail())

    # Imported methods
    from src.models.methods._albedo import get_albedo
    from src.models.methods._height_steps import get_height_steps
    from src.models.methods._discharge import get_discharge
    from src.models.methods._area import get_area
    from src.models.methods._energy import get_energy
    from src.models.methods._figures import summary_figures

    def derive_parameters(
        self,
    ):  # Derives additional parameters required for simulation
        if self.name in ["guttannen21", "guttannen20"]:
        # if self.name in ["guttannen21"]:
            df_c,df_cam = get_calibration(site=self.name, input=self.input)
            self.r_spray = df_c.loc[0, "dia"] / 2
            self.h_i = 3 * df_c.loc[0, "DroneV"] / (math.pi * self.r_spray ** 2)
            self.df = pd.merge(self.df, df_c, on="When", how="left")
            self.df = pd.merge(self.df, df_cam, on="When", how="left")
        else:
            df_c= get_calibration(site=self.name, input=self.input)
            self.df = pd.merge(self.df, df_c, on="When", how="left")

        unknown = ["a", "vp_a", "LW_in", "cld"]  # Possible unknown variables
        for i in range(len(unknown)):
            if unknown[i] in list(self.df.columns):
                unknown[i] = np.NaN  # Removes known variable
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

        self.get_discharge()
        f_on = self.df.When[
            self.df.Discharge != 0
        ].tolist()  # List of all timesteps when fountain on
        self.start_date = f_on[0]
        logger.info("Model starts at %s" % (self.start_date))
        logger.warning("Fountain ends %s" % f_on[-1])

        mask = self.df["When"] >= self.start_date
        self.df = self.df.loc[mask]
        self.df = self.df.reset_index(drop=True)

        solar_df = get_solar(
            latitude=self.latitude,
            longitude=self.longitude,
            start=self.start_date,
            end=self.df["When"].iloc[-1],
            TIME_STEP=self.TIME_STEP,
        )
        self.df = pd.merge(solar_df, self.df, on="When")
        self.df.Prec = self.df.Prec * self.TIME_STEP  # mm

        """Albedo Decay parameters initialized"""
        self.T_DECAY = self.T_DECAY * 24 * 60 * 60 / self.TIME_STEP
        s = 0
        f = 0
        for row in tqdm(self.df[1:].itertuples(), total=self.df.shape[0]):
            s, f = self.get_albedo(row, s, f, site=self.name)

        self.df = self.df.round(3)
        self.df = self.df[
            self.df.columns.drop(list(self.df.filter(regex="Unnamed")))
        ]  # Remove junk columns

        self.df.to_hdf(
            self.input + "model_input_" + self.trigger + ".h5",
            key="df",
            mode="w",
        )
        self.df.to_csv(self.input + "model_input_" + self.trigger + ".csv")

    def summary(self):  # Summarizes results and saves output

        self.df = self.df[
            self.df.columns.drop(list(self.df.filter(regex="Unnamed")))
        ]  # Drops garbage columns
        Efficiency = (
            (self.df["meltwater"].iloc[-1] + self.df["ice"].iloc[-1])
            / self.df["input"].iloc[-1]
            * 100
        )

        Duration = self.df.index[-1] * self.TIME_STEP / (60*60 * 24)

        print("\nIce Volume Max", float(round(self.df["iceV"].max(), 2)))
        print("Fountain efficiency", round(Efficiency, 3))
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
            mode="w",
        )

    def read_input(self, report=False):  # Use processed input dataset

        self.df = pd.read_hdf(self.input + "model_input_" + self.trigger + ".h5", "df")

        logger.debug(self.df.head())

        if report == True:
            prof = ProfileReport(self.df)
            prof.to_file(output_file=self.output + "input_report.html")

        if self.df.isnull().values.any():
            logger.warning("Null values present")

    def read_output(
        self, report=False
    ):  # Reads output and Displays outputs useful for manuscript

        self.df = pd.read_hdf(
            self.output + "model_output_" + self.trigger + ".h5", "df"
        )

        Efficiency = (
            (self.df["meltwater"].iloc[-1] + self.df["ice"].iloc[-1])
            / self.df["input"].iloc[-1]
            * 100
        )

        Duration = self.df.index[-1] * self.TIME_STEP / (60*60 * 24)

        print("\nIce Volume Max", float(round(self.df["iceV"].max(), 2)))
        print("Fountain efficiency", round(Efficiency, 3))
        print("Ice Mass Remaining", round(self.df["ice"].iloc[-1], 2))
        print("Meltwater", round(self.df["meltwater"].iloc[-1], 2))
        print("Input", round(self.df["input"].iloc[-1], 2))
        print("Ppt", round(self.df["ppt"].sum(), 2))
        print("Duration", round(Duration, 2))

        # self.df = self.df.set_index("When").resample("D").mean().reset_index()

        if report == True:
            prof = ProfileReport(self.df)
            prof.to_file(output_file=self.output + "output_report.html")

    def melt_freeze(self):  # Main function

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

        if hasattr(self, "r_spray"):  # Provide discharge
            self.discharge = get_droplet_projectile(
                dia=self.dia_f, h=self.h_f, x=self.r_spray
            )
        else:  # Provide spray radius
            self.r_spray = get_droplet_projectile(
                dia=self.dia_f, h=self.h_f, d=self.discharge
            )

        logger.debug("AIR simulation begins...")
        for row in tqdm(self.df[1:-1].itertuples(), total=self.df.shape[0]):
            i = row.Index
            ice_melted = self.df.loc[i, "ice"] < 1

            if (
                ice_melted and STATE == 1
            ):  # Break loop when ice melted and simulation done
                self.df.loc[i - 1, "meltwater"] += self.df.loc[i - 1, "ice"]
                self.df.loc[i - 1, "ice"] = 0
                logger.info("Model ends at %s" % (self.df.When[i]))
                self.df = self.df[self.start : i - 1]
                self.df = self.df.reset_index(drop=True)
                break

            if self.df.Discharge[i] > 0 and STATE == 0:
                STATE = 1

                # Special Initialisaton for specific sites
                if self.name == "schwarzsee19":
                    self.df.loc[i - 1, "r_ice"] = self.r_spray
                    self.df.loc[i - 1, "h_ice"] = self.DX


                if self.name in ["guttannen21", "guttannen20"]:
                    if hasattr(self, "h_i"):
                        self.df.loc[i - 1, "h_ice"] = self.h_i
                        self.df.loc[i - 1, "r_ice"] = self.r_spray
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
                logger.warning(
                    "Initialise: radius %s, height %s, iceV %s"
                    % (
                        self.df.loc[i - 1, "r_ice"],
                        self.df.loc[i - 1, "h_ice"],
                        self.df.loc[i, "iceV"],
                    )
                )

                self.start = i - 1

            if STATE == 1:
                # Change in fountain height
                if not np.isnan(self.df.loc[i, "h_s"]):
                    self.h_f += row.h_s
                    logger.warning(
                        "Height increased to %s on %s" % (self.h_f, self.df.When[i])
                    )
                    # self.get_height_steps(i)
                    self.r_spray = get_droplet_projectile(
                        dia=self.dia_f, h=self.h_f, d=self.discharge
                    )
                    self.df.loc[i - 1, "r_ice"] = self.r_spray
                    self.df.loc[i - 1, "h_ice"] = (
                        3 * self.df.loc[i, "iceV"] / (math.pi * self.r_spray ** 2)
                    )

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
                self.liquid = (
                    self.df.Discharge.loc[i] * (1 - self.ftl) * self.TIME_STEP / 60
                )


                if self.df.loc[i, "SA"]:
                    self.get_energy(row)
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
                    # Change in paper
                    self.df.loc[i, "TotalE"] += (
                        self.df.loc[i, "T_s"]
                        * self.RHO_I
                        * self.DX
                        * self.C_I
                        / self.TIME_STEP
                    )
                    # DUE TO qF force surface temperature zero
                    self.df.loc[i, "$q_{T}$"] -= (
                        self.df.loc[i, "T_s"]
                        * self.RHO_I
                        * self.DX
                        * self.C_I
                        / self.TIME_STEP
                        # - self.df.loc[i, "Ql"]
                    )

                    self.liquid += (
                        self.df.loc[i, "TotalE"] * self.TIME_STEP * self.df.loc[i, "SA"]
                    ) / (self.L_F)


                    if self.liquid < 0:
                        # Cooling Ice
                        self.df.loc[i, "$q_{T}$"] = 0
                        self.df.loc[i, "$q_{T}$"] += (self.liquid * self.L_F) / (
                            self.TIME_STEP * self.df.loc[i, "SA"]
                        )
                        self.liquid -= (
                            self.df.loc[i, "TotalE"]
                            * self.TIME_STEP
                            * self.df.loc[i, "SA"]
                        ) / (self.L_F)
                        self.df.loc[i, "$q_{melt}$"] -= (self.liquid * self.L_F) / (
                            self.TIME_STEP * self.df.loc[i, "SA"]
                        )
                        self.liquid = 0
                        logger.warning("Discharge froze completely")
                    else:
                        self.df.loc[i, "$q_{melt}$"] += self.df.loc[i, "TotalE"]

                else:
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

                if math.fabs(self.df.delta_T_s[i]) > 50:
                    logger.error("%s,Surface Temperature %s,Mass %s"%(self.df.loc[i, "When"], self.df.loc[i, "T_s"], self.df.loc[i, "ice"]))
                    sys.exit("High temperature changes")

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
                    + self.liquid
                )
                self.df.loc[i + 1, "thickness"] = (
                    self.df.loc[i, "solid"]
                    + self.df.loc[i, "dpt"]
                    - self.df.loc[i, "melted"]
                    + self.df.loc[i, "ppt"]
                ) / (self.df.loc[i, "SA"] * self.RHO_I)

                self.liquid = [0] * 1
