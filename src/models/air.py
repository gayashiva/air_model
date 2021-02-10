import pandas as pd
import sys
from datetime import datetime
from tqdm import tqdm
import os
import math
import time
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.offsetbox import AnchoredText
from matplotlib.ticker import AutoMinorLocator
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import seaborn as sns
from pvlib import location
dirname = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

sys.path.append(dirname)
from src.data.config import SITE, FOUNTAIN, FOLDERS
import logging
import coloredlogs

pd.plotting.register_matplotlib_converters()


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

    """Model constants"""
    TIME_STEP = 5 * 60  # s Model time steps
    DX = 5e-03  # Ice layer thickness

    """Surface"""
    IE = 0.95  # Ice Emissivity IE
    A_I = 0.35  # Albedo of Ice A_I
    A_S = 0.85  # Albedo of Fresh Snow A_S
    T_DECAY = 10  # Albedo decay rate decay_t_d
    Z_I = 0.0017  # Ice Momentum and Scalar roughness length
    T_RAIN = 1  # Temperature condition for liquid precipitation

    "Simulation"
    dia_f = 0
    T_w = 0
    h_f = 0
    h_aws = 0

    """Miscellaneous"""
    STATE = 0

    def __init__(self, **kwds):
        self.__dict__.update(kwds)

        if SITE["name"] == "schwarzsee":
            input_file = FOLDERS["input_folder"] + "raw_input_extended.csv"
        else:
            input_file = FOLDERS["input_folder"] + "raw_input_ERA5.csv"

        self.df = pd.read_csv(input_file, sep=",", header=0, parse_dates=["When"])

        if SITE["name"] == "guttannen":
            self.df_cam = pd.read_csv(
                FOLDERS["input_folder"] + "cam.csv",
                sep=",",
                header=0,
                parse_dates=["When"],
            )

    def config_update(self):
        parameters = {
            "dia_f": self.dia_f,
            "T_w": self.T_w,
            "h_f": self.h_f,
            "h_aws": self.h_aws,
        }
        for var in parameters:
            if parameters[var]:
                logger.info(var, "->", parameters[var])
                FOUNTAIN[var] = parameters[var]

    def get_solar(self):

        self.df["ghi"] = self.df["SW_direct"] + self.df["SW_diffuse"]
        self.df["dif"] = self.df["SW_diffuse"]

        site_location = location.Location(SITE["latitude"], SITE["longitude"])

        times = pd.date_range(
            start=SITE["start_date"], end=self.df["When"].iloc[-1], freq="5T"
        )
        clearsky = site_location.get_clearsky(times)
        # Get solar azimuth and zenith to pass to the transposition function
        solar_position = site_location.get_solarposition(
            times=times, method="ephemeris"
        )

        solar_df = pd.DataFrame(
            {
                "ghics": clearsky["ghi"],
                "difcs": clearsky["dhi"],
                "zen": solar_position["zenith"],
                "sea": np.radians(solar_position["elevation"]),
            }
        )
        solar_df.loc[solar_df["sea"] < 0, "sea"] = 0
        solar_df.index = solar_df.index.set_names(["When"])
        solar_df = solar_df.reset_index()

        self.df = pd.merge(solar_df, self.df, on="When")

    def projectile_xy(self, v, h=0):
        if h == 0:
            hs = FOUNTAIN["h_f"]
        else:
            hs = h
        g = 9.81
        data_xy = []
        t = 0.0
        theta_f = math.radians(FOUNTAIN["theta_f"])
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

        """Albedo"""
        # Precipitation
        if (row.Discharge == 0) & (row.Prec > 0):
            if row.T_a < self.T_RAIN:  # Snow
                s = 0
                f = 0

        if row.Discharge > 0:
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

    def spray_radius(self, r_mean=0):
        Area = math.pi * math.pow(FOUNTAIN["dia_f"], 2) / 4
        v = self.df["Discharge"].replace(0, np.NaN).mean() / (60 * 1000 * Area)

        if r_mean != 0:
            self.r_mean = r_mean
        else:
            self.r_mean = self.projectile_xy(v=v)

        return self.r_mean

    def derive_parameters(self):

        self.get_solar()

        missing = ["a", "vp_a", "LW_in"]
        for col in missing:
            if col in list(self.df.columns):
                missing.remove(col)
            else:
                logger.info(col, "is missing")
                self.df[col] = 0

        """Albedo Decay"""
        self.T_DECAY = (
            self.T_DECAY * 24 * 60 * 60 / self.TIME_STEP
        )  # convert to 5 minute time steps
        s = 0
        f = 0

        logger.info("Creating model input file...")
        for row in tqdm(self.df[1:].itertuples(), total=self.df.shape[0]):

            """ Vapour Pressure"""
            if "vp_a" in missing:
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
            if "LW_in" in missing:

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

            s, f = self.albedo(row, s, f)

        self.df = self.df.round(5)

        self.df = self.df[
            [
                "When",
                "sea",
                "T_a",
                "RH",
                "v_a",
                "Discharge",
                "SW_direct",
                "SW_diffuse",
                "Prec",
                "p_a",
                "cld",
                "a",
                # "e_a",
                "vp_a",
                "LW_in",
                "missing",
            ]
        ]

        if SITE["name"] == "schwarzsee":
            self.df.to_hdf(
                FOLDERS["input_folder"] + "model_input_extended.h5", key="df", mode="w"
            )
        else:
            self.df.to_hdf(
                FOLDERS["input_folder"] + "model_input_ERA5.h5", key="df", mode="w"
            )

    def surface_area(self, i):

        if (
            self.df.solid[i - 1]
            + self.df.dpt[i - 1]
            - self.df.melted[i - 1]
            + self.df.ppt[i - 1]
            > 0
        ) & (self.df.loc[i - 1, "r_ice"] > self.r_mean):
            # Ice Radius
            self.df.loc[i, "r_ice"] = self.df.loc[i - 1, "r_ice"]

            # Ice Height
            self.df.loc[i, "h_ice"] = (
                3 * self.df.loc[i, "iceV"] / (math.pi * self.df.loc[i, "r_ice"] ** 2)
            )

            # Height by Radius ratio
            self.df.loc[i, "s_cone"] = (
                self.df.loc[i - 1, "h_ice"] / self.df.loc[i - 1, "r_ice"]
            )

        else:

            # Height to radius ratio
            self.df.loc[i, "s_cone"] = self.df.loc[i - 1, "s_cone"]

            # Ice Radius
            # logger.info(self.df.loc[i, "iceV"], self.df.loc[i, "s_cone"])
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

    def energy_balance(self, row):
        i = row.Index

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

        self.df.loc[i, "Ql"] = (
            0.623
            * self.L_S
            * self.RHO_A
            / self.P0
            * math.pow(self.VAN_KARMAN, 2)
            * self.df.loc[i, "v_a"]
            * (row.vp_a - self.df.loc[i, "vp_ice"])
            / ((np.log(SITE["h_aws"] / self.Z_I)) ** 2)
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
            / ((np.log(SITE["h_aws"] / self.Z_I)) ** 2)
        )

        # Short Wave Radiation SW
        self.df.loc[i, "SW"] = (1 - row.a) * (
            row.SW_direct * self.df.loc[i, "f_cone"] + row.SW_diffuse
        )

        # Long Wave Radiation LW

        self.df.loc[i, "LW"] = row.LW_in - self.IE * self.STEFAN_BOLTZMAN * math.pow(
            self.df.loc[i, "T_s"] + 273.15, 4
        )

        if np.isnan(self.df.loc[i, "LW"]):
            logger.info(
                f"LW {self.df.LW[i]}, LW_in {self.df.LW_in[i]}, T_s {self.df.T_s[i - 1]}"
            )

        if self.liquid > 0:
            self.df.loc[i, "Qf"] = (
                (self.df.loc[i - 1, "solid"])
                * self.C_W
                * FOUNTAIN["T_w"]
                / (self.TIME_STEP * self.df.loc[i, "SA"])
            )

            self.df.loc[i, "Qf"] += (
                self.RHO_I
                * self.DX
                * self.C_I
                * (self.df.loc[i, "T_s"])
                / self.TIME_STEP
            )

        self.df.loc[i, "Qg"] = (
            self.K_I
            * (self.df.loc[i, "T_bulk"] - self.df.loc[i, "T_s"])
            / (self.df.loc[i, "r_ice"] / 2)
        )

        # Bulk Temperature
        self.df.loc[i + 1, "T_bulk"] = self.df.loc[i, "T_bulk"] - self.df.loc[
            i, "Qg"
        ] * self.TIME_STEP * self.df.loc[i, "SA"] / (self.df.loc[i, "ice"] * self.C_I)

        # Total Energy W/m2
        self.df.loc[i, "TotalE"] = (
            self.df.loc[i, "SW"]
            + self.df.loc[i, "LW"]
            + self.df.loc[i, "Qs"]
            + self.df.loc[i, "Qf"]
            + self.df.loc[i, "Qg"]
        )

        # if np.isnan(self.df.loc[i, "TotalE"]) :
        #     logger.info(f"When {self.df.When[i]}, SW {self.df.SW[i]}, LW {self.df.LW[i]}, Qs {self.df.Qs[i]}, Qf {self.df.Qf[i]}, Qg {self.df.Qg[i]}, SA {self.df.SA[i]}")

    def summary(self):

        self.df = self.df[
            [
                "When",
                "sea",
                "T_a",
                "RH",
                "v_a",
                "Discharge",
                "SW_direct",
                "SW_diffuse",
                "Prec",
                "p_a",
                "cld",
                "a",
                # "e_a",
                "vp_a",
                "LW_in",
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
                "missing",
                "s_cone",
                "input",
                "vp_ice",
                "thickness",
                "$q_{T}$",
                "$q_{melt}$",
            ]
        ]

        Efficiency = (
            (self.df["meltwater"].iloc[-1] + self.df["ice"].iloc[-1])
            / self.df["input"].iloc[-1]
            * 100
        )

        Duration = self.df.index[-1] * 5 / (60 * 24)

        logger.info("\nIce Volume Max", float(self.df["iceV"].max()))
        logger.info("Fountain efficiency", Efficiency)
        logger.info("Ice Mass Remaining", self.df["ice"].iloc[-1])
        logger.info("Meltwater", self.df["meltwater"].iloc[-1])
        logger.info("Ppt", self.df["ppt"].sum())
        logger.info("Deposition", self.df["dpt"].sum())
        logger.info("Duration", Duration)

        # Full Output
        filename4 = FOLDERS["output_folder"] + "model_results.csv"
        self.df.to_csv(filename4, sep=",")

        self.df.to_hdf(FOLDERS["output_folder"] + "model_output.h5", key="df", mode="w")

    def read_input(self):

        if SITE["name"] == "schwarzsee":
            self.df = pd.read_hdf(
                FOLDERS["input_folder"] + "model_input_extended.h5", "df"
            )
        else:
            self.df = pd.read_hdf(FOLDERS["input_folder"] + "model_input_ERA5.h5", "df")

        # self.TIME_STEP=15*60
        # self.df = self.df.set_index('When').resample(str(int(self.TIME_STEP/60))+'T').mean().reset_index()

        logger.info(self.df.head())

        if self.df.isnull().values.any():
            logger.info("Warning: Null values present")

    def read_output(self):

        self.df = pd.read_hdf(FOLDERS["output_folder"] + "model_output.h5", "df")

        if self.df.isnull().values.any():
            logger.info("Warning: Null values present")
            # logger.info(self.df.columns)
            # logger.info(self.df[['s_cone']].isnull().sum())
            # logger.info(self.df[self.df['s_cone'].isnull()])

        # Table outputs
        f_off = self.df.index[self.df.Discharge.astype(bool)].tolist()
        f_off = f_off[-1] + 1
        logger.info(
            "When %s \n M_U %.2f\n M_solid %.2f\n M_gas %.2f\n M_liquid %.2f\n M_R %.2f\n M_D %.2f\n M_F %.2f\n" %(self.df.When.iloc[f_off], self.df.unfrozen_water.iloc[f_off], self.df.ice.iloc[f_off], self.df.vapour.iloc[f_off], self.df.meltwater.iloc[f_off], self.df.ppt[: f_off + 1].sum(), self.df.dpt[: f_off + 1].sum(), self.df.Discharge[: f_off + 1].sum() * 5+ self.df.iceV.iloc[0] * self.RHO_I)
        )
        logger.info(
            f"M_input {self.df.input.iloc[-1]}, M_R {self.df.ppt.sum()}, M_D {self.df.dpt.sum()}, M_F {self.df.Discharge.sum() * 5 + self.df.iceV.iloc[0] * self.RHO_I}"
        )
        logger.info(
            f"When {self.df.When.iloc[-1]},M_U {self.df.unfrozen_water.iloc[-1]}, M_solid {self.df.ice.iloc[-1]}, M_gas {self.df.vapour.iloc[-1]}, M_liquid {self.df.meltwater.iloc[-1]}"
        )
        # logger.info("AE", self.df["e_a"].mean())
        logger.info(f"Temperature of spray : %.2f" %(self.df.loc[self.df["TotalE"] < 0, "T_a"].mean()))
        logger.info(f"Temperature minimum : {self.df.T_a.min()}")
        logger.info(f"Energy mean : {self.df.TotalE.mean()}")

        dfd = self.df.set_index("When").resample("D").mean().reset_index()

        dfd["Global"] = dfd["SW_diffuse"] + dfd["SW_direct"]
        logger.info(f"Global max: {dfd.Global.max()}\n SW max: {dfd.SW.max()}")
        logger.info(f"Nonzero Qf: {self.df.Qf.astype(bool).sum(axis=0)}")
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
        logger.info("Qmelt: %.2f, Qt: %.2f" %(dfd["$q_{melt}$"].mean(), dfd["$q_{T}$"].mean()))
        logger.info(
            "Qmelt: %.2f \n Qt: %.2f" %(dfd["$q_{melt}$"].abs().sum() / Total, dfd["$q_{T}$"].abs().sum() / Total)
        )
        logger.info(
            f"% of SW {dfd.SW.abs().sum()/Total}, LW {dfd.LW.abs().sum()/Total}, Qs {dfd.Qs.abs().sum()/Total}, Ql {dfd.Ql.abs().sum()/Total}, Qf {dfd.Qf.abs().sum()/Total}, Qg {dfd.Qg.abs().sum()/Total}"
        )

        # logger.info(
        #     f"Mean of SW {self.df.SW.mean()}, LW {self.df.LW.mean()}, Qs {self.df.Qs.mean()}, Ql {self.df.Ql.mean()}, Qf {self.df.Qf.mean()}, Qg {self.df.Qg.mean()}"
        # )
        # logger.info(
        #     f"Range of SW {self.df.SW.min()}-{self.df.SW.max()}, LW {self.df.LW.min()}-{self.df.LW.max()}, Qs {self.df.Qs.min()}-{self.df.Qs.max()}, Ql {self.df.Ql.min()}-{self.df.Ql.max()}, Qf {self.df.Qf.min()}-{self.df.Qf.max()}, Qg {self.df.Qg.min()}-{self.df.Qg.max()}"
        # )
        logger.info(
            f"Mean of SW {dfd.SW.mean()}, LW {dfd.LW.mean()}, Qs {dfd.Qs.mean()}, Ql {dfd.Ql.mean()}, Qf {dfd.Qf.mean()}, Qg {dfd.Qg.mean()}"
        )
        logger.info(
            f"Range of SW {dfd.SW.min()}-{dfd.SW.max()}, LW {dfd.LW.min()}-{dfd.LW.max()}, Qs {dfd.Qs.min()}-{dfd.Qs.max()}, Ql {dfd.Ql.min()}-{dfd.Ql.max()}, Qf {dfd.Qf.min()}-{dfd.Qf.max()}, Qg {dfd.Qg.min()}-{dfd.Qg.max()}"
        )
        # logger.info(
        #     f"Mean of emissivity {self.df.e_a.mean()}, Range of f_cone {self.df.e_a.min()}-{self.df.e_a.max()}"
        # )
        logger.info(f"Max SA {self.df.SA.max()}")
        logger.info(
            f"M_input {self.df.input.iloc[-1]}, M_R {self.df.ppt.sum()}, M_D {self.df.dpt.sum()}, M_F {self.df.Discharge.sum() * 5 + self.df.iceV.iloc[0] * self.RHO_I}"
        )
        logger.info(
            f"Max_growth {self.df.solid.max() / 5}, average_discharge {self.df.Discharge.replace(0, np.NaN).mean()}"
        )

        logger.info(f"Duration {self.df.index[-1] * 5 / (60 * 24)}")
        logger.info(f"Ended {self.df.When.iloc[-1]}")

        # Output for manim
        filename2 = os.path.join(
            FOLDERS["output_folder"], SITE["name"] + "_model_gif.csv"
        )
        self.df["h_f"] = FOUNTAIN["h_f"]
        cols = ["When", "h_ice", "h_f", "r_ice", "ice", "T_a", "Discharge"]
        self.df[cols].to_csv(filename2, sep=",")

    def melt_freeze(self):

        col = [
            "T_s",  # Surface Temperature
            "T_bulk",  # Bulk Temperature
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
            "$q_{T}$",
            "$q_{melt}$",
        ]

        for column in col:
            self.df[column] = 0

        self.liquid = [0] * 1

        self.start = -10

        self.sum_T_s = 0  # weighted_sums
        self.sum_SA = 0  # weighted_sums

        logger.info("AIR simulation begins...")
        for row in tqdm(self.df[1:-1].itertuples(), total=self.df.shape[0]):
            i = row.Index

            # Initialize
            if self.df.Discharge[i] > 0 and self.STATE == 0:
                self.STATE = 1

                if SITE["name"] == "guttannen":
                    # self.df.loc[i - 1, "r_ice"] = self.spray_radius()
                    # self.df.loc[i - 1, "h_ice"] = self.DX
                    self.spray_radius()
                    self.df.loc[i - 1, "r_ice"] = 9.8655
                    # self.df.loc[i - 1, "r_ice"] =self.spray_radius()
                    self.df.loc[i - 1, "h_ice"] = self.DX
                    self.hollow_V = (
                        math.pi
                        / 3
                        # * FOUNTAIN['tree_radius'] **2
                        * self.df.loc[i - 1, "r_ice"] ** 2
                        * FOUNTAIN["tree_height"]
                    )

                if SITE["name"] == "schwarzsee":
                    self.config_update()
                    self.df.loc[i - 1, "r_ice"] = self.spray_radius()
                    self.df.loc[i - 1, "h_ice"] = self.DX

                if SITE["name"] == "leh":
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
                    math.pi / 3 * self.df.loc[i - 1, "r_ice"] ** 2 * self.DX
                )
                self.df.loc[i, "input"] = self.df.loc[i, "ice"]

                self.start = i - 1

            ice_melted = (self.df.loc[i, "ice"] < 0.01) or (
                self.df.loc[i, "T_s"] < -100
            )

            fountain_off = self.df.Discharge[i:].sum() == 0
            if ice_melted & fountain_off:
                self.df.loc[i - 1, "meltwater"] += self.df.loc[i - 1, "ice"]
                self.df.loc[i - 1, "ice"] = 0
                self.df = self.df[self.start : i - 1]
                self.df = self.df.reset_index(drop=True)
                break

            if self.STATE == 1:

                if SITE["name"] == "guttannen" and i != self.start + 1:
                    self.df.loc[i, "iceV"] += self.hollow_V

                self.surface_area(i)

                # Precipitation to ice quantity
                if row.T_a < self.T_RAIN and row.Prec > 0:
                    self.df.loc[i, "ppt"] = (
                        self.RHO_W
                        * row.Prec
                        * self.TIME_STEP
                        / 1000
                        * math.pi
                        * math.pow(self.df.loc[i, "r_ice"], 2)
                    )

                # Fountain water output
                self.liquid = (
                    row.Discharge * (1 - FOUNTAIN["ftl"]) * self.TIME_STEP / 60
                )

                self.energy_balance(row)

                # Latent Heat
                self.df.loc[i, "$q_{T}$"] = self.df.loc[i, "Ql"]

                if self.df.loc[i, "Ql"] < 0:
                    if self.df.loc[i, "RH"] < 60:
                        L = self.L_S  # Sublimation
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

                else:  # Deposition

                    self.df.loc[i, "dpt"] += (
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

                """Hot Ice"""
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
                    self.df.loc[i, "meltwater"] + self.df.loc[i, "melted"]
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
                    + self.df.loc[i, "Discharge"] * 5
                )
                self.df.loc[i + 1, "thickness"] = (
                    self.df.loc[i, "solid"]
                    + self.df.loc[i, "dpt"]
                    - self.df.loc[i, "melted"]
                    + self.df.loc[i, "ppt"]
                ) / (self.df.loc[i, "SA"] * self.RHO_I)

                # logger.info(self.df.loc[i, "When"], self.df.loc[i, "ice"])

                self.liquid = [0] * 1

    def corr_plot(self):

        data = self.df

        data = data[data.columns.drop(list(data.filter(regex="Unnamed")))]

        data["$q_{net}$"] = data["TotalE"] + data["Ql"]

        data["$\\Delta M_{input}$"] = data["Discharge"] * 5 + data["dpt"] + data["ppt"]

        data["$SW_{in}$"] = data["SW_direct"] + data["SW_diffuse"]

        data["$\\Delta M_{ice}$"] = (
            self.df["solid"] + self.df["dpt"] - self.df["melted"] + self.df["ppt"]
        )

        # data = data.drop(["When", "input", "ppt", "ice", "T_s", "vapour", "Discharge", "TotalE", "T_a", "sea", "SW_direct", "a", "cld", "sea", "e_a", "vp_a", "LW_in", "vp_ice", "f_cone", "SW_diffuse", "s_cone", "RH", "iceV", "melted", "Qf", "SW", "LW", "Qs", "Ql", "dpt", "p_a", "thickness", "h_ice", "r_ice", "Prec", "v_a", "unfrozen_water", "meltwater"], axis=1)

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

        logger.info(
            data.drop("$q_{net}$", axis=1).apply(
                lambda x: x.corr(data["$q_{net}$"]) ** 2
            )
        )
        logger.info(
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


class PDF(Icestupa):
    def print_input(self, filename="derived_parameters.pdf"):
        if filename == "derived_parameters.pdf":
            filename = FOLDERS["input_folder"]

        """Input Plots"""

        filename = FOLDERS["input_folder"] + "data.pdf"

        pp = PdfPages(filename)

        fig, (ax1, ax2, ax3, ax4, ax5, ax6, ax7) = plt.subplots(
            nrows=7, ncols=1, sharex="col", sharey="row", figsize=(16, 14)
        )

        x = self.df.When

        y1 = self.df.Discharge
        ax1.plot(x, y1, "k-", linewidth=0.5)
        ax1.set_ylabel("Fountain Spray [$l\\, min^{-1}$]")
        ax1.grid()

        ax1t = ax1.twinx()
        ax1t.plot(x, self.df.Prec * 1000, "b-", linewidth=0.5)
        ax1t.set_ylabel("Precipitation [$mm\\, s^{-1}$]", color="b")
        for tl in ax1t.get_yticklabels():
            tl.set_color("b")

        y2 = self.df.T_a
        ax2.plot(x, y2, "k-", linewidth=0.5)
        ax2.set_ylabel("Temperature [ea]")
        ax2.grid()

        y3 = self.df.SW_direct + self.df.SW_diffuse
        ax3.plot(x, y3, "k-", linewidth=0.5)
        ax3.set_ylabel("Global Rad.[$W\\,m^{-2}$]")
        ax3.grid()

        ax3t = ax3.twinx()
        ax3t.plot(x, self.df.SW_diffuse, "b-", linewidth=0.5)
        ax3t.set_ylim(ax3.get_ylim())
        ax3t.set_ylabel("Diffuse Rad.[$W\\,m^{-2}$]", color="b")
        for tl in ax3t.get_yticklabels():
            tl.set_color("b")

        y4 = self.df.RH
        ax4.plot(x, y4, "k-", linewidth=0.5)
        ax4.set_ylabel("Humidity [$\\%$]")
        ax4.grid()

        y5 = self.df.p_a
        ax5.plot(x, y5, "k-", linewidth=0.5)
        ax5.set_ylabel("Pressure [$hPa$]")
        ax5.grid()

        y6 = self.df.v_a
        ax6.plot(x, y6, "k-", linewidth=0.5)
        ax6.set_ylabel("Wind [$m\\,s^{-1}$]")
        ax6.grid()

        y7 = self.df.cld
        ax7.plot(x, y7, "k-", linewidth=0.5)
        ax7.set_ylabel("Cloudiness")
        ax7.grid()

        ax1.xaxis.set_major_locator(mdates.WeekdayLocator())
        ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
        ax1.xaxis.set_minor_locator(mdates.DayLocator())
        fig.autofmt_xdate()
        pp.savefig(bbox_inches="tight")

        plt.clf()

        ax1 = fig.add_subplot(111)
        y1 = self.df.T_a
        ax1.plot(x, y1, "k-", linewidth=0.5)
        ax1.set_ylabel("Temperature [$\\degree C$]")
        ax1.grid()
        ax1.xaxis.set_major_locator(mdates.WeekdayLocator())
        ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
        ax1.xaxis.set_minor_locator(mdates.DayLocator())

        fig.autofmt_xdate()
        pp.savefig(bbox_inches="tight")
        plt.clf()

        ax1 = fig.add_subplot(111)

        y2 = self.df.Discharge
        ax1.plot(x, y2, "k-", linewidth=0.5)
        ax1.set_ylabel("Discharge Rate ")
        ax1.grid()
        ax1.xaxis.set_major_locator(mdates.WeekdayLocator())
        ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
        ax1.xaxis.set_minor_locator(mdates.DayLocator())

        fig.autofmt_xdate()
        pp.savefig(bbox_inches="tight")
        plt.clf()

        ax1 = fig.add_subplot(111)
        y3 = self.df.SW_direct
        ax1.plot(x, y3, "k-", linewidth=0.5)
        ax1.set_ylabel("Direct SWR [$W\\,m^{-2}$]")
        ax1.grid()
        ax1.xaxis.set_major_locator(mdates.WeekdayLocator())
        ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
        ax1.xaxis.set_minor_locator(mdates.DayLocator())

        fig.autofmt_xdate()
        pp.savefig(bbox_inches="tight")
        plt.clf()

        ax1 = fig.add_subplot(111)
        y31 = self.df.SW_diffuse
        ax1.plot(x, y31, "k-", linewidth=0.5)
        ax1.set_ylabel("Diffuse SWR [$W\\,m^{-2}$]")
        ax1.grid()
        ax1.xaxis.set_major_locator(mdates.WeekdayLocator())
        ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
        ax1.xaxis.set_minor_locator(mdates.DayLocator())

        fig.autofmt_xdate()
        pp.savefig(bbox_inches="tight")
        plt.clf()

        ax1 = fig.add_subplot(111)
        y4 = self.df.Prec * 1000
        ax1.plot(x, y4, "k-", linewidth=0.5)
        ax1.set_ylabel("Ppt [$mm$]")
        ax1.grid()
        ax1.xaxis.set_major_locator(mdates.WeekdayLocator())
        ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
        ax1.xaxis.set_minor_locator(mdates.DayLocator())

        fig.autofmt_xdate()
        pp.savefig(bbox_inches="tight")
        plt.clf()

        ax1 = fig.add_subplot(111)
        y5 = self.df.p_a
        ax1.plot(x, y5, "k-", linewidth=0.5)
        ax1.set_ylabel("Pressure [$hPa$]")
        ax1.grid()
        ax1.xaxis.set_major_locator(mdates.WeekdayLocator())
        ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
        ax1.xaxis.set_minor_locator(mdates.DayLocator())
        fig.autofmt_xdate()
        pp.savefig(bbox_inches="tight")
        plt.clf()

        ax1 = fig.add_subplot(111)
        y6 = self.df.v_a
        ax1.plot(x, y6, "k-", linewidth=0.5)
        ax1.set_ylabel("Wind [$m\\,s^{-1}$]")

        ax1.xaxis.set_major_locator(mdates.WeekdayLocator())
        ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
        ax1.xaxis.set_minor_locator(mdates.DayLocator())
        fig.autofmt_xdate()
        pp.savefig(bbox_inches="tight")
        plt.clf()

        ax1 = fig.add_subplot(111)
        y6 = self.df.cld
        ax1.plot(x, y6, "k-", linewidth=0.5)
        ax1.set_ylabel("Cloudiness")

        ax1.xaxis.set_major_locator(mdates.WeekdayLocator())
        ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
        ax1.xaxis.set_minor_locator(mdates.DayLocator())
        fig.autofmt_xdate()
        pp.savefig(bbox_inches="tight")
        plt.clf()

        plt.close("all")
        pp.close()

    def print_output(self, filename="model_results.pdf"):

        if filename == "model_results.pdf":
            if self.TIME_STEP != 5 * 60:
                filename = (
                    FOLDERS["output_folder"]
                    + "model_results_"
                    + str(self.TIME_STEP)
                    + ".pdf"
                )
            else:
                filename = FOLDERS["output_folder"] + "model_results" + ".pdf"

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

        self.df["$q_{T}$"] *= -1
        self.df["$q_{melt}$"] *= -1

        # Plots
        pp = PdfPages(filename)

        fig = plt.figure()
        x = self.df.When
        y1 = self.df.iceV

        ax1 = fig.add_subplot(111)
        ax1.plot(x, y1, "k-")
        ax1.set_ylabel("Ice Volume [$m^3$]")
        # ax1.set_xlabel("Days")

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
        ax1.grid()
        ax1.xaxis.set_major_locator(mdates.WeekdayLocator())
        ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
        ax1.xaxis.set_minor_locator(mdates.DayLocator())
        fig.autofmt_xdate()
        pp.savefig(bbox_inches="tight")
        plt.clf()

        y1 = self.df.SA
        ax1 = fig.add_subplot(111)
        ax1.plot(x, y1, "k-")
        ax1.set_ylabel("Surface Area [$m^2$]")
        ax1.grid()
        ax1.xaxis.set_major_locator(mdates.WeekdayLocator())
        ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
        ax1.xaxis.set_minor_locator(mdates.DayLocator())
        fig.autofmt_xdate()
        pp.savefig(bbox_inches="tight")
        plt.clf()

        y1 = self.df.h_ice
        y2 = self.df.r_ice
        ax1 = fig.add_subplot(111)
        ax1.plot(x, y1, "k-")
        ax1.set_ylabel("Ice Cone Height [$m$]")

        ax2 = ax1.twinx()
        ax2.plot(x, y2, "b-", linewidth=0.5)
        ax2.set_ylabel("Ice Radius[$m$]", color="b")
        for tl in ax2.get_yticklabels():
            tl.set_color("b")

        # Include Validation line segment 1

        ax1.scatter(datetime(2019, 2, 14, 16), 0.7, color="black", marker="o")
        ax2.scatter(datetime(2019, 2, 14, 16), 1.15, color="blue", marker="o")
        ax1.grid()
        ax1.xaxis.set_major_locator(mdates.WeekdayLocator())
        ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
        ax1.xaxis.set_minor_locator(mdates.DayLocator())
        ax1.set_ylim([0, 2])
        ax2.set_ylim([0, 2])

        fig.autofmt_xdate()
        pp.savefig(bbox_inches="tight")
        plt.clf()

        y1 = self.df.a
        y2 = self.df.f_cone
        ax1 = fig.add_subplot(111)
        ax1.plot(x, y1, "k-")
        ax1.set_ylabel("Albedo")
        # ax1.set_xlabel("Days")
        ax1t = ax1.twinx()
        ax1t.plot(x, y2, "b-", linewidth=0.5)
        ax1t.set_ylabel("$f_{cone}$", color="b")
        for tl in ax1t.get_yticklabels():
            tl.set_color("b")
        ax1.grid()

        ax1.xaxis.set_major_locator(mdates.WeekdayLocator())
        ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
        ax1.xaxis.set_minor_locator(mdates.DayLocator())
        ax1.set_ylim([0, 1])
        ax1t.set_ylim([0, 1])
        fig.autofmt_xdate()
        pp.savefig(bbox_inches="tight")
        plt.clf()

        y1 = self.df.T_s
        y2 = self.df.T_bulk
        ax1 = fig.add_subplot(111)
        ax1.plot(x, y1, "k-", linewidth=0.5, alpha=0.5)
        ax1.set_ylabel("Surface Temperature [$\\degree C$]")
        # ax1.grid()
        ax2 = ax1.twinx()
        ax2.plot(x, y2, "b-", linewidth=0.5)
        ax2.set_ylabel("Bulk Temperature [$\\degree C$]", color="b")
        for tl in ax2.get_yticklabels():
            tl.set_color("b")
        ax1.xaxis.set_major_locator(mdates.WeekdayLocator())
        ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
        ax1.xaxis.set_minor_locator(mdates.DayLocator())
        ax1.set_ylim([-20, 1])
        ax2.set_ylim([-20, 1])
        fig.autofmt_xdate()
        pp.savefig(bbox_inches="tight")
        plt.close("all")

        fig, (ax1, ax3, ax4) = plt.subplots(
            nrows=3, ncols=1, sharex="col", sharey="row", figsize=(15, 12)
        )

        x = self.df.When

        y1 = self.df.a
        y2 = self.df.f_cone
        ax1.plot(x, y1, "k-")
        ax1.set_ylabel("Albedo")
        # ax1.set_xlabel("Days")
        ax1t = ax1.twinx()
        ax1t.plot(x, y2, "b-", linewidth=0.5)
        ax1t.set_ylabel("$f_{cone}$", color="b")
        for tl in ax1t.get_yticklabels():
            tl.set_color("b")
        ax1.grid()
        ax1.set_ylim([0, 1])
        ax1t.set_ylim([0, 1])

        # y1 = self.df.e_a
        # y2 = self.df.cld
        # ax2.plot(x, y1, "k-")
        # ax2.set_ylabel("Atmospheric Emissivity")
        # # ax1.set_xlabel("Days")
        # ax2t = ax2.twinx()
        # ax2t.plot(x, y2, "b-", linewidth=0.5)
        # ax2t.set_ylabel("Cloudiness", color="b")
        # for tl in ax2t.get_yticklabels():
        #     tl.set_color("b")
        # ax2.grid()

        y3 = self.df.vp_a - self.df.vp_ice
        ax3.plot(x, y3, "k-", linewidth=0.5)
        ax3.set_ylabel("Vapour gradient [$hPa$]")
        ax3.grid()

        y4 = self.df.T_a - self.df.T_s
        ax4.plot(x, y4, "k-", linewidth=0.5)
        ax4.set_ylabel("Temperature gradient [$\\degree C$]")
        ax4.grid()

        ax1.xaxis.set_major_locator(mdates.WeekdayLocator())
        ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
        ax1.xaxis.set_minor_locator(mdates.DayLocator())
        fig.autofmt_xdate()
        pp.savefig(bbox_inches="tight")
        plt.close("all")

        dfd = self.df.set_index("When").resample("D").mean().reset_index()
        dfd["Discharge"] = dfd["Discharge"] == 0
        dfd["Discharge"] = dfd["Discharge"].astype(int)
        dfd["Discharge"] = dfd["Discharge"].astype(str)
        dfd["When"] = dfd["When"].dt.strftime("%b %d")

        dfd["label"] = " "
        labels = [
            "Jan 30",
            "Feb 05",
            "Feb 12",
            "Feb 19",
            "Feb 26",
            "Mar 05",
            "Mar 12",
            "Mar 19",
            "Mar 26",
            "Apr 02",
        ]
        for i in range(0, dfd.shape[0]):
            for item in labels:
                if dfd.When[i] == item:
                    dfd.loc[i, "label"] = dfd.When[i]

        dfd = dfd.set_index("label")

        z = dfd[
            [
                "$q_{SW}$",
                "$q_{LW}$",
                "$q_S$",
                "$q_L$",
                "$q_{F}$",
                "$q_{G}$",
                "$q_{T}$",
                "$q_{melt}$",
            ]
        ]
        ax = z.plot.bar(stacked=True, edgecolor=dfd["Discharge"], linewidth=0.5)
        ax.xaxis.set_label_text("")
        plt.grid(axis="y", color="black", alpha=0.3, linewidth=0.5, which="major")
        # plt.xlabel('Date')
        plt.ylabel("Energy Flux [$W\\,m^{-2}$]")
        plt.legend(loc="upper right", ncol=4)
        plt.ylim(-150, 150)
        plt.xticks(rotation=45)
        pp.savefig(bbox_inches="tight")
        plt.clf()

        fig, (ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8) = plt.subplots(
            nrows=8, ncols=1, sharex="col", sharey="row", figsize=(16, 14)
        )

        x = self.df.When

        y1 = self.df["$q_{SW}$"]
        ax1.plot(x, y1, "k-", linewidth=0.5)
        ax1.set_ylabel("SW")
        ax1.grid()

        y2 = self.df["$q_{LW}$"]
        ax2.plot(x, y2, "k-", linewidth=0.5)
        ax2.set_ylabel("LW")
        ax2.grid()

        y3 = self.df["$q_S$"]
        ax3.plot(x, y3, "k-", linewidth=0.5)
        ax3.set_ylabel("S")
        ax3.grid()

        y4 = self.df["$q_L$"]
        ax4.plot(x, y4, "k-", linewidth=0.5)
        ax4.set_ylabel("L")
        ax4.grid()

        y5 = self.df["$q_{F}$"]
        ax5.plot(x, y5, "k-", linewidth=0.5)
        ax5.set_ylabel("F")
        ax5.grid()
        ax5.set_ylim([-10, 10])

        y6 = self.df["$q_{G}$"]
        ax6.plot(x, y6, "k-", linewidth=0.5)
        ax6.set_ylabel("G")
        ax6.grid()
        ax6.set_ylim([-150, 150])

        y7 = self.df["$q_{T}$"]
        ax7.plot(x, y7, "k-", linewidth=0.5)
        ax7.set_ylabel("T")
        ax7.grid()
        ax7.set_ylim([-150, 150])

        y8 = self.df["$q_{melt}$"]
        ax8.plot(x, y8, "k-", linewidth=0.5)
        ax8.set_ylabel("melt")
        ax8.grid()
        ax8.set_ylim([-150, 150])

        ax1.xaxis.set_major_locator(mdates.WeekdayLocator())
        ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
        ax1.xaxis.set_minor_locator(mdates.DayLocator())
        fig.autofmt_xdate()
        pp.savefig(bbox_inches="tight")

        plt.clf()

        ax2 = fig.add_subplot()
        y1 = self.df.unfrozen_water
        ax2.plot(x, y1, "k-", linewidth=0.5, alpha=0.5)
        ax2.set_ylabel("Surface Temperature [$\\degree C$]")
        ax2.xaxis.set_major_locator(mdates.WeekdayLocator())
        ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
        ax2.xaxis.set_minor_locator(mdates.DayLocator())
        fig.autofmt_xdate()
        ax2.grid(axis="x", color="black", alpha=0.3, linewidth=0.5, which="major")
        plt.tight_layout()
        pp.savefig(bbox_inches="tight")
        plt.clf()

        plt.close("all")
        pp.close()

    def print_output_guttannen(self, filename="model_results.pdf"):

        if filename == "model_results.pdf":
            filename = FOLDERS["output_folder"] + "model_results.pdf"

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

        df_temp = pd.read_csv(FOLDERS["input_folder"] + "lumtemp.csv")

        df_temp["When"] = pd.to_datetime(df_temp["When"])

        # Plots
        pp = PdfPages(filename)

        fig = plt.figure()
        x = self.df.When
        y1 = self.df.iceV

        ax1 = fig.add_subplot(111)
        ax1.plot(x, y1, "k-")
        ax1.set_ylabel("Ice Volume [$m^3$]")

        ax1t = ax1.twinx()
        ax1t.plot(
            self.df_cam.When,
            self.df_cam.Volume,
            "o-",
            color="b",
            alpha=0.5,
            linewidth=0.5,
        )
        ax1t.set_ylabel("Cam Volume [$m^3$]", color="b")
        for tl in ax1t.get_yticklabels():
            tl.set_color("b")

        ax1.set_ylim([0, 500])
        ax1t.set_ylim([0, 500])

        ax1.scatter(
            datetime(2020, 1, 3),
            (54.15),
            color="green",
            marker="o",
            label="Drone Estimate",
        )
        ax1.scatter(datetime(2020, 1, 24), (120.61), color="green", marker="o")
        ax1.scatter(datetime(2020, 2, 15), (128.32), color="green", marker="o")
        ax1.scatter(datetime(2020, 4, 14, 18), 0, color="green", marker="o")

        ax1.grid()
        ax1.legend()

        ax1.xaxis.set_major_locator(mdates.WeekdayLocator())
        ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
        ax1.xaxis.set_minor_locator(mdates.DayLocator())
        fig.autofmt_xdate()
        pp.savefig(bbox_inches="tight")
        plt.clf()

        y1 = self.df.SA
        ax1 = fig.add_subplot(111)
        ax1.plot(x, y1, "k-")
        ax1.set_ylabel("Surface Area [$m^2$]")
        ax1.grid()

        ax1t = ax1.twinx()
        ax1t.plot(
            self.df_cam.When, self.df_cam.SA, "o-", color="b", alpha=0.5, linewidth=0.5
        )
        ax1t.set_ylabel("Cam SA [$m^2$]", color="b")
        for tl in ax1t.get_yticklabels():
            tl.set_color("b")

        ax1.set_ylim([0, 600])
        ax1t.set_ylim([0, 600])

        ax1.scatter(
            datetime(2020, 1, 3),
            (334.78),
            color="green",
            marker="o",
            label="Drone Estimate",
        )
        ax1.scatter(datetime(2020, 1, 24), (374.61), color="green", marker="o")
        ax1.scatter(datetime(2020, 2, 15), (564.12), color="green", marker="o")
        ax1.legend()

        ax1.xaxis.set_major_locator(mdates.WeekdayLocator())
        ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
        ax1.xaxis.set_minor_locator(mdates.DayLocator())
        fig.autofmt_xdate()
        pp.savefig(bbox_inches="tight")
        plt.clf()

        y1 = self.df.h_ice
        y2 = self.df.r_ice
        ax1 = fig.add_subplot(111)
        ax1.plot(x, y1, "k-")
        ax1.set_ylabel("Ice Cone Height [$m$]")
        ax1.grid()

        ax2 = ax1.twinx()
        ax2.plot(x, y2, "b-", linewidth=0.5)
        ax2.set_ylabel("Ice Radius[$m$]", color="b")
        for tl in ax2.get_yticklabels():
            tl.set_color("b")

        ax1.xaxis.set_major_locator(mdates.WeekdayLocator())
        ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
        ax1.xaxis.set_minor_locator(mdates.DayLocator())

        fig.autofmt_xdate()
        pp.savefig(bbox_inches="tight")
        plt.clf()

        y1 = self.df.s_cone
        ax1 = fig.add_subplot(111)
        ax1.plot(x, y1, "k-")
        ax1.set_ylabel("Ice Cone Slope [$m$]")
        ax1.grid()

        ax1.xaxis.set_major_locator(mdates.WeekdayLocator())
        ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
        ax1.xaxis.set_minor_locator(mdates.DayLocator())

        fig.autofmt_xdate()
        pp.savefig(bbox_inches="tight")
        plt.clf()

        y1 = self.df.T_s
        y2 = self.df.T_bulk
        ax1 = fig.add_subplot(111)
        ax1.plot(x, y1, "k-", linewidth=0.5, alpha=0.5)
        ax1.set_ylabel("Surface Temperature[$\\degree C$]")
        # ax1.grid()
        ax2 = ax1.twinx()
        ax2.plot(x, y2, "b-", linewidth=0.5)
        ax2.set_ylabel("Bulk Temperature[$\\degree C$]", color="b")
        for tl in ax2.get_yticklabels():
            tl.set_color("b")
        ax1.xaxis.set_major_locator(mdates.WeekdayLocator())
        ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
        ax1.xaxis.set_minor_locator(mdates.DayLocator())
        ax1.set_ylim([-20, 1])
        ax2.set_ylim([-20, 1])
        fig.autofmt_xdate()
        pp.savefig(bbox_inches="tight")
        plt.clf()

        y1 = self.df.T_s
        ax1 = fig.add_subplot(111)
        ax1.plot(x, y1, "k-", linewidth=0.5, alpha=0.5)
        ax1.set_ylabel("Surface Temperature[$\\degree C$]")

        ax1t = ax1.twinx()
        ax1t.scatter(df_temp.When, df_temp.Temp, color="b", alpha=0.5, s=1)
        ax1t.set_ylabel("Cam Temp [$\\degree C$]", color="b")
        for tl in ax1t.get_yticklabels():
            tl.set_color("b")

        ax1.xaxis.set_major_locator(mdates.WeekdayLocator())
        ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
        ax1.xaxis.set_minor_locator(mdates.DayLocator())
        ax1.set_ylim([-20, 5])
        ax1t.set_ylim([-20, 5])
        fig.autofmt_xdate()
        pp.savefig(bbox_inches="tight")
        plt.close("all")

        fig, (ax1, ax3, ax4) = plt.subplots(
            nrows=3, ncols=1, sharex="col", sharey="row", figsize=(15, 12)
        )

        x = self.df.When

        y1 = self.df.a
        y2 = self.df.f_cone
        ax1.plot(x, y1, "k-")
        ax1.set_ylabel("Albedo")
        # ax1.set_xlabel("Days")
        ax1t = ax1.twinx()
        ax1t.plot(x, y2, "b-", linewidth=0.5)
        ax1t.set_ylabel("$f_{cone}$", color="b")
        for tl in ax1t.get_yticklabels():
            tl.set_color("b")
        ax1.grid()

        # y1 = self.df.e_a
        # y2 = self.df.cld
        # ax2.plot(x, y1, "k-")
        # ax2.set_ylabel("Atmospheric Emissivity")
        # # ax1.set_xlabel("Days")
        # ax2t = ax2.twinx()
        # ax2t.plot(x, y2, "b-", linewidth=0.5)
        # ax2t.set_ylabel("Cloudiness", color="b")
        # for tl in ax2t.get_yticklabels():
        #     tl.set_color("b")
        # ax2.grid()

        y3 = self.df.vp_a - self.df.vp_ice
        ax3.plot(x, y3, "k-", linewidth=0.5)
        ax3.set_ylabel("Vapour gradient [$hPa$]")
        ax3.grid()

        y4 = self.df.T_a - self.df.T_s
        ax4.plot(x, y4, "k-", linewidth=0.5)
        ax4.set_ylabel("Temperature gradient [$\\degree C$]")
        ax4.grid()

        ax1.xaxis.set_major_locator(mdates.WeekdayLocator())
        ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
        ax1.xaxis.set_minor_locator(mdates.DayLocator())
        fig.autofmt_xdate()
        pp.savefig(bbox_inches="tight")
        plt.close("all")

        dfd = self.df.set_index("When").resample("D").mean().reset_index()
        dfd["Discharge"] = dfd["Discharge"] == 0
        dfd["Discharge"] = dfd["Discharge"].astype(int)
        dfd["Discharge"] = dfd["Discharge"].astype(str)
        dfd["When"] = dfd["When"].dt.strftime("%b %d")

        dfd["label"] = " "

        for i in range(0, dfd.shape[0]):
            if i % 7 == 0:
                dfd.loc[i, "label"] = dfd.When[i]

        dfd = dfd.set_index("label")

        z = dfd[["$q_{SW}$", "$q_{LW}$", "$q_S$", "$q_L$", "$q_{F}$", "$q_{G}$"]]
        ax = z.plot.bar(stacked=True, edgecolor=dfd["Discharge"], linewidth=0.5)
        ax.xaxis.set_label_text("")
        plt.grid(axis="y", color="black", alpha=0.3, linewidth=0.5, which="major")
        # plt.xlabel('Date')
        plt.ylabel("Energy Flux Density [$W\\,m^{-2}$]")
        plt.legend(loc="best")
        plt.xticks(rotation=45)
        pp.savefig(bbox_inches="tight")
        plt.close("all")

        dfds = self.df[["When", "thickness", "SA"]]

        with pd.option_context("mode.chained_assignment", None):
            dfds["negative"] = dfds.loc[dfds.thickness < 0, "thickness"]
            dfds["positive"] = dfds.loc[dfds.thickness >= 0, "thickness"]

        dfds1 = dfds.set_index("When").resample("D").sum().reset_index()
        dfds2 = dfds.set_index("When").resample("D").mean().reset_index()

        dfds2["When"] = dfds2["When"].dt.strftime("%b %d")
        dfds2["label"] = " "

        for i in range(0, dfd.shape[0]):
            if i % 7 == 0:
                dfds2.loc[i, "label"] = dfd.When[i]

        dfds2 = dfds2.set_index("label")

        dfds1 = dfds1.rename(columns={"positive": "Ice", "negative": "Meltwater"})
        y1 = dfds1[["Ice", "Meltwater"]]
        y3 = dfds2["SA"]

        fig = plt.figure()
        ax1 = fig.add_subplot(2, 1, 1)
        y1.plot(
            kind="bar",
            stacked=True,
            edgecolor="black",
            linewidth=0.5,
            color=["#D9E9FA", "#0C70DE"],
            ax=ax1,
        )
        plt.ylabel("Thickness ($m$ w. e.)")
        plt.xticks(rotation=45)
        # plt.legend(loc='upper right')
        ax1.set_ylim(-0.055, 0.055)
        ax1.yaxis.set_minor_locator(AutoMinorLocator())
        ax1.grid(axis="y", color="black", alpha=0.3, linewidth=0.5, which="major")
        x_axis = ax1.axes.get_xaxis()
        x_axis.set_visible(False)

        ax3 = fig.add_subplot(2, 1, 2)
        ax = y3.plot.bar(
            y="SA", linewidth=0.5, edgecolor="black", color="xkcd:grey", ax=ax3
        )
        ax.xaxis.set_label_text("")
        ax3.set_ylabel("Surface Area ($m^2$)")
        ax3.grid(axis="y", color="black", alpha=0.3, linewidth=0.5, which="major")
        plt.xticks(rotation=45)
        pp.savefig(bbox_inches="tight")
        plt.clf()

        plt.close("all")

        pp.close()

    def paper_figures(self):
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

        # pp = PdfPages(FOLDERS["output_folder"] + "Figure_3.pdf")

        fig, (ax1, ax2, ax3, ax4, ax5, ax6) = plt.subplots(
            nrows=6, ncols=1, sharex="col", sharey="row", figsize=(12, 18)
        )

        x = self.df.When

        y1 = self.df.Discharge
        ax1.plot(x, y1, linestyle="-", color="#284D58", linewidth=1)
        ax1.set_ylabel("Fountain Spray [$l\\, min^{-1}$]")
        # ax1.grid()

        ax1t = ax1.twinx()
        # ax1t.plot(x, self.df.Prec * 1000, linestyle='--', color='#118ab2', linewidth=1)
        ax1t.plot(
            x,
            self.df.Prec * 1000 * self.TIME_STEP,
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
        # ax2.plot(x, y2_ERA5, linestyle='-', color='#e76f51', linewidth=1)
        for ev in events:
            ax2.axvspan(
                ev.head(1).values, ev.tail(1).values, facecolor="xkcd:grey", alpha=0.25
            )
        ax2.plot(x, y2_ERA5, linestyle="-", color="#284D58")
        ax2.set_ylabel("Temperature [$\\degree C$]")
        # ax2.grid()

        y3 = self.df.SW_direct
        # lns1 = ax3.plot(x, y3, linestyle='--', color='#e76f51', linewidth=0.5,  label = 'Shortwave Direct')
        # lns2 = ax3.plot(x, self.df.SW_diffuse, color='#e76f51',linestyle=':', linewidth=2, label = 'Shortwave Diffuse')
        # lns3 = ax3.plot(x, self.df.LW_in, color='#e76f51',linestyle='-', linewidth=1, label = 'Longwave')
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
        # ax3.grid()

        # added these three lines
        lns = lns1 + lns2 + lns3
        labs = [l.get_label() for l in lns]
        ax3.legend(lns, labs, ncol=3, loc="best")

        y4 = df_SZ.RH
        y4_ERA5 = df_ERA5.RH
        ax4.plot(x, y4, linestyle="-", color="#284D58", linewidth=1)
        # ax4.plot(x, y4_ERA5, linestyle='-', color='#e76f51', linewidth=1)
        ax4.plot(x, y4_ERA5, linestyle="-", color="#284D58")
        for ev in events:
            ax4.axvspan(
                ev.head(1).values, ev.tail(1).values, facecolor="grey", alpha=0.25
            )
        ax4.set_ylabel("Humidity [$\\%$]")
        # ax4.grid()

        y5 = df_SZ.p_a
        y5_ERA5 = df_ERA5.p_a
        ax5.plot(x, y5, linestyle="-", color="#264653", linewidth=1)
        # ax5.plot(x, y5_ERA5, linestyle='-', color='#e76f51', linewidth=1)
        ax5.plot(x, y5_ERA5, linestyle="-", color="#284D58")
        for ev in events:
            ax5.axvspan(
                ev.head(1).values, ev.tail(1).values, facecolor="grey", alpha=0.25
            )
        ax5.set_ylabel("Pressure [$hPa$]")
        # ax5.grid()

        y6 = df_SZ.v_a
        y6_ERA5 = df_ERA5.v_a
        y6_ERA52 = df_ERA52.v_a
        ax6.plot(x, y6, linestyle="-", color="#264653", linewidth=1, label="Schwarzsee")
        # ax6.plot(x, y6_ERA5, linestyle='-', color='#e76f51', linewidth=1)
        # ax6.plot(x, y6_ERA52, linestyle='-', color='#e76f51', linewidth=1)
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
        # ax6.grid()

        # y7 = self.df.cld
        # # ax7.plot(x, y7, linestyle='-', color='#e76f51', linewidth=1)
        # ax7.plot(x, y7, linestyle='-',color='#284D58', label = "ERA5")
        # ax7.set_ylabel("Cloudiness []")
        # ax7.axvspan(self.df.When.head(1).values, self.df.When.tail(1).values,facecolor='grey', alpha=0.25)
        # # ax7.grid()

        # ERA5= mpatches.Patch(color='grey', alpha=0.25, label='ERA5 data')
        # SZ= mpatches.Patch(color='#264653', label='Schwarzsee data')
        # PLF= mpatches.Patch(color=CB91_Blue, label='Plaffeien data')

        # ax1.legend(handles = [ERA5, PLF], bbox_to_anchor=(0,1.02,1,0.2), loc="lower left",
        #          borderaxespad=1, ncol=4, fancybox=True)

        ax1.xaxis.set_major_locator(mdates.WeekdayLocator())
        ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
        ax1.xaxis.set_minor_locator(mdates.DayLocator())
        fig.autofmt_xdate()
        plt.savefig(
            FOLDERS["output_folder"] + "jpg/Figure_3.jpg", dpi=300, bbox_inches="tight"
        )
        # pp.savefig(bbox_inches="tight")
        plt.clf()
        # pp.close()

        # pp = PdfPages(FOLDERS["output_folder"] + "Figure_6.pdf")
        fig = plt.figure(figsize=(12, 14))
        dfds = self.df[
            ["When", "solid", "ppt", "dpt", "melted", "gas", "SA", "iceV", "Discharge"]
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

        dfds = dfds.set_index("When").resample("D").sum().reset_index()
        dfds["When"] = dfds["When"].dt.strftime("%b %d")

        dfds["label"] = " "
        labels = [
            "Jan 30",
            "Feb 05",
            "Feb 12",
            "Feb 19",
            "Feb 26",
            "Mar 05",
            "Mar 12",
            "Mar 19",
            "Mar 26",
            "Apr 02",
        ]
        for i in range(0, dfds.shape[0]):
            for item in labels:
                if dfds.When[i] == item:
                    dfds.loc[i, "label"] = dfds.When[i]

        dfds = dfds.set_index("label")
        dfds = dfds.rename(
            columns={
                "solid": "Ice",
                "ppt": "Snow",
                "dpt": "Vapour cond./dep.",
                "melted": "Melt",
                "gas": "Vapour sub./evap.",
            }
        )

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

        dfd["label"] = " "
        labels = [
            "Jan 30",
            "Feb 05",
            "Feb 12",
            "Feb 19",
            "Feb 26",
            "Mar 05",
            "Mar 12",
            "Mar 19",
            "Mar 26",
            "Apr 02",
        ]
        for i in range(0, dfd.shape[0]):
            for item in labels:
                if dfd.When[i] == item:
                    dfd.loc[i, "label"] = dfd.When[i]

        dfd = dfd.set_index("label")

        dfds2 = self.df.set_index("When").resample("D").mean().reset_index()
        dfds2["When"] = dfds2["When"].dt.strftime("%b %d")
        dfds2["label"] = " "
        labels = [
            "Jan 30",
            "Feb 05",
            "Feb 12",
            "Feb 19",
            "Feb 26",
            "Mar 05",
            "Mar 12",
            "Mar 19",
            "Mar 26",
            "Apr 02",
        ]
        for i in range(0, dfds2.shape[0]):
            for item in labels:
                if dfds2.When[i] == item:
                    dfds2.loc[i, "label"] = dfds2.When[i]
        dfds2 = dfds2.set_index("label")
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
        # plt.legend(bbox_to_anchor=(0,1.02,1,0.2), loc="lower left",
        #            mode="expand", borderaxespad=0, ncol=6)
        plt.legend(loc="upper center", ncol=6)
        plt.ylim(-125, 125)
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
        # plt.legend(bbox_to_anchor=(0,1.02,1,0.2), loc="lower left",
        #            mode="expand", borderaxespad=0, ncol=6)
        plt.legend(loc="upper center", ncol=6)
        ax2.set_ylim(-0.03, 0.03)
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
        plt.xticks(rotation=45)
        plt.tight_layout()
        # pp.savefig(bbox_inches="tight")
        plt.savefig(
            FOLDERS["output_folder"] + "jpg/Figure_6.jpg", dpi=300, bbox_inches="tight"
        )
        plt.clf()
        # pp.close()

        # pp = PdfPages(FOLDERS["output_folder"] + "Figure_7.pdf")
        fig, (ax1, ax2) = plt.subplots(
            nrows=2, ncols=1, sharex="col", sharey="row", figsize=(15, 12)
        )

        x = self.df.When

        y1 = self.df.a
        y2 = self.df.f_cone
        ax1.plot(x, y1, color="#16697a")
        ax1.set_ylabel("Albedo")
        # ax1.set_xlabel("Days")
        ax1t = ax1.twinx()
        ax1t.plot(x, y2, color="#ff6d00", linewidth=0.5)
        ax1t.set_ylabel("$f_{cone}$", color="#ff6d00")
        for tl in ax1t.get_yticklabels():
            tl.set_color("#ff6d00")
        # ax1.grid()
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
            FOLDERS["output_folder"] + "jpg/Figure_7.jpg", dpi=300, bbox_inches="tight"
        )
        # pp.savefig(bbox_inches="tight")
        plt.close("all")
        # pp.close()


if __name__ == "__main__":
    start = time.time()
    logger = logging.getLogger(__name__)
    coloredlogs.install(
        fmt="%(levelname)s %(message)s",
        logger=logger,
    )
    logger.error('Model begins')

    icestupa = PDF()

    # icestupa.derive_parameters()

    icestupa.read_input()

    # icestupa.melt_freeze()

    icestupa.read_output()

    # icestupa.corr_plot()

    # icestupa.summary()

    # icestupa.print_input()
    # icestupa.paper_figures()

    # icestupa.print_output()

    total = time.time() - start

    logger.info("Total time  : %.2f", total / 60)
