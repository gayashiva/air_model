"""Icestupa class object definition
"""

# External modules
import pickle
pickle.HIGHEST_PROTOCOL = 4  # For python version 2.7
import pandas as pd
import sys, os, math, json
import numpy as np
import logging
import pytz
from tqdm import tqdm
from codetiming import Timer
from datetime import timedelta

# Locals
dirname = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.append(dirname)
from src.models.methods.solar import get_solar
from src.utils.settings import config

# Module logger
logger = logging.getLogger("__main__")
logger.propagate = False

class Icestupa:
    def __init__(self, location="Guttannen 2021", spray="unscheduled_field"):

        self.spray = spray

        with open("data/common/constants.json") as f:
            CONSTANTS = json.load(f)

        SITE, FOLDER = config(location, spray)
        diff = SITE["expiry_date"] - SITE["start_date"]
        days, seconds = diff.days, diff.seconds
        self.total_hours = days * 24 + seconds // 3600

        initialize = [CONSTANTS, SITE, FOLDER]

        for dictionary in initialize:
            for key in dictionary:
                setattr(self, key, dictionary[key])
                logger.info(f"%s -> %s" % (key, str(dictionary[key])))

        # Initialize input dataset
        self.df = pd.read_csv(self.input + "aws.csv", sep=",", header=0, parse_dates=["time"])
        if "Discharge" not in list(self.df.columns):
            df_f = pd.read_csv(self.input + "discharge_types.csv", sep=",", header=0, parse_dates=["time"])
            df_f["Discharge"] = df_f[self.spray]
            df_f = df_f[["time", "Discharge"]]
            if np.count_nonzero(df_f["Discharge"]) <= 24:
                logger.error("Less than 24 hours of spray")
                sys.exit()

            # Perform discharge atleast one night check
            self.df = self.df.set_index("time")
            df_f = df_f.set_index("time")
            self.df["Discharge"] = df_f["Discharge"]
            self.df["Discharge"] = self.df["Discharge"].replace(np.NaN, 0)
            self.df = self.df.reset_index()

            self.D_F = df_f.Discharge[df_f.Discharge != 0].mean()
            print("\n") 
            logger.warning("Discharge mean of %s method is %.1f\n" % (self.spray, self.D_F))
        else:
            self.D_F = self.df.Discharge[self.df.Discharge != 0].mean()
            print("\n") 
            logger.warning("Discharge mean of %s method is %.1f\n" % (self.spray, self.D_F))

        # Reset date range
        self.df = self.df.set_index("time")
        self.df = self.df[SITE["start_date"] : SITE["expiry_date"]]
        self.df = self.df.reset_index()

        logger.debug(self.df.head())
        logger.debug(self.df.tail())

    # Imported methods
    from src.models.methods._freq import change_freq
    from src.models.methods._self_attributes import self_attributes
    from src.models.methods._albedo import get_albedo
    # from src.models.methods._discharge import get_discharge
    from src.models.methods._area import get_area
    from src.models.methods._temp import get_temp, test_get_temp
    from src.models.methods._energy import get_energy, test_get_energy
    from src.models.methods._figures import summary_figures

    @Timer(
        text="Preprocessed data in {:.2f} seconds",
        logger=logging.getLogger("__main__").warning,
    )
    def gen_input(self):  # Use processed input dataset
        unknown = [
            "alb",
            "vp_a",
            "LW_in",
            "SW_global",
            # "T_F",
        ]  # Possible unknown variables

        for i in range(len(unknown)):
            if unknown[i] in list(self.df.columns):
                unknown[i] = np.NaN  # Removes known variable
            else:
                logger.warning(" %s is unknown\n" % (unknown[i]))
                self.df[unknown[i]] = 0

        solar_df = get_solar(
            coords=self.coords,
            start=self.start_date,
            end=self.df["time"].iloc[-1],
            DT=self.DT,
            alt=self.alt,
            # ghi=self.df.set_index("time")["SW_global"],
            # press=self.df["press"].mean(),
        )

        self.df = pd.merge(solar_df, self.df, on="time", how="left")

        if "SW_global" in unknown:
            self.df["SW_global"] = self.df["ghi"]
            logger.warning(f"Estimated global solar from pvlib\n")

        # self.df["SW_diffuse"]= np.where(self.df.ppt > 0, self.df["SW_global"], 0)
        # self.df["SW_direct"] = self.df["SW_global"] - self.df["SW_diffuse"]
        # logger.warning(f"Estimated solar components using precipitation\n")

        if 'cld' in list(self.df.columns):
            self.df["SW_direct"] = (1- self.df["cld"]) * self.df["SW_global"]
            self.df["SW_diffuse"] = self.df["cld"] * self.df["SW_global"]
            logger.warning(f"Estimated solar components with ERA5 cloudiness with mean {self.df.cld.mean()}\n")
        else:
            self.df["SW_direct"] = (1- self.cld) * self.df["SW_global"]
            self.df["SW_diffuse"] = self.cld * self.df["SW_global"]
            logger.warning(f"Estimated solar components with constant cloudiness of {self.cld}\n")

        if "T_G" in list(self.df.columns):
            self.df["T_F"] = self.df["T_G"]
            logger.warning(f"Measured ground temp is fountain water temp with mean {self.df.T_F.mean()}\n")
        else:
            self.df["T_F"] = float(self.T_F)
            logger.warning(f"Estimated constant fountain water temp is {self.T_F}\n")

            
        for row in tqdm(
            self.df[1:].itertuples(),
            total=self.df.shape[0],
            desc="Creating AIR input...",
        ):
            i = row.Index

            """ Vapour Pressure"""
            if "vp_a" in unknown:

                self.df.loc[i, "vp_a"] = np.exp(
                    34.494 - 4924.99/ (row.temp + 237.1)
                ) / ((row.temp + 105) ** 1.57 * 100)
                self.df.loc[i, "vp_a"] *= row.RH/100

            """LW incoming"""
            if "LW_in" in unknown:

                self.df.loc[i, "e_a"] = (
                    1.24
                    * math.pow(abs(self.df.loc[i, "vp_a"] / (row.temp + 273.15)), 1 / 7)
                )

                if 'cld' in list(self.df.columns):
                    self.df.loc[i, "e_a"] *= (1 + 0.22 * math.pow(self.df.loc[i,"cld"], 2))
                else:
                    self.df.loc[i, "e_a"] *= (1 + 0.22 * math.pow(self.cld, 2))


                self.df.loc[i, "LW_in"] = (
                    self.df.loc[i, "e_a"] * self.sigma * math.pow(row.temp + 273.15, 4)
                )

            """Water temperature"""
            if row.temp < 0:
                self.df.loc[i,"T_F"] = 0

        logger.warning(f"Variable fountain water temp mean is {self.df.T_F.mean()}\n")

        self.self_attributes()

        if "alb" in unknown:
            self.A_DECAY = self.A_DECAY * 24 * 60 * 60 / self.DT
            s = 0
            f = 1
            for row in self.df.itertuples():
                i = row.Index
                s, f = self.get_albedo(i, s, f)

        self.df = self.df.round(3)

        if self.df.isnull().values.any():
            for column in self.df.columns:
                if self.df[column].isna().sum() > 0:
                    logger.warning(" Null values interpolated in %s" % column)
                    self.df.loc[:, column] = self.df[column].interpolate()

        self.df.to_csv(
            self.input_sim + "/input.csv",
        )
        self.df.to_hdf(
            self.input_sim + "/input.h5",
            key="df",
            mode="a",
        )
        logger.debug(self.df.head())
        logger.debug(self.df.tail())
        if 'index' in self.df.columns:
            logger.error("Index present")

    def gen_output(self):  # Use processed input dataset

        results_dict = {}
        results = [
            "iceV_max",
            "last_hour",
            "M_input",
            "M_F",
            "M_ppt",
            "M_dep",
            "M_water",
            "M_waste",
            "M_sub",
            "M_ice",
            "last_hour",
            "R_F",
            "D_F",
            "WUE",
        ]
        iceV_max = self.df["iceV"].max()
        M_input = self.df["input"].iloc[-1]
        M_F = self.df["Discharge"].sum() * self.DT / 60 + self.df.loc[0, "input"]
        M_ppt = self.df["snow2ice"].sum()
        M_dep = self.df["dep"].sum()
        M_water = self.df["meltwater"].iloc[-1]
        M_waste = self.df["wastewater"].iloc[-1]
        M_sub = self.df["vapour"].iloc[-1]
        M_ice = self.df["ice"].iloc[-1]
        last_hour = self.df.shape[0]
        R_F = self.R_F
        D_F = self.D_F
        WUE = int((M_ice + M_water) / M_input * 100)

        if self.spray.split('_')[1] == 'field':
            results.append("RMSE")
            if self.name in ["guttannen22"]:
                df_c = pd.read_hdf(self.input_sim  + "/input.h5", "df_c")
            else:
                df_c = pd.read_hdf(self.input  + "/input.h5", "df_c")
            df_c = df_c[["time", "DroneV", "DroneVError"]]

            df_c = df_c.set_index("time")
            df = self.df.set_index("time")
            RMSE = ((df['iceV'].subtract(df_c['DroneV'],axis=0))**2).mean()**.5

        # For web app
        for variable in results:
            results_dict[variable] = float(round(eval(variable), 1))

        logger.warning("Summary of results for %s with scheduler %s  :" %(self.name, self.spray))
        for var in sorted(results_dict.keys()):
            logger.warning("\t%s: %r" % (var, results_dict[var]))
        # print()

        with open(self.output + "results.json", "w") as f:
            json.dump(results_dict, f, sort_keys=True, indent=4)

        if last_hour > self.total_hours + 1:
            self.df = self.df[: self.total_hours]
        else:
            for j in range(last_hour, self.total_hours):
                for col in self.df.columns:
                    if col not in ["temp"]:
                        self.df.loc[j, col] = 0
                    if col in ["iceV"]:
                        self.df.loc[j, col] = self.V_dome
                    if col in ["time"]:
                        self.df.loc[j, col] = self.df.loc[j - 1, col] + timedelta(
                            hours=1
                        )

        self.df = self.df.reset_index(drop=True)

        # Full Output
        self.df.to_csv(self.output + "/output.csv", sep=",")
        self.df.to_hdf(
            self.output  + "/output.h5",
            key="df",
            mode="w",
        )

    def read_input(self):  # Use processed input dataset

        self.df = pd.read_hdf(self.input_sim  + "/input.h5", "df")

        if self.df.isnull().values.any():
            logger.warning("\n Null values present\n")

    def read_output(self):  # Reads output

        self.df = pd.read_hdf(self.output + "/output.h5", "df")

        self.self_attributes()

        with open(self.output + "/results.json", "r") as read_file:
            results_dict = json.load(read_file)

        # Initialise all variables of dictionary
        for key in results_dict:
            setattr(self, key, results_dict[key])
            logger.warning(f"%s -> %s" % (key, str(results_dict[key])))


    # @Timer(text="Simulation executed in {:.2f} seconds", logger=logging.NOTSET)
    def sim_air(self, test=False):

        # Initialisaton for sites
        all_cols = [
            "T_s",
            "T_bulk",
            "f_cone",
            "ice",
            "iceV",
            "sub",
            "vapour",
            "melted",
            "delta_T_s",
            "wastewater",
            "Qtotal",
            "SW",
            "LW",
            "Qs",
            "Ql",
            "Qf",
            "Qg",
            "meltwater",
            "A_cone",
            "h_cone",
            "r_cone",
            "dr",
            "snow2ice",
            "rain2ice",
            "dep",
            "j_cone",
            "wasted",
            "fountain_froze",
            "Qt",
            "Qmelt",
            "Qfreeze",
            "input",
            "event",
            "rho_air",
        ]

        for column in all_cols:
            if column in ["event"]:
                self.df[column] = np.nan
            else:
                self.df[column] = 0

        # Initialise first model time step
        self.df.loc[0, "h_cone"] = self.h_i
        self.df.loc[0, "r_cone"] = self.R_F
        self.df.loc[0, "dr"] = self.DX
        self.df.loc[0, "s_cone"] = self.df.loc[0, "h_cone"] / self.df.loc[0, "r_cone"]
        V_initial = math.pi / 3 * self.R_F ** 2 * self.h_i
        self.df.loc[1, "rho_air"] = self.RHO_I
        self.df.loc[1, "ice"] = V_initial* self.df.loc[1, "rho_air"]
        self.df.loc[1, "iceV"] = V_initial
        self.df.loc[1, "input"] = self.df.loc[1, "ice"]

        logger.warning(
            "Initialise: time %s, radius %.3f, height %.3f, iceV %.3f\n"
            % (
                self.df.loc[0, "time"],
                self.df.loc[0, "r_cone"],
                self.df.loc[0, "h_cone"],
                self.df.loc[1, "iceV"],
            )
        )

        t = tqdm(
            self.df[1:-1].itertuples(),
            total=self.total_hours,
        )

        t.set_description("Simulating %s Icestupa" % self.name)

        for row in t:
            i = row.Index

            ice_melted = self.df.loc[i, "iceV"] < self.V_dome

            if ice_melted:
                if (
                    self.df.loc[i - 1, "time"] < self.fountain_off_date
                    and self.df.loc[i - 1, "melted"] > 0
                ):
                    logger.error("Skipping %s" % self.df.loc[i, "time"])

                    # Initialise first model time step
                    for column in all_cols:
                        if column in ["event"]:
                            self.df[column] = np.nan
                        else:
                            self.df[column] = 0

                    # Initialise first model time step
                    self.df.loc[i-1, "h_cone"] = self.h_i
                    self.df.loc[i-1, "r_cone"] = self.R_F
                    self.df.loc[i-1, "dr"] = self.DX
                    self.df.loc[i-1, "s_cone"] = self.df.loc[0, "h_cone"] / self.df.loc[0, "r_cone"]
                    V_initial = math.pi / 3 * self.R_F ** 2 * self.h_i
                    self.df.loc[i, "rho_air"] = self.RHO_I
                    self.df.loc[i, "ice"] = V_initial* self.df.loc[i, "rho_air"]
                    self.df.loc[i, "iceV"] = V_initial
                    self.df.loc[i, "input"] = self.df.loc[i, "ice"]

                    logger.warning(
                        "Initialise: time %s, radius %.3f, height %.3f, iceV %.3f\n"
                        % (
                            self.df.loc[i-1, "time"],
                            self.df.loc[i-1, "r_cone"],
                            self.df.loc[i-1, "h_cone"],
                            self.df.loc[i, "iceV"],
                        )
                    )

                    # self.df.loc[i - 1, "h_cone"] = self.h_i
                    # self.df.loc[i - 1, "r_cone"] = self.R_F
                    # self.df.loc[i - 1, "s_cone"] = (
                    #     self.df.loc[i - 1, "h_cone"] / self.df.loc[i - 1, "r_cone"]
                    # )
                    # self.df.loc[i, "ice"] = V_initial * self.RHO_I
                    # self.df.loc[i, "iceV"] = V_initial
                    # self.df.loc[i, "input"] = self.df.loc[i, "ice"]

                    # logger.warning(
                    #     "Initialise again: time %s, radius %.3f, height %.3f, iceV %.3f\n"
                    #     % (
                    #         self.df.loc[i - 1, "time"],
                    #         self.df.loc[i - 1, "r_cone"],
                    #         self.df.loc[i - 1, "h_cone"],
                    #         self.df.loc[i, "iceV"],
                    #     )
                    # )

                else:

                    col_list = [
                        "dep",
                        "snow2ice",
                        "fountain_froze",
                        "wasted",
                        "sub",
                        "melted",
                    ]
                    for column in col_list:
                        self.df.loc[i - 1, column] = 0

                    last_hour = i - 1
                    self.df = self.df[1:i]
                    self.df = self.df.reset_index(drop=True)
                    break

            self.get_area(i)

            # Precipitation 
            if self.df.loc[i, "ppt"] > 0:

                if self.df.loc[i, "temp"] < self.T_PPT:
                    self.df.loc[i, "snow2ice"] = (
                        self.RHO_W
                        * self.df.loc[i, "ppt"]
                        / 1000
                        * math.pi
                        * math.pow(self.df.loc[i, "r_cone"], 2)
                    )
                else:
                # If rain add to discharge and change temperature
                    self.df.loc[i, "rain2ice"] = (
                        self.RHO_W
                        * self.df.loc[i, "ppt"]
                        / 1000
                        * math.pi
                        * math.pow(self.df.loc[i, "r_cone"], 2)
                    )
                    # self.df.loc[i, "Discharge"] += self.df.loc[i, "rain2ice"]/60
                    self.df.loc[i, "snow2ice"] = 0
                    logger.info(f"Rain event on {self.df.time.loc[i]} with temp {self.df.temp.loc[i]}")
            else:
                self.df.loc[i, "snow2ice"] = 0

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
                    self.df.loc[i, "Ql"] * self.DT * self.df.loc[i, "A_cone"] / L
                )
            else:
                L = self.L_S
                self.df.loc[i, "dep"] = (
                    self.df.loc[i, "Ql"] * self.DT * self.df.loc[i, "A_cone"] / self.L_S
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
                + self.df.loc[i, "fountain_froze"]
                + self.df.loc[i, "dep"]
                + self.df.loc[i, "snow2ice"]
                - self.df.loc[i, "sub"]
                - self.df.loc[i, "melted"]
            )

            self.df.loc[i + 1, "vapour"] = (
                self.df.loc[i, "vapour"] + self.df.loc[i, "sub"]
            )
            self.df.loc[i + 1, "wastewater"] = (
                self.df.loc[i, "wastewater"] + self.df.loc[i, "wasted"]
            )

            # if self.df.loc[:i, "fountain_froze"].sum() + self.df.loc[:i,"dep"].sum() + self.df.loc[:i,"snow2ice"].sum() == 0:
            #     self.df.loc[i + 1, "rho_air"] = self.RHO_I
            # else:

            if self.name in ['guttannen21', 'gangles21']:
                self.df.loc[i + 1, "rho_air"] = self.RHO_I
            else:
                self.df.loc[i + 1, "rho_air"] =(
                        (self.df.loc[1, "ice"] + self.df.loc[:i, "fountain_froze"].sum()+self.df.loc[:i,"dep"].sum()+self.df.loc[:i,"snow2ice"].sum())
                        /(( self.df.loc[1, "ice"] + self.df.loc[:i, "fountain_froze"].sum()+self.df.loc[:i, "dep"].sum())/self.RHO_I
                        +(self.df.loc[:i, "snow2ice"].sum()/self.RHO_S))
                )

            self.df.loc[i + 1, "iceV"] = self.df.loc[i + 1, "ice"]/self.df.loc[i+1, "rho_air"]

            self.df.loc[i + 1, "input"] = (
                self.df.loc[i, "input"]
                + self.df.loc[i, "snow2ice"]
                + self.df.loc[i, "rain2ice"]
                + self.df.loc[i, "dep"]
                + self.df.loc[i, "Discharge"] * self.DT / 60
            )
            self.df.loc[i + 1, "j_cone"] = (
                self.df.loc[i + 1, "iceV"] - self.df.loc[i, "iceV"]
            ) / (self.df.loc[i, "A_cone"])

            if test and not ice_melted:
                logger.info(f"time {self.df.time[i]}, rho_air {self.df.rho_air[i+1]}, iceV {self.df.iceV[i]}")
        # else:
            # print(self.df.loc[i, "time"], self.df.loc[i, "iceV"])
