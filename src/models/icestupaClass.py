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
from pvlib import atmosphere

# Locals
dirname = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.append(dirname)
from src.models.methods.solar import get_solar
from src.utils.settings import config
from src.plots.data import plot_input

# Module logger
logger = logging.getLogger("__main__")
logger.propagate = False

class Icestupa:
    def __init__(self, location="Guttannen 2021", spray="none_none"):

        self.spray = spray

        with open("constants.json") as f:
            CONSTANTS = json.load(f)

        SITE, FOLDER = config(location, spray)

        initialize = [CONSTANTS, SITE, FOLDER]

        for dictionary in initialize:
            for key in dictionary:
                setattr(self, key, dictionary[key])
                logger.info(f"%s -> %s" % (key, str(dictionary[key])))

        # Initialize input dataset
        self.df = pd.read_csv(self.input + "aws.csv", sep=",", header=0, parse_dates=["time"])
        if self.spray == "ERA5_":
            self.start_date = self.df.time[0]
            self.expiry_date = self.df.time[self.df.shape[0]-1]

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
        # self.df = self.df[self.start_date : self.expiry_date]
        self.df = self.df.reset_index()

        logger.debug(self.df.head())
        logger.debug(self.df.tail())

    # Imported methods
    from src.models.methods._freq import change_freq
    from src.models.methods._self_attributes import self_attributes
    # from src.models.methods._albedo import get_albedo
    from src.models.methods._area import get_area
    from src.models.methods._temp import get_temp, test_get_temp
    from src.models.methods._energy import get_energy, test_get_energy
    from src.models.methods._figures import summary_figures

    def read_input(self):  # Use processed input dataset

        self.df = pd.read_hdf(self.input_sim  + "/input.h5", "df")

        if self.df.isnull().values.any():
            logger.warning("\n Null values present\n")

    def read_output(self):  # Reads output

        self.df = pd.read_hdf(self.output + "output.h5", "df")

        self.self_attributes()

        # with open(self.output + "/results.json", "r") as read_file:
        #     results_dict = json.load(read_file)

        # # Initialise all variables of dictionary
        # for key in results_dict:
        #     setattr(self, key, results_dict[key])
        #     logger.warning(f"%s -> %s" % (key, str(results_dict[key])))


    # @Timer(text="Simulation executed in {:.2f} seconds", logger=logging.NOTSET)
    def sim_air(self, test=False):

        """Solar radiation"""
        solar_df = get_solar(
            coords=self.coords,
            start=self.start_date,
            end=self.df["time"].iloc[-1],
            DT=self.DT,
            alt=self.alt,
        )
        self.df = pd.merge(solar_df, self.df, on="time", how="left")
        if self.df.isna().values.any():
            logger.warning(self.df[self.df.columns].isna().sum())
            self.df= self.df.interpolate(method='ffill', axis=0)
            logger.warning(f"Filling nan values created by solar module\n")

        self.df["tau_atm"]= self.df["SW_global"]/self.df["SW_extra"]
        # self.df= self.df.rename(columns={"ghi": "SW_global"})
        # plot_input(self.df, self.fig, self.name)
        # logger.warning(f"Estimated global solar from pvlib\n")
        self.df["SW_direct"] = (1- self.df["tcc"]) * self.df["SW_global"]
        self.df["SW_diffuse"] = self.df["tcc"] * self.df["SW_global"]
        logger.error(f"Estimated solar components with average cloudiness of {self.df.tcc.mean():.2f}\n")
        # logger.warning(f"Estimated solar components with constant cloudiness of {self.cld}\n")
        # self.df["SW_direct"] = self.df["tau_atm"] * self.df["SW_global"]
        # self.df["SW_diffuse"] = self.df["SW_global"] - self.df["SW_direct"]
        # logger.warning(f"Estimated solar components with mean atmospheric transmittivity of {self.df.tau_atm.mean()}\n")

        """Pressure"""
        self.df["press"] = atmosphere.alt2pres(self.alt) / 100
        logger.warning(f"Estimated pressure from altitude\n")

        # """Albedo"""
        # self.A_DECAY = self.A_DECAY * 24 * 60 * 60 / self.DT
        # s = 0
        # f = 1
        # for row in self.df.itertuples():
        #     i = row.Index
        #     s, f = self.get_albedo(i, s, f)

        self.df = self.df.round(3)

        # Initialisaton for sites
        all_cols = [
            "T_F",
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
            # "snow2ice",
            # "rain2ice",
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
            "tau_atm",
        ]

        for column in all_cols:
            if column in ["event"]:
                self.df[column] = np.nan
            else:
                self.df[column] = 0

        # Resample to daily minimum temperature
        daily_min_temps = self.df.set_index("time")['temp'].resample('D').min()

        # Find longest consecutive period
        current_period = 0
        start_date_list = []
        crit_temp = 0

        for date, temp in daily_min_temps.items():
            if temp < crit_temp:
                current_period += 1
                if current_period == self.minimum_period:
                    start_date_list.append(date - pd.DateOffset(days=current_period - 1))
            else:
                current_period = 0

        logger.warning(f"Cold windows: {start_date_list}")
        self.self_attributes()

        day_index = self.df.index[self.df['time'].dt.date==start_date_list[0]][0]

        # Initialise first model time step
        self.df.loc[day_index, "h_cone"] = self.h_i
        self.df.loc[day_index, "r_cone"] = self.R_F
        self.df.loc[day_index, "dr"] = self.DX
        self.df.loc[day_index, "s_cone"] = self.df.loc[day_index, "h_cone"] / self.df.loc[day_index, "r_cone"]
        V_initial = math.pi / 3 * self.R_F ** 2 * self.h_i
        self.df.loc[day_index +1, "rho_air"] = self.RHO_I
        self.df.loc[day_index + 1, "ice"] = V_initial* self.df.loc[day_index + 1, "rho_air"]
        self.df.loc[day_index + 1, "iceV"] = V_initial
        self.df.loc[day_index + 1, "input"] = self.df.loc[day_index + 1, "ice"]

        logger.warning(
            "Initialise: time %s, radius %.3f, height %.3f, iceV %.3f\n"
            % (
                self.df.loc[day_index, "time"],
                self.df.loc[day_index, "r_cone"],
                self.df.loc[day_index, "h_cone"],
                self.df.loc[day_index + 1, "iceV"],
            )
        )

        pbar = tqdm(total = self.df.shape[0])
        pbar.set_description("%s AIR" % self.name)

        i = day_index+1
        pbar.update(i)
        end = self.df.shape[0]-1

        while i <= end:

            ice_melted = self.df.loc[i, "iceV"] < self.V_dome

            if ice_melted:
                # No further cold windows
                if self.df.loc[i, "time"] > start_date_list[-1]:
                    logger.warning("\tNo further cold windows after %s\n" %self.df.loc[i, "time"] )

                    col_list = [
                        "dep",
                        # "snow2ice",
                        "fountain_froze",
                        "wasted",
                        "sub",
                        "melted",
                    ]
                    for column in col_list:
                        self.df.loc[i - 1, column] = 0

                    # last_hour = i - 1
                    # self.df = self.df[1:i]
                    # self.df = self.df.reset_index(drop=True)
                    pbar.update(end - i)

                    # Full Output
                    self.df.to_hdf(
                        self.output  + "/output.h5",
                        key="df",
                        mode="w",
                    )
                    break
                else:
                    for day in start_date_list:
                        if day >= self.df.loc[i+1, "time"]: 
                            day_index = self.df.index[self.df['time'].dt.date==day][0]
                            pbar.update(day_index - i)
                            i = day_index
                            logger.warning("\tNext cold window at %s\n" % self.df.loc[day_index, "time"])
                            break

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

            self.get_area(i)

            # # Precipitation 
            # if self.df.loc[i, "ppt"] > 0:

            #     if self.df.loc[i, "temp"] < self.T_PPT:
            #         self.df.loc[i, "snow2ice"] = (
            #             self.RHO_W
            #             * self.df.loc[i, "ppt"]
            #             / 1000
            #             * math.pi
            #             * math.pow(self.df.loc[i, "r_cone"], 2)
            #         )
            #     else:
            #     # If rain add to discharge and change temperature
            #         self.df.loc[i, "rain2ice"] = (
            #             self.RHO_W
            #             * self.df.loc[i, "ppt"]
            #             / 1000
            #             * math.pi
            #             * math.pow(self.df.loc[i, "r_cone"], 2)
            #         )
            #         # self.df.loc[i, "Discharge"] += self.df.loc[i, "rain2ice"]/60
            #         self.df.loc[i, "snow2ice"] = 0
            #         logger.info(f"Rain event on {self.df.time.loc[i]} with temp {self.df.temp.loc[i]}")
            # else:
            #     self.df.loc[i, "snow2ice"] = 0

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
                # + self.df.loc[i, "snow2ice"]
                - self.df.loc[i, "sub"]
                - self.df.loc[i, "melted"]
            )

            self.df.loc[i + 1, "vapour"] = (
                self.df.loc[i, "vapour"] + self.df.loc[i, "sub"]
            )
            self.df.loc[i + 1, "wastewater"] = (
                self.df.loc[i, "wastewater"] + self.df.loc[i, "wasted"]
            )

            if self.name in ['guttannen21', 'gangles21']:
                self.df.loc[i + 1, "rho_air"] = self.RHO_I
            else:
                self.df.loc[i + 1, "rho_air"] =(
                        # (self.df.loc[1, "ice"] + self.df.loc[:i, "fountain_froze"].sum()+self.df.loc[:i,"dep"].sum()+self.df.loc[:i,"snow2ice"].sum())
                        (self.df.loc[1, "ice"] + self.df.loc[:i, "fountain_froze"].sum()+self.df.loc[:i,"dep"].sum())
                        /(( self.df.loc[1, "ice"] + self.df.loc[:i, "fountain_froze"].sum()+self.df.loc[:i,
                                                                                                        "dep"].sum())/self.RHO_I)
                        # +(self.df.loc[:i, "snow2ice"].sum()/self.RHO_S))
                )

            self.df.loc[i + 1, "iceV"] = self.df.loc[i + 1, "ice"]/self.df.loc[i+1, "rho_air"]

            self.df.loc[i + 1, "input"] = (
                self.df.loc[i, "input"]
                # + self.df.loc[i, "snow2ice"]
                # + self.df.loc[i, "rain2ice"]
                + self.df.loc[i, "dep"]
                + self.df.loc[i, "Discharge"] * self.DT / 60
            )
            self.df.loc[i + 1, "j_cone"] = (
                self.df.loc[i + 1, "iceV"] - self.df.loc[i, "iceV"]
            ) / (self.df.loc[i, "A_cone"])

            # if test and not ice_melted:
                # logger.error(f"time {self.df.time[i]}, iceV {self.df.iceV[i+1]}")
            i = i+1
            pbar.update(1)
        else:
            # Full Output
            self.df.to_hdf(
                self.output  + "/output.h5",
                key="df",
                mode="w",
            )
            results_dict = {}
            results = [
                "iceV_max",
                "iceV_sum",
                "survival_days",
            ]
            iceV_max = self.df["iceV"].max()
            iceV_sum = self.df["iceV"].sum()
            survival_days= self.df.iceV[self.df["iceV"]>0].sum()/24

            for var in results:
                results_dict[var] = float(round(eval(var), 1))

            print("Summary of results for %s :" %(self.name))
            for var in sorted(results_dict.keys()):
                print("\t%s: %r" % (var, results_dict[var]))

            with open(self.output + "results.json", "w") as f:
                json.dump(results_dict, f, sort_keys=True, indent=4)
            # print(self.df.loc[i, "time"], self.df.loc[i, "iceV"])
