import pandas as pd
import math
import sys
import os
import logging
import coloredlogs
import numpy as np
import multiprocessing
from multiprocessing import Pool

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
)

from src.utils.settings import config
from src.models.icestupaClass import Icestupa
from src.models.methods.calibration import get_calibration
from src.models.methods.metadata import get_parameter_metadata
from src.models.methods.solar import get_solar
from src.models.methods.droplet import get_droplet_projectile


class DX_Icestupa(Icestupa):
    def __init__(self, location="Guttannen 2021", trigger="Manual"):
        SITE, FOUNTAIN, FOLDER = config(location, trigger)
        initial_data = [SITE, FOUNTAIN, FOLDER]

        # Initialise all variables of dictionary
        for dictionary in initial_data:
            for key in dictionary:
                setattr(self, key, dictionary[key])
                logger.info(f"%s -> %s" % (key, str(dictionary[key])))

        self.TIME_STEP = 15 * 60
        
        result = []
       

    def run(self, experiment):

        key = experiment.get("DX")
        self.DX = key
        if self.name in ["gangles21"]:
            df_c = get_calibration(site=self.name, input=self.raw)
            df_c.loc[0, "DroneV"] = 2/3 * math.pi * 4 ** 3 # Volume of dome
            self.r_spray = df_c.loc[1, "dia"] / 2
            self.h_i = 3 * df_c.loc[0, "DroneV"] / (math.pi * self.r_spray ** 2)

        if self.name in ["guttannen21", "guttannen20"]:
            df_c, df_cam = get_calibration(site=self.name, input=self.raw)
            self.r_spray = df_c.loc[0, "dia"] / 2
            self.h_i = 3 * df_c.loc[0, "DroneV"] / (math.pi * self.r_spray ** 2)

        if hasattr(self, "r_spray"):  # Provide discharge
            self.discharge = get_droplet_projectile(
                dia=self.dia_f, h=self.h_f, x=self.r_spray
            )
        else:  # Provide spray radius
            self.r_spray = get_droplet_projectile(
                dia=self.dia_f, h=self.h_f, d=self.discharge
            )

        self.df = pd.read_hdf(self.input + "model_input_" + self.trigger + ".h5", "df")

        self.melt_freeze()

        Efficiency = (
            (self.df["meltwater"].iloc[-1] + self.df["ice"].iloc[-1])
            / self.df["input"].iloc[-1]
            * 100
        )

        Max_IceV = self.df["iceV"].max()
        Min_T_s = self.df["T_s"].min()
        Min_T_bulk = self.df["T_bulk"].min()
        rmse_V = (((self.df.DroneV - self.df.iceV) ** 2).mean() ** .5)
        if self.name in ["guttannen21", "guttannen20"]:
            rmse_T = (((self.df.cam_temp - self.df.T_s) ** 2).mean() ** .5)
        else:
            rmse_T = 0
        Efficiency = (
            (self.df["meltwater"].iloc[-1] + self.df["ice"].iloc[-1])
            / (self.df["input"].iloc[-1])
            * 100
        )
        Duration = self.df.index[-1] * self.TIME_STEP / (60 * 60 * 24)

        print("\nDX", key)
        print("Ice Volume Max", float(self.df["iceV"].max()))
        print("Ice Mass Remaining", self.df["ice"].iloc[-1])
        print("Meltwater", self.df["meltwater"].iloc[-1])
        print("Duration", Duration)
        print("\n")

        result = pd.Series(
            [
                experiment.get("DX") * 1000,
                Max_IceV,
                Duration,
                Min_T_s,
                Min_T_bulk,
                rmse_V,
                rmse_T,
            ]
        )
        self.df = self.df.set_index("When").resample("1H").mean().reset_index()

        return (
            key,
            self.df["When"].values,
            self.df["SA"].values,
            self.df["iceV"].values,
            self.df["T_s"].values,
            result,
        )


if __name__ == "__main__":
    # Main logger
    logger = logging.getLogger(__name__)
    coloredlogs.install(
        fmt="%(funcName)s %(levelname)s %(message)s",
        level=logging.ERROR,
        logger=logger,
    )

    answers = dict(
        # location="Schwarzsee 2019",
        # location="Guttannen 2020",
        location="Gangles 2021",
        trigger="Manual",
        # trigger="None",
        # trigger="Temperature",
        # trigger="Weather",
        run="yes",
    )

    # Get settings for given location and trigger
    SITE, FOUNTAIN, FOLDER = config(answers["location"], answers["trigger"])

    # Initialise icestupa object
    model = DX_Icestupa(location=answers["location"], trigger=answers["trigger"])
    # model = DX_Icestupa()

    param_values = np.arange(0.001, 0.010, 0.0005).tolist()

    experiments = pd.DataFrame(param_values, columns=["DX"])
    variables = ["When", "SA", "iceV", "T_s"]

    df_out = pd.DataFrame()

    results = []

    logger.info("CPUs running %s" % multiprocessing.cpu_count())
    with Pool(12) as executor:

        for (
            key,
            When,
            SA,
            iceV,
            T_s,
            result,
        ) in executor.map(model.run, experiments.to_dict("records")):
            iterables = [[key], variables]
            index = pd.MultiIndex.from_product(iterables, names=["DX", "variables"])
            data = pd.DataFrame(
                {
                    (key, "When"): When,
                    (key, "SA"): SA,
                    (key, "iceV"): iceV,
                    (key, "T_s"): T_s,
                },
                columns=index,
            )
            df_out = pd.concat([df_out, data], axis=1, join="outer", ignore_index=False)

            results.append(result)

        results = pd.DataFrame(results)
        results = results.rename(
            columns={
                0: "DX",
                1: "Max_IceV",
                2: "Duration",
                3: "Min_T_s",
                4: "Min_T_bulk",
                5: "rmse_V",
                6: "rmse_T",
            }
        )

        print(results)

        results.round(3).to_csv(FOLDER["sim"] + "/DX_sim.csv", sep=",")

        data_store = pd.HDFStore(FOLDER["sim"] + "/DX_sim.h5")
        data_store["dfd"] = df_out
        data_store.close()
