import pandas as pd
import math
import sys
import os
import logging
import coloredlogs
import numpy as np
import multiprocessing
from multiprocessing import Pool
from sklearn.model_selection import ParameterGrid

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
)

from src.utils.settings import config
from src.models.icestupaClass import Icestupa
from src.models.methods.calibration import get_calibration
from src.models.methods.metadata import get_parameter_metadata
from src.models.methods.solar import get_solar
from src.models.methods.droplet import get_droplet_projectile


class Tune_Icestupa(Icestupa):
    def __init__(self, location, trigger="Manual"):
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

        print(experiment)

        self.df = pd.read_hdf(self.input + "model_input_" + self.trigger + ".h5", "df")

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
                dia=self.dia_f, h=self.df.loc[1,"h_f"], x=self.r_spray
            )
        else:  # Provide spray radius
            self.r_spray = get_droplet_projectile(
                dia=self.dia_f, h=self.df.loc[1,"h_f"], d=self.discharge
            )

        for key in experiment:
            setattr(self, key, experiment[key])
            # logger.warning(f"%s -> %s" % (key, str(experiment[key])))
            if key == 'TIME_STEP':
                self.df = self.df.set_index('When')
                self.df= self.df.resample(str(int(self.TIME_STEP/60))+'T').mean()
                # self.df.loc[:, self.df.columns != 'h_f'] = self.df.loc[:, self.df.columns != 'h_f'].resample(str(int(self.TIME_STEP/60))+'T').mean()
                # self.df.h_f = self.df.h_f.resample(str(int(self.TIME_STEP/60))+'T').sum()
                self.df = self.df.reset_index()


        self.melt_freeze()

        rmse_V = (((self.df.DroneV - self.df.iceV) ** 2).mean() ** .5)
        corr_V = self.df['DroneV'].corr(self.df['iceV'])
        Max_IceV = self.df["iceV"].max()
        Duration = self.df.index[-1] * self.TIME_STEP / (60 * 60 * 24)

        if self.name in ["guttannen21", "guttannen20"]:
            rmse_T = (((self.df.cam_temp - self.df.T_s) ** 2).mean() ** .5)
            corr_T = self.df['cam_temp'].corr(self.df['T_s'])
        else:
            rmse_T = 0
            corr_T = 0

        result = pd.Series(
            [
                experiment['DX'] * 1000,
                experiment['TIME_STEP'] / 60,
                rmse_V,
                corr_V,
                rmse_T,
                corr_T,
                Max_IceV,
                Duration,
            ]
        )

        return result


if __name__ == "__main__":
    # Main logger
    logger = logging.getLogger(__name__)
    coloredlogs.install(
        fmt="%(funcName)s %(levelname)s %(message)s",
        level=logging.ERROR,
        logger=logger,
    )

    answers = dict(
        location="Schwarzsee 2019",
        # location="Guttannen 2021",
        # location="Gangles 2021",
        trigger="Manual",
    )

    # locations = ["Schwarzsee 2019", "Guttannen 2021", "Guttannen 2020", "Gangles 2021"]
    locations = ["Guttannen 2021", "Guttannen 2020"]
    param_grid = {'DX': np.arange(0.003, 0.050, 0.0005).tolist(), 'TIME_STEP': np.arange(15 * 60, 65*60, 15*60).tolist()}

    experiments = []
    for params in ParameterGrid(param_grid):
        experiments.append(params)

    for location in locations:

        SITE, FOUNTAIN, FOLDER = config(location, "Manual")

        model = Tune_Icestupa(location)

        results = []

        with Pool(multiprocessing.cpu_count()) as executor:

            for result in executor.map(model.run, experiments):
                results.append(result)

        results = pd.DataFrame(results)
        results = results.rename(
            columns={
                0: "DX",
                1: "TIME_STEP",
                2: "rmse_V",
                3: "corr_V",
                4: "rmse_T",
                5: "corr_T",
                6: "Max_IceV",
                7: "Duration",
            }
        )

        results.round(2).to_csv(FOLDER["sim"] + "/Tune_sim.csv", sep=",")

        print(results)
