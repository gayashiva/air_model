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
from codetiming import Timer

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
        super().__init__(location)
        # SITE, FOUNTAIN, FOLDER, df_h = config(location, trigger)
        # initial_data = [SITE, FOUNTAIN, FOLDER]

        # for dictionary in initial_data:
        #     for key in dictionary:
        #         setattr(self, key, dictionary[key])
        #         logger.info(f"%s -> %s" % (key, str(dictionary[key])))

        # self.TIME_STEP = 15 * 60
        
        result = []
       

    @Timer(text="Simulation executed in {:.2f} seconds")
    def run(self, experiment):

        print(experiment)
        for key in experiment:
            setattr(self, key, experiment[key])

        self.df = pd.read_hdf(self.input + "model_input_" + self.trigger + ".h5", "df")
        self.read_input()
        self.melt_freeze()
        Max_IceV = self.df["iceV"].max()
        Duration = self.df.index[-1] * self.TIME_STEP / (60 * 60 * 24)

        df_c = pd.read_hdf(self.input + "model_input_" + self.trigger + ".h5", "df_c")
        df_c = df_c.set_index("When")
        self.df= self.df.set_index("When")
        tol = pd.Timedelta('1T')
        df = pd.merge_asof(left=self.df,right=df_c,right_index=True,left_index=True,direction='nearest',tolerance=tol)
        
        ctr = 0
        while (df[df.DroneV.notnull()].shape[0] == 0 and ctr !=5):
            tol += pd.Timedelta('15T')
            ctr+=1
            print("Timedelta increase as shape %s" %(df[df.DroneV.notnull()].shape[0]))
            df = pd.merge_asof(left=self.df,right=df_c,right_index=True,left_index=True,direction='nearest',tolerance=tol)

        rmse_V = (((df.DroneV - df.iceV) ** 2).mean() ** .5)
        corr_V = df['DroneV'].corr(df['iceV'])
        

        if self.name in ["guttannen21", "guttannen20"]:
            df_cam = pd.read_hdf(self.input + "model_input_" + self.trigger + ".h5", "df_cam")
            df = pd.merge_asof(left=self.df,right=df_cam,right_index=True,left_index=True,direction='nearest',tolerance=tol)
            rmse_T = (((df.cam_temp - df.T_s) ** 2).mean() ** .5)
            corr_T = df['cam_temp'].corr(df['T_s'])
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

    locations = ["Schwarzsee 2019", "Guttannen 2021", "Guttannen 2020", "Gangles 2021"]
    # locations = ["Guttannen 2021"]
    # locations = ["Schwarzsee 2019"]
    param_grid = {'DX': np.arange(0.003, 0.080, 0.001).tolist(), 'TIME_STEP': np.arange(15*60, 65*60, 15*60).tolist()}

    experiments = []
    for params in ParameterGrid(param_grid):
        experiments.append(params)

    for location in locations:

        SITE, FOUNTAIN, FOLDER, df_h = config(location)

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
