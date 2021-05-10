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
import chaospy as cp

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
)

from src.utils.settings import config
from src.models.icestupaClass import Icestupa
from src.models.methods.calibration import get_calibration
from src.models.methods.metadata import get_parameter_metadata
from src.models.methods.solar import get_solar
from src.models.methods.droplet import get_droplet_projectile

def uniform(parameter, interval):
    if parameter == 0:
        raise ValueError("Creating a percentage distribution around 0 does not work")

    return cp.Uniform(
        parameter - abs(interval / 2.0 * parameter),
        parameter + abs(interval / 2.0 * parameter),
    )

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

        key_to_lookup = ['A_I', "T_DECAY", "A_S", "T_RAIN"]

        if any(x in key_to_lookup for x in experiment):
            print("Recalculating albedo")
            """Albedo Decay parameters initialized"""
            self.T_DECAY = self.T_DECAY * 24 * 60 * 60 / self.TIME_STEP
            s = 0
            if self.name in ["schwarzsee19", "guttannen20"]:
                f = 0  # Start with snow event
            else:
                f = 1
            for i, row in self.df.iterrows():
                s, f = self.get_albedo(i, s, f, site=self.name)

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
            df = pd.merge_asof(left=self.df,right=df_c,right_index=True,left_index=True,direction='nearest',tolerance=tol)

        vals = df[df.DroneV.notnull()].shape[0]

        if vals:
            rmse_V = (((df.DroneV - df.iceV) ** 2).mean() ** .5)
            corr_V = df['DroneV'].corr(df['iceV'])
            predicted_vols = df.loc[df.DroneV.notnull(), "iceV"].values.tolist()
            measured_vols = df.loc[df.DroneV.notnull(), "DroneV"].values.tolist()
        

        if self.name in ["guttannen21", "guttannen20"]:
            df_cam = pd.read_hdf(self.input + "model_input_" + self.trigger + ".h5", "df_cam")
            df = pd.merge_asof(left=self.df,right=df_cam,right_index=True,left_index=True,direction='nearest',tolerance=tol)
            rmse_T = (((df.cam_temp - df.T_s) ** 2).mean() ** .5)
            corr_T = df['cam_temp'].corr(df['T_s'])

        result = pd.Series(
            [
                rmse_V,
                corr_V,
                rmse_T,
                corr_T,
                Max_IceV,
                Duration,
                vals,
                predicted_vols,
                measured_vols,
            ],
            index = 
            [
                "rmse_V",
                "corr_V",
                "rmse_T",
                "corr_T",
                "max_V",
                "duration",
                "points",
                "predicted_vols",
                "measured_vols",
            ],
        )
        experiment['DX'] *= 1000
        experiment['TIME_STEP'] /= 60

        params = pd.Series(
            experiment,
            index = 
            [
                'DX', 
                'TIME_STEP',
                'dia_f',
                'IE',
                'A_I',
                'A_S',
                'T_RAIN',
                'T_DECAY',
                # 'v_a_limit',
                # 'Z_I',
            ],
        )
        result = result.append(params)

        return result


if __name__ == "__main__":
    # Main logger
    logger = logging.getLogger(__name__)
    coloredlogs.install(
        fmt="%(funcName)s %(levelname)s %(message)s",
        level=logging.WARNING,
        logger=logger,
    )

    # locations = ["Schwarzsee 2019", "Guttannen 2021", "Guttannen 2020", "Gangles 2021"]
    locations = ["Guttannen 2021"]
    # locations = ["Schwarzsee 2019"]

    # param_grid = {'DX': np.arange(0.003, 0.004, 0.001).tolist(), 'TIME_STEP': np.arange(30*60, 35*60, 15*60).tolist()}
    param_grid = {
        'DX': np.arange(0.005, 0.015, 0.001).tolist(), 
        'TIME_STEP': np.arange(15*60, 65*60, 15*60).tolist(),
        'dia_f': np.arange(0.003, 0.010 , 0.001).tolist(),
        'IE': np.arange(0.9, 0.99 , 0.01).tolist(),
        'A_I': np.arange(0.3, 0.4 , 0.01).tolist(),
        'A_S': np.arange(0.8, 0.9 , 0.01).tolist(),
        'T_RAIN': np.arange(0, 2 , 1).tolist(),
        'T_DECAY': np.arange(1, 22 , 1).tolist(),
        # 'v_a_limit': np.arange(4, 10, 1).tolist(),
        # 'Z_I': np.arange(0.0010, 0.0020, 0.0001).tolist(),
    }

    experiments = []
    for params in ParameterGrid(param_grid):
        experiments.append(params)

    for location in locations:

        SITE, FOLDER, df_h = config(location)

        model = Tune_Icestupa(location)

        results = []

        with Pool(multiprocessing.cpu_count()) as executor:

            for result in executor.map(model.run, experiments):
                results.append(result)

        results = pd.DataFrame(results)

        results.round(3).to_csv(FOLDER["sim"] + "/Full_tune_sim.csv", sep=",")

        print(results)
