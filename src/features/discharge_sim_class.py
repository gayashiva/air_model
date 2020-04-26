import logging
import os
import math
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from src.data.config import site, option, folders, fountain, surface

from multiprocessing import Pool
from src.models.air import Icestupa

class Discharge_Icestupa(Icestupa):

    def __init__(self):

        data_store = pd.HDFStore("/home/surya/Programs/PycharmProjects/air_model/data/interim/schwarzsee/model_input.h5")
        self.df = data_store['df']
        data_store.close()


    def run(self, experiment):

        print(experiment)

        Area = math.pi * math.pow(self.aperture_f, 2) / 4

        for row in self.df.itertuples():
            if row.Discharge > 0:
                self.df.loc[row.Index, "Discharge"] = experiment.get("Discharge")
                v_f = experiment.get("Discharge") / (60 * 1000 * Area)
                self.df.loc[row.Index, "r_f"] = self.projectile_xy(v_f)

        self.melt_freeze()

        Efficiency = float(
            (self.df["meltwater"].tail(1) + self.df["ice"].tail(1))
            / (self.df["Discharge"].sum() * self.time_steps / 60 + self.df["ppt"].sum() + self.df["deposition"].sum())
            * 100
        )

        print("\nIce Volume Max", float(self.df["iceV"].max()))
        print("Fountain efficiency", Efficiency)
        print("Ice Mass Remaining", float(self.df["ice"].tail(1)))
        print("Meltwater", float(self.df["meltwater"].tail(1)))
        print("Ppt", self.df["ppt"].sum())

        self.df = self.df.set_index('When').resample('1H').mean().reset_index()

        key = experiment.get("Discharge")
        results = self.df["ice"].values

        return key, results

param_values = np.arange(2, 16, 2).tolist()


experiments = pd.DataFrame(param_values,
                           columns=["Discharge"])

model = Discharge_Icestupa()

df_out = pd.DataFrame()
i = 0

with Pool(8) as executor:

    for key, entry in executor.map(model.run, experiments.to_dict('records')):
        # data = pd.DataFrame({str(key): entry})
        # df_out = df_out.append(data)
        df_out[str(key)] = entry

    filename2 = "/home/surya/Programs/PycharmProjects/air_model/data/processed/schwarzsee/simulations/discharge.csv"
    df_out.to_csv(filename2, sep=",")