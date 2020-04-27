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

        self.print_input(filename = "/home/surya/Programs/PycharmProjects/air_model/data/processed/schwarzsee/simulations/")

        self.melt_freeze()

        self.print_output(filename = "/home/surya/Programs/PycharmProjects/air_model/data/processed/schwarzsee/simulations/discharge_output.pdf")

        Max_IceV = self.df["iceV"].max()
        Efficiency = float(
            self.df["water"].tail(1)
            / (self.df["Discharge"].sum() * self.time_steps / 60)
            * 100
        )
        Duration = self.df.index[-1] * 5 /(60 * 24)

        print("\nIce Volume Max", float(self.df["iceV"].max()))
        print("Fountain efficiency", Efficiency)
        print("Ice Mass Remaining", float(self.df["ice"].tail(1)))
        print("Meltwater", float(self.df["meltwater"].tail(1)))
        print("Ppt", self.df["ppt"].sum())
        print("Deposition", self.df["deposition"].sum() )

        result = pd.Series([experiment.get("Discharge"),
                             Max_IceV,
                             Efficiency,
                            Duration]
                            )

        self.df = self.df.set_index('When').resample("H").mean().reset_index()

        key = experiment.get("Discharge")

        return key, self.df["SA"].values, self.df["iceV"].values, self.df["solid"].values, self.df["Discharge"].values, result

param_values = np.arange(15, 25, 1).tolist()


experiments = pd.DataFrame(param_values,
                           columns=["Discharge"])

model = Discharge_Icestupa()

df_out = pd.DataFrame()
results = []
with Pool(8) as executor:

    for key, SA, iceV, solid, Discharge, result in executor.map(model.run, experiments.to_dict('records')):
        data = pd.DataFrame({str(key) + "_SA":SA, str(key)+ "_iceV": iceV, str(key)+ "_solid":solid, str(key)+ "_Discharge": Discharge})
        df_out = pd.concat([df_out, data], axis=1)
        results.append(result)

    results = pd.DataFrame(results)
    results = results.rename(
        columns={0: 'Discharge', 1: 'Max_IceV', 2: 'Efficiency', 3 : 'Duration'})

    print(results)
    filename = "/home/surya/Programs/PycharmProjects/air_model/data/processed/schwarzsee/simulations/results_2.csv"
    results.to_csv(filename, sep=",")

    filename2 = "/home/surya/Programs/PycharmProjects/air_model/data/processed/schwarzsee/simulations/discharge_2.csv"
    df_out.to_csv(filename2, sep=",")

