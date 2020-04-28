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
            (self.df["meltwater"].tail(1) + self.df["ice"].tail(1))
            / (
                    self.df["Discharge"].sum() * self.time_steps / 60
                    + self.df["ppt"].sum()
                    + self.df["deposition"].sum()
            )
            * 100
        )

        Duration = self.df.index[-1] * 5 /(60 * 24)
        h_r = self.df.h_ice.max()/self.df.r_ice.max()

        print("\nIce Volume Max", float(self.df["iceV"].max()))
        print("Fountain efficiency", Efficiency)
        print("Ice Mass Remaining", float(self.df["ice"].tail(1)))
        print("Meltwater", float(self.df["meltwater"].tail(1)))
        print("Ppt", self.df["ppt"].sum())
        print("Deposition", self.df["deposition"].sum() )

        result = pd.Series([experiment.get("Discharge"),
                             Max_IceV,
                             Efficiency,
                            Duration,
                            h_r]
                            )

        self.df = self.df.set_index('When').resample("H").mean().reset_index()

        key = experiment.get("Discharge")

        return key, self.df["When"].values, self.df["SA"].values, self.df["iceV"].values, self.df["solid"].values, self.df["thickness"].values, self.df["Discharge"].values, result

param_values = np.arange(5, 7, 1).tolist()


experiments = pd.DataFrame(param_values,
                           columns=["Discharge"])

model = Discharge_Icestupa()

variables = ["When", 'SA', 'iceV', 'solid', 'thickness', 'Discharge', 'result']

iterables = [param_values, variables]
index = pd.MultiIndex.from_product(iterables, names=['discharge_rate', 'variables'])

data = pd.DataFrame(columns=index)

results = []
with Pool(7) as executor:

    for key, When, SA, iceV, solid, thickness, Discharge, result in executor.map(model.run, experiments.to_dict('records')):

        data.append({ (key, "When"):When, (key, "SA"):SA, (key, "iceV"): iceV, (key, "solid"):solid, (key, "thickness"):thickness, (key, "Discharge"): Discharge}, ignore_index=True)
        # data = pd.DataFrame({"discharge_rate":,"When":When, "SA":SA, "iceV": iceV, "solid":solid, "thickness":thickness, "Discharge": Discharge}, columns = index)
        print(data.head())
        # df_out = pd.concat([df_out, data], axis = 1, levels=0).sort_index(axis=1)
        # print(df_out.head())
        results.append(result)

    results = pd.DataFrame(results)
    results = results.rename(
        columns={0: 'Discharge', 1: 'Max_IceV', 2: 'Efficiency', 3 : 'Duration'})

    print(results)
    filename = "/home/surya/Programs/PycharmProjects/air_model/data/processed/schwarzsee/simulations/results.csv"
    results.to_csv(filename, sep=",")

    filename2 = "/home/surya/Programs/PycharmProjects/air_model/data/processed/schwarzsee/simulations/discharge.csv"
    df_out.to_csv(filename2, sep=",")

