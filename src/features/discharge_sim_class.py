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
        key = experiment.get("spray_radius")

        self.r_mean = key

        # self.print_input(filename = "/home/surya/Programs/PycharmProjects/air_model/data/processed/schwarzsee/simulations/")

        self.melt_freeze()

        # self.print_output(filename = "/home/surya/Programs/PycharmProjects/air_model/data/processed/schwarzsee/simulations/discharge_output.pdf")

        Max_IceV = self.df["iceV"].max()
        Efficiency = (self.df["meltwater"].iloc[-1] + self.df["ice"].iloc[-1]) / self.df["input"].iloc[-1] * 100
        Duration = self.df.index[-1] * 5 /(60 * 24)
        h_r = self.df.h_ice.max()/self.df.r_ice.max()
        water_stored = (self.df["meltwater"].iloc[-1] + self.df["ice"].iloc[-1])
        water_lost = self.df["vapour"].iloc[-1]
        unfrozen_water = self.df["unfrozen_water"].iloc[-1]

        print("\nIce Volume Max", float(self.df["iceV"].max()))
        print("Fountain efficiency", Efficiency)
        print("Ice Mass Remaining", self.df["ice"].iloc[-1])
        print("Meltwater", self.df["meltwater"].iloc[-1])
        print("Ppt", self.df["ppt"].sum())
        print("Deposition", self.df["deposition"].sum() )

        result = pd.Series([experiment.get("spray_radius"),
                             Max_IceV,
                             Efficiency,
                            Duration,
                            h_r,
                            water_stored,
                            water_lost,
                            unfrozen_water]
                            )

        self.df = self.df.set_index('When').resample("H").mean().reset_index()

        return key, self.df["When"].values, self.df["SA"].values, self.df["iceV"].values, self.df["solid"].values, self.df["thickness"].values, self.df["Discharge"].values, self.df["input"].values, self.df["meltwater"].values, result

param_values = np.arange(2, 12, 2).tolist()


experiments = pd.DataFrame(param_values,
                           columns=["spray_radius"])

model = Discharge_Icestupa()

variables = ["When", 'SA', 'iceV', 'solid', 'thickness', 'Discharge', 'input', 'meltwater']

df_out = pd.DataFrame()

results = []

if __name__ == "__main__":
    with Pool(8) as executor:

        for key, When, SA, iceV, solid, thickness, Discharge, input, meltwater, result in executor.map(model.run, experiments.to_dict('records')):
            iterables = [[key], variables]
            index = pd.MultiIndex.from_product(iterables, names=['discharge_rate', 'variables'])
            data = pd.DataFrame({ (key, "When"):When, (key, "SA"):SA, (key, "iceV"): iceV, (key, "solid"):solid, (key, "thickness"):thickness, (key, "Discharge"): Discharge, (key, "input"): input, (key, "meltwater"): meltwater}, columns = index)
            df_out = pd.concat([df_out, data], axis=1, join='outer', ignore_index=False)
            print(result)
            results.append(result)

        results = pd.DataFrame(results)
        results = results.rename(
            columns={0: 'spray_radius', 1: 'Max_IceV', 2: 'Efficiency', 3 : 'Duration', 4:'h_r', 5:'water_stored', 6:'water_lost', 7:'unfrozen_water'})

        filename = "/home/surya/Programs/PycharmProjects/air_model/data/processed/schwarzsee/simulations/spray_radius_results.csv"
        results.to_csv(filename, sep=",")

        filename2 = "/home/surya/Programs/PycharmProjects/air_model/data/processed/schwarzsee/simulations/spray_radius.h5"
        data_store = pd.HDFStore(filename2)
        data_store["dfd"] = df_out
        data_store.close()

