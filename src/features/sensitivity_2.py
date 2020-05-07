from multiprocessing import Pool
import os
import numpy as np
import pandas as pd
import math

from src.data.config import site, option, folders, fountain, surface
from src.models.air_forecast import icestupa
from src.data.make_dataset import projectile_xy, discharge_rate

from SALib.sample import saltelli


def run_simulation(experiment):

    #Set the input parameters
    # Set the input parameters
    for key, value in experiment.items():
        index = key
        surface['dx'] = value


    df = icestupa(df_in, fountain, surface)

    Max_IceV = df["iceV"].max()
    Efficiency = float((df["meltwater"].tail(1) + df["ice"].tail(1)) / (
                df["sprayed"].tail(1) + df["ppt"].sum() + df["deposition"].sum()))

    max_melt_thickness = float(df["thickness"].max())
    mean_melt_thickness = float(df["thickness"].replace(0, np.NaN).mean())

    results = pd.Series([surface['dx'],
                         Max_IceV,
                         Efficiency,
                         max_melt_thickness,
                         mean_melt_thickness],
                         )
    return results


if __name__ == '__main__':
    problem = {"num_vars": 1, "names": ["dx"],
               "bounds": [[1e-02, 1e-01]]}

    # Generate samples
    param_values = saltelli.sample(problem, 4, calc_second_order=False)

    # Output file Initialise
    columns = ["ie", "a_i", "a_s", "decay_t", "dx", "Max_IceV", "Efficiency"]
    index = range(0, param_values.shape[0])
    dfo = pd.DataFrame(index=index, columns=columns)
    dfo = dfo.fillna(0)
    tasks = []

    #  read files
    global df_in
    filename0 = os.path.join(folders["input_folder"] + site + "_raw_input.csv")
    df_in = pd.read_csv(filename0, sep=",")
    df_in["When"] = pd.to_datetime(df_in["When"], format="%Y.%m.%d %H:%M:%S")

    """ Derived Parameters"""

    l = [
        "a",
        "r_f",
        "Fountain",
    ]
    for col in l:
        df_in[col] = 0

    """Discharge Rate"""
    df_in["Fountain"], df_in["Discharge"] = discharge_rate(df_in, fountain)

    """Albedo Decay"""
    surface["decay_t"] = (
            surface["decay_t"] * 24 * 60 / 5
    )  # convert to 5 minute time steps
    s = 0
    f = 0

    """ Fountain Spray radius """
    Area = math.pi * math.pow(fountain["aperture_f"], 2) / 4

    for i in range(1, df_in.shape[0]):

        if option == "schwarzsee":

            ti = surface["decay_t"]
            a_min = surface["a_i"]

            # Precipitation
            if (df_in.loc[i, "Fountain"] == 0) & (df_in.loc[i, "Prec"] > 0):
                if df_in.loc[i, "T_a"] < surface["rain_temp"]:  # Snow
                    s = 0
                    f = 0

            if df_in.loc[i, "Fountain"] > 0:
                f = 1
                s = 0

            if f == 0:  # last snowed
                df_in.loc[i, "a"] = a_min + (surface["a_s"] - a_min) * math.exp(-s / ti)
                s = s + 1
            else:  # last sprayed
                df_in.loc[i, "a"] = a_min
                s = s + 1
        else:
            df_in.loc[i, "a"] = surface["a_i"]

        """ Fountain Spray radius """
        v_f = df_in.loc[i, "Discharge"] / (60 * 1000 * Area)
        df_in.loc[i, "r_f"] = projectile_xy(
            v_f, fountain["h_f"]
        )

    # cast the param_values to a dataframe to
    # include the column labels
    experiments = pd.DataFrame(param_values,
                               columns=problem['names'])

    with Pool(7) as executor:
        results = []
        for entry in executor.map(run_simulation, experiments.to_dict('records')):
            results.append(entry)

        results = pd.DataFrame(results)
        results = results.rename(columns={0: 'dx', 1: 'Max_IceV', 2: 'Efficiency', 3: 'max_melt_thickness', 4: 'mean_melt_thickness'})
        print(results)
        filename2 = os.path.join(
            folders['sim_folder'], site + "_simulations_" + str(problem["names"]) + str(param_values.shape[0]) + ".csv"
        )
        results.to_csv(filename2, sep=",")