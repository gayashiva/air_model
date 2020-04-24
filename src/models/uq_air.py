import uncertainpy as un
import chaospy as cp
import pandas as pd
import numpy as np
import os
import math
from src.models.air import Icestupa

def uniform(parameter, interval):
    """
    A closure that creates a function that takes a `parameter` as input and
    returns a uniform distribution with `interval` around `parameter`.
    Parameters
    ----------
    interval : int, float
        The interval of the uniform distribution around `parameter`.
    Returns
    -------
    distribution : function
        A function that takes `parameter` as input and returns a
        uniform distribution with `interval` around this `parameter`.
    Notes
    -----
    This function ultimately calculates:
    .. code-block:: Python
        cp.Uniform(parameter - abs(interval/2.*parameter),
                   parameter + abs(interval/2.*parameter)).
    """
    if parameter == 0:
        raise ValueError("Creating a percentage distribution around 0 does not work")

    return cp.Uniform(parameter - abs(interval/2.*parameter),
                      parameter + abs(interval/2.*parameter))

    return distribution

def max_volume(time, values):
    # Calculate the feature using time, values and info.
    icev_max = values.max()
    # Return the feature times and values.
    return None, icev_max #todo include efficiency

class UQ_Icestupa(un.Model, Icestupa):

    def __init__(self):

        super(UQ_Icestupa, self).__init__(labels=["Time (days)", "Ice Volume (m3)"], interpolate=False)

        data_store = pd.HDFStore("/home/surya/Programs/PycharmProjects/air_model/data/interim/schwarzsee/model_input.h5")
        self.df = data_store['df']
        data_store.close()

    def run(self, **parameters):

        self.set_parameters(**parameters)
        print(parameters.values())

        if 'aperture_f' in parameters.keys():  # todo change to general
            """ Fountain Spray radius """
            Area = math.pi * math.pow(self.aperture_f, 2) / 4

            for row in self.df[1:].itertuples():
                v_f = row.Discharge / (60 * 1000 * Area)
                self.df.loc[row.Index, "r_f"] = self.projectile_xy(v_f)

        if 'a_i' or 'rain_temp' in parameters.keys():
            """Albedo Decay"""
            self.decay_t = (
                    self.decay_t * 24 * 60 * 60 / self.time_steps
            )  # convert to 5 minute time steps
            s = 0
            f = 0

            for row in self.df[1:].itertuples():
                s, f = self.albedo(row, s, f)

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

        return self.df.index.values / 24, self.df["iceV"].values

list_of_feature_functions = [max_volume]

features = un.Features(new_features=list_of_feature_functions,
                       features_to_run=["max_volume"])


# # Set all parameters to have a uniform distribution
# # within a 20% interval around their fixed value
# parameters.set_all_distributions(un.uniform(0.05))

ie = 0.95
a_i =0.35
a_s = 0.85
decay_t = 10

interval = 0.05

ie_dist = uniform(ie, interval)
a_i_dist = uniform(a_i, interval)
a_s_dist = uniform(a_s, interval)
decay_t_dist = uniform(decay_t, interval)

rain_temp_dist = cp.Uniform(0, 2)
z0mi_dist = cp.Uniform(0.0007, 0.0027)
z0hi_dist = cp.Uniform(0.0007, 0.0027)
snow_fall_density_dist = cp.Uniform(200, 300)

interval = 0.01

aperture_f_dist = uniform(0.005, interval)
height_f_dist = uniform(1.35, interval)

dx_dist = cp.Uniform(0.0001, 0.01)

# parameters = {
#                 "ie": ie_dist,
#                 "a_i": a_i_dist,
#                 "a_s": a_s_dist,
#                 "decay_t": decay_t_dist,
#                 "dx": dx_dist
# }

# parameters = {
#               "rain_temp": rain_temp_dist,
#               "z0mi": z0mi_dist,
#               "z0hi": z0hi_dist,
#               "snow_fall_density": snow_fall_density_dist
#               }

parameters = {
              "aperture_f": aperture_f_dist,
              "height_f": height_f_dist
              }


# Create the parameters
parameters = un.Parameters(parameters)

# Initialize the model
model = UQ_Icestupa()

# Set up the uncertainty quantification
UQ = un.UncertaintyQuantification(model=model,
                                  parameters=parameters,
                                  features=features,
                                  )

# Perform the uncertainty quantification using
# polynomial chaos with point collocation (by default)
data = UQ.quantify(data_folder = "/home/surya/Programs/PycharmProjects/air_model/data/processed/schwarzsee/simulations/data/",
                    figure_folder="/home/surya/Programs/PycharmProjects/air_model/data/processed/schwarzsee/simulations/figures/",
                    filename="Fount")

# data = UQ.quantify(filename="Meteorological")

