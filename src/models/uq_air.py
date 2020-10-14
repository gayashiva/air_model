import uncertainpy as un
import chaospy as cp
import pandas as pd
import math
from src.models.air import Icestupa


def uniform(parameter, interval):
    if parameter == 0:
        raise ValueError("Creating a percentage distribution around 0 does not work")

    return cp.Uniform(parameter - abs(interval / 2. * parameter),
                      parameter + abs(interval / 2. * parameter))


def max_volume(time, values):
    # Calculate the feature using time, values and info.
    icev_max = values.max()
    # Return the feature times and values.
    return None, icev_max  # todo include efficiency


class UQ_Icestupa(un.Model, Icestupa):

    def __init__(self):

        super(UQ_Icestupa, self).__init__(labels=["Time (days)", "Ice Volume ($m^3$)"], interpolate=True)

        data_store = pd.HDFStore(
            "/home/surya/Programs/PycharmProjects/air_model/data/interim/schwarzsee/model_input_extended.h5")
        self.df = data_store['df']
        data_store.close()

    def run(self, **parameters):

        self.set_parameters(**parameters)
        print(parameters.values())

        if 'dia_f' or 'h_f' in parameters.keys():  # todo change to general
            """ Fountain Spray radius """
            Area = math.pi * math.pow(self.dia_f, 2) / 4

            for row in self.df[1:].itertuples():
                v_f = row.Discharge / (60 * 1000 * Area)
                self.df.loc[row.Index, "r_f"] = self.projectile_xy(v_f, self.h_f)

        if 'a_i' or 'T_rain' in parameters.keys():
            """Albedo Decay"""
            self.t_decay = (
                    self.t_decay * 24 * 60 * 60 / self.time_steps
            )  # convert to 5 minute time steps
            s = 0
            f = 0

            for row in self.df[1:].itertuples():
                s, f = self.albedo(row, s, f)

        self.melt_freeze()

        Efficiency = (self.df["meltwater"].iloc[-1] + self.df["ice"].iloc[-1]) / self.df["input"].iloc[-1] * 100

        print("\nIce Volume Max", float(self.df["iceV"].max()))
        print("Fountain efficiency", Efficiency)
        print("Ice Mass Remaining", self.df["ice"].iloc[-1])
        print("Meltwater", self.df["meltwater"].iloc[-1])
        print("Ppt", self.df["ppt"].sum())
        print("Number of days", self.df.index[-1] * 5 / (60*24))

        self.df = self.df.set_index('When').resample('1H').mean().reset_index()

        for i in range(len(self.df), 65 * 24):
            self.df.loc[i, "iceV"] = 0

        return self.df.index.values / 24, self.df["iceV"].values


list_of_feature_functions = [max_volume]

features = un.Features(new_features=list_of_feature_functions,
                       features_to_run=["max_volume"])

# # Set all parameters to have a uniform distribution
# # within a 20% interval around their fixed value
# parameters.set_all_distributions(un.uniform(0.05))

ie = 0.95
a_i = 0.35
a_s = 0.85
t_decay = 10

interval = 0.05

ie_dist = uniform(ie, interval)
a_i_dist = uniform(a_i, interval)
a_s_dist = uniform(a_s, interval)

t_decay_dist = cp.Uniform(1, 22)
T_rain_dist = cp.Uniform(0, 2)

dia_f = 0.005
h_f = 1.35
h_aws = 3
T_w = 5

interval = 0.01

dia_f_dist = uniform(0.005, interval)
h_f_dist = uniform(1.35, interval)
h_aws_dist = uniform(3, interval)
T_w_dist = cp.Uniform(0, 9)

dx_dist = cp.Uniform(0.001, 0.01)

parameters_single = {
    "ie": ie_dist,
    "a_i": a_i_dist,
    "a_s": a_s_dist,
    "t_decay": t_decay_dist,
    "T_rain": T_rain_dist,
    "dia_f": dia_f_dist,
    "h_f": h_f_dist,
    "h_aws": h_aws_dist,
    "T_w": T_w_dist,
    "dx": dx_dist
}

# Create the parameters
for k,v in parameters_single.items():
    print(k,v)
    parameters = un.Parameters({k:v})


    # Initialize the model
    model = UQ_Icestupa()

    # Set up the uncertainty quantification
    UQ = un.UncertaintyQuantification(model=model,
                                      parameters=parameters,
                                      features=features,
                                      CPUs=6,
                                      )

    # Perform the uncertainty quantification using # polynomial chaos with point collocation (by default) data =
    UQ.quantify(seed=10, data_folder = "/home/surya/Programs/PycharmProjects/air_model/data/processed/schwarzsee/simulations/data/", figure_folder="/home/surya/Programs/PycharmProjects/air_model/data/processed/schwarzsee/simulations/figures/", filename=k)


# parameters = {
#     "T_rain": T_rain_dist,
#     "ie": ie_dist
#
# }
#
# parameters = un.Parameters(parameters)
#
# # Initialize the model
# model = UQ_Icestupa()
#
# # Set up the uncertainty quantification
# UQ = un.UncertaintyQuantification(model=model,
#                                   parameters=parameters,
#                                   features=features,
#                                   CPUs=6,
#                                   )
#
# # Perform the uncertainty quantification using
# # polynomial chaos with point collocation (by default)
# data = UQ.quantify(seed=10,
#     data_folder="/home/surya/Programs/PycharmProjects/air_model/data/processed/schwarzsee/simulations/data/",
#     figure_folder="/home/surya/Programs/PycharmProjects/air_model/data/processed/schwarzsee/simulations/figures/",
#     filename="full")
