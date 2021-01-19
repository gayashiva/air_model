import uncertainpy as un
import chaospy as cp
import pandas as pd
import math
import sys

sys.path.append("/home/surya/Programs/Github/air_model")
from src.data.config import SITE, FOUNTAIN, FOLDERS
from src.models.air import Icestupa


def uniform(parameter, interval):
    if parameter == 0:
        raise ValueError("Creating a percentage distribution around 0 does not work")

    return cp.Uniform(
        parameter - abs(interval / 2.0 * parameter),
        parameter + abs(interval / 2.0 * parameter),
    )


def max_volume(time, values):
    # Calculate the feature using time, values and info.
    icev_max = values.max()
    # Return the feature times and values.
    return None, icev_max  # todo include efficiency


class UQ_Icestupa(un.Model, Icestupa):
    def __init__(self):

        super(UQ_Icestupa, self).__init__(
            labels=["Time (days)", "Ice Volume ($m^3$)"], interpolate=True
        )

        self.df = pd.read_hdf(FOLDERS["input_folder"] + "model_input_extended.h5", "df")
        self.TIME_STEP = 10 * 60
        self.df = (
            self.df.set_index("When")
            .resample(str(int(self.TIME_STEP / 60)) + "T")
            .mean()
            .reset_index()
        )

    def run(self, **parameters):

        self.set_parameters(**parameters)
        print(parameters.values())

        # if "dia_f" or "h_f" or "T_w" in parameters.keys():  # todo change to general
        #     FOUNTAIN["dia_f"] = dia_f
        #     FOUNTAIN["h_f"] = h_f
        #     FOUNTAIN["T_w"] = T_w
        #     SITE["h_aws"] = h_aws

        if "dia_f" or "h_f" in parameters.keys():  # todo change to general
            """ Fountain Spray radius """
            Area = math.pi * math.pow(FOUNTAIN["dia_f"], 2) / 4

            for row in self.df[1:].itertuples():
                v_f = row.Discharge / (60 * 1000 * Area)
                self.df.loc[row.Index, "r_f"] = self.projectile_xy(v_f, FOUNTAIN["h_f"])

        if "A_I" or "T_RAIN" in parameters.keys():
            """Albedo Decay"""
            self.T_DECAY = (
                self.T_DECAY * 24 * 60 * 60 / self.TIME_STEP
            )  # convert to 5 minute time steps
            s = 0
            f = 0

        for row in self.df[1:].itertuples():
            s, f = self.albedo(row, s, f)

        self.melt_freeze()

        Efficiency = (
            (self.df["meltwater"].iloc[-1] + self.df["ice"].iloc[-1])
            / self.df["input"].iloc[-1]
            * 100
        )

        print("\nIce Volume Max", float(self.df["iceV"].max()))
        print("Fountain efficiency", Efficiency)
        print("Ice Mass Remaining", self.df["ice"].iloc[-1])
        print("Meltwater", self.df["meltwater"].iloc[-1])
        print("Ppt", self.df["ppt"].sum())
        print("Number of days", self.df.index[-1] * self.TIME_STEP / (60 * 60 * 24))
        print("\n")

        self.df = self.df.set_index("When").resample("1H").mean().reset_index()

        for i in range(len(self.df), 75 * 24):
            self.df.loc[i, "iceV"] = 0

        return self.df.index.values / 24, self.df["iceV"].values


list_of_feature_functions = [max_volume]

features = un.Features(
    new_features=list_of_feature_functions, features_to_run=["max_volume"]
)

# # Set all parameters to have a uniform distribution
# # within a 20% interval around their fixed value
# parameters.set_all_distributions(un.uniform(0.05))

IE = 0.95
A_I = 0.35
A_S = 0.85
T_DECAY = 10

interval = 0.05

# ie_dist = uniform(IE, interval)
ie_dist = cp.Uniform(0.949, 0.993)
a_i_dist = uniform(A_I, interval)
a_s_dist = uniform(A_S, interval)

t_decay_dist = cp.Uniform(1, 22)
T_rain_dist = cp.Uniform(0, 2)
# T_rain_dist = uniform(1, 1)

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
time_steps_dist = cp.Uniform(5 * 60, 30 * 60)

# parameters_single = {
#     "IE": ie_dist,
#     "A_I": a_i_dist,
#     "A_S": a_s_dist,
#     "T_DECAY": t_decay_dist,
#     "T_RAIN": T_rain_dist,
#     "dia_f": dia_f_dist,
#     "h_f": h_f_dist,
#     "h_aws": h_aws_dist,
#     "T_w": T_w_dist,
#     "DX": dx_dist,
#     "TIME_STEP": time_steps_dist,
# }

# Create the parameters
# for k, v in parameters_single.items():
#     print(k, v)
#     parameters = un.Parameters({k: v})

#     # Initialize the model
#     model = UQ_Icestupa()

#     # Set up the uncertainty quantification
#     UQ = un.UncertaintyQuantification(
#         model=model,
#         parameters=parameters,
#         features=features,
#         CPUs=4,
#     )

#     # Perform the uncertainty quantification using # polynomial chaos with point collocation (by default) data =
#     data = UQ.quantify(
#         seed=10,
#         data_folder=FOLDERS["sim_folder"],
#         figure_folder=FOLDERS["sim_folder"],
#         filename=k,
#     )

parameters = {
    "IE": ie_dist,
    "T_RAIN": T_rain_dist,
}
parameters = un.Parameters(parameters)

# Initialize the model
model = UQ_Icestupa()

# Set up the uncertainty quantification
UQ = un.UncertaintyQuantification(
    model=model,
    parameters=parameters,
    features=features,
    CPUs=4,
)

# Perform the uncertainty quantification using
# polynomial chaos with point collocation (by default)
data = UQ.quantify(
    seed=10,
    data_folder=FOLDERS["sim_folder"],
    figure_folder=FOLDERS["sim_folder"],
    filename="full",
)
