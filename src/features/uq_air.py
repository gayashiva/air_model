import uncertainpy as un
import chaospy as cp
import pandas as pd
import math
import sys
import os
import logging
import coloredlogs

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
)

from src.data.settings import config
from src.models.icestupaClass import Icestupa
from src.models.methods.calibration import get_calibration
from src.models.methods.metadata import get_parameter_metadata
from src.models.methods.solar import get_solar
from src.models.methods.droplet import get_droplet_projectile
from src.data.settings import config


def uniform(parameter, interval):
    if parameter == 0:
        raise ValueError("Creating a percentage distribution around 0 does not work")

    return cp.Uniform(
        parameter - abs(interval / 2.0 * parameter),
        parameter + abs(interval / 2.0 * parameter),
    )


def max_volume(time, values, info, result=[]):
    # Calculate the feature using time, values and info.
    icev_max = values.max()
    # result.append([info, icev_max])
    logger.error("%0.5f %0.4f"% (info, icev_max))
    # print(info, icev_max)
    # Return the feature times and values.
    return None, icev_max  # todo include efficiency


def optimum_dx(result, dx, icev_max, sim_folder):
    result.append([dx, icev_max])
    print(result)
    if result.len() == 12:
        df = pd.DataFrame(result, columns=["thickness", "Max vol"])
        df.to_csv(sim_folder + "result" + self.trigger + ".csv")
    else:
        logger.warning(result.len())
    return result


class UQ_Icestupa(un.Model, Icestupa):
    # def __init__(self):
    def __init__(self, location="Guttannen 2021", trigger="Manual"):
        super(UQ_Icestupa, self).__init__(
            labels=["Time (days)", "Ice Volume ($m^3$)"], interpolate=True
        )

        SITE, FOUNTAIN, FOLDER = config(location, trigger)
        initial_data = [SITE, FOUNTAIN, FOLDER]

        # Initialise all variables of dictionary
        for dictionary in initial_data:
            for key in dictionary:
                setattr(self, key, dictionary[key])
                logger.info(f"%s -> %s" % (key, str(dictionary[key])))

        self.TIME_STEP = 15 * 60
        self.df = pd.read_hdf(self.input + "model_input_" + self.trigger + ".h5", "df")
        if self.name in ["gangles21"]:
            df_c = get_calibration(site=self.name, input=self.input)
            self.r_spray = df_c.loc[1, "dia"] / 2
            self.h_i = self.DX

        if self.name in ["guttannen21", "guttannen20"]:
            df_c, df_cam = get_calibration(site=self.name, input=self.input)
            self.r_spray = df_c.loc[0, "dia"] / 2
            self.h_i = 3 * df_c.loc[0, "DroneV"] / (math.pi * self.r_spray ** 2)

        if hasattr(self, "r_spray"):  # Provide discharge
            self.discharge = get_droplet_projectile(
                dia=self.dia_f, h=self.h_f, x=self.r_spray
            )
        else:  # Provide spray radius
            self.r_spray = get_droplet_projectile(
                dia=self.dia_f, h=self.h_f, d=self.discharge
            )
        result = []

    def run(self, **parameters):

        self.set_parameters(**parameters)
        logger.info(parameters.values())
        logger.warning(self.DX)

        # self.df = pd.read_hdf(self.input + "model_input_" + self.trigger + ".h5", "df")
        # self.TIME_STEP = (
        #     int(pd.infer_freq(self.df["When"])[:-1]) * 60
        # )  # Extract time step from datetime column
        # logger.debug(f"Time steps -> %s minutes" % (str(self.TIME_STEP / 60)))
        # self.TIME_STEP = 15 * 60

        # self.df = (
        #     self.df.set_index("When")
        #     .resample(str(int(self.TIME_STEP / 60)) + "T")
        #     .mean()
        #     .reset_index()
        # )

        # if "dia_f" or "h_f" in parameters.keys():  # todo change to general
        #     """ Fountain Spray radius """ Area = math.pi * math.pow(FOUNTAIN["dia_f"], 2) / 4
        #     for row in self.df[1:].itertuples():
        #         v_f = row.Discharge / (60 * 1000 * Area)
        #         self.df.loc[row.Index, "r_f"] = self.projectile_xy(v_f, FOUNTAIN["h_f"])

        # if "A_I" or "T_RAIN" in parameters.keys():
        #     """Albedo Decay"""
        #     self.T_DECAY = (
        #         self.T_DECAY * 24 * 60 * 60 / self.TIME_STEP
        #     )  # convert to 5 minute time steps
        #     s = 0
        #     f = 0

        #     for row in self.df[1:].itertuples():
        #         s, f = self.albedo(row, s, f)

        self.melt_freeze()

        Efficiency = (
            (self.df["meltwater"].iloc[-1] + self.df["ice"].iloc[-1])
            / self.df["input"].iloc[-1]
            * 100
        )

        # print("\nIce Volume Max", float(self.df["iceV"].max()))
        # print("Ice Layer Thickness", float(self.DX))
        # print("Fountain efficiency", Efficiency)
        # print("Ice Mass Remaining", self.df["ice"].iloc[-1])
        # print("Meltwater", self.df["meltwater"].iloc[-1])
        # print("Ppt", self.df["ppt"].sum())
        # print("Number of days", self.df.index[-1] * self.TIME_STEP / (60 * 60 * 24))
        # print("\n")

        # result = optimum_dx(result, self.DX, self.df["iceV"].max(), self.sim)
        self.df = self.df.set_index("When").resample("1H").mean().reset_index()

        for i in range(len(self.df), 75 * 24):
            self.df.loc[i, "iceV"] = 0

        return self.df.index.values / 24, self.df["iceV"].values, self.DX


if __name__ == "__main__":
    # Main logger
    logger = logging.getLogger(__name__)
    coloredlogs.install(
        fmt="%(funcName)s %(levelname)s %(message)s",
        level=logging.ERROR,
        logger=logger,
    )

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

    ie_dist = cp.Uniform(0.949, 0.993)
    a_i_dist = uniform(A_I, interval)
    a_s_dist = uniform(A_S, interval)

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
    time_steps_dist = cp.Uniform(5 * 60, 30 * 60)

    parameters_single = {
        # "IE": ie_dist,
        # "A_I": a_i_dist,
        # "A_S": a_s_dist,
        # "T_DECAY": t_decay_dist,
        # "T_RAIN": T_rain_dist,
        # "dia_f": dia_f_dist,
        # "h_f": h_f_dist,
        # "h_aws": h_aws_dist,
        # "T_w": T_w_dist,
        "DX": dx_dist,
        # "TIME_STEP": time_steps_dist,
    }

    answers = dict(
        # location="Schwarzsee 2019",
        location="Guttannen 2021",
        # location="Gangles 2021",
        trigger="Manual",
        # trigger="None",
        # trigger="Temperature",
        # trigger="Weather",
        run="yes",
    )

    # Get settings for given location and trigger
    SITE, FOUNTAIN, FOLDER = config(answers["location"], answers["trigger"])

    # Create the parameters
    for k, v in parameters_single.items():
        print(k, v)
        parameters = un.Parameters({k: v})

        # Initialize the model
        model = UQ_Icestupa(location=answers["location"], trigger=answers["trigger"])

        # Set up the uncertainty quantification
        UQ = un.UncertaintyQuantification(
            model=model,
            parameters=parameters,
            features=features,
            # CPUs=1,
        )

        # Perform the uncertainty quantification using # polynomial chaos with point collocation (by default) data =
        data = UQ.quantify(
            seed=10,
            data_folder=FOLDER["sim"],
            figure_folder=FOLDER["sim"],
            filename=k,
        )

# parameters = {
#     "IE": ie_dist,
#     "T_RAIN": T_rain_dist,
# }
# parameters = un.Parameters(parameters)

# # Initialize the model
# model = UQ_Icestupa()

# # Set up the uncertainty quantification
# UQ = un.UncertaintyQuantification(
#     model=model,
#     parameters=parameters,
#     features=features,
#     CPUs=2,
# )

# # Perform the uncertainty quantification using
# # polynomial chaos with point collocation (by default)
# data = UQ.quantify(
#     seed=10,
#     data_folder=FOLDER["sim_folder"],
#     figure_folder=FOLDER["sim_folder"],
#     filename="full",
# )
