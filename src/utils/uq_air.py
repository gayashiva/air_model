""" UncertaintyQuantification of Icestupa class
"""
import uncertainpy as un
import chaospy as cp
import pandas as pd
import math
import sys
import os
import logging
import coloredlogs
from sklearn.metrics import mean_squared_error

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
)

from src.utils.settings import config
from src.models.icestupaClass import Icestupa
from src.models.methods.calibration import get_calibration
from src.models.methods.metadata import get_parameter_metadata
from src.models.methods.solar import get_solar
from src.models.methods.droplet import get_droplet_projectile

def setup_params(params):
    params_range = []
    for param in params:
        y_lim=get_parameter_metadata(param)['ylim']
        param_range = cp.Uniform(y_lim[0], y_lim[1])
        params_range.append(param_range)
        print(param, param_range)

    tuned_params = {params[i]: params_range[i] for i in range(len(params))}
    return tuned_params

# def max_volume(time, values, params, y_true, y_pred, se):
#     icev_max = values.max()
#     for param_name in sorted(params.keys()):
#         print("\n\t%s: %r" % (param_name, params[param_name]))
#     print("\n\tMax Ice Volume %0.1f\n" % (icev_max))
#     return None, icev_max


def rmse_V(time, values, params, y_true, y_pred, z_true, z_pred):
    print(y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = math.sqrt(mse)
    for param_name in sorted(params.keys()):
        print("\n\t%s: %r" % (param_name, params[param_name]))
    print("\n\tRMSE %0.1f\n" % (rmse))
    return None, rmse

def rmse_T(time, values, params, y_true, y_pred, z_true, z_pred):
    mse = mean_squared_error(z_true, z_pred)
    rmse_T = math.sqrt(mse)
    for param_name in sorted(params.keys()):
        print("\n\t%s: %r" % (param_name, params[param_name]))
    print("\n\tRMSE T %0.3f\n" % (rmse_T))
    return None, rmse_T

# def efficiency(time, values, params, y_true, y_pred, se):
#     for param_name in sorted(params.keys()):
#         print("\n\t%s: %r" % (param_name, params[param_name]))
#     print("\n\tSE %0.1f\n" % (se))
#     return None, se


class UQ_Icestupa(un.Model, Icestupa):
    def __init__(self, location, ignore=False):
        super(UQ_Icestupa, self).__init__(
            labels=["Time (days)", "Ice Volume ($m^3$)"], 
            interpolate=False,
            suppress_graphics=False,
            logger_level="debug",
            ignore = ignore
        )

        SITE, FOLDER = config(location)
        initial_data = [SITE, FOLDER]
        diff = SITE["end_date"] - SITE["start_date"]
        days, seconds = diff.days, diff.seconds
        self.total_hours = days * 24 + seconds // 3600

        # Initialise all variables of dictionary
        for dictionary in initial_data:
            for key in dictionary:
                setattr(self, key, dictionary[key])
                # logger.info(f"%s -> %s" % (key, str(dictionary[key])))

        self.read_input()
        self.self_attributes()

        self.df_c = pd.read_hdf(FOLDER["input"] + "model_input.h5", "df_c")
        self.df_c = self.df_c.iloc[1:]
        self.y_true = self.df_c.DroneV.values

        if location == 'gangles21':
            self.z_true = [0]
        else:
            self.df_cam = pd.read_hdf(FOLDER["input"] + "model_input.h5", "df_cam")
            self.df_cam = self.df_cam.reset_index()
            # self.df_cam = self.df_cam.iloc[1180:]
            self.z_true = self.df_cam.cam_temp.values

        print("Ice volume measurements for %s are %s\n" % (self.name, len(self.y_true)))
        print("Surface temp measurements for %s are %s\n" % (self.name, len(self.z_true)))

    def run(self, **parameters):

        self.set_parameters(**parameters)
        # logger.info(parameters.values())

        if "r_F" in parameters.keys():
            self.self_attributes()

        if "D_F" in parameters.keys():
            self.df.loc[self.df.Discharge !=0, "Discharge"] = self.D_F
            # self.get_discharge()

        if "A_I" or "A_S" or "T_PPT" or "A_DECAY" in parameters.keys():
            """Albedo Decay parameters initialized"""
            self.A_DECAY = self.A_DECAY * 24 * 60 * 60 / self.DT
            s = 0
            f = 1
            for i, row in self.df.iterrows():
                s, f = self.get_albedo(i, s, f)

        self.melt_freeze()

        if len(self.df) != 0:
            M_input = round(self.df["input"].iloc[-1], 1)
            M_water = round(self.df["meltwater"].iloc[-1], 1)
            M_ice = round(self.df["ice"].iloc[-1] - self.V_dome * self.RHO_I, 1)
            se = (M_water + M_ice) / M_input * 100

            self.last_hour = len(self.df) -2
            if len(self.df) >= self.total_hours:
                self.df = self.df[: self.total_hours]
            else:
                for i in range(len(self.df), self.total_hours):
                    self.df.loc[i, "iceV"] = 0
            y_pred = []
            z_pred = []
            for date in self.df_c.When.values:
                if self.df[self.df.When == date].shape[0]:
                    print(self.df.loc[self.df.When == date, "iceV"].values[0])
                    y_pred.append(self.df.loc[self.df.When == date, "iceV"].values[0])
                else:
                    # y_pred.append(self.V_dome)
                    # print("Error: Date not found")
                    y_pred.append(0)

            if self.name != 'gangles21':
                for date in self.df_cam.When.values:
                    if self.df[self.df.When == date].shape[0]:
                        z_pred.append(self.df.loc[self.df.When == date, "T_s"].values[0])
                    else:
                        # print("Error: Date not found")
                        z_pred.append(0)
            else:
                z_pred = [0]
        else:
            for i in range(0, self.total_hours):
                self.df.loc[i, "iceV"] = self.V_dome
            y_pred = [999] * len(self.df_c.When.values)
            z_pred = [999] * len(self.df_cam.When.values)
            se = 0

        return (
            self.df.index.values[self.last_hour],
            # self.df["iceV"].values,
            se,
            parameters,
            self.y_true,
            y_pred,
            self.z_true,
            [round(num, 3) for num in z_pred],
        )


if __name__ == "__main__":
    # Main logger
    logger = logging.getLogger(__name__)
    coloredlogs.install(
        fmt="%(funcName)s %(levelname)s %(message)s",
        level=logging.ERROR,
        logger=logger,
    )

    locations = ["gangles21", "guttannen20", "guttannen21"]
    # locations = ["guttannen21"]

    for location in locations:
        # Get settings for given location and trigger
        SITE, FOLDER = config(location)
        icestupa = Icestupa(location)
        icestupa.read_input()
        icestupa.self_attributes()

        list_of_feature_functions = [rmse_V, rmse_T]

        features = un.Features(
            # new_features=list_of_feature_functions, features_to_run=["max_volume"]
            # new_features=list_of_feature_functions, features_to_run=["rmse"]
            new_features=list_of_feature_functions,
            features_to_run=["rmse_T", "rmse_V"],
        )

        params = ['D_F', 'A_I', 'A_S','A_DECAY', 'T_PPT', 'Z', 'T_F', 'DX', 'IE', 'r_F']
        # params = ['IE', 'A_I', 'A_S', 'Z', 'A_DECAY', 'T_PPT', 'DX', 'T_F']
        parameters_full = setup_params(params)

        # Create the parameters
        for k, v in parameters_full.items():
            print(k, v)
            parameters_single = un.Parameters({k: v})

            # Initialize the model
            model = UQ_Icestupa(location=location)

            # Set up the uncertainty quantification
            UQ = un.UncertaintyQuantification(
                model=model,
                parameters=parameters_single,
                features=features,
                # CPUs=3,
            )

            # Perform the uncertainty quantification using # polynomial chaos with point collocation (by default) data =
            data = UQ.quantify(
                seed=10,
                data_folder=FOLDER["sim"],
                figure_folder=FOLDER["sim"],
                filename=k,
            )
