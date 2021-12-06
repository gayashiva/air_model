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
# from sklearn.metrics import mean_squared_error

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
)

from src.utils.settings import config
from src.models.icestupaClass import Icestupa
from src.models.methods.calibration import get_calibration
from src.models.methods.metadata import get_parameter_metadata
from src.models.methods.solar import get_solar
# from src.utils.cv import setup_params

def setup_params_dist(icestupa, params):
    params_range = []
    for param in params:
        y_lim = get_parameter_metadata(param)['ylim']
        if param in ['R_F', 'D_F']:
            param_range = cp.Uniform(getattr(icestupa, param) * y_lim[0],getattr(icestupa, param) * y_lim[1])
        else:
            param_range = cp.Uniform(y_lim[0], y_lim[1])
        params_range.append(param_range)
        print("\t%s : %s\n" %(param, param_range))
    tuned_params = {params[i]: params_range[i] for i in range(len(params))}
    return tuned_params

# def max_volume(time, values, params, y_true, y_pred, se):
#     icev_max = values.max()
#     for param_name in sorted(params.keys()):
#         print("\n\t%s: %r" % (param_name, params[param_name]))
#     print("\n\tMax Ice Volume %0.1f\n" % (icev_max))
#     return None, icev_max

def survival(time, values, params, y_true, y_pred, z_true, z_pred, total_hours, last_hour):
    diff = abs(total_hours - last_hour)
    for param_name in sorted(params.keys()):
        print("\n\t%s: %r" % (param_name, params[param_name]))
    print("\n\tHour diff %0.1f\n" % (diff))
    return None, diff

def rmse_V(time, values, params, y_true, y_pred, z_true, z_pred, total_hours,last_hour):
    mse = mean_squared_error(y_true, y_pred)
    rmse = math.sqrt(mse)
    for param_name in sorted(params.keys()):
        print("\n\t%s: %r" % (param_name, params[param_name]))
    print("\n\tRMSE %0.1f\n" % (rmse))
    return None, rmse

def rmse_T(time, values, params, y_true, y_pred, z_true, z_pred, total_hours,last_hour):
    mse = mean_squared_error(z_true, z_pred)
    rmse_T = math.sqrt(mse)
    for param_name in sorted(params.keys()):
        print("\n\t%s: %r" % (param_name, params[param_name]))
    print("\n\tRMSE T %0.3f\n" % (rmse_T))
    return None, rmse_T

def rmse(time, values, params, y_true, y_pred, z_true, z_pred, total_hours,last_hour):
    mse_T = mean_squared_error(z_true, z_pred)
    rmse_T = math.sqrt(mse_T)
    mse_V = mean_squared_error(y_true, y_pred)
    rmse_V = math.sqrt(mse_V)
    rmse = math.sqrt(rmse_V**2+rmse_T**2)
    for param_name in sorted(params.keys()):
        print("\n\t%s: %r" % (param_name, params[param_name]))
    print("\n\tRMSE T %0.3f\n" % (rmse))
    return None, rmse

def efficiency(time, values, params, se):
    for param_name in sorted(params.keys()):
        print("\n\t%s: %r" % (param_name, params[param_name]))
    print("\n\tSE %0.1f\n" % (se))
    return None, se


class UQ_Icestupa(un.Model, Icestupa):
    def __init__(self, location, ignore=False):
        super(UQ_Icestupa, self).__init__(
            # labels=["Time (days)", "Ice Volume ($m^3$)"], 
            interpolate=False,
            suppress_graphics=False,
            logger_level="warning",
            ignore = ignore
        )

        CONSTANTS, SITE, FOLDER = config(location)
        initial_data = [CONSTANTS, SITE, FOLDER]
        diff = SITE["melt_out"] - SITE["start_date"]
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

        # if location == 'gangles21':
        #     self.z_true = [0]
        # else:
        #     self.df_cam = pd.read_hdf(FOLDER["input"] + "model_input.h5", "df_cam")
        #     self.df_cam = self.df_cam.reset_index()
        #     # self.df_cam = self.df_cam.iloc[1180:]
        #     self.z_true = self.df_cam.cam_temp.values

        print("Ice volume measurements for %s are %s\n" % (self.name, len(self.y_true)))
        # print("Surface temp measurements for %s are %s\n" % (self.name, len(self.z_true)))

    def run(self, **parameters):

        self.set_parameters(**parameters)
        # logger.info(parameters.values())

        if "R_F" in parameters.keys():
            self.self_attributes()

        if "D_F" in parameters.keys():
            self.df.loc[self.df.Discharge !=0, "Discharge"] = self.D_F

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
            if M_input !=0:
                SE = (M_water + M_ice) / M_input * 100
            else:
                SE=0

            last_hour = len(self.df) -1
            if len(self.df) >= self.total_hours:
                self.df = self.df[: self.total_hours]
            else:
                for i in range(len(self.df), self.total_hours):
                    self.df.loc[i, "iceV"] = 0
            y_pred = []
            # z_pred = []
            for date in self.df_c.time.values:
                if self.df[self.df.time == date].shape[0]:
                    y_pred.append(self.df.loc[self.df.time == date, "iceV"].values[0])
                else:
                    # y_pred.append(self.V_dome)
                    # print("Error: Date not found")
                    y_pred.append(0)

            # if self.name != 'gangles21':
            #     for date in self.df_cam.time.values:
            #         if self.df[self.df.time == date].shape[0]:
            #             z_pred.append(self.df.loc[self.df.time == date, "T_s"].values[0])
            #         else:
            #             # print("Error: Date not found")
            #             z_pred.append(0)
            # else:
            #     z_pred = [0]
        else:
            for i in range(0, self.total_hours):
                self.df.loc[i, "iceV"] = self.V_dome
            y_pred = [999] * len(self.df_c.time.values)
            # z_pred = [999] * len(self.df_cam.time.values)
            SE = 0
            last_hour = 0

        return (
            self.df.index.values,
            self.df["iceV"].values,
            parameters,
            SE,
        )
        # return (
        #     self.df.index.values,
        #     self.df["iceV"].values,
        # )
        # return (
        #     # None,
        #     # se,
        #     self.df.index.values,
        #     self.df["iceV"].values,
        #     # parameters,
        #     # self.y_true,
        #     # y_pred,
        #     # self.z_true,
        #     # [round(num, 3) for num in z_pred],
        #     # self.total_hours,
        #     # last_hour,
        # )


if __name__ == "__main__":
    # Main logger
    logger = logging.getLogger(__name__)
    coloredlogs.install(
        fmt="%(funcName)s %(levelname)s %(message)s",
        level=logging.ERROR,
        logger=logger,
    )

    # locations = ["gangles21", "guttannen21"]
    locations = ["guttannen21"]

    for location in locations:
        # Get settings for given location and trigger
        CONSTANTS, SITE, FOLDER = config(location)
        icestupa = Icestupa(location)
        icestupa.read_input()
        icestupa.self_attributes()

        list_of_feature_functions = [rmse_V, rmse_T, rmse, efficiency]

        features = un.Features(
            # new_features=list_of_feature_functions, features_to_run=["max_volume"]
            # new_features=list_of_feature_functions, features_to_run=["rmse"]
            new_features=list_of_feature_functions,
            features_to_run=["efficiency"],
        )

        if location == 'gangles21':
            params = ['IE', 'A_I', 'Z', 'DX']
        else:
            params = ['IE', 'A_I', 'A_S','A_DECAY', 'T_PPT', 'Z', 'DX']
        parameters_full = setup_params_dist(icestupa, params)

        # Initialize the model
        model = UQ_Icestupa(location=location)

        # Set up the uncertainty quantification
        UQ = un.UncertaintyQuantification(
            model=model,
            parameters=parameters_full,
            features=features,
            # CPUs=3,
        )

        # Perform the uncertainty quantification using # polynomial chaos with point collocation (by default) data =
        data = UQ.quantify(
            seed=10,
            data_folder=FOLDER["sim"],
            figure_folder=FOLDER["sim"],
            filename="globalSA",
        )

        # # Create the parameters
        # for k, v in parameters_full.items():
        #     print(k, v)
        #     parameters_single = un.Parameters({k: v})

        #     # Initialize the model
        #     model = UQ_Icestupa(location=location)

        #     # Set up the uncertainty quantification
        #     UQ = un.UncertaintyQuantification(
        #         model=model,
        #         parameters=parameters_single,
        #         features=features,
        #         # CPUs=3,
        #     )

        #     # Perform the uncertainty quantification using # polynomial chaos with point collocation (by default) data =
        #     data = UQ.quantify(
        #         seed=10,
        #         data_folder=FOLDER["sim"],
        #         figure_folder=FOLDER["sim"],
        #         filename=k,
        #     )
