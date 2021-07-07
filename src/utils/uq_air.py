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


def max_volume(time, values, info, y_true, y_pred, se):
    icev_max = values.max()
    for param_name in sorted(info.keys()):
        print("\n\t%s: %r" % (param_name, info[param_name]))
    print("\n\tMax Ice Volume %0.1f\n"% (icev_max))
    return None, icev_max 

def rmse(time, values, info, y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = math.sqrt(mse)
    for param_name in sorted(info.keys()):
        print("\n\t%s: %r" % (param_name, info[param_name]))
    print("\n\tRMSE %0.1f\n"% (rmse))
    return None, rmse

def efficiency(time, values, info, y_true, y_pred, se):
    for param_name in sorted(info.keys()):
        print("\n\t%s: %r" % (param_name, info[param_name]))
    print("\n\tSE %0.1f\n"% (se))
    return None, se

class UQ_Icestupa(un.Model, Icestupa):
    def __init__(self, location):
        super(UQ_Icestupa, self).__init__(
            labels=["Time (days)", "Ice Volume ($m^3$)"], interpolate=True
        )

        SITE, FOLDER = config(location)
        initial_data = [SITE, FOLDER]

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
        print("Ice volume measurements for %s are %s\n"% (self.name, self.y_true))

        if location == "guttannen21":
            self.total_days = 180
        if location == "schwarzsee19":
            self.total_days = 60
        if location == "guttannen20":
            self.total_days = 110
        if location == "gangles21":
            self.total_days = 150

    def run(self, **parameters):

        self.set_parameters(**parameters)
        # logger.info(parameters.values())

        if "r_spray" in parameters.keys():
            self.self_attributes()

        if "D_MEAN" in parameters.keys():
            self.get_discharge()

        if "A_I" or "A_S" or "T_PPT" or "A_DECAY" in parameters.keys():
            """Albedo Decay parameters initialized"""
            self.A_DECAY = self.A_DECAY * 24 * 60 * 60 / self.DT
            s = 0
            f = 1
            for i, row in self.df.iterrows():
                s, f = self.get_albedo(i, s, f)

        self.melt_freeze()

        if len(self.df) != 0:
            if len(self.df) >= self.total_days * 24:
                self.df = self.df[: self.total_days * 24]
            else:
                for i in range(len(self.df), self.total_days * 24):
                    self.df.loc[i, "iceV"] = self.df.loc[i-1, "iceV"]
            y_pred = []
            for date in self.df_c.When.values :
                if (self.df[self.df.When == date].shape[0]): 
                    y_pred.append(self.df.loc[self.df.When == date, "iceV"].values[0])
                else:
                    # y_pred.append(self.V_dome)
                    y_pred.append(0)
        else:
            for i in range(0, self.total_days * 24):
                self.df.loc[i, "iceV"] = self.V_dome 
            y_pred = [999] * len(self.df_c.When.values)

        M_input = round(self.df["input"].iloc[-1],1)
        M_water = round(self.df["meltwater"].iloc[-1],1)
        M_ice = round(self.df["ice"].iloc[-1]- self.V_dome * self.RHO_I,1)
        se = (M_water + M_ice) / M_input

        return self.df.index.values, self.df["iceV"].values, parameters, self.y_true, y_pred, se

if __name__ == "__main__":
    # Main logger
    logger = logging.getLogger(__name__)
    coloredlogs.install(
        fmt="%(funcName)s %(levelname)s %(message)s",
        level=logging.ERROR,
        logger=logger,
    )

    locations = ['gangles21', 'guttannen20', 'guttannen21']

    for location in locations:
        # Get settings for given location and trigger
        SITE, FOLDER = config(location)
        icestupa = Icestupa(location)
        icestupa.read_input()
        icestupa.self_attributes()

        list_of_feature_functions = [max_volume, rmse, efficiency]

        features = un.Features(
            # new_features=list_of_feature_functions, features_to_run=["max_volume"]
            # new_features=list_of_feature_functions, features_to_run=["rmse"]
            new_features=list_of_feature_functions, features_to_run=["efficiency"]
        )

        a_i_dist = cp.Uniform(0.01, 0.35)
        a_s_dist = cp.Uniform(icestupa.A_S * .95, icestupa.A_S * 1.05)
        z_dist = cp.Uniform(0.001, 0.005)
        dx_dist = cp.Uniform(icestupa.DX * .95, icestupa.DX * 1.05)
        # r_spray_dist = cp.Uniform(icestupa.r_spray * .95, icestupa.r_spray * 1.05)
        ie_dist = cp.Uniform(0.95, 0.99)
        a_decay_dist = cp.Uniform(icestupa.A_DECAY * .95, icestupa.A_DECAY* 1.05)
        T_PPT_dist = cp.Uniform(0, 2)
        # MU_CONE_dist = cp.Uniform(0, 1)
        T_W_dist = cp.Uniform(0, 5)
        if location in ['guttannen21', 'guttannen20']:
            d_dist = cp.Uniform(3, 10)
        if location == 'gangles21':
            d_dist = cp.Uniform(20, 90)

        parameters_full = {
            "IE": ie_dist,
            "A_I": a_i_dist,
            "A_S": a_s_dist,
            "Z": z_dist,
            "A_DECAY": a_decay_dist,
            "T_PPT": T_PPT_dist,
            "DX": dx_dist,

            "T_W": T_W_dist,
#             "D_MEAN": d_dist,
#             "r_spray": r_spray_dist,
            # "MU_CONE": MU_CONE_dist,
        }


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
                CPUs=2,
            )

            # Perform the uncertainty quantification using # polynomial chaos with point collocation (by default) data =
            data = UQ.quantify(
                seed=10,
                data_folder=FOLDER["sim"],
                figure_folder=FOLDER["sim"],
                filename=k,
            )
