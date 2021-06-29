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

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
)

from src.utils.settings import config
from src.models.icestupaClass import Icestupa
from src.models.methods.calibration import get_calibration
from src.models.methods.metadata import get_parameter_metadata
from src.models.methods.solar import get_solar
from src.models.methods.droplet import get_droplet_projectile


def max_volume(time, values, info, result=[]):
    # Calculate the feature using time, values and info.
    icev_max = values.max()
    # result.append([info, icev_max])
    for param_name in sorted(info.keys()):
        print("\n\t%s: %r" % (param_name, info[param_name]))
    print("Max Ice Volume %0.1f\n"% (icev_max))
    # Return the feature times and values.
    return None, icev_max  # todo include efficiency

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
        # result = []

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

        if "A_I" or "A_S" or "T_PPT" or "T_DECAY" or "H_PPT" in parameters.keys():
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
        else:
            for i in range(0, self.total_days * 24):
                self.df.loc[i, "iceV"] = self.V_dome 

        return self.df.index.values, self.df["iceV"].values, parameters

if __name__ == "__main__":
    # Main logger
    logger = logging.getLogger(__name__)
    coloredlogs.install(
        fmt="%(funcName)s %(levelname)s %(message)s",
        level=logging.ERROR,
        logger=logger,
    )

    locations = ['guttannen20', 'guttannen21', 'gangles21']

    for location in locations:
        # Get settings for given location and trigger
        SITE, FOLDER = config(location)
        icestupa = Icestupa(location)
        icestupa.read_input()
        icestupa.self_attributes()

        list_of_feature_functions = [max_volume]

        features = un.Features(
            new_features=list_of_feature_functions, features_to_run=["max_volume"]
        )

        a_i_dist = cp.Uniform(icestupa.A_I * .95, icestupa.A_I * 1.05)
        a_s_dist = cp.Uniform(icestupa.A_S * .95, icestupa.A_S * 1.05)
        dx_dist = cp.Uniform(icestupa.DX * .95, icestupa.DX * 1.05)
        r_spray_dist = cp.Uniform(icestupa.r_spray * .95, icestupa.r_spray * 1.05)

        ie_dist = cp.Uniform(0.949, 0.993)
        a_decay_dist = cp.Uniform(1, 22)
        T_PPT_dist = cp.Uniform(0, 2)
        H_PPT_dist = cp.Uniform(0, 2)
        T_W_dist = cp.Uniform(0, 5)
        if location in ['guttannen21', 'guttannen20']:
            d_dist = cp.Uniform(5, 10)
        if location == 'gangles21':
            d_dist = cp.Uniform(30, 90)

        parameters_single = {
            # "IE": ie_dist,
            # "A_I": a_i_dist,
            # "A_S": a_s_dist,
            # "A_DECAY": a_decay_dist,
            # "T_PPT": T_PPT_dist,
            # "H_PPT": H_PPT_dist,
            # "T_W": T_W_dist,
            # "DX": dx_dist,
            # "d_mean": d_dist,
            "r_spray": r_spray_dist,
        }


        # Create the parameters
        for k, v in parameters_single.items():
            print(k, v)
            parameters = un.Parameters({k: v})

            # Initialize the model
            model = UQ_Icestupa(location=location)

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
