""" UncertaintyQuantification of Icestupa class
"""
import uncertainpy as un
import pickle
pickle.HIGHEST_PROTOCOL = 4 # For python version 2.7
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
from src.utils.uq_air import UQ_Icestupa, setup_params_dist, rmse_V

def max_volume(time, values, params, se):
    icev_max = values.max()
    for param_name in sorted(params.keys()):
        print("\n\t%s: %r" % (param_name, params[param_name]))
    print("\n\tMax Ice Volume %0.1f\n" % (icev_max))
    return None, icev_max

if __name__ == "__main__":
    # Main logger
    logger = logging.getLogger(__name__)
    logger.setLevel("ERROR")

    # locations = ["gangles21", "guttannen21", "guttannen20"]
    # locations = ["guttannen21", "guttannen20"]
    locations = ["guttannen20"]
    for location in locations:

        CONSTANTS, SITE, FOLDER = config(location)
        icestupa = Icestupa(location)
        icestupa.read_input()
        icestupa.self_attributes()

        types = ["fountain", "weather"]
        params = [['D_F', 'T_F'], ['IE', 'A_I', 'A_S','A_DECAY', 'T_PPT', 'Z'] ]

        if location == 'gangles21':
            params = [['D_F', 'T_F'], ['IE', 'A_I', 'Z'] ]

        list_of_feature_functions = [max_volume]
        features = un.Features(
            new_features=list_of_feature_functions, features_to_run=["max_volume"]
        )

        for i, type in enumerate(types):

            print(f"\n\tBegin uncertainty quantification of {type} in {location}\n")

            param = params[i]

            parameters = un.Parameters(setup_params_dist(icestupa, param))

            # Initialize the model
            model = UQ_Icestupa(location)

            # Set up the uncertainty quantification
            UQ = un.UncertaintyQuantification(
                model=model,
                parameters=parameters,
                features=features,
                # CPUs=12,
            )

            # Perform the uncertainty quantification using
            # polynomial chaos with point collocation (by default)
            data = UQ.quantify(
                seed=10,
                data_folder=FOLDER["sim"],
                figure_folder=FOLDER["sim"],
                filename=type,
                method="pc",
                # pc_method="spectral",
                rosenblatt=True           
            )
