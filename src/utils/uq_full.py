""" UncertaintyQuantification of Icestupa class
"""
import uncertainpy as un
# import pickle
# pickle.HIGHEST_PROTOCOL = 4 # For python version 2.7
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
from src.utils.uq_air import UQ_Icestupa

def setup_params():
    params = ['IE', 'A_I', 'T_PPT', 'Z', 'T_W']
    params_range = []
    for param in params:
        y_lim=get_parameter_metadata(param)['ylim']
        param_range = cp.Uniform(y_lim[0], y_lim[1])
        params_range.append(param_range)
        print(param, param_range)

    tuned_params = {params[i]: params_range[i] for i in range(len(params))}
    return tuned_params

if __name__ == "__main__":
    # Main logger
    logger = logging.getLogger(__name__)
    coloredlogs.install(
        fmt="%(funcName)s %(levelname)s %(message)s",
        level=logging.ERROR,
        logger=logger,
    )

    locations = ["gangles21", "guttannen21", "guttannen20"]
    for location in locations:

        # Get settings for given location and trigger
        SITE, FOLDER = config(location)
        icestupa = Icestupa(location)
        icestupa.read_input()
        icestupa.self_attributes()

        parameters = un.Parameters(setup_params())

        # Initialize the model
        model = UQ_Icestupa(location)

        # Set up the uncertainty quantification
        UQ = un.UncertaintyQuantification(
            model=model,
            parameters=parameters,
            # CPUs=2,
        )

        # Perform the uncertainty quantification using
        # polynomial chaos with point collocation (by default)
        data = UQ.quantify(
            seed=10,
            data_folder=FOLDER["sim"],
            figure_folder=FOLDER["sim"],
            filename="efficiency",
        )
