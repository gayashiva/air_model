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
from src.utils.uq_air import UQ_Icestupa, max_volume


if __name__ == "__main__":
    # Main logger
    logger = logging.getLogger(__name__)
    coloredlogs.install(
        fmt="%(funcName)s %(levelname)s %(message)s",
        level=logging.ERROR,
        logger=logger,
    )

    location="gangles21"

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
    MU_CONE_PPT_dist = cp.Uniform(0, 1)
    T_W_dist = cp.Uniform(0, 5)
    if location in ['guttannen21', 'guttannen20']:
        d_dist = cp.Uniform(5, 10)
    if location == 'gangles21':
        d_dist = cp.Uniform(30, 90)

    parameters = {
            "IE": ie_dist,
            "A_I": a_i_dist,
            # "A_S": a_s_dist,
            # "A_DECAY": a_decay_dist,
            "T_PPT": T_PPT_dist,
            "MU_CONE": MU_CONE_dist,
            "DX": dx_dist,
            "T_W": T_W_dist,
            # "d_mean": d_dist,
            "r_spray": r_spray_dist,
    }
    parameters = un.Parameters(parameters)

    # Initialize the model
    model = UQ_Icestupa(location)

    # Set up the uncertainty quantification
    UQ = un.UncertaintyQuantification(
        model=model,
        parameters=parameters,
        features=features,
        # CPUs=2,
    )

    # Perform the uncertainty quantification using
    # polynomial chaos with point collocation (by default)
    data = UQ.quantify(
        seed=10,
        data_folder=FOLDER["sim"],
        figure_folder=FOLDER["sim"],
        filename="full",
    )
