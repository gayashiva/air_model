""" UncertaintyQuantification of Icestupa class input
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

    def run(self, **parameters):

        self.set_parameters(**parameters)

        if "r_F" in parameters.keys():
            self.self_attributes()

        if "D_F" in parameters.keys():
            self.df.loc[self.df.Discharge !=0, "Discharge"] = self.D_F
            print("Discharge changed to %s"%self.D_F)

        self.melt_freeze()

        if len(self.df) != 0:
            if len(self.df) >= self.total_hours:
                self.df = self.df[: self.total_hours]
            else:
                for i in range(len(self.df), self.total_hours):
                    self.df.loc[i, "iceV"] = 0
        else:
            for i in range(0, self.total_hours):
                self.df.loc[i, "iceV"] = self.V_dome

        return (
            self.df.index.values,
            self.df["iceV"].values,
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
    # locations = ["gangles21"]

    for location in locations:
        # Get settings for given location and trigger
        SITE, FOLDER = config(location)
        icestupa = Icestupa(location)
        icestupa.read_input()
        icestupa.self_attributes()

        if location in ['guttannen21', 'guttannen20']:
            D_F_dist = cp.Uniform(3.5, 15)

        if location in ['gangles21']:
            D_F_dist = cp.Uniform(30, 120)

        parameters= {
            "D_F":D_F_dist,
            "r_F":cp.Uniform(icestupa.r_F * 0.9,icestupa.r_F * 1.1),
            "T_F":cp.Uniform(0,3),
        }

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
            filename="input",
            method="pc",
        )
