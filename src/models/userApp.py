"""Command line interface to create or display Icestupa class
"""

# External modules
import os, sys
import matplotlib.pyplot as plt

# Locals
dirname = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.append(dirname)
from src.models.icestupaClass import Icestupa
from src.utils.settings import config
from src.utils import setup_logger
import logging, coloredlogs


if __name__ == "__main__":

    # Main logger
    logger = logging.getLogger(__name__)
    logger.setLevel("WARNING")

    test = True
    # test = False

    # location="Schwarzsee 2019"
    # location="phortse20"
    location = "Guttannen 2020"
    # location = "Guttannen 2021"
    # location = "Gangles 2021"

    # Initialise icestupa object
    icestupa = Icestupa(location)

    if test:
        # Derive all the input parameters
        icestupa.derive_parameters()

        # Generate results
        icestupa.melt_freeze(test)
        # icestupa.melt_freeze()

        # Summarise and save model results
        icestupa.save()

        # Create figures for web interface
        icestupa.summary_figures()
    else:
        # Use output parameters from cache
        icestupa.read_output()

        print(icestupa.df.Qfreeze.min())

        # plt.figure()
        # ax = plt.gca()
        # plt.scatter(icestupa.df.Qt, icestupa.df.fountain_froze / 60, s=1)
        # plt.legend()
        # plt.grid()
        # plt.savefig("data/tests/T_relation.jpg")

        icestupa.summary_figures()
