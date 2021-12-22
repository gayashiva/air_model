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
    # logger.setLevel("INFO")

    test = True
    # test = False

    # location="Schwarzsee 2019"
    # location="phortse20"
    location = "Guttannen 2020"
    # location = "Guttannen 2021"
    # location = "Gangles 2021"

    # Initialise icestupa object
    icestupa = Icestupa(location)
    # icestupa.DX = 12e-03
    # print("DX change")

    if test:
        icestupa.gen_input()

        icestupa.sim_air(test)

        icestupa.gen_output()

        icestupa.summary_figures()
    else:
        icestupa.read_output()

        icestupa.summary_figures()
