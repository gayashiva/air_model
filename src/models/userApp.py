"""Command line interface to create or display Icestupa class
"""

# External modules
import os, sys
import logging
import coloredlogs

# Locals
dirname = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.append(dirname)
from src.models.icestupaClass import Icestupa
from src.utils.settings import config


if __name__ == "__main__":

    # Main logger
    logger = logging.getLogger(__name__)
    coloredlogs.install(
        fmt="%(funcName)s %(levelname)s %(message)s",
        level=logging.WARNING,
        # level=logging.INFO,
        logger=logger,
    )

    answers = dict(
        # location="Schwarzsee 2019",
        location="Guttannen 2021",
        # location="Gangles 2021",
        # location="Diavolezza 2021",
        # location="Ravat 2020",
        # run="yes",
        run="no",
    )

    # Initialise icestupa object
    icestupa = Icestupa(answers["location"])

    if answers["run"] == "yes":
        # Derive all the input parameters
        icestupa.derive_parameters()
        # icestupa.read_input()

        # Generate results
        icestupa.melt_freeze(test=True)
        # icestupa.melt_freeze()

        # Summarise and save model results
        icestupa.summary()

        # Create figures for web interface
        icestupa.summary_figures()
    else:
        # Use output parameters from cache
        # icestupa.derive_parameters()
        # icestupa.read_input()
        # icestupa.melt_freeze()
        # icestupa.summary()
        icestupa.read_output()
        icestupa.summary_figures()
