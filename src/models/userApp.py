"""Command line interface to create or display Icestupa class
"""

# External modules
import os, sys, time
import logging
import coloredlogs
import inquirer

# Locals
dirname = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.append(dirname)
from src.models.icestupaClass import Icestupa
from src.data.settings import config


if __name__ == "__main__":

    # Main logger
    logger = logging.getLogger(__name__)
    coloredlogs.install(
        fmt="%(funcName)s %(levelname)s %(message)s",
        level=logging.INFO,
        logger=logger,
    )

    # q = [
    #     inquirer.List(
    #         "location",
    #         message="Where is the Icestupa?",
    #         choices=["Guttannen 2020", "Guttannen 2021", "Schwarzsee 2019"],
    #         default="Guttannen 2020",
    #     ),
    #     inquirer.List(
    #         "trigger",
    #         message="How is fountain switched on?",
    #         choices=["None", "Manual", "Temperature", "Weather"],
    #         default="None",
    #     ),
    #     inquirer.List(
    #         "run", message="Regenerate results?", choices=["yes", "no"], default="yes"
    #     ),
    # ]

    # answers = inquirer.prompt(q)

    answers = dict(
        location="Schwarzsee 2019",
        # location="Guttannen 2021",
        # location="Gangles 2021",
        trigger="Manual",
        # trigger="None",
        # trigger="Temperature",
        # trigger="Weather",
        run="yes",
    )

    # Get settings for given location and trigger
    SITE, FOUNTAIN, FOLDER = config(answers["location"], answers["trigger"])

    # Initialise icestupa object
    icestupa = Icestupa(SITE, FOUNTAIN, FOLDER)

    if answers["run"] == "yes":
        # Derive all the input parameters
        icestupa.derive_parameters()

        # Generate results
        icestupa.melt_freeze()

        # Summarise and save model results
        icestupa.summary()

        # Create figures for web interface
        icestupa.summary_figures()
    else:
        # Use output parameters from cache
        icestupa.read_output()
        icestupa.summary_figures()
