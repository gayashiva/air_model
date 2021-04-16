# External modules
import os, sys, time
import logging
import coloredlogs
import multiprocessing
from multiprocessing import Pool
import time

# Locals
dirname = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.append(dirname)
from src.models.icestupaClass import Icestupa
from src.data.settings import config

locations = ["Guttannen 2021", "Guttannen 2020", "Schwarzsee 2019"]
triggers = ["Manual", "None", "Temperature", "Weather"]


def work_log(location, trigger):
    # Initialise icestupa object
    icestupa = Icestupa(location, trigger)
    # Derive all the input parameters
    icestupa.derive_parameters()

    # Generate results
    icestupa.melt_freeze()

    # Summarise and save model results
    icestupa.summary()

    # Create figures for web interface
    icestupa.summary_figures()


def pool_handler():
    logger.info("CPUs running %s" % multiprocessing.cpu_count())
    p = Pool(multiprocessing.cpu_count())
    for location in locations:
        for trigger in triggers:
            p.apply(work_log, [location, trigger])


if __name__ == "__main__":
    # Main logger
    logger = logging.getLogger(__name__)
    coloredlogs.install(
        fmt="%(funcName)s %(levelname)s %(message)s",
        level=logging.INFO,
        logger=logger,
    )
    pool_handler()
