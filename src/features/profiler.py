"""Outputs time taken by each function call of class
"""

# External modules
import os, sys
import logging
import coloredlogs
import cProfile
import pstats

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

    location = 'schwarzsee19'

    # Initialise icestupa object
    icestupa = Icestupa(location)
    # Derive all the input parameters
    icestupa.derive_parameters()
    with cProfile.Profile() as pr:
        # Generate results
        icestupa.melt_freeze()
    # # Summarise and save model results
    # icestupa.summary()
    # # Create figures for web interface
    # icestupa.summary_figures()

    stats = pstats.Stats(pr)
    stats.sort_stats(pstats.SortKey.TIME)
    stats.print_stats("get_temp")
