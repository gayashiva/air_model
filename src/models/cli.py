import os, sys, time
from src.models.air import Icestupa
from src.data.settings import config

import logging
import coloredlogs

if __name__ == "__main__":
    # Initialise logger
    logger = logging.getLogger(__name__)
    coloredlogs.install(
        fmt="%(funcName)s %(levelname)s %(message)s",
        level=logging.INFO,
        logger=logger,
    )

    start = time.time()

    SITE, FOUNTAIN, FOLDER = config("Guttannen")

    icestupa = Icestupa(SITE, FOUNTAIN, FOLDER)

    icestupa.derive_parameters()

    # icestupa.read_input()

    icestupa.melt_freeze()

    # icestupa.read_output()

    # icestupa.corr_plot()

    icestupa.summary()

    # icestupa.print_input()

    # icestupa.summary_figures()

    # icestupa.print_output()

    total = time.time() - start

    logger.debug("Total time  : %.2f", total / 60)
