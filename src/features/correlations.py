"""Command line interface to create or display Icestupa class
"""

# External modules
import os, sys
import logging, coloredlogs

# Locals
dirname = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.append(dirname)
from src.models.icestupaClass import Icestupa
from src.utils.settings import config
from src.utils import setup_logger


if __name__ == "__main__":

    # Main logger
    logger = logging.getLogger(__name__)
    coloredlogs.install(
        fmt="%(funcName)s %(levelname)s %(message)s",
        # level=logging.WARNING,
        level=logging.INFO,
        logger=logger,
    )

    location = 'guttannen21'

    SITE, FOLDER = config(location)

    icestupa = Icestupa(location)
    icestupa.self_attributes()
    icestupa.read_output()
    column_1 = "r_ice"
    column_2 = "SA"
    correlation = icestupa.df[column_1].corr(icestupa.df[column_2])
    print("Correlation between %s and %s is %0.2f"%(column_1, column_2, correlation))
