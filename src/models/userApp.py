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

    answers = dict(
        # location="Schwarzsee 2019",
        location="guttannen21",
        # location="Gangles 2021",
        run="yes",
        # run="no",
    )

    # Initialise icestupa object
    # icestupa = Icestupa(answers["location"], params='best')
    icestupa = Icestupa(answers["location"])

    if answers["run"] == "yes":
        # Derive all the input parameters
        icestupa.derive_parameters()

        # Generate results
        icestupa.melt_freeze(test=True)
        # icestupa.melt_freeze()

        # Summarise and save model results
        icestupa.save()

        # Create figures for web interface
        icestupa.summary_figures()
    else:
        # Use output parameters from cache
        icestupa.read_output()
        # icestupa.summary_figures()

    # for location in ["gangles21", "guttannen21", "guttannen20",
    #     ds = icestupa.df.rename(columns={"TIMESTAMP": "time"})
    #     ds = ds.set_index("time")
    #     ds = ds.to_xarray()
    #     ds.coords["locs"] = icestupa.name
    #     # ds.to_netcdf(FOLDER["input"] + SITE["name"] + "_input_field.nc")
    #     print(ds)
