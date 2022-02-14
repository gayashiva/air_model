"""Command line interface to create or display Icestupa class
"""

# External modules
import os, sys
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd

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
    # location = "Guttannen 2020"
    # location = "Guttannen 2021"
    location = "Guttannen 2022"
    # location = "Gangles 2021"

    # Initialise icestupa object
    # icestupa = Icestupa(location, spray="auto")
    icestupa = Icestupa(location, spray="man")

    if test:
        icestupa.gen_input()

        icestupa.sim_air(test)

        icestupa.gen_output()

        icestupa.summary_figures()
    else:
        icestupa.read_output()
        df = icestupa.df


        # fig, ax = plt.subplots()
        # x1 = df.time
        # y1 = df.Discharge
        # # x2 = dfr.time
        # # y2 = dfr.rad
        # ax.plot(x1,y1)
        # # ax.scatter(x2,y2)

        # ax.xaxis.set_major_locator(mdates.WeekdayLocator())
        # ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
        # ax.xaxis.set_minor_locator(mdates.DayLocator())
        # fig.autofmt_xdate()
        # plt.savefig(
        #     icestupa.fig + icestupa.spray + "/discharge.jpg",
        #     bbox_inches="tight",
        # )
        # plt.clf()

        icestupa.summary_figures()
