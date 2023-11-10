"""Command line interface to create or display Icestupa class
"""

# External modules
import os, sys, shutil, time
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd

# Locals
dirname = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.append(dirname)
from src.models.icestupaClass import Icestupa
from src.utils.settings import config
from src.utils import setup_logger
from src.utils.eff_criterion import nse
import logging, coloredlogs


if __name__ == "__main__":

    # Main logger
    logger = logging.getLogger(__name__)
    logger.setLevel("ERROR")
    # logger.setLevel("WARNING")
    st = time.time()

    # locations = ["north_america20"]
    locations = ["leh20"]

    # locations = ["central_asia20"]
    # locations = ["chuapalca20"]
    # locations = [ "north_america20", "europe20", "central_asia20","leh20", "south_america20"]
    locations = [ "north_america20", "europe20", "central_asia20","leh20"]
    # locations = ["north_america20"]

    spray = "ERA5_"

    for location in locations:

        icestupa = Icestupa(location, spray)
        SITE, FOLDER = config(location)
        icestupa.sim_air(test=False)
        icestupa.summary_figures()
        # icestupa.read_output()
        # print(icestupa.df.LW.min())
        # get the end time
        et = time.time()

        # get the execution time
        elapsed_time = (et - st)/60
        print('Execution time:', round(elapsed_time,2), 'min')
