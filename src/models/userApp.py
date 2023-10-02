"""Command line interface to create or display Icestupa class
"""

# External modules
import os, sys, shutil, time
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
# from sklearn.metrics import mean_squared_error


# Locals
dirname = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.append(dirname)
from src.models.icestupaClass import Icestupa
from src.utils.settings import config
from src.utils import setup_logger
from src.utils.eff_criterion import nse
import logging, coloredlogs
from src.plots.data import plot_input


if __name__ == "__main__":

    # Main logger
    logger = logging.getLogger(__name__)
    # logger.setLevel("ERROR")
    logger.setLevel("WARNING")
    st = time.time()

    locations = ["south_america20"]

    # locations = ["central_asia20"]
    # locations = ["chuapalca20"]
    # locations = ["candarave20"]
    # locations = [ "north_america20", "europe20", "central_asia20","leh20", "south_america20"]

    spray = "ERA5_"

    for location in locations:

        icestupa = Icestupa(location, spray)
        SITE, FOLDER = config(location)
        icestupa.sim_air()
        # icestupa.gen_output()
        # icestupa.read_output()
        # icestupa.summary_figures()
        # get the end time
        et = time.time()

        # get the execution time
        elapsed_time = (et - st)/60
        print('Execution time:', round(elapsed_time,2), 'min')
