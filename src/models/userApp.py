"""Command line interface to create or display Icestupa class
"""

# External modules
import os, sys, shutil, time, argparse
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

def parse_args():
    parser = argparse.ArgumentParser(description="Command line interface to create or display Icestupa class")

    parser.add_argument("--location", required=True, help="Specify the location (e.g., leh20)")
    parser.add_argument("--spray", default="ERA5_", help="Specify the spray parameter (default: ERA5_)")

    # Add more arguments as needed, such as start date, end date, altitude, etc.

    return parser.parse_args()

def merge_with_settings(args):
    # Merge command-line arguments with settings file
    SITE, FOLDER = config(args.location, args.spray)

    # If the argument is not provided, use the value from the settings file
    # You can extend this logic for other parameters
    if not hasattr(args, 'start_date') or args.start_date is None:
        args.start_date = SITE.get('start_date', None)

    if not hasattr(args, 'end_date') or args.end_date is None:
        args.end_date = SITE.get('end_date', None)

    if not hasattr(args, 'altitude') or args.altitude is None:
        args.altitude = SITE.get('alt', None)

    return args, SITE, FOLDER


if __name__ == "__main__":

    # Main logger
    logger = logging.getLogger(__name__)
    logger.setLevel("ERROR")
    # logger.setLevel("WARNING")
    st = time.time()

    args = parse_args()

    # Merge command-line arguments with settings file
    args, SITE, FOLDER = merge_with_settings(args)

    # locations = ["north_america20"]
    locations = ["leh20"]

    # locations = ["central_asia20"]
    # locations = ["chuapalca20"]
    # locations = [ "north_america20", "europe20", "central_asia20","leh20", "south_america20"]
    # locations = [ "north_america20", "europe20", "central_asia20","leh20"]
    # locations = ["leh20", "south_america20"]

    spray = "ERA5_"

    # for location in locations:

    icestupa = Icestupa(args.location, args.spray)
    SITE, FOLDER = config(location)
    # icestupa.sim_air(test=False)
    icestupa.read_output()
    icestupa.summary_figures()
    # print(icestupa.df.LW.min())
    # get the end time
    et = time.time()

    # get the execution time
    elapsed_time = (et - st)/60
    print('\n\tExecution time:', round(elapsed_time,2), 'min\n')
